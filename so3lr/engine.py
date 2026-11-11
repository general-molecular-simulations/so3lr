"""Energy and forces of one set of atoms, call after call: the force callback an MD code
needs (OpenMM, i-PI, LAMMPS, a loop of your own).

    import jax
    jax.config.update("jax_enable_x64", True)     # precision mixed (the default) and float64 need it
    from so3lr import So3lrEngine
    engine = So3lrEngine(numbers, model="so3lr-tauri-m", charge=0, multiplicity=1)
    energy, forces = engine(positions, box)       # A, lattice vectors as rows in A (None: no box)
                                                  # -> eV, eV/A

The atoms, charge and spin are fixed; positions and box change from call to call. For each
box the engine takes the first of so3lr's three ways that can compute it:

* flash (``so3lr.flash``): GPU kernels, for orthorhombic boxes at least twice the cutoffs
  across, or no box (``flash.check.flash_problem`` says where else it declines: e.g. Ewald,
  or a CPU). Once the box moves, the cell grid is laid out with NPT's margin, so boxes
  within a few per cent compile nothing (with PME, each box gets its own mesh, as on the
  other ways: a new mesh compiles again);
* neighbour lists (``to_jax_md``): one compiled step that updates the lists on the device;
  a list that overflows is allocated again here. Orthorhombic boxes at least twice the
  cutoffs (plus the skin) across, or no box;
* exact images (``So3lrCalculator`` with ``neighbor_backend="exact"``): any box, triclinic or
  thinner than that; its neighbour list is built on the CPU at every call.

``engine``: auto (that order), flash (raise where flash cannot run), or standard (never flash),
as for ``so3lr md`` and ``so3lr eval``: auto also falls back where flash does not import or
fails to start.
"""
from __future__ import annotations

import functools
import gc
import warnings

import jax
import jax.numpy as jnp
import numpy as np

_UNSET = object()
OVERFLOW_EXTRA_SR, OVERFLOW_EXTRA_LR = 2, 8     # as so3lr md's (so3lr.cli.so3lr_md, a heavier import)


def _coordinates(positions, lengths, dtype):
    """Fractional coordinates in a diagonal box matrix, or Cartesian ones without a box."""
    positions = jnp.asarray(positions, dtype)
    if lengths is None:
        return positions, {}
    lengths = jnp.asarray(lengths, dtype)
    return jnp.mod(positions / lengths, 1.0), {"box": jnp.diag(lengths)}


class So3lrEngine:
    """Energy (eV) and forces (eV/A) of atoms with atomic numbers ``numbers``.

    model: a bundled name or a model directory (None: the default model); charge: the total
    charge (e); multiplicity: 2S + 1; lr_cutoff: the long-range cutoff (A); dispersion_damping:
    the long-range damping of the dispersion (A); precision: float32 or float64 throughout, or
    mixed, a float32 model around float64 positions and sums (None: mixed with x64, float32
    without; mixed and float64 need x64); kspace_electrostatics: ewald or pme for periodic
    electrostatics in k-space (None: in real space up to lr_cutoff), with kspace_smearing and
    kspace_spacing in A (None: so3lr's defaults); engine: auto, flash or standard (above);
    skin: the neighbour lists' skin (A; they are updated at every call either way);
    matmul_precision: what the model's matrix products run in (highest: full float32).

    Pickles as its arguments: an unpickled engine compiles again at its first call.
    """

    def __init__(self, numbers, model=None, *, charge=0.0, multiplicity=1, lr_cutoff=12.0,
                 dispersion_damping=2.0, precision=None, kspace_electrostatics=None,
                 kspace_smearing=None, kspace_spacing=None, engine="auto", skin=0.0,
                 matmul_precision="highest"):
        from so3lr.flash.check import model_cutoff, validate_options
        from so3lr.md_settings_validation import available_precision, model_precision, state_precision
        from so3lr.mlff.nn.observable.dispersion_ref_data import (
            check_atomic_numbers_supported, supported_atomic_numbers_for_model)

        # the arguments as given, for pickling: precision is resolved again where it is unpickled
        self._kwargs = dict(model=model, charge=charge, multiplicity=multiplicity, lr_cutoff=lr_cutoff,
                            dispersion_damping=dispersion_damping, precision=precision,
                            kspace_electrostatics=kspace_electrostatics, kspace_smearing=kspace_smearing,
                            kspace_spacing=kspace_spacing, engine=engine, skin=skin,
                            matmul_precision=matmul_precision)
        self.numbers = np.asarray(numbers, dtype=np.int32).reshape(-1)
        check_atomic_numbers_supported(self.numbers, supported_atomic_numbers_for_model(model),
                                       context=f"SO3LR model {model}")
        validate_options(dict(engine=engine))
        if kspace_electrostatics not in (None, "ewald", "pme", "p3m"):
            raise ValueError(f"kspace_electrostatics must be None, ewald, pme or p3m (received {kspace_electrostatics!r})")
        if kspace_electrostatics is None and (kspace_smearing is not None or kspace_spacing is not None):
            raise ValueError("kspace_smearing and kspace_spacing need kspace_electrostatics")
        self._kspace = {}               # the k-space settings, as so3lr's settings and calculator name them
        if kspace_electrostatics is not None:
            from so3lr.kspace_defaults import DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING
            self._kspace = dict(
                kspace_electrostatics=kspace_electrostatics,
                kspace_smearing=DEFAULT_KSPACE_SMEARING if kspace_smearing is None else float(kspace_smearing),
                kspace_spacing=DEFAULT_KSPACE_SPACING if kspace_spacing is None else float(kspace_spacing))
        self.model, self.engine = model, engine
        self.charge, self.multiplicity = float(charge), int(multiplicity)
        self.lr_cutoff, self.dispersion_damping, self.skin = float(lr_cutoff), float(dispersion_damping), float(skin)
        self.precision = available_precision(dict(precision=precision))
        self.model_dtype = np.dtype(model_precision(self.precision))
        self.dtype = np.dtype(state_precision(self.precision))
        self.matmul_precision = matmul_precision
        self._model_cutoff = model_cutoff(model)
        # the lists are built at both cutoffs (and the skin), so the larger one counts
        self._list_cutoff = max(self._model_cutoff, self.lr_cutoff) + self.skin
        self._flash_settings = dict(model_path=model, lr_cutoff=self.lr_cutoff,
                                    dispersion_damping=self.dispersion_damping, precision=self.precision,
                                    engine=engine, **self._kspace)
        self.backend = None                 # the way the last call took: flash, neighbour-list or exact
        self._last_box = _UNSET             # the box the way was chosen for
        self._flash_refusal = _UNSET        # why flash cannot run whatever the box (None: it can)
        self._flash = self._atoms = self._calc = self._step = self._step_periodic = None
        self._flash_box = _UNSET            # the box flash was built for
        self._neighbor_fn = self._neighbor_fn_lr = self._nbrs = self._nbrs_lr = None
        self._extra = np.zeros(2, int)      # the lists' extra capacity, grown at each overflow
        self._k_box, self._k_args = None, {}

    def __reduce__(self):
        return functools.partial(type(self), **self._kwargs), (self.numbers,)

    def __call__(self, positions, box=None):
        """(energy in eV, forces in eV/A as an [N, 3] float64 array) at positions (A, [N, 3]) in
        box (lattice vectors as rows, A, [3, 3]; None or all zeros: free space)."""
        positions = np.asarray(positions, np.float64).reshape(len(self.numbers), 3)
        if box is not None:
            box = np.asarray(box, np.float64).reshape(3, 3)
            if not box.any():
                box = None                  # ASE's cell for a molecule: free space
        key = None if box is None else box.tobytes()
        if key != self._last_box:           # under a barostat every call: only the last box kept
            if box is not None and abs(np.linalg.det(box)) < 1e-6:
                raise ValueError(f"So3lrEngine: the box spans no volume (lattice vectors {box.tolist()})")
            self.backend = self._route(positions, box)
            self._last_box = key
        with jax.default_matmul_precision(self.matmul_precision):
            if self.backend == "flash":
                result = self._call_flash(positions, box, key)
                if result is not None:
                    return result
            if self.backend == "exact":
                return self._call_exact(positions, box)
            return self._call_lists(positions, box)

    # ---- which way ------------------------------------------------------------------
    def _route(self, positions, box):
        if self.engine != "standard":
            problem = self._flash_problem(positions, box)
            if problem is None:
                return "flash"
            if self.engine == "flash":
                raise ValueError(f"flash: {problem}")
        return "neighbour-list" if box is None or self._minimum_image_works(box) else "exact"

    def _flash_problem(self, positions, box):
        """Why flash cannot take this box (None if it can): the settings, model, device and
        import checked once, the structure for each box."""
        from so3lr.flash.check import flash_problem, structure_problem
        if self._flash_refusal is _UNSET:
            problem = flash_problem(self._flash_settings, [], task="ase")
            if problem is None:
                try:
                    import so3lr.flash.md  # noqa: F401
                except Exception as exc:    # e.g. a JAX build without Pallas-Triton
                    problem = f"it does not import here ({type(exc).__name__}: {exc})"
            self._flash_refusal = problem
        return self._flash_refusal or structure_problem(
            self._flash_settings, self._ase_atoms(positions, box), self._model_cutoff)

    def _minimum_image_works(self, box):
        from so3lr.pbc import minimum_image_problem
        orthorhombic = np.allclose(box - np.diag(np.diag(box)), 0.0, atol=1e-6)
        return orthorhombic and minimum_image_problem(box, self._list_cutoff) is None

    def _ase_atoms(self, positions, box):
        """The atoms at positions in box: one ase.Atoms, updated."""
        if self._atoms is None:
            import ase
            self._atoms = ase.Atoms(numbers=self.numbers)
            self._atoms.info.update(charge=self.charge, multiplicity=self.multiplicity)
        self._atoms.pbc = box is not None
        self._atoms.set_cell(np.zeros((3, 3)) if box is None else box)
        self._atoms.set_positions(positions)
        return self._atoms

    # ---- flash ----------------------------------------------------------------------
    def _call_flash(self, positions, box, key):
        """Energy and forces on flash; None where flash failed to start (built, or built again
        for a moving box) under auto: the way is then chosen again without it, as ``so3lr md``
        falls back."""
        from so3lr.flash.check import NPT_BOX_MARGIN
        from so3lr.flash.md import SinglePoints
        if self._flash is not None and not self._flash.box_margin and key != self._flash_box:
            # the box moves (a barostat): built again once, with NPT's margin, so later boxes
            # within it compile nothing (a fixed box keeps every cell the margin would cost)
            self._flash = None
            gc.collect()
        starting, failed = self._flash is None, None
        try:
            if starting:
                margin = 0.0 if self._flash_box is _UNSET else NPT_BOX_MARGIN
                self._flash = SinglePoints(self._flash_settings, task="ase", matmul_precision=self.matmul_precision,
                                           box_margin=margin)
                self._flash_box = key
            energy, forces, _ = self._flash(self._ase_atoms(positions, box), self.charge, self.multiplicity - 1.0)
        except Exception as exc:
            if not starting or self.engine == "flash":
                raise
            failed = f"it failed to start ({type(exc).__name__}: {exc})"
        if failed is None:
            return energy, forces
        # out of the except block, so the traceback no longer holds the failed engine
        self._flash, self._flash_refusal = None, failed
        gc.collect()
        warnings.warn(f"So3lrEngine: flash {failed}; running without it", stacklevel=3)
        self.backend = self._route(positions, box)
        return None

    # ---- neighbour lists ------------------------------------------------------------
    def _call_lists(self, positions, box):
        lengths = None if box is None else box.diagonal().copy()
        kwargs = self._k_grid(lengths) if self._kspace and lengths is not None else {}
        if self._step_periodic != (lengths is not None):
            self._build_lists(lengths)
            self._nbrs = None
        if self._nbrs is None:
            self._allocate(positions, lengths, (True, True))
        for _ in range(2):
            try:
                energy, forces, overflow, self._nbrs, self._nbrs_lr = self._step(
                    positions, lengths, self._nbrs, self._nbrs_lr, **kwargs)
            except BaseException:
                self._nbrs = self._nbrs_lr = None   # donated to the step: allocated again next call
                raise
            energy, forces, overflow = jax.device_get((energy, forces, overflow))
            if not overflow.any():
                return float(energy), np.asarray(forces, np.float64)
            # a list overflowed, or the box shrank below its cells: that list allocated again
            # for this box with more room, as so3lr md does, and the step again (it compiles
            # again: a new list's cell-list functions are static, new objects)
            self._extra += np.where(overflow, (OVERFLOW_EXTRA_SR, OVERFLOW_EXTRA_LR), 0)
            self._allocate(positions, lengths, overflow)
        raise RuntimeError("So3lrEngine: the neighbour lists overflow right after their allocation")

    def _build_lists(self, lengths):
        from jax_md import space

        from so3lr import So3lrPotential, to_jax_md

        if lengths is None:
            displacement, _ = space.free()
            options = dict(box_size=None, disable_cell_list=True, fractional_coordinates=False)
        else:
            box = jnp.diag(jnp.asarray(lengths, self.dtype))
            displacement, _ = space.periodic_general(box, fractional_coordinates=True)
            options = dict(box_size=box, fractional_coordinates=True)
        potential = So3lrPotential(
            model=self.model, lr_cutoff=self.lr_cutoff, dispersion_energy_cutoff_lr_damping=self.dispersion_damping,
            dtype=self.model_dtype, coulomb_kspace_method=self._kspace.get("kspace_electrostatics", "pme"))
        self._neighbor_fn, self._neighbor_fn_lr, energy_fn = to_jax_md(
            potential, displacement, species=jnp.asarray(self.numbers), dr_threshold=self.skin,
            total_charge=self.charge, num_unpaired_electrons=float(self.multiplicity - 1),
            precision=self.model_dtype, **options)
        dtype = self.dtype

        def step(positions, lengths, nbrs, nbrs_lr, **kwargs):
            R, box = _coordinates(positions, lengths, dtype)
            nbrs = nbrs.update(R, **box)
            nbrs_lr = nbrs_lr.update(R, **box)
            energy, gradient = jax.value_and_grad(energy_fn)(R, nbrs.idx, nbrs_lr.idx, **box, **kwargs)
            overflow = jnp.stack([nbrs.did_buffer_overflow | nbrs.cell_size_too_small,
                                  nbrs_lr.did_buffer_overflow | nbrs_lr.cell_size_too_small])
            # in jax-md, the gradient with respect to fractional coordinates is the Cartesian one
            return energy, -gradient, overflow, nbrs, nbrs_lr

        self._step = jax.jit(step, donate_argnums=(2, 3))
        self._step_periodic = lengths is not None

    def _allocate(self, positions, lengths, which):
        """Allocate the short- and long-range lists where ``which`` says, for these positions."""
        R, box = _coordinates(positions, lengths, self.dtype)
        if which[0]:
            self._nbrs = self._neighbor_fn.allocate(R, extra_capacity=int(self._extra[0]), **box)
        if which[1]:
            self._nbrs_lr = self._neighbor_fn_lr.allocate(R, extra_capacity=int(self._extra[1]), **box)

    def _k_grid(self, lengths):
        """The k-space grid for this box, set up again only when the box changes."""
        from so3lr.mlff.utils.kspace_utils import setup_kspace_grid

        key = lengths.tobytes()
        if key != self._k_box:
            k_grid, k_smearing = setup_kspace_grid(
                self._kspace["kspace_electrostatics"], self._kspace["kspace_smearing"],
                self._kspace["kspace_spacing"], jnp.asarray(lengths, self.dtype),
                k_grid=self._k_args.get("k_grid"), dtype=self.dtype)
            self._k_box, self._k_args = key, {"k_grid": k_grid, "k_smearing": k_smearing}
        return self._k_args

    # ---- exact images ---------------------------------------------------------------
    def _call_exact(self, positions, box):
        if self._calc is None:
            from so3lr import So3lrCalculator
            self._calc = So3lrCalculator(model=self.model, lr_cutoff=self.lr_cutoff,
                                         dispersion_energy_cutoff_lr_damping=self.dispersion_damping,
                                         neighbor_backend="exact", precision=self.precision, **self._kspace)
        atoms = self._ase_atoms(positions, box)
        atoms.calc = self._calc
        return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces(), np.float64)
