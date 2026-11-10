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
  or a CPU);
* neighbour lists (``to_jax_md``): one compiled step that updates the lists on the device;
  they are allocated again here when they overflow. Orthorhombic boxes at least twice the
  cutoffs (plus the skin) across, or no box;
* exact images (``So3lrCalculator`` with ``neighbor_backend="exact"``): any box, triclinic or
  thinner than that; its neighbour list is built on the CPU at every call.

``engine``: auto (that order), flash (raise where flash cannot run), or standard (never flash),
as for ``so3lr md`` and ``so3lr eval``.
"""
from __future__ import annotations

import functools

import numpy as np

from so3lr.flash.check import ENGINES

KSPACE_METHODS = ("ewald", "pme")


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
        from so3lr.md_settings_validation import available_precision, model_precision, state_precision
        from so3lr.mlff.nn.observable.dispersion_ref_data import (
            check_atomic_numbers_supported, supported_atomic_numbers_for_model)

        self._kwargs = dict(model=model, charge=charge, multiplicity=multiplicity, lr_cutoff=lr_cutoff,
                            dispersion_damping=dispersion_damping, precision=precision,
                            kspace_electrostatics=kspace_electrostatics, kspace_smearing=kspace_smearing,
                            kspace_spacing=kspace_spacing, engine=engine, skin=skin,
                            matmul_precision=matmul_precision)
        self.numbers = np.asarray(numbers, dtype=np.int32).reshape(-1)
        check_atomic_numbers_supported(self.numbers, supported_atomic_numbers_for_model(model),
                                       context=f"SO3LR model {model}")
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {', '.join(ENGINES)} (received {engine!r})")
        if kspace_electrostatics not in (None, *KSPACE_METHODS):
            raise ValueError(f"kspace_electrostatics must be None, ewald or pme (received {kspace_electrostatics!r})")
        if kspace_electrostatics is None and (kspace_smearing is not None or kspace_spacing is not None):
            raise ValueError("kspace_smearing and kspace_spacing need kspace_electrostatics")
        if kspace_electrostatics is not None:
            from so3lr.kspace_defaults import DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING
            kspace_smearing = DEFAULT_KSPACE_SMEARING if kspace_smearing is None else float(kspace_smearing)
            kspace_spacing = DEFAULT_KSPACE_SPACING if kspace_spacing is None else float(kspace_spacing)
        self.model, self.engine = model, engine
        self.charge, self.multiplicity = float(charge), int(multiplicity)
        self.lr_cutoff, self.dispersion_damping, self.skin = float(lr_cutoff), float(dispersion_damping), float(skin)
        self.kspace = None if kspace_electrostatics is None else (kspace_electrostatics, kspace_smearing,
                                                                  kspace_spacing)
        self.precision = available_precision(dict(precision=precision))
        self.model_dtype = np.dtype(model_precision(self.precision))
        self.dtype = np.dtype(state_precision(self.precision))
        self.matmul_precision = matmul_precision
        self.backend = None                 # the way the last call took: flash, neighbour-list or exact
        self._routes = {}                   # box bytes -> way
        self._flash = self._potential = self._step = self._atoms = None
        self._k_box, self._k_args = None, {}

    def __reduce__(self):
        return functools.partial(So3lrEngine, **self._kwargs), (self.numbers,)

    def __call__(self, positions, box=None):
        """(energy in eV, forces in eV/A as an [N, 3] float64 array) at positions (A, [N, 3]) in
        box (lattice vectors as rows, A, [3, 3]; None: free space)."""
        import jax

        positions = np.asarray(positions, np.float64).reshape(len(self.numbers), 3)
        box = None if box is None else np.asarray(box, np.float64).reshape(3, 3)
        self.backend = self._route(positions, box)
        with jax.default_matmul_precision(self.matmul_precision):
            if self.backend == "flash":
                return self._call_flash(positions, box)
            if self.backend == "exact":
                return self._call_exact(positions, box)
            return self._call_lists(positions, box)

    # ---- which way ------------------------------------------------------------------
    def _route(self, positions, box):
        key = None if box is None else box.tobytes()
        if key not in self._routes:
            way = None
            if self.engine != "standard":
                from so3lr.flash.check import flash_problem
                problem = flash_problem(self._flash_settings(), self._ase_atoms(positions, box), task="ase")
                if problem is None:
                    way = "flash"
                elif self.engine == "flash":
                    raise ValueError(f"flash: {problem}")
            if way is None:
                way = "neighbour-list" if box is None or self._minimum_image_works(box) else "exact"
            self._routes[key] = way
        return self._routes[key]

    def _minimum_image_works(self, box):
        from so3lr.pbc import minimum_image_problem
        # the lists are built at both cutoffs (and the skin), so the larger one counts
        cutoff = max(self._model().cutoff, self.lr_cutoff) + self.skin
        orthorhombic = np.allclose(box - np.diag(np.diag(box)), 0.0, atol=1e-6)
        return orthorhombic and minimum_image_problem(box, cutoff) is None

    def _ase_atoms(self, positions, box):
        import ase
        atoms = ase.Atoms(numbers=self.numbers, positions=positions, cell=box, pbc=box is not None)
        atoms.info.update(charge=self.charge, multiplicity=self.multiplicity)
        return atoms

    # ---- flash ----------------------------------------------------------------------
    def _flash_settings(self):
        settings = dict(model_path=self.model, lr_cutoff=self.lr_cutoff, dispersion_damping=self.dispersion_damping,
                        precision=self.precision, engine=self.engine)
        if self.kspace is not None:
            method, smearing, spacing = self.kspace
            settings.update(kspace_electrostatics=method, kspace_smearing=smearing, kspace_spacing=spacing)
        return settings

    def _call_flash(self, positions, box):
        if self._flash is None:
            from so3lr.flash.check import NPT_BOX_MARGIN
            from so3lr.flash.md import SinglePoints
            # the cell grid laid out with NPT's margin, so a box that moves a little compiles nothing
            self._flash = SinglePoints(self._flash_settings(), task="ase", matmul_precision=self.matmul_precision,
                                       box_margin=NPT_BOX_MARGIN)
        energy, forces, _ = self._flash(self._ase_atoms(positions, box), self.charge, self.multiplicity - 1.0)
        return energy, forces

    # ---- neighbour lists ------------------------------------------------------------
    def _model(self):
        """so3lr's potential, loaded at first use."""
        if self._potential is None:
            from so3lr import So3lrPotential
            self._potential = So3lrPotential(
                model=self.model, lr_cutoff=self.lr_cutoff, dispersion_energy_cutoff_lr_damping=self.dispersion_damping,
                dtype=self.model_dtype, coulomb_kspace_method=self.kspace[0] if self.kspace else "pme")
        return self._potential

    def _coordinates(self, positions, lengths):
        """Fractional coordinates in a diagonal box matrix, or Cartesian ones without a box."""
        import jax.numpy as jnp
        positions = jnp.asarray(positions, self.dtype)
        if lengths is None:
            return positions, {}
        lengths = jnp.asarray(lengths, self.dtype)
        return jnp.mod(positions / lengths, 1.0), {"box": jnp.diag(lengths)}

    def _call_lists(self, positions, box):
        import jax

        lengths = None if box is None else box.diagonal().copy()
        kwargs = self._k_grid(lengths) if self.kspace is not None and lengths is not None else {}
        if self._step is None or self._step_periodic != (lengths is not None):
            self._build_lists(positions, lengths)
        energy, forces, overflow, self._nbrs, self._nbrs_lr = self._step(
            positions, lengths, self._nbrs, self._nbrs_lr, **kwargs)
        energy, forces, overflow = jax.device_get((energy, forces, overflow))
        if overflow:
            # a list overflowed, or the box shrank below the cell list's cells: the lists
            # allocated again for this box, and the step again (it compiles only for list
            # sizes it has not seen)
            self._allocate(positions, lengths)
            energy, forces, _, self._nbrs, self._nbrs_lr = self._step(
                positions, lengths, self._nbrs, self._nbrs_lr, **kwargs)
            energy, forces = jax.device_get((energy, forces))
        return float(energy), np.asarray(forces, np.float64)

    def _build_lists(self, positions, lengths):
        import jax
        import jax.numpy as jnp
        from jax_md import space

        from so3lr import to_jax_md

        R, box = self._coordinates(positions, lengths)
        if lengths is None:
            displacement, _ = space.free()
            options = dict(box_size=None, disable_cell_list=True, fractional_coordinates=False)
        else:
            # a diagonal box matrix: every jax-md version takes it
            displacement, _ = space.periodic_general(box["box"], fractional_coordinates=True)
            options = dict(box_size=box["box"], fractional_coordinates=True)
        self._neighbor_fn, self._neighbor_fn_lr, energy_fn = to_jax_md(
            self._model(), displacement, species=jnp.asarray(self.numbers), dr_threshold=self.skin,
            total_charge=self.charge, num_unpaired_electrons=float(self.multiplicity - 1),
            precision=self.model_dtype, **options)

        def step(positions, lengths, nbrs, nbrs_lr, **kwargs):
            R, box = self._coordinates(positions, lengths)
            nbrs = nbrs.update(R, **box)
            nbrs_lr = nbrs_lr.update(R, **box)
            energy, gradient = jax.value_and_grad(energy_fn)(R, nbrs.idx, nbrs_lr.idx, **box, **kwargs)
            # not error.code as a whole: some jax-md versions flag every box matrix as malformed
            overflow = (nbrs.did_buffer_overflow | nbrs.cell_size_too_small
                        | nbrs_lr.did_buffer_overflow | nbrs_lr.cell_size_too_small)
            # in jax-md, the gradient with respect to fractional coordinates is the Cartesian one
            return energy, -gradient, overflow, nbrs, nbrs_lr

        self._step = jax.jit(step, donate_argnums=(2, 3))
        self._step_periodic = lengths is not None
        self._allocate(positions, lengths)

    def _allocate(self, positions, lengths):
        R, box = self._coordinates(positions, lengths)
        self._nbrs = self._neighbor_fn.allocate(R, **box)
        self._nbrs_lr = self._neighbor_fn_lr.allocate(R, **box)

    def _k_grid(self, lengths):
        """The k-space grid for this box, set up again only when the box changes."""
        import jax.numpy as jnp
        from so3lr.mlff.utils.kspace_utils import setup_kspace_grid

        key = lengths.tobytes()
        if key != self._k_box:
            method, smearing, spacing = self.kspace
            k_grid, k_smearing = setup_kspace_grid(method, smearing, spacing, jnp.asarray(lengths, self.dtype),
                                                   k_grid=self._k_args.get("k_grid"), dtype=self.dtype)
            self._k_box, self._k_args = key, {"k_grid": k_grid, "k_smearing": k_smearing}
        return self._k_args

    # ---- exact images ---------------------------------------------------------------
    def _call_exact(self, positions, box):
        if self._atoms is None:
            from so3lr import So3lrCalculator
            args = dict(model=self.model, lr_cutoff=self.lr_cutoff,
                        dispersion_energy_cutoff_lr_damping=self.dispersion_damping,
                        neighbor_backend="exact", precision=self.precision)
            if self.kspace is not None:
                method, smearing, spacing = self.kspace
                args.update(kspace_electrostatics=method, kspace_smearing=smearing, kspace_spacing=spacing)
            self._atoms = self._ase_atoms(positions, box)
            self._atoms.calc = So3lrCalculator(**args)
        self._atoms.set_cell(box)
        self._atoms.set_positions(positions)
        return float(self._atoms.get_potential_energy()), np.asarray(self._atoms.get_forces(), np.float64)
