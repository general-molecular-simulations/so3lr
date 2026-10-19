"""An ASE calculator on the flash engine: energy and forces of so3lr-tauri-s/m/l.

    from so3lr import FlashCalculator
    atoms.calc = FlashCalculator(model="so3lr-tauri-m", kspace_electrostatics="pme")
    atoms.get_potential_energy(), atoms.get_forces()

For periodic orthorhombic cells at least twice the long-range cutoff across, on an NVIDIA
GPU (``check.flash_problem`` with task "ase"); anything else raises, naming the reason
(``so3lr.So3lrCalculator``, the neighbour-list model, takes any structure). The engine is
built at the first call and kept while the atoms, charge and spin stay the same: a new
cell is laid out again only where it leaves the cell grid's margin or gets another PME
mesh, and other atoms build a new engine (and compile). The charge and spin come from
``atoms.info`` as for ``So3lrCalculator`` (``charge``, default 0; ``multiplicity`` - 1,
default 0). No stress: the engine's strain derivative is isotropic only.
"""
import gc

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from so3lr.mlff.nn.observable.dispersion_ref_data import (
    check_atomic_numbers_supported, supported_atomic_numbers_for_model)

from .check import NPT_BOX_MARGIN, flash_problem, validate_options


class FlashCalculator(Calculator):
    """Energy and forces from SO3LR-Flash. model: a bundled name or a model directory (None:
    the default model); precision float32 (positions in float64 with flash_mixed, which
    needs jax_enable_x64) or float64 (needs jax_enable_x64); k-space: PME or none, for
    electrostatics and dispersion (spacing and smearing in A, None: so3lr's defaults);
    flash_mode: a memory mode of ``engine.MODES`` or auto (the fastest that fits);
    flash_devices: GPUs of this node to split the atoms over; matmul_precision: what the
    engine's matrix products run in (highest: full float32, as so3lr's MD runs them)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, model=None, *, lr_cutoff=12.0, dispersion_damping=2.0, precision="float32",
                 kspace_electrostatics=None, kspace_smearing=None, kspace_spacing=None,
                 kspace_dispersion=None, kspace_dispersion_smearing=None, kspace_dispersion_spacing=None,
                 charge_embed_total_charge=None, charge_equilibration_total_charge=None,
                 flash_mode="auto", flash_mixed=True, flash_devices=1, matmul_precision="highest", **kwargs):
        super().__init__(**kwargs)
        if precision not in ("float32", "float64"):
            raise ValueError(f"precision must be float32 or float64 (received {precision!r})")
        if precision == "float64" and not jax.config.jax_enable_x64:
            raise ValueError("precision float64 needs jax.config.update('jax_enable_x64', True) first")
        self.settings = dict(
            model_path=model, lr_cutoff=lr_cutoff, dispersion_damping=dispersion_damping, precision=precision,
            kspace_electrostatics=kspace_electrostatics, kspace_smearing=kspace_smearing,
            kspace_spacing=kspace_spacing, kspace_dispersion=kspace_dispersion,
            kspace_dispersion_smearing=kspace_dispersion_smearing,
            kspace_dispersion_spacing=kspace_dispersion_spacing,
            charge_embed_total_charge=charge_embed_total_charge,
            charge_equilibration_total_charge=charge_equilibration_total_charge,
            engine="flash", flash_mode=flash_mode, flash_mixed=flash_mixed, flash_devices=flash_devices)
        validate_options(self.settings)
        self.dtype = jnp.float64 if precision == "float64" else jnp.float32
        mixed = self.dtype == jnp.float32 and flash_mixed and jax.config.jax_enable_x64
        self.position_dtype = jnp.float64 if mixed else self.dtype
        self.matmul_precision = matmul_precision
        self.supported = supported_atomic_numbers_for_model(model)
        self.engine, self._ef, self._key = None, None, None

    def _system(self, atoms):
        """What the engine is built for: the atoms, charge and spin."""
        charge = float(atoms.info.get("charge", 0))
        multiplicity = atoms.info.get("multiplicity")
        spin = 0.0 if multiplicity is None else float(multiplicity) - 1.0
        return atoms.numbers.tobytes(), charge, spin

    def _build(self, atoms, charge, spin):
        from .md import build_engine
        problem = flash_problem(self.settings, atoms, task="ase")
        if problem:
            raise ValueError(f"FlashCalculator: flash cannot take this structure: {problem}")
        check_atomic_numbers_supported(atoms.numbers, self.supported, context="FlashCalculator")
        self.engine = self._ef = None
        gc.collect()                         # the last engine's arrays, before the next one's
        with jax.default_matmul_precision(self.matmul_precision):
            self.engine, staged = build_engine(
                atoms, dict(self.settings, total_charge=charge, num_unpaired_electrons=spin), self.dtype,
                box_margin=NPT_BOX_MARGIN)
        self._ef = self.engine.energy_forces_staged if staged else self.engine.energy_forces

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        key = self._system(atoms)
        if key != self._key:
            self._build(atoms, *key[1:])
            self._key = key
        elif "cell" in system_changes or "pbc" in system_changes:
            problem = flash_problem(self.settings, atoms, task="ase")
            if problem:
                raise ValueError(f"FlashCalculator: flash cannot take this structure: {problem}")
            self.engine.set_box(np.diag(np.asarray(atoms.cell[:])), exact_mesh=True)
        R = jnp.asarray(atoms.get_positions(wrap=True), self.position_dtype)
        with jax.default_matmul_precision(self.matmul_precision):
            e, F = self._ef(R)
        e = float(e)
        self.results = {"energy": e, "free_energy": e, "forces": np.asarray(F, np.float64)}
