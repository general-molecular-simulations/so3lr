"""An ASE calculator on the flash engine: energy and forces of so3lr-tauri-s/m/l.

    from so3lr import FlashCalculator
    atoms.calc = FlashCalculator(model="so3lr-tauri-m", kspace_electrostatics="pme")
    atoms.get_potential_energy(), atoms.get_forces()

For periodic orthorhombic cells at least twice the long-range cutoff across, on an NVIDIA
GPU (``check.flash_problem`` with task "ase"); anything else raises, naming the reason
(``so3lr.So3lrCalculator``, the neighbour-list model, takes any structure). The engine is
built at the first call and kept while the atoms, charge and spin stay the same
(``md.SinglePoints``): a new cell is laid out again (and compiles), and other atoms build a
new engine (and compile). The
charge and spin come from ``atoms.info`` as for ``So3lrCalculator`` (``charge``, default
0; ``multiplicity`` - 1, default 0). No stress: the engine's strain derivative is
isotropic only.
"""
import jax
from ase.calculators.calculator import Calculator, all_changes

from .check import validate_options


class FlashCalculator(Calculator):
    """Energy and forces from SO3LR-Flash. model: a bundled name or a model directory (None:
    the default model); precision float32 (positions in float64 with flash_mixed, which
    needs jax_enable_x64) or float64 (needs jax_enable_x64); k-space: PME or none, for
    electrostatics and dispersion (spacing and smearing in A, None: so3lr's defaults);
    flash_mode: a memory mode of ``engine.MODES`` or auto (the fastest that fits);
    flash_devices: GPUs of this node to split the atoms over; matmul_precision: what the
    engine's matrix products run in (highest: full float32, as so3lr's MD runs them).
    Raises at once if flash cannot take these settings or this model on this device."""

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
        settings = dict(
            model_path=model, lr_cutoff=lr_cutoff, dispersion_damping=dispersion_damping, precision=precision,
            kspace_electrostatics=kspace_electrostatics, kspace_smearing=kspace_smearing,
            kspace_spacing=kspace_spacing, kspace_dispersion=kspace_dispersion,
            kspace_dispersion_smearing=kspace_dispersion_smearing,
            kspace_dispersion_spacing=kspace_dispersion_spacing,
            charge_embed_total_charge=charge_embed_total_charge,
            charge_equilibration_total_charge=charge_equilibration_total_charge,
            engine="flash", flash_mode=flash_mode, flash_mixed=flash_mixed, flash_devices=flash_devices)
        validate_options(settings)
        from .md import SinglePoints
        # no grid margin: a cell that changes is laid out again (and compiles)
        self.points = SinglePoints(settings, task="ase", matmul_precision=matmul_precision)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        info = self.atoms.info
        multiplicity = info.get("multiplicity")
        e, F, _ = self.points(self.atoms, float(info.get("charge", 0)),
                              0.0 if multiplicity is None else float(multiplicity) - 1.0)
        self.results = {"energy": e, "free_energy": e, "forces": F}
