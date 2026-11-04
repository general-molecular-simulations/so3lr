import numpy as np

from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse, model_dtype
from so3lr.model_registry import resolve_model


def make_ase_calculator(
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        calculate_stress=False,
        calculate_hessian=False,
        dtype=None,
        model=None,
        charge_embed_total_charge=None,
        charge_equilibration_total_charge=None,
        precision=None,
        **kwargs
):
    """precision: float32 or float64 throughout, or mixed, a float32 model around float64
    positions and results (the default; float32 without x64). dtype: the older spelling of the
    model's dtype (default float32)."""
    # ``workdir`` (legacy kwarg) takes precedence over ``model``; either may be a
    # bundled model name or a filesystem path. None -> DEFAULT_MODEL.
    workdir = kwargs.pop('workdir', None)
    spec = resolve_model(workdir if workdir is not None else model)

    calc = AseCalculatorSparse.create_from_workdir(
        workdir=spec.dir,
        lr_cutoff=lr_cutoff,
        dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
        from_file=kwargs.pop('from_file', spec.from_file),
        calculate_stress=calculate_stress,
        # Was silently dropped: accepted here but never forwarded, so
        # So3lrCalculator(calculate_hessian=True) built a calculator that could
        # not produce a hessian even though 'hessian' is in implemented_properties.
        calculate_hessian=calculate_hessian,
        dtype=model_dtype(precision, dtype, np.float32),
        precision=precision,
        head_index=spec.head_index,
        charge_embed_total_charge=charge_embed_total_charge,
        charge_equilibration_total_charge=charge_equilibration_total_charge,
        **kwargs
    )

    return calc
