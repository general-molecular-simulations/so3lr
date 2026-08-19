import numpy as np

from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse
from so3lr.model_registry import resolve_model


def make_ase_calculator(
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        calculate_stress=False,
        calculate_hessian=False,
        dtype=np.float32,
        model=None,
        charge_embed_total_charge=None,
        **kwargs
):
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
        dtype=dtype,
        head_index=spec.head_index,
        charge_embed_total_charge=charge_embed_total_charge,
        **kwargs
    )

    return calc
