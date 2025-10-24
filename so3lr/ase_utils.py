import numpy as np
import pathlib

from mlff.md import mlffCalculatorSparse


def make_ase_calculator(
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        calculate_stress=False,
        calculate_hessian=False,
        dtype=np.float32,
        **kwargs
):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=package_dir / 'so3lr' / 'params',
        lr_cutoff=lr_cutoff,
        dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
        from_file=True,
        calculate_stress=calculate_stress,
        calculate_hessian=calculate_hessian,
        dtype=dtype,
        **kwargs
    )

    return calc
