import numpy as np
import pathlib

from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse


def make_ase_calculator(
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        calculate_stress=False,
        calculate_hessian=False,
        dtype=np.float32,
        **kwargs
):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    calc = AseCalculatorSparse.create_from_workdir(
        workdir=package_dir / 'so3lr' / 'params',
        lr_cutoff=lr_cutoff,
        dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
        from_file=True,
        calculate_stress=calculate_stress,
        dtype=dtype,
        **kwargs
    )

    return calc
