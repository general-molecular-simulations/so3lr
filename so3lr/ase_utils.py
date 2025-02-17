import numpy as np
import pathlib
from typing import Dict

from mlff.md import mlffCalculatorSparse


def make_ase_calculator(
        lr_cutoff=12.,
        dispersion_energy_lr_cutoff_damping=2.,
        calculate_stress=False,
        calculate_charges=False,
        dtype=np.float32,
        obs_fn_kwargs: Dict[str, Dict[str, int]] = {},
        has_aux=False
):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=package_dir / 'so3lr' / 'params',
        lr_cutoff=lr_cutoff,
        dispersion_energy_lr_cutoff_damping=dispersion_energy_lr_cutoff_damping,
        from_file=True,
        calculate_stress=calculate_stress,
        calculate_charges=calculate_charges,
        dtype=dtype,
        obs_fn_kwargs=obs_fn_kwargs,
        has_aux=has_aux
    )

    return calc
