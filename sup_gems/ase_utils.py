import numpy as np
import pathlib

from mlff.md import mlffCalculatorSparse


def make_ase_calculator(
        lr_cutoff=12.,
        dispersion_energy_lr_cutoff_damping=2.,
        dtype=np.float32
):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=package_dir / 'sup_gems' / 'sup_gems_params',
        lr_cutoff=lr_cutoff,
        dispersion_energy_lr_cutoff_damping=dispersion_energy_lr_cutoff_damping,
        from_file=True,
        dtype=dtype
    )

    return calc
