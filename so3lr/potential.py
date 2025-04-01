import numpy as np
import pathlib

from mlff.mdx.potential import MLFFPotentialSparse


def make_potential_fn(
    lr_cutoff=12.,
        dispersion_energy_lr_cutoff_damping=2.,
        dtype=np.float32,
        workdir= None
):
    if workdir == None:
        package_dir = pathlib.Path(__file__).parent.parent.resolve()
        workdir=package_dir / 'so3lr' / 'params'

    return MLFFPotentialSparse.create_from_workdir(
        workdir=workdir,
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_lr_cutoff_damping,
            neighborlist_format_lr='ordered_sparse'
        ),
        dtype=dtype
    )
