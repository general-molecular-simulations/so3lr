import numpy as np
import pathlib

from mlff.mdx.potential import MLFFPotentialSparse


def make_potential_fn(
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        dtype=np.float32,
        coulomb_kspace_do_ewald=False,
        coulomb_kspace_interp_nodes=4,
        **kwargs
):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    return MLFFPotentialSparse.create_from_workdir(
        workdir=package_dir / 'so3lr' / 'params',
        from_file=True,
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            neighborlist_format_lr='ordered_sparse',
            coulomb_kspace_do_ewald=coulomb_kspace_do_ewald,
            coulomb_kspace_interp_nodes=coulomb_kspace_interp_nodes,
        ),
        dtype=dtype,
        **kwargs
    )
