import numpy as np

from so3lr.mlff.calculators.potential import PotentialSparse
from so3lr.model_registry import resolve_model


def make_potential_fn(
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        dtype=np.float32,
        coulomb_kspace_method='pme',  # 'ewald', 'pme', or 'p3m'
        coulomb_kspace_interp_nodes=4,
        dispersion_kspace_method=None,  # None, 'ewald', or 'pme'
        dispersion_kspace_interp_nodes=4,
        model=None,
        **kwargs
):
    # ``workdir`` (legacy kwarg) takes precedence over ``model``; either may be a
    # bundled model name or a filesystem path. None -> DEFAULT_MODEL.
    workdir = kwargs.pop('workdir', None)
    workdir_path, from_file, _ = resolve_model(
        workdir if workdir is not None else model
    )

    long_range_kw = dict(
        cutoff_lr=lr_cutoff,
        dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
        neighborlist_format_lr='ordered_sparse',
        coulomb_kspace_method=coulomb_kspace_method,
        coulomb_kspace_interp_nodes=coulomb_kspace_interp_nodes,
    )

    if dispersion_kspace_method is not None:
        do_ewald = (dispersion_kspace_method == 'ewald')
        long_range_kw['dispersion_kspace_do_ewald'] = do_ewald
        long_range_kw['dispersion_kspace_interp_nodes'] = dispersion_kspace_interp_nodes

    return PotentialSparse.create_from_workdir(
        workdir=workdir_path,
        from_file=kwargs.pop('from_file', from_file),
        long_range_kwargs=long_range_kw,
        dtype=dtype,
        **kwargs
    )
