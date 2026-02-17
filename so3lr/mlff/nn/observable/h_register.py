from .observable_sparse import EnergySparse
from .observable_sparse import DipoleVecSparse
from .observable_sparse import HirshfeldSparse
from .observable_sparse import C6RatiosSparse


def get_observable_module(name, h):
    if name == 'energy_sparse':
        return EnergySparse(**h)
    elif name == 'dipole_vec_sparse':
        return DipoleVecSparse(**h)
    elif name == 'hirsh_ratios_sparse':
        return HirshfeldSparse(**h)
    elif name == 'c6_ratios_sparse':
        return C6RatiosSparse(**h)
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
