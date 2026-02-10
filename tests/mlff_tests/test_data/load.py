import pkg_resources
from pathlib import Path


def load_data(filename):
    p_filename = Path(filename)
    if p_filename.suffix == '.npz':
        import numpy as np
        stream = pkg_resources.resource_stream(__name__, filename)
        return np.load(stream)
    else:
        from ase.io import iread
        f = pkg_resources.resource_filename(__name__, filename)
        return iread(f, ':')
