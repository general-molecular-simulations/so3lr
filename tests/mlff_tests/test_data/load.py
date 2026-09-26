from pathlib import Path


def load_data(filename):
    p_filename = Path(__file__).parent / filename
    if p_filename.suffix == '.npz':
        import numpy as np
        return np.load(p_filename)
    else:
        from ase.io import iread
        return iread(p_filename, ':')
