from .data import DataTuple
from .dataset import DataSet
from .preprocessing import get_per_atom_shift
from .dataloader_sparse_ase import AseDataLoaderSparse
from .dataloader_sparse_npz import NpzDataLoaderSparse
from .dataloader_sparse_spice import SpiceDataLoaderSparse
from . import transformations
def __getattr__(name):
    if name == "QCMLDataLoaderSparseParallel":
        from .dataloader_sparse_tfds import QCMLDataLoaderSparseParallel
        return QCMLDataLoaderSparseParallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
