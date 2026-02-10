import jax.numpy as jnp
import numpy as np
from ase import Atoms
from .test_data import load_data


def compare_distances(a, b, cutoff, atol=0.0, rtol=1e-7):
    a = filter_and_sort(a, cutoff)
    b = filter_and_sort(b, cutoff)

    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def filter_and_sort(distances, cutoff):
    d = distances[distances < cutoff]
    d = d[d > 0]
    return np.sort(d)


def test_neighborhood_list():
    from jax import config
    config.update("jax_enable_x64", True)

    # load all data from the new ZrO2 data set
    data = load_data('test_solid.npz')

    zs = data['z']
    cells = data['unit_cell']
    Rs = data['R']

    cutoff = 4
    mic = False

    from so3lr.mlff.indexing.indices import get_indices
    from tqdm import tqdm

    indices = get_indices(np.array(Rs), np.array(zs), r_cut=cutoff, cell=np.array(cells), mic=mic)

    for i in tqdm(range(len(Rs))):
        idx_i = indices['idx_i'][i]
        idx_j = indices['idx_j'][i]

        v_ij = Rs[i][idx_j] - Rs[i][idx_i]
        d_ij = jnp.linalg.norm(v_ij, axis=-1)

        atoms = Atoms(zs[i], positions=Rs[i])
        ase_d_ij = atoms.get_all_distances(mic=mic)
        compare_distances(d_ij, ase_d_ij, cutoff=cutoff)
