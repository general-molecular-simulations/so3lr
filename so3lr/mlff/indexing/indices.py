import jax
import jax.numpy as jnp
import numpy as np
import logging

from ase.neighborlist import primitive_neighbor_list
from functools import partial
from typing import Dict
from tqdm import tqdm

from so3lr.mlff.geometric.metric import coordinates_to_distance_matrix, coordinates_to_distance_matrix_mic
from so3lr.mlff.utils import PrimitiveNeighbors


def get_indices(R: np.ndarray, z: np.ndarray, r_cut: float, cell: np.ndarray = None, mic: bool = False):
    """
        For the `n_data` data frames, return index lists for centering and neighboring atoms given some cutoff radius
        `r_cut` for each structure. As atoms may leave or enter the neighborhood for a given atom within the dataset,
        one can have different lengths for the index lists, even for same structures. Thus, the index lists are padded
        wrt the length `n_pairs_max` of the longest index list observed over all frames in the coordinates `R` across
        all structures. Note, that this is suboptimal if one has a wide range number of atoms in the same dataset. The
        coordinates `R` and the atomic types `z` can also be already padded (see `mlff.src.padding.padding`) and are
        assumed to be padded with `0` if padded. Index values are padded with `-1`.

        Args:
            R (Array): Atomic coordinates, shape: (n_data,n,3)
            z (Array): Atomic types, shape: (n_data, n)
            r_cut (float): Cutoff distance
            cell (Array): Unit cell with lattice vectors along rows (ASE default), shape: (n_data,3,3)
            mic (bool): Minimal image convention for periodic boundary conditions

        Returns: Tuple of centering and neighboring indices, shape: Tuple[(n_pairs_max), (n_pairs_max)]

        """
    if mic is True:
        raise DeprecationWarning('`get_indices()` with `mic=True` is deprecated in favor of `get_pbc_neighbors()`.')

    n = R.shape[-2]
    n_data = R.shape[0]
    idx = np.indices((n, n))

    if mic:
        cell_lengths = np.linalg.norm(cell, axis=-1).reshape(-1)  # shape: (n_data*3)
        if r_cut < 0.5 * min(cell_lengths):
            distance_fn = lambda r, c: coordinates_to_distance_matrix_mic(r, c)
        else:
            raise NotImplementedError(f'Minimal image convention currently only implemented for '
                                      f'r_cut < 0.5*min(cell_lengths), but r_cut={r_cut} and 0.5*min(cell_lengths) = '
                                      f'{0.5 * min(cell_lengths)}. Consider using `get_pbc_indices` which uses ASE under '
                                      f'the hood. However, the latter takes ~15 times longer so maybe reduce r_cut.')
    else:
        distance_fn = lambda r, _: coordinates_to_distance_matrix(r)
        cell = np.zeros(len(R))

    @jax.jit
    def neigh_filter(_z, _R, _c):
        Dij = distance_fn(_R, _c).squeeze(axis=-1)  # shape: (n,n)
        msk_ij = (jnp.einsum('i, j -> ij', _z, _z) != 0).astype(np.int16)  # shape: (n,n)
        Dij_x_msk_ij = Dij * msk_ij  # shape: (n,n)
        return jnp.where((Dij_x_msk_ij <= r_cut) & (Dij_x_msk_ij > 0), True, False)

    def get_idx(i):
        idx_ = idx[:, neigh_filter(z[i], R[i], cell[i])]  # shape: (2,n_pairs)
        # pad_idx = np.pad(idx_, ((0, 0), (0, int(pad_length[i]))), mode='constant', constant_values=((0, 0), (0, -1)))
        # shape: (2,n_pair+pad_length)
        return idx_

    logging.info('Generate neighborhood lists for {} geometries: '.format(n_data))
    idxs = list(map(get_idx, tqdm(range(n_data))))  # [(2,*), ..., (2,*)]  * is number of pairs per point

    n_pairs = [u.shape[-1] for u in idxs]
    n_pairs_max = max(n_pairs)
    pad_length = [n_pairs_max - u for u in n_pairs]

    def pad_idxs(t):
        u, l = t
        return np.pad(u, ((0, 0), (0, int(l))), mode='constant', constant_values=((0, 0), (0, -1)))

    pad_idx_i, pad_idx_j = map(np.squeeze,
                               np.split(np.array(list(map(pad_idxs, zip(idxs, pad_length)))),
                                        indices_or_sections=2,
                                        axis=-2))

    return {'idx_i': pad_idx_i, 'idx_j': pad_idx_j}


def pad_index_list(idx, n_pair_max, pad_value=-1):
    n_pair = idx.shape[-1]

    pad_length = n_pair_max - n_pair
    assert pad_length >= 0

    pad = partial(np.pad, pad_width=(0, pad_length), mode='constant',
                  constant_values=(0, pad_value))
    pad_idx = pad(idx)
    return pad_idx


def pad_shift(_shift, n_pair_max, pad_value=0):
    n_pair = _shift.shape[0]

    pad_length = n_pair_max - n_pair
    assert pad_length >= 0

    pad = partial(np.pad, pad_width=((0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, pad_value), (0, 0)))
    pad_s = pad(_shift)
    return pad_s


def get_pbc_neighbors(pos, node_mask, pbc, cell, cutoff) -> Dict:
    """
    Calculate the neighbors in the graph if periodic boundary conditions (PBC) are present. Under the hood, this
    function uses the ase.primitive_neighbors function from ASE to calculate the central index `idx_i`, the neighboring
    indices `idx_j` as well as shifts vectors `shifts`.

    Args:
        pos (Array): Positions that, if structures of different sizes are present, are already padded to the same
            length n, shape: (B,n,3)
        node_mask (Array): Mask that indicates which nodes are true nodes and which are only present due to padding,
            shape: (B,n)
        cell (Array): Cells are assumed to be row-wise (ASE default), shape: (B,3,3)
        pbc (Array): Directions of periodic boundary conditions, shape: (B,3)
        cutoff (float): Cutoff used to calculate the neighbors

    Returns: Dictionary, with central, neighboring indices as well as the shift vectors.

    """

    # Cells must be transposed in the following, since mlff neighbor list assume cells to be column wise to make it
    # consistent with GLP and jax_md neighbor list

    # TODO: check for 2*cutoff < min(||lattice_vector||_2)

    def estimate_capacity_multiplier():
        return 1.1 * node_mask.sum(-1).max() / node_mask.sum(-1).min()

    allocate_fn, update_fn = pbc_neighbor_list(cell=cell[0].T,
                                               cutoff=cutoff,
                                               capacity_multiplier=1.)

    neighbors = allocate_fn(pos=pos[0], node_mask=node_mask[0], pbc=pbc[0], new_cell=cell[0].T)

    tmp_idx_i_list = []
    tmp_idx_j_list = []
    tmp_shifts_list = []

    idx_i_list = []
    idx_j_list = []
    shifts_list = []
    
    print('Construct neighbors using minimal image convention ...')
    for (R, msk, pb, ce) in tqdm(zip(pos, node_mask, pbc, cell)):
        neighbors = update_fn(pos=R, node_mask=msk, pbc=pb, primitive_neighbors=neighbors, new_cell=ce.T)

        # check for overflow in the neighbor list
        if neighbors.overflow:
            # print('Overflow detected. Re-allocating neighbor list!')
            # re-allocate the neighborhood list
            # allocate_fn, update_fn = pbc_neighbor_list(cell=ce.T, cutoff=cutoff, capacity_multiplier=1.)
            neighbors = allocate_fn(pos=R, node_mask=msk, pbc=pb, new_cell=ce.T)

            # save the primitive neighbors calculated so far
            idx_i_list += [np.stack(tmp_idx_i_list, axis=0)]
            idx_j_list += [np.stack(tmp_idx_j_list, axis=0)]
            shifts_list += [np.stack(tmp_shifts_list, axis=0)]

            # clear the temporary lists
            tmp_idx_i_list = []
            tmp_idx_j_list = []
            tmp_shifts_list = []

        # append neighbors and shifts
        tmp_idx_i_list += [neighbors.idx_i]
        tmp_idx_j_list += [neighbors.idx_j]
        tmp_shifts_list += [neighbors.shifts]

    # append the remaining temporary lists
    idx_i_list += [np.stack(tmp_idx_i_list, axis=0)]
    idx_j_list += [np.stack(tmp_idx_j_list, axis=0)]
    shifts_list += [np.stack(tmp_shifts_list, axis=0)]

    n_edges_max = max([x.shape[-1] for x in idx_j_list])

    idx_i_list = [np.array(list(map(partial(pad_index_list, n_pair_max=n_edges_max), x))) for x in idx_i_list]
    idx_j_list = [np.array(list(map(partial(pad_index_list, n_pair_max=n_edges_max), x))) for x in idx_j_list]
    shifts_list = [np.array(list(map(partial(pad_shift, n_pair_max=n_edges_max), x))) for x in shifts_list]
    print('... done!')
    return {'idx_i': np.concatenate(idx_i_list, axis=0),
            'idx_j': np.concatenate(idx_j_list, axis=0),
            'shifts': np.concatenate(shifts_list, axis=0)}


def pbc_neighbor_list(cell, cutoff, skin: float = None, capacity_multiplier: float = 1.25):
    """

    Args:
        cell (Array): Lattice vectors, are assumed to be column-wise!! Differs from ASE default
        cutoff ():
        skin ():
        capacity_multiplier ():

    Returns:

    """

    def allocate_fn(pos, pbc, new_cell=None, node_mask=None):
        if new_cell is None:
            new_cell = cell

        if node_mask is None:
            node_mask = np.ones(pos.shape[0], dtype=pos.dtype).astype(bool)

        ce_ase = new_cell.T  # switch to ASE default, lattice vectors are now row-wise

        idx_i, idx_j, shifts = primitive_neighbor_list(quantities='ijS',
                                                       pbc=pbc,
                                                       cell=ce_ase,
                                                       positions=pos[node_mask],
                                                       cutoff=cutoff,
                                                       numbers=None,
                                                       self_interaction=False,
                                                       use_scaled_positions=False,
                                                       max_nbins=1000000.0)

        n_edges_max = int(np.ceil(capacity_multiplier * len(idx_i)))
        return PrimitiveNeighbors(idx_i=pad_index_list(idx_i, n_pair_max=n_edges_max),
                                  idx_j=pad_index_list(idx_j, n_pair_max=n_edges_max),
                                  shifts=pad_shift(shifts, n_pair_max=n_edges_max),
                                  overflow=False
                                  )

    def update_fn(pos, pbc, primitive_neighbors: PrimitiveNeighbors, new_cell, node_mask=None):
        """

        Args:
            pos (Array): Atomic positions, shape: (n,3)
            pbc (Array):
            primitive_neighbors (PrimitiveNeighbors)
            new_cell (Array): Unit cell, lattice vectors are assumed to be column-wise (transpose of ASE default),
                shape: (3,3)
            node_mask (Array):

        Returns:

        """
        if node_mask is None:
            node_mask = np.ones(pos.shape[0], dtype=pos.dtype).astype(bool)

        n_edges_max = len(primitive_neighbors.idx_i)

        ce_ase = new_cell.T  # switch to ASE default, lattice vectors are now row-wise

        idx_i, idx_j, shifts = primitive_neighbor_list(quantities='ijS',
                                                       pbc=pbc,
                                                       cell=ce_ase,
                                                       positions=pos[node_mask],
                                                       cutoff=cutoff,
                                                       numbers=None,
                                                       self_interaction=False,
                                                       use_scaled_positions=False,
                                                       max_nbins=1000000.0)

        if len(idx_i) > n_edges_max:
            overflow = True
        else:
            overflow = False
            idx_i = pad_index_list(idx_i, n_pair_max=n_edges_max)
            idx_j = pad_index_list(idx_j, n_pair_max=n_edges_max)
            shifts = pad_shift(shifts, n_pair_max=n_edges_max)

        return PrimitiveNeighbors(idx_i=idx_i, idx_j=idx_j, shifts=shifts, overflow=overflow)

    # note that the update function still returns correct neighbors and shifts, even when overflow is present.
    return allocate_fn, update_fn
