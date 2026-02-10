import jax
import numpy as np
import logging
import os

from functools import partial
from flax.traverse_util import flatten_dict, unflatten_dict

from typing import Dict, Sequence

from so3lr.mlff.indexing.indices import get_indices, get_pbc_neighbors
from so3lr.mlff.utils.training_utils import set_seeds
from so3lr.mlff.io import read_json, save_dict, merge_dicts
from so3lr.mlff.data.preprocessing import get_per_atom_shift
from so3lr.mlff.properties import properties as pn


class DataSet:
    def __init__(self, prop_keys: Dict, data: Dict):
        self.prop_keys = prop_keys
        self.data = data
        self.data, self.n_data = self._correct_shapes()
        self.splits = {}
        self.splits_info = {}
        self.data_split = {}
        self.scales = self._init_scales()

        self.track_shift_x_by_mean_x = []
        self.track_divide_x_by_std_y = []
        self.track_shift_x_by_type = []

    def summary(self):
        raise NotImplementedError

    def _correct_shapes(self):
        # get n_data, kind of messy but best way I could think of
        q_data = {k: v for k, v in self.data.items()}

        # if energy exists in data, make sure it has the correct dimensions
        try:
            q_data[self.prop_keys[pn.energy]] = q_data[self.prop_keys[pn.energy]].reshape(-1, 1)
        except KeyError:
            pass

        def reshape(y):
            if len(y.shape) <= 1:
                return y.reshape(1, -1)
            else:
                return y

        q_data = jax.tree_util.tree_map(lambda y: reshape(y), q_data)
        max_key = max(jax.tree_util.tree_map(lambda y: len(y), q_data), key=jax.tree_util.tree_map(lambda y: len(y), q_data).get)
        n_data = len(q_data[max_key])

        def repeat(name, y, repeats):
            if len(y) == 1:
                print('Detected missing data dimension (0-th axis) for {}. Assume that the '
                      'data dimension is missing and repeat the entry {} times. Reshaped '
                      'array to ({}, {})'.format(name, n_data, n_data, y.shape[0]))
                return np.repeat(y, repeats=repeats, axis=0)
            elif 1 < len(y) < n_data:
                print('Detected missing data dimension (0-th axis) for {}. Assume that the '
                      'data dimension is missing and repeat the entry {} times. Reshaped '
                      'array to ({}, {})'.format(name, n_data, n_data, y.shape))
                return np.repeat(y[None], repeats=repeats, axis=0)
            else:
                return y

        return {k: repeat(name=k, y=v, repeats=n_data) for (k, v) in q_data.items()}, n_data

    def _init_scales(self):
        return {k: {'per_atom_shift': [0.] * 100,
                    'scale': 1.}
                for k in self.prop_keys.keys()}

    def save_scales(self, path, filename='scales.json'):
        save_dict(path=path, filename=filename, data=self.scales, exists_ok=True)

    def neighborhood_list(self, r_cut):
        R_key = self.prop_keys[pn.atomic_position]
        z_key = self.prop_keys[pn.atomic_type]
        neigh_idx = get_indices(R=self.data[R_key], z=self.data[z_key], r_cut=r_cut)
        return neigh_idx

    def average_number_of_neighbors(self) -> float:
        _d = self.data_split['train']
        _n_atoms = self.number_of_atoms()
        idx_i = _d['idx_i']  # shape: (n_data,P)
        p_segment_sum = partial(jax.ops.segment_sum, num_segments=_n_atoms)
        neighs = jax.vmap(p_segment_sum)(np.ones_like(idx_i), segment_ids=idx_i)  # shape: (n)
        return neighs.mean().item()

    def number_of_atoms(self) -> int:
        _n_atoms = (self.data_split['train'][self.prop_keys[pn.atomic_type]] != -1).sum(-1)  # shape: (D)
        if _n_atoms.std() != 0:
            logging.warning('Dataset contains structures with different structure sizes. Number of atoms is thus not'
                            'well defined.')
            return _n_atoms.item()
        else:
            return int(_n_atoms.mean().item())

    def all_atomic_types(self) -> Sequence[int]:
        z_key = self.prop_keys['atomic_type']
        z_unique = np.unique(self.data[z_key])
        return z_unique[z_unique > 0].tolist()

    def index_split(self,
                    data_idx_train,
                    data_idx_valid,
                    data_idx_test,
                    training: bool,
                    r_cut: float = None,
                    mic: str = None,
                    split_name: str = 'split'):

        if mic == 'bins':
            logging.warning(f'mic={mic} is deprecated in favor of mic=True.')
        if mic == 'naive':
            raise DeprecationWarning(f'mic={mic} is not longer supported.')

        node_mask_present = False
        d = {}

        for i, i_n in zip([data_idx_train, data_idx_valid, data_idx_test], ['train', 'valid', 'test']):
            _d = {}
            for k, v in self.data.items():

                if k == self.prop_keys.get(pn.node_mask):
                    node_mask_present = True

                if len(v.shape) <= 1:
                    logging.warning(
                        f'Array with shape {v.shape} for quantity {k} detected, such that we assume that the data '
                        f'dimension is missing. Reshaped to ({len(i)}, {v.shape[-1]}).'
                    )
                    v = np.repeat(v[None, :], repeats=len(i), axis=0)
                _d.update({k: v[i]})

                d.update({i_n: _d})

        z_key = self.prop_keys.get(pn.atomic_type)
        node_msk_needed = node_mask_required(self.data[z_key])

        if r_cut is not None:
            R_key = self.prop_keys.get(pn.atomic_position)
            n_msk_key = self.prop_keys.get(pn.node_mask)

            if training:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key]])

                if node_msk_needed | node_mask_present:
                    n_msk_dat = np.concatenate([d['train'][n_msk_key], d['valid'][n_msk_key]])
                else:
                    n_msk_dat = np.ones_like(z_dat).astype(bool)

            else:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key], d['test'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key], d['test'][z_key]])

                if node_msk_needed | node_mask_present:
                    n_msk_dat = np.concatenate([d['train'][n_msk_key], d['valid'][n_msk_key], d['test'][n_msk_key]])
                else:
                    n_msk_dat = np.ones_like(z_dat).astype(bool)

            if mic:
                uc_key = self.prop_keys.get(pn.unit_cell)
                pbc_key = self.prop_keys.get(pn.pbc)

                unit_cell_dat = np.concatenate([d['train'][uc_key], d['valid'][uc_key], d['test'][uc_key]])
                pbc_dat = np.concatenate([d['train'][pbc_key], d['valid'][pbc_key], d['test'][pbc_key]])

                cell_lengths = np.linalg.norm(unit_cell_dat, axis=-1).reshape(-1)
                if 0.5 * min(cell_lengths) <= r_cut:
                    # raise NotImplementedError(f'Minimal image convention currently only implemented for '
                    #                           f'r_cut < 0.5*min(cell_lengths), but r_cut={r_cut} and '
                    #                           f'0.5*min(cell_lengths) = {0.5 * min(cell_lengths)}. Consider '
                    #                           f'using `get_pbc_indices` which uses ASE under the hood. '
                    #                           f'However, the latter takes ~15 times longer so maybe '
                    #                           f'reduce r_cut.')
                    logging.warning(f'r_cut > 0.5*min(cell_lengths) detected! r_cut={r_cut} and '
                                    f'0.5*min(cell_lengths) = {0.5 * min(cell_lengths)}. '
                                    f'This case has not been tested rigorously yet and might leave to unwanted'
                                    f'artifacts, so use with care!')

                neigh_idxs = get_pbc_neighbors(pos=R_dat,
                                               node_mask=n_msk_dat,
                                               cell=unit_cell_dat,
                                               cutoff=r_cut,
                                               pbc=pbc_dat)
            else:
                neigh_idxs = get_indices(R_dat, z_dat, r_cut=r_cut)

            n_train = len(data_idx_train)
            n_valid = len(data_idx_valid)

            idx_i_train, idx_i_valid, idx_i_test = np.split(neigh_idxs['idx_i'],
                                                            indices_or_sections=[n_train, n_train + n_valid])
            idx_j_train, idx_j_valid, idx_j_test = np.split(neigh_idxs['idx_j'],
                                                            indices_or_sections=[n_train, n_train + n_valid])

            if mic:
                c_off_train, c_off_valid, c_off_test = np.split(neigh_idxs['shifts'],
                                                                indices_or_sections=[n_train, n_train + n_valid])
                d_idx = {'train': {'idx_i': idx_i_train, 'idx_j': idx_j_train, 'cell_offset': c_off_train},
                         'valid': {'idx_i': idx_i_valid, 'idx_j': idx_j_valid, 'cell_offset': c_off_valid},
                         'test': {'idx_i': idx_i_test, 'idx_j': idx_j_test, 'cell_offset': c_off_test}}
            else:
                d_idx = {'train': {'idx_i': idx_i_train, 'idx_j': idx_j_train},
                         'valid': {'idx_i': idx_i_valid, 'idx_j': idx_j_valid},
                         'test': {'idx_i': idx_i_test, 'idx_j': idx_j_test}}

            _d = {k: merge_dicts(v, d_idx[k])
                  for k, v in d.items() if k in ['train', 'valid', 'test']}
            d.update(_d)

            if node_mask_present:
                pass
            else:
                n_msk_train, n_msk_valid, n_msk_test = np.split(n_msk_dat,
                                                                indices_or_sections=[n_train, n_train + n_valid])

                d_n_msk = {'train': {'node_mask': n_msk_train},
                           'valid': {'node_mask': n_msk_valid},
                           'test': {'node_mask': n_msk_test}}

                _d_n_msk = {k: merge_dicts(v, d_n_msk[k])
                            for k, v in d.items() if k in ['train', 'valid', 'test']}

                d.update(_d_n_msk)

        n_train = len(data_idx_train)
        n_valid = len(data_idx_valid)
        n_test = len(data_idx_test)
        self.splits.update({split_name: {'data_idx_train': np.array(data_idx_train),
                                         'data_idx_valid': np.array(data_idx_valid),
                                         'data_idx_test': np.array(data_idx_test)}})
        self.splits_info.update({split_name: {'n_train': n_train,
                                              'n_valid': n_valid,
                                              'n_test': n_test,
                                              'n_data': self.n_data,
                                              'r_cut': r_cut,
                                              'mic': mic}})
        self.data_split = d

    def random_split(self,
                     n_train,
                     n_valid,
                     n_test,
                     r_cut=None,
                     mic=None,
                     seed=0,
                     training=True):

        idx_train, idx_valid, idx_test = self.generate_split_indices(self.data,
                                                                     n_train=n_train,
                                                                     n_valid=n_valid,
                                                                     n_test=n_test,
                                                                     seed=seed
                                                                     )

        self.index_split(data_idx_train=idx_train,
                         data_idx_valid=idx_valid,
                         data_idx_test=idx_test,
                         training=training,
                         r_cut=r_cut,
                         mic=mic)

    @staticmethod
    def generate_split_indices(data, n_train, n_valid, n_test=None, seed=0, draw_strat=None):
        set_seeds(seed)
        _k = list(data.keys())[0]
        n_data = len(data[_k])
        perm = np.random.RandomState(seed).permutation(n_data)
        idx_all = np.arange(n_data)[perm]

        if draw_strat:
            raise NotImplementedError('Stratified data set sampling is not implemented.')
        else:
            idx = idx_all[:n_train + n_valid]
            idx_valid, idx_train = np.split(idx, indices_or_sections=[n_valid])

        # set sorts the indices, so we have to permute them again
        idx_test = np.array(list(set(idx_all) - set(idx_train) - set(idx_valid)))
        test_perm = np.random.RandomState(seed).permutation(len(idx_test))
        idx_test = idx_test[test_perm][:n_test]  # array[:None] returns all elements of array

        if n_test is None:
            n_test = len(idx_test)

        # assert no duplicates per subset
        assert len(set(idx_train)) == n_train
        assert len(set(idx_valid)) == n_valid
        assert len(set(idx_test)) == n_test
        # make sure there is no overlap
        assert len(set(idx_test) & set(idx_train)) == 0
        assert len(set(idx_test) & set(idx_valid)) == 0
        assert len(set(idx_train) & set(idx_valid)) == 0

        return idx_train, idx_valid, idx_test

    def shift_x_by_mean_x(self, x):
        n_atoms = self.data[self.prop_keys[pn.atomic_type]].shape[-1]
        if x in self.track_shift_x_by_mean_x:
            logging.warning(f'You already called `shift_x_by_mean` for `x={x}`. It is not shifted again.')
        else:
            p_key = self.prop_keys[x]
            p_mean = self.data_split['train'][p_key].reshape(-1).mean()
            self.data_split['train'][p_key] -= p_mean
            self.data_split['valid'][p_key] -= p_mean
            self.data_split['test'][p_key] -= p_mean
            self.scales[x]['per_atom_shift'] = [0] + [p_mean / n_atoms] * 100
            self.track_shift_x_by_mean_x += [x]

    def divide_x_by_std_y(self, x, y):
        if x in self.track_divide_x_by_std_y:
            logging.warning(f'You already called `divide_x_by_std_y` for `x={x}`. It is not divided again.')
        else:
            x_key = self.prop_keys[x]
            y_key = self.prop_keys[y]
            y_scale = self.data_split['train'][y_key].reshape(-1).std()
            self.data_split['train'][x_key] /= y_scale
            self.data_split['valid'][x_key] /= y_scale
            self.data_split['test'][x_key] /= y_scale
            self.scales[x]['scale'] = y_scale.item()
            self.track_divide_x_by_std_y += [x]

    def shift_x_by_type(self, x, shifts=None):
        if shifts is not None:
            self.shift_x_by_type_hand(x, shifts=shifts)
        else:
            self.shift_x_by_type_lse(x)

    def shift_x_by_type_hand(self, x, shifts: Dict[int, float]):
        if x in self.track_shift_x_by_type:
            logging.warning(f'You already called `shift_x_by_type` for `x={x}`. It is not shifted again.')
        else:
            x_key = self.prop_keys[x]
            z_key = self.prop_keys[pn.atomic_type]

            shifts_arr = np.zeros(int(max(list(shifts.keys()))) + 1)
            for k, v in shifts.items():
                shifts_arr[k] = v

            def apply_shifts(q, _z):
                q_scaled = q - np.take(shifts_arr, _z).sum(axis=-1)
                return q_scaled

            self.data_split['train'][x_key] = apply_shifts(self.data_split['train'][x_key].reshape(-1),
                                                           self.data_split['train'][z_key]).reshape(
                self.data_split['train'][x_key].shape)

            self.data_split['valid'][x_key] = apply_shifts(self.data_split['valid'][x_key].reshape(-1),
                                                           self.data_split['valid'][z_key]).reshape(
                self.data_split['valid'][x_key].shape)

            self.scales[x]['per_atom_shift'] = shifts_arr.reshape(-1).tolist()
            self.track_shift_x_by_type += [x]

    def shift_x_by_type_lse(self, x):
        if x in self.track_shift_x_by_type:
            logging.warning(f'You already called `shift_x_by_type` for `x={x}`. It is not shifted again.')
        else:
            x_key = self.prop_keys[x]
            z_key = self.prop_keys[pn.atomic_type]

            z = self.data_split['train'][z_key]

            shifts, x_shift_lse = get_per_atom_shift(z=z,
                                                     q=self.data_split['train'][x_key].reshape(-1),
                                                     pad_value=0)

            def apply_shifts(q, _z):
                q_scaled = q - np.take(shifts, _z).sum(axis=-1)
                return q_scaled

            self.data_split['train'][x_key] = x_shift_lse.reshape(self.data_split['train'][x_key].shape)
            self.data_split['valid'][x_key] = apply_shifts(self.data_split['valid'][x_key].reshape(-1),
                                                           self.data_split['valid'][z_key]).reshape(
                self.data_split['valid'][x_key].shape)

            self.scales[x]['per_atom_shift'] = shifts.reshape(-1).tolist()
            self.track_shift_x_by_type += [x]

    def get_data_split(self):
        return self.data_split

    def load_split(self, file, r_cut, split_name, mic=None, n_train=None, n_valid=None, n_test=None):
        path, filename = os.path.split(file)
        split_idx = self.load_splits_from_file(path=path, filename=filename)[split_name]
        key_2_n = {'data_idx_train': n_train, 'data_idx_valid': n_valid, 'data_idx_test': n_test}

        valid_keys = list(key_2_n.keys())

        subset_split = {k: v[:key_2_n[k]] for (k, v) in split_idx.items() if k in valid_keys}
        self.index_split(r_cut=r_cut, mic=mic, training=False, **subset_split)

    def save_splits_to_file(self, path, filename):
        splits_ = jax.tree_util.tree_map(lambda y: y.tolist(), self.splits)
        save_dict(path=path, filename=filename, data=splits_, exists_ok=True)
        print('Saved the data indices of the splits to {}'.format(os.path.join(path, filename)))

    @staticmethod
    def load_splits_from_file(path, filename):
        _data_idx = read_json(path=os.path.join(path, filename))
        return _data_idx

    def __dict_repr__(self):
        return {'dataset': self.splits_info}


def tree_map_by_key(fn, x, keys):
    x_flat = flatten_dict(x)
    apply_mask = unflatten_dict({p: (p[-1] in keys) for p in x_flat})
    msk_fn = lambda y, m: fn(y) if m else y
    return jax.tree_util.tree_map(msk_fn, x, apply_mask)


def node_mask_required(z):
    """
    Check if node mask needs to passed explicitly.

    Args:
        z (Array): Array with the atomic types, shape: (B,n)

    Returns:

    """

    if z.min() < 1:
        return True
    if z.max() == np.inf:
        return True

    return False
