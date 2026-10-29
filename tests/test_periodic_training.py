"""The training data path on periodic data: molecules and crystals in one batch, and
the k-space mesh a training run uses for cells of different sizes.

`so3lr finetune`, `so3lr train` and `so3lr eval --engine standard` all load with
AseDataLoaderSparse and batch with dynamically_batch_with_lr. A molecule has no cell
and no image offsets; a batch used to take the long-range offsets from its first
item, so a molecule first silently dropped the image offsets of every crystal after
it, and a crystal first failed on the molecule's missing ones. Training with k-space
also sized one mesh from the first periodic cell for every structure.
"""
import pathlib

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import numpy.testing as npt
import pytest
from ase.io import read, write
from ml_collections import config_dict

from so3lr.base_calculator import make_so3lr
from so3lr.mlff.config.from_config import _compute_kspace_constants, _validate_kspace_graph_pbc
from so3lr.mlff.data import AseDataLoaderSparse
from so3lr.mlff.data.dataloader_sparse_ase import ASE_to_jraph
from so3lr.mlff.utils import jraph_utils
from so3lr.mlff.utils.kspace_utils import kspace_grid_shape, setup_kspace_grid

TEST_DATA = pathlib.Path(__file__).parent / 'test_data'
MODEL, CUTOFF, LR = 'so3lr-tauri-s', 4.5, 6.0


def _crystal():
    return read(TEST_DATA / 'water_pbc_dummy.xyz', index=0)  # 192 atoms, 12.4 A cube


def _molecule():
    return read(TEST_DATA / 'ac.xyz', index=0)


def _load(frames, tmp_path):
    path = tmp_path / 'frames.extxyz'
    write(path, frames)
    data, stats = AseDataLoaderSparse(input_file=path).load(
        cutoff=CUTOFF, cutoff_lr=LR, calculate_neighbors_lr=True, pick_idx=np.arange(len(frames)))
    return data, stats


def _predict(data, stats, batch_size, kspace=None, k_grid=None):
    """Energies and forces through the eval/training batching, as `so3lr eval` runs it."""
    calc = jax.jit(make_so3lr(model=MODEL, lr_cutoff=LR, calculate_forces=True,
                              kspace_electrostatics=kspace))
    kconst = None
    if kspace is not None:
        kconst = {'k_grid': k_grid, 'k_smearing': jnp.array([2.0]),
                  'k_grid_disp': None, 'k_smearing_disp': None}
    energies, forces = [], []
    for batch in jraph_utils.dynamically_batch_with_lr(
            data, n_node=stats['max_num_of_nodes'] * batch_size + 1,
            n_edge=stats['max_num_of_edges'] * batch_size + 1, n_graph=batch_size + 1,
            n_pairs=stats['max_num_of_pairs'] * batch_size + 1):
        inputs = jraph_utils.graph_to_batch_fn(*batch, kspace_constants=kconst)
        out = jax.device_get(calc(inputs))
        n = int(inputs['num_of_non_padded_graphs'])
        f = np.asarray(out['forces'])[np.asarray(inputs['node_mask'], bool)]
        energies += list(np.asarray(out['energy'])[:n])
        forces += np.split(f, np.cumsum(batch[0].n_node[:n])[:-1])
    return np.array(energies), forces


@pytest.mark.parametrize('molecule_first', [True, False])
def test_molecules_and_crystals_share_a_batch(tmp_path, molecule_first):
    frames = [_molecule(), _crystal()] if molecule_first else [_crystal(), _molecule()]
    data, stats = _load(frames, tmp_path)
    e_one, f_one = _predict(data, stats, batch_size=1)
    e_two, f_two = _predict(data, stats, batch_size=2)
    npt.assert_allclose(e_two, e_one, atol=2e-3)
    for a, b in zip(f_two, f_one):
        npt.assert_allclose(a, b, atol=2e-3)


def test_molecule_keeps_its_energy_next_to_a_crystal(tmp_path):
    """Zero cell offsets given to the molecule change none of its distances."""
    alone, stats_alone = _load([_molecule()], tmp_path)
    mixed, stats_mixed = _load([_molecule(), _crystal()], tmp_path)
    assert 'cell_offset' not in alone[0][0].edges
    assert 'cell_offset' in mixed[0][0].edges
    (e_alone,), (f_alone,) = _predict(alone, stats_alone, batch_size=1)
    e_mixed, f_mixed = _predict(mixed, stats_mixed, batch_size=1)
    npt.assert_allclose(e_mixed[0], e_alone, atol=1e-4)
    npt.assert_allclose(f_mixed[0], f_alone, atol=1e-4)


def _graph(atoms):
    graph, lr = ASE_to_jraph(atoms, cutoff=CUTOFF, calculate_neighbors_lr=True, cutoff_lr=LR)
    return graph, lr


@pytest.mark.parametrize('method', ['pme', 'ewald'])
def test_kspace_mesh_covers_the_largest_cell(method):
    small, large = _crystal(), _crystal().repeat((2, 1, 1))
    config = config_dict.ConfigDict({'model': {
        'kspace_electrostatics': method, 'kspace_smearing': 2.0, 'kspace_spacing': 1.0}})
    k = _compute_kspace_constants(config, [_graph(small)], False, validation_data=[_graph(large)])
    want = np.maximum(kspace_grid_shape(method, 1.0, small.cell.array),
                      kspace_grid_shape(method, 1.0, large.cell.array))
    assert k['k_grid'].shape == tuple(want)
    assert k['k_grid'].shape != kspace_grid_shape(method, 1.0, small.cell.array)
    assert k['k_grid_disp'].shape == k['k_grid'].shape


@pytest.mark.parametrize('method', ['pme', 'ewald'])
def test_a_finer_kspace_mesh_gives_the_same_energy(tmp_path, method):
    """The larger cell's mesh, used for a smaller cell, changes its energy by no more than
    convergence noise: the k-grid only carries a shape, the cell comes from the graph."""
    crystal = _crystal()
    data, stats = _load([crystal], tmp_path)
    own, _ = setup_kspace_grid(method, 2.0, 1.0, jnp.asarray(crystal.cell.array))
    finer = jnp.ones(tuple(2 * n for n in own.shape))
    (e_own,), (f_own,) = _predict(data, stats, batch_size=1, kspace=method, k_grid=own)
    (e_fine,), (f_fine,) = _predict(data, stats, batch_size=1, kspace=method, k_grid=finer)
    assert abs(e_fine - e_own) / len(crystal) < 1e-4
    npt.assert_allclose(f_fine, f_own, atol=5e-3)


def test_kspace_training_refuses_gas_phase_structures():
    graph, _ = _graph(_molecule())
    with pytest.raises(ValueError, match='vacuum box'):
        _validate_kspace_graph_pbc(graph, padded=False)
    # in a padded batch, the padding graph is exempt
    crystal, _ = _graph(_crystal())
    batch = jraph.pad_with_graphs(jraph.batch_np([crystal]),
                                  n_node=200, n_edge=int(crystal.n_edge[0]) + 1, n_graph=2)
    _validate_kspace_graph_pbc(batch)
