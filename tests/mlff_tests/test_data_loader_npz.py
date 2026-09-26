import shutil
from pathlib import Path

import numpy as np
import jraph
import numpy.testing as npt
from so3lr.mlff.data import NpzDataLoaderSparse
from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr
import pytest


TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_data_load(calculate_neighbors_lr: bool):
    filename = 'test_data/ethanol.npz'
    f = TEST_DIR / filename

    loader = NpzDataLoaderSparse(input_file=f)
    all_data, data_stats = loader.load(
        cutoff=5.,
        calculate_neighbors_lr=calculate_neighbors_lr,
        cutoff_lr=10.
    )
    npt.assert_equal(len(all_data), 2000)
    npt.assert_equal(loader.cardinality(), 2000)

    npt.assert_equal(data_stats['max_num_of_nodes'], 9)
    npt.assert_equal(data_stats['max_num_of_edges'], 72)

    # all_data is now a list of (graph, long_range) tuples
    graph0, lr0 = all_data[0]
    npt.assert_(isinstance(graph0, jraph.GraphsTuple))
    npt.assert_(isinstance(lr0, dict))

    for i in range(0, 100)[::10]:
        graph, lr = all_data[i]
        forces = graph.nodes.get('forces')
        positions = graph.nodes.get('positions')
        atomic_numbers = graph.nodes.get('atomic_numbers')
        hirshfeld_ratios = graph.nodes['hirshfeld_ratios']
        senders = graph.senders
        receivers = graph.receivers
        energy = graph.globals.get('energy')
        cell = graph.edges.get('cell')
        cell_offset = graph.edges.get('cell_offset')
        stress = graph.globals['stress']
        total_charge = graph.globals['total_charge']
        num_unpaired_electrons = graph.globals['num_unpaired_electrons']
        dipole_vec = graph.globals['dipole_vec']

        num_pairs = lr['n_pairs']
        idx_i_lr = lr['idx_i_lr']
        idx_j_lr = lr['idx_j_lr']

        npt.assert_(isinstance(forces, np.ndarray))
        npt.assert_(isinstance(energy, np.ndarray))
        npt.assert_(isinstance(atomic_numbers, np.ndarray))
        npt.assert_(isinstance(senders, np.ndarray))
        npt.assert_(isinstance(receivers, np.ndarray))
        npt.assert_(isinstance(idx_i_lr, np.ndarray))
        npt.assert_(isinstance(idx_j_lr, np.ndarray))

        npt.assert_equal(forces.dtype, np.float64)
        npt.assert_equal(energy.dtype, np.float64)
        npt.assert_equal(positions.dtype, np.float64)
        npt.assert_equal(atomic_numbers.dtype, np.int64)
        npt.assert_equal(senders.dtype, np.int64)
        npt.assert_equal(receivers.dtype, np.int64)
        npt.assert_equal(total_charge.dtype, np.int16)
        npt.assert_equal(num_unpaired_electrons.dtype, np.int16)

        npt.assert_equal(cell, None)
        npt.assert_equal(cell_offset, None)

        stress_expected = np.empty((1, 6))
        stress_expected[:] = np.nan
        npt.assert_equal(stress, stress_expected)

        num_atoms = 9
        hirshfeld_ratios_expected = np.empty((num_atoms,))
        hirshfeld_ratios_expected[:] = np.nan
        npt.assert_equal(hirshfeld_ratios, hirshfeld_ratios_expected)

        dipole_vec_expected = np.empty((1, 3))
        dipole_vec_expected[:] = np.nan
        npt.assert_equal(dipole_vec, dipole_vec_expected)

        npt.assert_equal(atomic_numbers.shape, (num_atoms,))
        npt.assert_equal(positions.shape, (num_atoms, 3))
        npt.assert_equal(energy.shape, (1,))
        npt.assert_equal(forces.shape, (num_atoms, 3))
        npt.assert_equal(total_charge.shape, (1,))
        npt.assert_equal(total_charge, 0)
        npt.assert_equal(num_unpaired_electrons.shape, (1,))
        npt.assert_equal(num_unpaired_electrons, 0)
        npt.assert_equal(len(senders), len(receivers))
        npt.assert_equal(len(senders), 72)

        npt.assert_equal(len(idx_j_lr), len(idx_j_lr))
        if calculate_neighbors_lr:
            npt.assert_equal(num_pairs, num_atoms * num_atoms - num_atoms)
            npt.assert_equal(len(idx_i_lr), num_pairs)
            npt.assert_equal(len(idx_j_lr), num_pairs)
            npt.assert_equal(senders, idx_j_lr)
            npt.assert_equal(receivers, idx_i_lr)
        else:
            npt.assert_equal(idx_i_lr, np.array([]).reshape(-1))
            npt.assert_equal(idx_j_lr, np.array([]).reshape(-1))
            npt.assert_equal(num_pairs, 0)


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_data_load_with_pbc(calculate_neighbors_lr: bool):
    filename = 'test_data/data_set_pbc.npz'
    f = TEST_DIR / filename
    loader = NpzDataLoaderSparse(input_file=f)
    if calculate_neighbors_lr:
        # PBC long-range is now supported, so we just load and check
        all_data, data_stats = loader.load(
            cutoff=4.,
            calculate_neighbors_lr=calculate_neighbors_lr,
            cutoff_lr=15.
        )

        npt.assert_equal(len(all_data), 50)
        npt.assert_equal(data_stats['max_num_of_nodes'], 110)

        graph0, lr0 = all_data[0]
        npt.assert_(isinstance(graph0, jraph.GraphsTuple))
        npt.assert_(isinstance(lr0, dict))
        # PBC long-range should have cell_offsets
        npt.assert_(lr0.get('cell_offsets_lr') is not None)
    else:
        all_data, data_stats = loader.load(
            cutoff=4.,
            calculate_neighbors_lr=calculate_neighbors_lr,
            cutoff_lr=15.
        )
        npt.assert_equal(len(all_data), 50)
        npt.assert_equal(data_stats['max_num_of_nodes'], 110)

        graph0, lr0 = all_data[0]
        npt.assert_(isinstance(graph0, jraph.GraphsTuple))

        for i in range(0, 50)[::5]:
            graph, lr = all_data[i]
            positions = graph.nodes.get('positions')
            atomic_numbers = graph.nodes.get('atomic_numbers')
            senders = graph.senders
            receivers = graph.receivers
            cell = graph.globals.get('cell')
            cell_offset = graph.edges.get('cell_offset')
            energy = graph.globals.get('energy')
            forces = graph.nodes.get('forces')
            stress = graph.globals['stress']

            npt.assert_(isinstance(energy, np.ndarray))
            npt.assert_(isinstance(forces, np.ndarray))
            npt.assert_(isinstance(cell, np.ndarray))
            npt.assert_(isinstance(cell_offset, np.ndarray))
            npt.assert_(isinstance(atomic_numbers, np.ndarray))
            npt.assert_(isinstance(senders, np.ndarray))
            npt.assert_(isinstance(receivers, np.ndarray))

            npt.assert_equal(positions.dtype, np.float64)
            npt.assert_equal(atomic_numbers.dtype, np.int64)
            npt.assert_equal(energy.dtype, np.float64)
            npt.assert_equal(forces.dtype, np.float64)
            npt.assert_equal(senders.dtype, np.int64)
            npt.assert_equal(receivers.dtype, np.int64)
            npt.assert_equal(cell.dtype, np.float64)
            npt.assert_equal(cell_offset.dtype, np.int64)

            stress_expected = np.empty((1, 6))
            stress_expected[:] = np.nan
            npt.assert_equal(stress, stress_expected)

            num_atoms = 110
            npt.assert_equal(atomic_numbers.shape, (num_atoms,))
            npt.assert_equal(positions.shape, (num_atoms, 3))
            npt.assert_equal(energy.shape, (1,))
            npt.assert_equal(forces.shape, (num_atoms, 3))
            npt.assert_equal(cell.shape, (1, 3, 3))
            npt.assert_equal(len(senders), len(receivers))
            npt.assert_equal(cell_offset.shape, (len(senders), 3))


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_jraph_dynamically_batch(calculate_neighbors_lr: bool):
    """
    Loaded data is compatible with dynamically_batch_with_lr(...).

    Args:
        calculate_neighbors_lr ():

    Returns:

    """
    filename = 'test_data/ethanol.npz'
    f = TEST_DIR / filename

    loader = NpzDataLoaderSparse(input_file=f)
    all_data, data_stats = loader.load(
        cutoff=4.,
        calculate_neighbors_lr=calculate_neighbors_lr,
        cutoff_lr=75.
    )

    num_atoms = 9

    if calculate_neighbors_lr is False:
        for x, lr in dynamically_batch_with_lr(
            iter(all_data),
            n_node=3 * num_atoms + 1,
            n_edge=3 * data_stats['max_num_of_edges'] + 1,
            n_graph=10,
            n_pairs=0
        ):
            npt.assert_equal(lr['idx_i_lr'].shape, (0, ))
            npt.assert_equal(lr['idx_j_lr'].shape, (0, ))
    else:
        # Too small n_pairs - single graph won't fit
        with npt.assert_raises(RuntimeError):
            for x, lr in dynamically_batch_with_lr(
                iter(all_data),
                n_node=3 * num_atoms + 1,
                n_edge=3 * data_stats['max_num_of_edges'] + 1,
                n_graph=10,
                n_pairs=5
            ):
                pass

        for k in [1, 5, 9]:
            # n_node and n_edge have space for 9 graphs so n_pairs determines maximum.
            for x, lr in dynamically_batch_with_lr(
                    iter(all_data[:k*3]),  # always do three batches
                    n_node=9 * num_atoms + 1,
                    n_edge=9 * data_stats['max_num_of_edges'] + 1,
                    n_graph=10,
                    n_pairs=k * (num_atoms * num_atoms - num_atoms) + 1
            ):
                npt.assert_equal(np.sum(jraph.get_graph_padding_mask(x)), np.array([k]))


def test_data_load_input_folder_aggregates(tmp_path):
    """Folder mode aggregates every .npz in the directory.

    Regression: load() previously read self.input_file directly, so
    input_folder mode (used by from_config.data_loader_from_config for
    directory inputs) crashed with `self.input_file is None`.
    """
    src = TEST_DIR / 'test_data/ethanol.npz'
    # Two copies (2000 frames each) -> aggregated dataset of 4000 frames.
    shutil.copy(src, tmp_path / 'a.npz')
    shutil.copy(src, tmp_path / 'b.npz')

    loader = NpzDataLoaderSparse(input_folder=str(tmp_path))
    # cardinality already aggregated across files; load() must match it.
    npt.assert_equal(loader.cardinality(), 4000)

    # Pick one frame from each file (index 2500 is in the second file),
    # proving load() iterates the whole multi-file dataset with a global index
    # (and no longer crashes in folder mode). Only two graphs are built, so the
    # test stays fast despite the 4000-frame iteration.
    all_data, _ = loader.load(
        cutoff=5.,
        calculate_neighbors_lr=False,
        cutoff_lr=10.,
        pick_idx=np.array([0, 2500]),
    )
    npt.assert_equal(len(all_data), 2)
    graph0, lr0 = all_data[0]
    npt.assert_(isinstance(graph0, jraph.GraphsTuple))
    npt.assert_(isinstance(lr0, dict))
