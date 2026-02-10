import numpy as np
import jraph
import numpy.testing as npt
from so3lr.mlff.data import AseDataLoaderSparse
from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr
import pkg_resources
import pytest


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_data_load_no_pbc(calculate_neighbors_lr: bool):
    filename = 'test_data/data_set.xyz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = AseDataLoaderSparse(input_file=f)
    all_data, data_stats = loader.load(
        cutoff=4.,
        calculate_neighbors_lr=calculate_neighbors_lr,
        cutoff_lr=75.
    )

    npt.assert_equal(len(all_data), 100)
    npt.assert_equal(loader.cardinality(), 100)
    npt.assert_equal(data_stats['max_num_of_nodes'], 24)

    # all_data is now a list of (graph, long_range) tuples
    graph0, lr0 = all_data[0]
    npt.assert_(isinstance(graph0, jraph.GraphsTuple))
    npt.assert_(isinstance(lr0, jraph.GraphsTuple))

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

        num_pairs = lr.n_edge
        idx_i_lr = lr.receivers
        idx_j_lr = lr.senders

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

        num_atoms = 24
        hirshfeld_ratios_expected = np.empty((num_atoms, ))
        hirshfeld_ratios_expected[:] = np.nan
        npt.assert_equal(hirshfeld_ratios, hirshfeld_ratios_expected)

        dipole_vec_expected = np.empty((1, 3))
        dipole_vec_expected[:] = np.nan
        npt.assert_equal(dipole_vec, dipole_vec_expected)

        npt.assert_equal(atomic_numbers.shape, (num_atoms,))
        npt.assert_equal(positions.shape, (num_atoms, 3))
        npt.assert_equal(energy.shape, (1,))
        npt.assert_equal(forces.shape, (num_atoms, 3))
        npt.assert_equal(total_charge.shape, (1, ))
        npt.assert_equal(total_charge, 0)
        npt.assert_equal(num_unpaired_electrons.shape, (1, ))
        npt.assert_equal(num_unpaired_electrons, 0)
        npt.assert_equal(len(senders), len(receivers))

        npt.assert_equal(len(idx_j_lr), len(idx_j_lr))
        if calculate_neighbors_lr:
            npt.assert_equal(num_pairs, num_atoms*num_atoms - num_atoms)
            npt.assert_equal(len(idx_i_lr), num_pairs)
            npt.assert_equal(len(idx_j_lr), num_pairs)
            npt.assert_equal(np.isnan(idx_i_lr).any(), False)
            npt.assert_equal(np.isnan(idx_j_lr).any(), False)
        else:
            npt.assert_equal(idx_i_lr, np.array([]).reshape(-1))
            npt.assert_equal(idx_j_lr, np.array([]).reshape(-1))
            npt.assert_equal(num_pairs, 0)


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_data_load_with_pbc(calculate_neighbors_lr: bool):
    filename = 'test_data/data_set_pbc.xyz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = AseDataLoaderSparse(input_file=f)
    if calculate_neighbors_lr:
        # PBC long-range is now supported, so we just load and check
        all_data, data_stats = loader.load(
            cutoff=4.,
            calculate_neighbors_lr=calculate_neighbors_lr,
            cutoff_lr=15.
        )

        npt.assert_equal(len(all_data), 50)
        npt.assert_equal(data_stats['max_num_of_nodes'], 192)

        graph0, lr0 = all_data[0]
        npt.assert_(isinstance(graph0, jraph.GraphsTuple))
        npt.assert_(isinstance(lr0, jraph.GraphsTuple))
        # PBC long-range should have cell_offsets
        npt.assert_(lr0.edges is not None and lr0.edges.get('cell_offsets') is not None)
    else:
        all_data, data_stats = loader.load(
            cutoff=4.,
            calculate_neighbors_lr=calculate_neighbors_lr,
            cutoff_lr=15.
        )

        npt.assert_equal(len(all_data), 50)
        npt.assert_equal(data_stats['max_num_of_nodes'], 192)

        graph0, lr0 = all_data[0]
        npt.assert_(isinstance(graph0, jraph.GraphsTuple))

        for i in range(0, 50)[::5]:
            graph, lr = all_data[i]
            positions = graph.nodes.get('positions')
            atomic_numbers = graph.nodes.get('atomic_numbers')
            senders = graph.senders
            receivers = graph.receivers
            cell = graph.edges.get('cell')
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

            num_atoms = 192
            npt.assert_equal(atomic_numbers.shape, (num_atoms,))
            npt.assert_equal(positions.shape, (num_atoms, 3))
            npt.assert_equal(energy.shape, (1,))
            npt.assert_equal(forces.shape, (num_atoms, 3))
            npt.assert_equal(cell.shape, (len(senders), 3, 3))
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
    filename = 'test_data/data_set.xyz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = AseDataLoaderSparse(input_file=f)
    all_data, data_stats = loader.load(
        cutoff=4.,
        calculate_neighbors_lr=calculate_neighbors_lr,
        cutoff_lr=75.
    )

    num_atoms = 24

    if calculate_neighbors_lr is False:
        for x, lr in dynamically_batch_with_lr(
            iter(all_data),
            n_node=75,
            n_edge=1500,
            n_graph=10,
            n_pairs=0
        ):
            npt.assert_equal(
                lr.n_edge,
                np.zeros(len(jraph.get_graph_padding_mask(x)))
            )
            npt.assert_equal(lr.receivers.shape, (0, ))
            npt.assert_equal(lr.senders.shape, (0, ))
    else:
        # Too small n_pairs - single graph won't fit
        with npt.assert_raises(RuntimeError):
            for x, lr in dynamically_batch_with_lr(
                iter(all_data),
                n_node=75,
                n_edge=1500,
                n_graph=10,
                n_pairs=5
            ):
                pass

        for k in [1, 5, 9]:
            # n_node and n_edge have space for 9 graphs so n_pairs determines maximum.
            for x, lr in dynamically_batch_with_lr(
                    iter(all_data[:k*3]),  # always do three batches
                    n_node=num_atoms * 9 + 1,
                    n_edge=num_atoms * num_atoms * 9,
                    n_graph=10,
                    n_pairs=k * (num_atoms * num_atoms - num_atoms) + 1
            ):
                npt.assert_equal(np.sum(jraph.get_graph_padding_mask(x)), np.array([k]))
