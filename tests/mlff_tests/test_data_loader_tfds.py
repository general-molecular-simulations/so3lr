"""Tests for the TFDS dataloader and its optional runtime dependencies.

Validates compute_edges_np, standardize_dataset_fields, and create_graph_tuple_tf
for both non-PBC and PBC cases, including a real temporary TFDS disk round trip.
"""

import jraph
import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
tfds = pytest.importorskip('tensorflow_datasets')

from so3lr.mlff.data.dataloader_sparse_tfds import (
    QCMLDataLoaderSparseParallel,
    combine_epoch_datasets,
    compute_edges_np,
    convert_dataset_units_tf,
    create_graph_tuple_tf,
    force_is_valid_tf,
    minimum_distance_is_valid_tf,
    standardize_dataset_fields,
)


def _make_water_element():
    """Non-PBC water-like molecule (3 atoms)."""
    return {
        'atomic_numbers': tf.constant([8, 1, 1], dtype=tf.int32),
        'positions': tf.constant(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=tf.float32,
        ),
        'cell': tf.zeros((3, 3), dtype=tf.float32),
        'pbc': tf.constant([False, False, False], dtype=tf.bool),
        'energy': tf.constant([-0.5], dtype=tf.float32),
        'forces': tf.constant(
            [[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
            dtype=tf.float32,
        ),
        'theory_level': tf.constant([0], dtype=tf.int32),
        'charge': tf.constant([0], dtype=tf.int32),
        'multiplicity': tf.constant([1], dtype=tf.int32),
    }


def _make_nacl_element():
    """PBC NaCl-like structure (4 atoms in a cubic cell)."""
    a = 5.6
    positions = [
        [0.0, 0.0, 0.0],
        [a / 2, a / 2, 0.0],
        [a / 2, 0.0, a / 2],
        [0.0, a / 2, a / 2],
    ]
    return {
        'atomic_numbers': tf.constant([11, 17, 11, 17], dtype=tf.int32),
        'positions': tf.constant(positions, dtype=tf.float32),
        'cell': tf.constant(np.diag([a, a, a]), dtype=tf.float32),
        'pbc': tf.constant([True, True, True], dtype=tf.bool),
        'energy': tf.constant([-4.0], dtype=tf.float32),
        'forces': tf.constant(np.zeros((4, 3)), dtype=tf.float32),
        'theory_level': tf.constant([0], dtype=tf.int32),
        'charge': tf.constant([0], dtype=tf.int32),
        'multiplicity': tf.constant([1], dtype=tf.int32),
    }


def test_tfds_extra_builder_from_directory_round_trip(tmp_path):
    """The ``tfds`` extra must write and reopen a prepared dataset."""
    features = tfds.features.FeaturesDict({
        'atomic_numbers': tfds.features.Tensor(
            shape=(3,), dtype=np.int32
        ),
        'positions': tfds.features.Tensor(
            shape=(3, 3), dtype=np.float32
        ),
        'cell': tfds.features.Tensor(shape=(3, 3), dtype=np.float32),
        'pbc': tfds.features.Tensor(shape=(3,), dtype=np.bool_),
        'energy': tfds.features.Tensor(shape=(1,), dtype=np.float32),
        'forces': tfds.features.Tensor(shape=(3, 3), dtype=np.float32),
        'theory_level': tfds.features.Tensor(shape=(1,), dtype=np.int32),
        'charge': tfds.features.Tensor(shape=(1,), dtype=np.int32),
        'multiplicity': tfds.features.Tensor(shape=(1,), dtype=np.int32),
    })
    stored = tfds.dataset_builders.store_as_tfds_dataset(
        name='so3lr_extra_round_trip',
        version='1.0.0',
        features=features,
        split_datasets={
            'train': tf.data.Dataset.from_tensors(_make_water_element()),
        },
        data_dir=tmp_path,
        disable_shuffling=True,
    )

    reopened = tfds.builder_from_directory(stored.data_dir)
    assert reopened.info.full_name == 'so3lr_extra_round_trip/1.0.0'
    assert reopened.info.splits['train'].num_examples == 1

    sample = next(iter(reopened.as_dataset(split='train')))
    np.testing.assert_array_equal(sample['atomic_numbers'].numpy(), [8, 1, 1])
    np.testing.assert_allclose(
        sample['positions'].numpy(),
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    np.testing.assert_allclose(sample['energy'].numpy(), [-0.5])


# ---------------------------------------------------------------------------
# Test 1: Non-PBC graph creation
# ---------------------------------------------------------------------------
def test_create_graph_no_pbc():
    element = _make_water_element()
    graph, lr_dict = create_graph_tuple_tf(
        element, cutoff=5.0, calculate_neighbors_lr=False
    )

    # cell_offset exists with shape (num_edges, 3), all zeros
    cell_offset = graph.edges['cell_offset'].numpy()
    assert cell_offset.ndim == 2 and cell_offset.shape[1] == 3
    np.testing.assert_array_equal(cell_offset, 0)

    # cell in globals has shape (1, 3, 3), all zeros
    cell = graph.globals['cell'].numpy()
    assert cell.shape == (1, 3, 3)
    np.testing.assert_array_equal(cell, 0)

    # senders / receivers have same length, no self-connections
    senders = graph.senders.numpy()
    receivers = graph.receivers.numpy()
    assert len(senders) == len(receivers)
    assert len(senders) > 0
    assert not np.any(senders == receivers)

    # nodes carry positions, atomic_numbers, forces
    assert 'positions' in graph.nodes
    assert 'atomic_numbers' in graph.nodes
    assert 'forces' in graph.nodes

    # LR offsets empty
    lr_offsets = lr_dict['cell_offsets_lr'].numpy()
    assert lr_offsets.shape == (0, 3)


# ---------------------------------------------------------------------------
# Test 2: PBC graph creation
# ---------------------------------------------------------------------------
def test_create_graph_with_pbc():
    element = _make_nacl_element()
    graph, lr_dict = create_graph_tuple_tf(
        element, cutoff=4.0, calculate_neighbors_lr=True, cutoff_lr=6.0
    )

    # cell_offset exists with correct shape
    cell_offset = graph.edges['cell_offset'].numpy()
    assert cell_offset.ndim == 2 and cell_offset.shape[1] == 3
    # Some offsets should be non-zero for PBC
    assert np.any(cell_offset != 0), "PBC should produce non-zero cell offsets"

    # cell in globals has correct values
    cell = graph.globals['cell'].numpy()
    assert cell.shape == (1, 3, 3)
    np.testing.assert_allclose(np.diag(cell[0]), [5.6, 5.6, 5.6])

    # LR edges should exist and have non-zero offsets
    lr_offsets = lr_dict['cell_offsets_lr'].numpy()
    assert lr_offsets.ndim == 2 and lr_offsets.shape[1] == 3
    assert lr_offsets.shape[0] > 0
    assert np.any(lr_offsets != 0), "LR PBC should produce non-zero offsets"

    # LR edges >= SR edges (larger cutoff)
    num_sr = graph.n_edge.numpy()[0]
    num_lr = lr_dict['n_pairs'].numpy()[0]
    assert num_lr >= num_sr, f"LR edges ({num_lr}) should be >= SR edges ({num_sr})"


# ---------------------------------------------------------------------------
# Test 3: standardize_dataset_fields adds missing cell/pbc
# ---------------------------------------------------------------------------
def test_standardize_adds_missing_cell_pbc():
    # Dataset without cell or pbc keys
    raw = {
        'atomic_numbers': tf.constant([6, 1, 1], dtype=tf.int32),
        'positions': tf.constant([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32),
        'energy': tf.constant([-1.0], dtype=tf.float32),
        'forces': tf.constant(np.zeros((3, 3)), dtype=tf.float32),
        'theory_level': tf.constant([0], dtype=tf.int32),
        'charge': tf.constant([0], dtype=tf.int32),
        'multiplicity': tf.constant([1], dtype=tf.int32),
    }
    ds = tf.data.Dataset.from_tensors(raw)
    ds_std = standardize_dataset_fields(ds)

    for sample in ds_std.take(1):
        assert 'cell' in sample, "standardize should add 'cell'"
        assert 'pbc' in sample, "standardize should add 'pbc'"
        np.testing.assert_array_equal(sample['cell'].numpy(), np.zeros((3, 3)))
        np.testing.assert_array_equal(sample['pbc'].numpy(), [False, False, False])


# ---------------------------------------------------------------------------
# Test 6: standardize infers pbc=[True,True,True] when cell is present
# ---------------------------------------------------------------------------
def test_standardize_infers_pbc_from_cell():
    """Alexandria converter provides cell but no pbc — pbc should be inferred."""
    a = 5.6
    raw = {
        'atomic_numbers': tf.constant([11, 17], dtype=tf.int32),
        'positions': tf.constant([[0.0, 0.0, 0.0], [2.8, 2.8, 0.0]], dtype=tf.float32),
        'cell': tf.constant(np.diag([a, a, a]), dtype=tf.float32),
        'energy': tf.constant([-2.0], dtype=tf.float32),
        'forces': tf.constant(np.zeros((2, 3)), dtype=tf.float32),
        'theory_level': tf.constant([0], dtype=tf.int32),
        'charge': tf.constant([0], dtype=tf.int32),
        'multiplicity': tf.constant([1], dtype=tf.int32),
    }
    ds = tf.data.Dataset.from_tensors(raw)
    ds_std = standardize_dataset_fields(ds)

    for sample in ds_std.take(1):
        assert 'pbc' in sample, "standardize should add 'pbc'"
        np.testing.assert_array_equal(
            sample['pbc'].numpy(), [True, True, True],
            err_msg="pbc should be inferred as [True,True,True] from non-zero cell"
        )


# ---------------------------------------------------------------------------
# Test 4: numpy iterator strips PBC fields for non-PBC
# ---------------------------------------------------------------------------
def test_numpy_tuple_iterator_strips_non_pbc():
    element = _make_water_element()
    graph, lr_dict = create_graph_tuple_tf(
        element, cutoff=5.0, calculate_neighbors_lr=False
    )

    # Build a one-element TF dataset and iterate as numpy (mimics _numpy_tuple_iterator)
    ds = tf.data.Dataset.from_tensors((graph, lr_dict))
    for np_graph, np_lr in ds.as_numpy_iterator():
        cell = np_graph.globals.get('cell')
        has_pbc = cell is not None and np.any(cell != 0.0)
        assert not has_pbc

        # Apply the same stripping logic as _numpy_tuple_iterator
        if not has_pbc:
            globals_new = {k: v for k, v in np_graph.globals.items() if k != 'cell'}
            edges_new = {k: v for k, v in np_graph.edges.items() if k != 'cell_offset'}
            np_graph = np_graph._replace(globals=globals_new, edges=edges_new)

        offsets = np.asarray(np_lr['cell_offsets_lr'])
        cell_offsets_lr = offsets if has_pbc and offsets.shape[0] > 0 else None

        assert 'cell' not in np_graph.globals
        assert 'cell_offset' not in np_graph.edges
        assert cell_offsets_lr is None


# ---------------------------------------------------------------------------
# Test 5: numpy iterator keeps PBC fields for periodic systems
# ---------------------------------------------------------------------------
def test_numpy_tuple_iterator_keeps_pbc():
    element = _make_nacl_element()
    graph, lr_dict = create_graph_tuple_tf(
        element, cutoff=4.0, calculate_neighbors_lr=True, cutoff_lr=6.0
    )

    ds = tf.data.Dataset.from_tensors((graph, lr_dict))
    for np_graph, np_lr in ds.as_numpy_iterator():
        cell = np_graph.globals.get('cell')
        has_pbc = cell is not None and np.any(cell != 0.0)
        assert has_pbc

        # Do NOT strip for PBC
        offsets = np.asarray(np_lr['cell_offsets_lr'])
        cell_offsets_lr = offsets if has_pbc and offsets.shape[0] > 0 else None

        assert 'cell' in np_graph.globals
        assert 'cell_offset' in np_graph.edges
        assert cell_offsets_lr is not None
        assert cell_offsets_lr.shape[1] == 3


@pytest.mark.parametrize('pbc', [[False, False, False], [True, False, True]])
def test_preprocess_preserves_explicit_pbc_with_nonzero_cell(pbc):
    element = _make_water_element()
    element['cell'] = tf.constant(np.diag([10.0] * 3), tf.float32)
    element['pbc'] = tf.constant(pbc)
    graph, lr = next(iter(QCMLDataLoaderSparseParallel._preprocess(
        standardize_dataset_fields(tf.data.Dataset.from_tensors(element)),
        batch_max_num_nodes=8, batch_max_num_edges=16,
        batch_max_num_graphs=4, batch_max_num_pairs=16,
        cutoff=2.0, calculate_neighbors_lr=True, cutoff_lr=2.0,
        max_force_filter=None, min_distance_filter=None,
        energy_unit=1.0, length_unit=1.0, dipole_vec_unit=1.0,
        train_seed=0, shuffle=False,
    )))

    np.testing.assert_array_equal(graph.globals['pbc'][0], pbc)
    assert ('cell' in graph.globals) is any(pbc)
    assert ('cell_offset' in graph.edges) is any(pbc)
    assert (lr['cell_offsets_lr'] is not None) is any(pbc)


def test_periodic_batch_preserves_offsets_after_graph_with_no_lr_neighbors():
    elements = []
    for positions in (
        [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
        [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]],
    ):
        element = _make_nacl_element()
        element['atomic_numbers'] = tf.constant([11, 17], tf.int32)
        element['positions'] = tf.constant(positions, tf.float32)
        element['forces'] = tf.zeros((2, 3), tf.float32)
        element['cell'] = tf.constant(np.diag([10.0] * 3), tf.float32)
        elements.append(element)
    dataset = tf.data.Dataset.from_tensors(elements[0]).concatenate(
        tf.data.Dataset.from_tensors(elements[1])
    )
    graph, lr = next(iter(QCMLDataLoaderSparseParallel._preprocess(
        standardize_dataset_fields(dataset),
        batch_max_num_nodes=8, batch_max_num_edges=8,
        batch_max_num_graphs=4, batch_max_num_pairs=8,
        cutoff=1.0, calculate_neighbors_lr=True, cutoff_lr=1.0,
        max_force_filter=None, min_distance_filter=None,
        energy_unit=1.0, length_unit=1.0, dipole_vec_unit=1.0,
        train_seed=0, shuffle=False,
    )))

    assert int(np.sum(jraph.get_graph_padding_mask(graph))) == 2
    assert lr['cell_offsets_lr'] is not None
    np.testing.assert_array_equal(
        np.sort(lr['cell_offsets_lr'][:2, 0]), [-1, 1]
    )


def test_non_pbc_edges_match_ase_neighbor_list():
    """The numpy all-pairs rewrite must reproduce ase.neighbor_list exactly.

    The property assertions above cannot catch a systematically wrong edge
    list (e.g. one direction of each pair dropped), so compare pair sets
    directly on random gas-phase geometries, including a pair close to the
    cutoff boundary.
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    cutoff = 5.0
    rng = np.random.RandomState(0)
    geometries = [rng.uniform(0.0, 8.0, (n, 3)) for n in (2, 5, 16)]
    # A pair near the cutoff boundary (exact ties are float-representation
    # dependent in ase's binning, so probe just inside and just outside).
    geometries.append(np.array([[0.0, 0.0, 0.0], [cutoff - 1e-6, 0.0, 0.0]]))
    geometries.append(np.array([[0.0, 0.0, 0.0], [cutoff + 1e-6, 0.0, 0.0]]))

    for positions in geometries:
        # Both sides must see bit-identical coordinates, or a pair within
        # float32 rounding of the cutoff would make the comparison flaky.
        positions = positions.astype(np.float32)
        n = len(positions)
        i, j, offsets = compute_edges_np(
            positions,
            np.full(n, 6, dtype=np.int32),
            np.zeros((3, 3), dtype=np.float32),
            np.array([False, False, False]),
            cutoff,
        )
        mol = Atoms(numbers=[6] * n, positions=positions)
        i_ref, j_ref = neighbor_list('ij', mol, cutoff, self_interaction=False)
        assert set(zip(i.tolist(), j.tolist())) == set(zip(i_ref.tolist(), j_ref.tolist()))
        np.testing.assert_array_equal(offsets, np.zeros((len(i), 3), dtype=np.int32))

    # n = 0 and n = 1 return empty edge lists with the right shapes/dtypes.
    for n in (0, 1):
        i, j, offsets = compute_edges_np(
            np.zeros((n, 3), dtype=np.float32),
            np.full(n, 6, dtype=np.int32),
            np.zeros((3, 3), dtype=np.float32),
            np.array([False, False, False]),
            cutoff,
        )
        assert i.shape == (0,) and i.dtype == np.int32
        assert j.shape == (0,) and j.dtype == np.int32
        assert offsets.shape == (0, 3) and offsets.dtype == np.int32


def test_weighted_epoch_dataset_is_finite_and_seeded():
    def collect(seed):
        dataset = combine_epoch_datasets(
            [
                tf.data.Dataset.from_tensor_slices([0, 1]),
                tf.data.Dataset.from_tensor_slices([10, 11, 12, 13]),
            ],
            weights=[1.0, 3.0],
            mode='train',
            epoch_size=6,
            seed=seed,
        )
        return list(dataset.as_numpy_iterator())

    first = collect(17)
    assert len(first) == 6
    assert first == collect(17)

    validation = combine_epoch_datasets(
        [
            tf.data.Dataset.from_tensor_slices([0, 1]),
            tf.data.Dataset.from_tensor_slices([10, 11, 12]),
        ],
        weights=[1.0, 100.0],
        mode='validation',
        epoch_size=5,
        seed=3,
    )
    assert sorted(validation.as_numpy_iterator()) == [0, 1, 10, 11, 12]


@pytest.mark.parametrize(
    ('forces', 'expected'),
    [
        ([[25.0, -25.0, 0.0]], True),
        ([[-25.01, -1.0, -2.0]], False),
        ([[np.nan, 0.0, 0.0]], False),
        ([[np.inf, 0.0, 0.0]], False),
    ],
)
def test_force_filter_uses_finite_absolute_maximum(forces, expected):
    actual = force_is_valid_tf(tf.constant(forces, tf.float32), 25.0)
    assert bool(actual.numpy()) is expected


def test_minimum_distance_filter_uses_minimum_image():
    positions = tf.constant([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], tf.float32)
    cell = tf.constant(np.diag([10.0, 10.0, 10.0]), tf.float32)

    periodic = minimum_distance_is_valid_tf(
        positions, cell, tf.constant([True, False, False]), 0.5
    )
    nonperiodic = minimum_distance_is_valid_tf(
        positions, cell, tf.constant([False, False, False]), 0.5
    )
    assert not bool(periodic.numpy())
    assert bool(nonperiodic.numpy())


def test_units_are_converted_before_periodic_neighbor_construction():
    element = _make_nacl_element()
    element.update({
        'positions': tf.constant([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]], tf.float32),
        'atomic_numbers': tf.constant([11, 17], tf.int32),
        'cell': tf.constant(np.diag([1.0, 1.0, 1.0]), tf.float32),
        'energy': tf.constant([4.0], tf.float32),
        'forces': tf.constant([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], tf.float32),
        'stress': tf.constant(np.ones(6), tf.float32),
        'dipole_vec': tf.constant([1.0, 2.0, 3.0], tf.float32),
    })
    converted = convert_dataset_units_tf(
        element, energy_unit=2.0, length_unit=0.5, dipole_vec_unit=3.0
    )
    graph, _ = create_graph_tuple_tf(converted, cutoff=0.3)

    np.testing.assert_allclose(graph.nodes['positions'].numpy(), [[0.125, 0, 0], [0.375, 0, 0]])
    np.testing.assert_allclose(graph.globals['cell'].numpy()[0], np.diag([0.5] * 3))
    np.testing.assert_allclose(graph.globals['energy'].numpy(), [8.0])
    np.testing.assert_allclose(graph.nodes['forces'].numpy()[:, 0], [4.0, -4.0])
    np.testing.assert_allclose(graph.globals['stress'].numpy(), np.full((1, 6), 16.0))
    np.testing.assert_allclose(graph.globals['dipole_vec'].numpy(), [[3.0, 6.0, 9.0]])
    assert np.any(graph.edges['cell_offset'].numpy() != 0)


def test_preprocess_filters_before_batching():
    elements = {
        'atomic_numbers': tf.constant([[1, 1], [1, 1], [1, 1]], tf.int32),
        'positions': tf.constant([
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]],
        ], tf.float32),
        'cell': tf.constant([np.zeros((3, 3)), np.zeros((3, 3)), np.diag([10.0] * 3)], tf.float32),
        'pbc': tf.constant([[False] * 3, [False] * 3, [True, False, False]]),
        'energy': tf.constant([[1.0], [2.0], [3.0]], tf.float32),
        'forces': tf.constant([
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            # Raw magnitude 7 is below the threshold, but converts to 28 eV/A.
            [[-7.0, 0.0, 0.0], [-7.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ], tf.float32),
        'theory_level': tf.constant([[0], [0], [0]], tf.int32),
        'charge': tf.constant([[0], [0], [0]], tf.int32),
        'multiplicity': tf.constant([[1], [1], [1]], tf.int32),
    }
    batches = list(QCMLDataLoaderSparseParallel._preprocess(
        standardize_dataset_fields(tf.data.Dataset.from_tensor_slices(elements)),
        batch_max_num_nodes=8,
        batch_max_num_edges=16,
        batch_max_num_graphs=4,
        batch_max_num_pairs=0,
        cutoff=3.0,
        calculate_neighbors_lr=False,
        cutoff_lr=None,
        max_force_filter=25.0,
        min_distance_filter=0.5,
        energy_unit=2.0,
        length_unit=0.5,
        dipole_vec_unit=1.0,
        train_seed=0,
    ))

    assert len(batches) == 1
    graph, _ = batches[0]
    np.testing.assert_allclose(graph.globals['energy'][0], 2.0)
    assert int(np.sum(jraph.get_graph_padding_mask(graph))) == 1


def test_validation_preprocess_order_is_seed_independent():
    elements = {
        'atomic_numbers': tf.constant([[1, 1]] * 3, tf.int32),
        'positions': tf.constant([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ], tf.float32),
        'cell': tf.constant([np.zeros((3, 3))] * 3, tf.float32),
        'pbc': tf.constant([[False] * 3] * 3),
        'energy': tf.constant([[1.0], [2.0], [3.0]], tf.float32),
        'forces': tf.zeros((3, 2, 3), tf.float32),
        'theory_level': tf.constant([[0], [0], [0]], tf.int32),
        'charge': tf.constant([[0], [0], [0]], tf.int32),
        'multiplicity': tf.constant([[1], [1], [1]], tf.int32),
    }

    def collect(seed):
        batches = QCMLDataLoaderSparseParallel._preprocess(
            standardize_dataset_fields(
                tf.data.Dataset.from_tensor_slices(elements)
            ),
            batch_max_num_nodes=8,
            batch_max_num_edges=16,
            batch_max_num_graphs=4,
            batch_max_num_pairs=0,
            cutoff=2.0,
            calculate_neighbors_lr=False,
            cutoff_lr=None,
            max_force_filter=None,
            min_distance_filter=None,
            energy_unit=1.0,
            length_unit=1.0,
            dipole_vec_unit=1.0,
            train_seed=seed,
            shuffle=False,
        )
        graph, _ = next(iter(batches))
        mask = np.asarray(jraph.get_graph_padding_mask(graph), dtype=bool)
        return np.asarray(graph.globals['energy'])[mask].reshape(-1)

    np.testing.assert_array_equal(collect(1), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(collect(999), [1.0, 2.0, 3.0])
