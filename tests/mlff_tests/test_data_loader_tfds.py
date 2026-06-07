"""Tests for TFDS dataloader PBC support.

Validates compute_edges_np, standardize_dataset_fields, and create_graph_tuple_tf
for both non-PBC and PBC cases using synthetic tf.data.Dataset elements.
"""

import numpy as np
import pytest
import tensorflow as tf

from so3lr.mlff.data.dataloader_sparse_tfds import (
    compute_edges_np,
    create_graph_tuple_tf,
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
