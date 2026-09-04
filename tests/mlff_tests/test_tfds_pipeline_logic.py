"""Portable tests for TFDS-pipeline logic that does not require TensorFlow."""

import jraph
import numpy as np
import numpy.testing as npt
import pytest
import queue
import threading

from so3lr.mlff.data.dataloader_sparse_tfds import (
    QCMLDataLoaderSparseParallel,
    epoch_size_from_dataset_sizes,
    minimum_distance_np,
)
from so3lr.mlff.data.transformations import unit_conversion


def test_weighted_epoch_size_is_finite_and_ignores_disabled_sources():
    sizes = [3, 5, 7]
    weights = [1.0, 0.0, 2.0]

    assert epoch_size_from_dataset_sizes(sizes, weights, 'train') == 10
    assert epoch_size_from_dataset_sizes(sizes, weights, 'validation') == 15

    with pytest.raises(ValueError, match='same length'):
        epoch_size_from_dataset_sizes(sizes, weights[:2], 'train')
    with pytest.raises(ValueError, match='non-negative'):
        epoch_size_from_dataset_sizes([3, -1], [1.0, 1.0], 'train')


def test_epoch_seed_progression_is_deterministic_and_probe_safe():
    loader = QCMLDataLoaderSparseParallel.__new__(QCMLDataLoaderSparseParallel)
    loader.train_seed = 41
    loader._epoch_counters = {'train': 0, 'validation': 0}
    loader._generator_sync = lambda split, mode, seed: seed

    assert loader.next_epoch('train', 'train', parallel=False, advance_epoch=False) == 41
    assert loader.next_epoch('train', 'train', parallel=False) == 41
    assert loader.next_epoch('train', 'train', parallel=False) == 42
    assert loader.next_epoch('validation', 'validation', parallel=False) == 41


def test_safe_put_honours_worker_cancellation():
    output_queue = queue.Queue(maxsize=1)
    output_queue.put('already full')
    cancelled = threading.Event()
    cancelled.set()

    assert not QCMLDataLoaderSparseParallel._safe_put(
        output_queue, 'must not block', cancelled
    )


def test_minimum_distance_respects_exact_pbc_axes():
    positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    cell = np.diag([10.0, 10.0, 10.0])

    assert minimum_distance_np(positions, cell, [False, False, False]) == pytest.approx(9.8)
    assert minimum_distance_np(positions, cell, [True, False, False]) == pytest.approx(0.2)
    assert minimum_distance_np(positions, cell, [False, True, True]) == pytest.approx(9.8)
    assert minimum_distance_np(positions[:1], cell, [True, True, True]) == np.inf
    assert np.isnan(minimum_distance_np([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]], cell, [False] * 3))


def test_unit_conversion_scales_periodic_cell_and_labels_together():
    graph = jraph.GraphsTuple(
        nodes={
            'positions': np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
            'forces': np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]]),
        },
        edges={'cell_offset': np.array([[1, 0, 0]], dtype=np.int32)},
        receivers=np.array([0]),
        senders=np.array([1]),
        globals={
            'energy': np.array([4.0]),
            'stress': np.ones((1, 6)),
            'dipole_vec': np.array([[1.0, 2.0, 3.0]]),
            'cell': np.array([np.diag([1.0, 2.0, 3.0])]),
        },
        n_node=np.array([2]),
        n_edge=np.array([1]),
    )

    converted = list(unit_conversion(
        [graph], energy_unit=2.0, length_unit=0.5, dipole_vec_unit=3.0
    ))[0]

    npt.assert_allclose(converted.nodes['positions'], [[0.125, 0.0, 0.0], [0.375, 0.0, 0.0]])
    npt.assert_allclose(converted.globals['cell'], [np.diag([0.5, 1.0, 1.5])])
    npt.assert_allclose(converted.nodes['forces'], [[4.0, -8.0, 12.0], [-4.0, 8.0, -12.0]])
    npt.assert_allclose(converted.globals['energy'], [8.0])
    npt.assert_allclose(converted.globals['stress'], np.full((1, 6), 16.0))
    npt.assert_allclose(converted.globals['dipole_vec'], [[3.0, 6.0, 9.0]])
    npt.assert_array_equal(converted.edges['cell_offset'], [[1, 0, 0]])
