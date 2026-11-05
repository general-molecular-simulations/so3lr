"""Portable tests for TFDS-pipeline logic that does not require TensorFlow."""

import jraph
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import optax
import orbax.checkpoint as ocp
import pytest
import queue
import threading
from types import SimpleNamespace

from ase import Atoms
from ml_collections import config_dict

from so3lr.mlff.data.dataloader_sparse_tfds import (
    QCMLDataLoaderSparseParallel,
    epoch_size_from_dataset_sizes,
    minimum_distance_is_valid_np,
)
from so3lr.mlff.data.transformations import unit_conversion
from so3lr.mlff.config import from_config
from so3lr.mlff.config.from_config import prepare_training_and_validation_data
from so3lr.mlff.utils import checkpoint_utils
from so3lr.mlff.utils import training_utils
from so3lr.mlff.utils.training_utils import (
    _make_tfds_training_state,
    _restore_tfds_training_state,
)


def test_weighted_epoch_size_is_finite_and_ignores_disabled_sources():
    sizes = [3, 5, 7]
    weights = [1.0, 0.0, 2.0]

    assert epoch_size_from_dataset_sizes(sizes, weights, 'train') == 10
    assert epoch_size_from_dataset_sizes(sizes, weights, 'validation') == 15

    with pytest.raises(ValueError, match='same length'):
        epoch_size_from_dataset_sizes(sizes, weights[:2], 'train')
    with pytest.raises(ValueError, match='non-negative'):
        epoch_size_from_dataset_sizes([3, -1], [1.0, 1.0], 'train')


@pytest.mark.parametrize('global_counts', [(None, None), (950, 50)])
def test_tfds_preparation_uses_existing_splits_not_global_counts(
        tmp_path, global_counts):
    class PreparedLoader:
        def cardinality(self):
            raise AssertionError('TFDS split selection is owned by the loader')

    loader = PreparedLoader()
    config = config_dict.ConfigDict({
        'workdir': str(tmp_path),
        'model': {},
        'data': {
            'datasets': [{'path': '/prepared/tfds', 'num_train': 3, 'num_valid': 1}],
            'energy_unit': 'eV', 'length_unit': 'Angstrom',
            'dipole_vec_unit': 'e * Angstrom',
            'neighbors_lr_bool': False, 'shift_mode': None, 'energy_shifts': None,
        },
        'training': {
            'num_train': global_counts[0], 'num_valid': global_counts[1],
            'batch_max_num_nodes': 8, 'batch_max_num_edges': 16,
            'batch_max_num_pairs': None,
        },
    })
    training, validation, stats = prepare_training_and_validation_data(
        config, loader, tf_record_present=True
    )
    assert training is validation is loader
    assert stats is None


def test_kspace_batch_conversion_validates_later_training_records(monkeypatch):
    constants = {'k_grid': 'test-grid'}
    monkeypatch.setattr(from_config, '_compute_kspace_constants', lambda *args: constants)
    converted = []
    monkeypatch.setattr(
        from_config.jraph_utils, 'graph_to_batch_fn',
        lambda graph, lr, **kwargs: converted.append(graph) or kwargs,
    )
    convert = from_config._make_graph_to_batch_fn(None, None, True)
    periodic = SimpleNamespace(
        globals={'pbc': np.array([[True, True, True], [False, False, False]])},
        n_node=np.array([2, 1]),
    )
    assert convert(periodic, {}) == {'kspace_constants': constants}
    mixed = SimpleNamespace(
        globals={'pbc': np.array([[True, True, True], [True, False, True]])},
        n_node=np.array([2, 2]),
    )
    with pytest.raises(ValueError, match='mixed-axis'):
        convert(mixed, {})
    assert converted == [periodic]


def test_epoch_index_selects_train_seed_and_validation_is_fixed():
    loader = QCMLDataLoaderSparseParallel.__new__(QCMLDataLoaderSparseParallel)
    loader.train_seed = 41
    loader._generator_sync = lambda split, mode, seed: seed

    assert loader.next_epoch('train', 'train', parallel=False) == 41
    assert loader.next_epoch('train', 'train', parallel=False, epoch_index=7) == 48
    assert loader.next_epoch('validation', 'validation', epoch_index=7) == 41


# Two of the four combinations: the plain optimizer with a current checkpoint,
# and the masked optimizer with a legacy fingerprint leaf. The two options act on
# different parts of the restored state, so the cross product adds nothing.
@pytest.mark.parametrize('masked_sgd, legacy_fingerprint', [(False, False), (True, True)])
def test_atomic_training_state_round_trip(tmp_path, masked_sgd, legacy_fingerprint):
    params = {'w': jnp.array([1.0, -2.0])}
    optimizer = optax.adam(1e-3)
    if masked_sgd:
        optimizer = training_utils.freeze_parameters(
            optax.sgd(1e-3), trainable_subset_keys=['w']
        )
    optimizer_state = optimizer.init(params)
    state = _make_tfds_training_state(
        step=120,
        raw_params=params,
        optimizer_state=optimizer_state,
        ema_params={'w': jnp.array([0.9, -1.8])},
        jax_rng=jax.random.PRNGKey(7),
        epoch=3,
        batches_consumed=17,
        processed_graphs=71,
        processed_nodes=811,
        best_eval_loss=0.25,
        evals_without_improvement=2,
        stop_training=False,
        plateau_best_loss=0.2,
        plateau_wait=3,
        plateau_cooldown_counter=1,
        plateau_scale=0.5,
        plateau_active=True,
        reduce_lr_enabled=True,
        stage_two_active=True,
        header_printed=True,
        use_ema=True,
    )
    if legacy_fingerprint:
        # Checkpoints written by earlier versions carry a 32-byte digest leaf.
        state['restart_fingerprint'] = np.arange(32, dtype=np.uint8)
    manager = checkpoint_utils.make_training_checkpoint_manager(tmp_path)
    assert manager.save(
        120,
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(state['ema_params']),
            training_state=ocp.args.StandardSave(state),
        ),
    )
    manager.wait_until_finished()

    restored = _restore_tfds_training_state(
        manager, step=120, optimizer=optimizer
    )
    assert restored['checkpoint_step'] == 120
    npt.assert_allclose(restored['raw_params']['w'], params['w'])
    npt.assert_allclose(restored['ema_params']['w'], [0.9, -1.8])
    assert type(restored['optimizer_state']) is type(optimizer_state)
    assert restored['epoch'] == 3
    assert restored['batches_consumed'] == 17
    assert restored['processed_graphs'] == 71
    assert restored['processed_nodes'] == 811
    assert restored['best_eval_loss'] == pytest.approx(0.25)
    assert restored['evals_without_improvement'] == 2
    assert restored['plateau_best_loss'] == pytest.approx(0.2)
    assert restored['plateau_wait'] == 3
    assert restored['plateau_cooldown_counter'] == 1
    assert restored['plateau_scale'] == pytest.approx(0.5)
    assert restored['plateau_active']
    assert restored['stage_two_active']
    assert restored['header_printed']
    npt.assert_array_equal(restored['jax_rng'], jax.random.PRNGKey(7))
    manager.close()

    # Inference asks only for the deployable item and remains compatible with
    # the richer atomic checkpoint.
    deployed = checkpoint_utils.load_params_from_checkpoint(tmp_path)
    npt.assert_allclose(deployed['w'], [0.9, -1.8])


def test_params_only_checkpoint_has_no_training_state(tmp_path):
    manager = checkpoint_utils.make_checkpoint_manager(tmp_path)
    assert manager.save(
        4,
        args=ocp.args.Composite(
            params=ocp.args.StandardSave({'w': jnp.array(1.0)})
        ),
    )
    manager.wait_until_finished()
    manager.close()

    loaded = checkpoint_utils.load_params_from_checkpoint(tmp_path)
    assert float(loaded['w']) == pytest.approx(1.0)

    training_manager = checkpoint_utils.make_training_checkpoint_manager(tmp_path)
    assert _restore_tfds_training_state(
        training_manager, step=4, optimizer=optax.sgd(1e-2)
    ) is None
    training_manager.close()


def test_composite_checkpoint_prunes_params_and_training_state_together(tmp_path):
    manager = checkpoint_utils.make_training_checkpoint_manager(
        tmp_path,
        ocp.CheckpointManagerOptions(step_prefix='ckpt', max_to_keep=1),
    )
    for step in (1, 2):
        assert manager.save(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave({'w': jnp.array(step)}),
                training_state=ocp.args.StandardSave({'step': jnp.array(step)}),
            ),
        )
        manager.wait_until_finished()
    manager.close()

    assert not (tmp_path / 'ckpt_1').exists()
    assert (tmp_path / 'ckpt_2' / 'params').is_dir()
    assert (tmp_path / 'ckpt_2' / 'training_state').is_dir()


def test_fit_from_iterator_restart_matches_uninterrupted_updates(
        tmp_path, monkeypatch):
    """A stop at an epoch checkpoint must not alter the next optimizer step."""

    class FiniteIterator:
        def __init__(self, values):
            self.values = list(values)
            self.calls = []

        def next_epoch(self, split, mode, epoch_index=None, **kwargs):
            self.calls.append((mode, epoch_index))
            values = self.values if mode == 'train' else [0.0]
            return iter((value, None) for value in values)

    class Model:
        @staticmethod
        def init(rng, batch):
            return {'w': jnp.array(0.0)}

    optimizer = optax.sgd(learning_rate=0.1, momentum=0.9)

    def fake_training_step_factory(selected_optimizer, loss_fn, log_gradients):
        def step(params, optimizer_state, batch):
            gradients = {'w': batch['gradient']}
            updates, optimizer_state = selected_optimizer.update(
                gradients, optimizer_state, params
            )
            params = optax.apply_updates(params, updates)
            return params, optimizer_state, {'loss': jnp.square(params['w'])}
        return step

    def fake_validation_step_factory(val_fn):
        return lambda params, batch: {'loss': jnp.square(params['w'])}

    monkeypatch.setattr(
        training_utils, 'make_training_step_fn', fake_training_step_factory
    )
    monkeypatch.setattr(
        training_utils, 'make_validation_step_fn', fake_validation_step_factory
    )
    monkeypatch.setattr(
        training_utils.jraph,
        'get_number_of_padding_with_graphs_nodes',
        lambda graph: 0,
    )

    def graph_to_batch(graph, long_range):
        return {
            'num_of_non_padded_graphs': 1,
            'gradient': jnp.asarray(graph),
        }

    def run(checkpoint_dir, epochs, training, validation, allow_restart=False):
        training_utils.fit_from_iterator(
            model=Model(),
            optimizer=optimizer,
            loss_fn=None,
            val_fn=None,
            graph_to_batch_fn=graph_to_batch,
            training_iterator=training,
            validation_iterator=validation,
            batch_max_num_nodes=10,
            batch_max_num_edges=20,
            batch_max_num_graphs=3,
            batch_max_num_pairs=0,
            num_epochs=epochs,
            ckpt_dir=checkpoint_dir,
            ckpt_manager_options={'max_to_keep': 10},
            eval_every_num_steps=1,
            allow_restart=allow_restart,
            use_wandb=False,
            use_ema=True,
            ema_decay=0.5,
        )

    uninterrupted_dir = tmp_path / 'uninterrupted'
    uninterrupted_train = FiniteIterator([1.0, 2.0])
    uninterrupted_validation = FiniteIterator([])
    run(
        uninterrupted_dir, 2, uninterrupted_train, uninterrupted_validation
    )

    restarted_dir = tmp_path / 'restarted'
    first_train = FiniteIterator([1.0, 2.0])
    first_validation = FiniteIterator([])
    run(restarted_dir, 1, first_train, first_validation)

    resumed_train = FiniteIterator([1.0, 2.0])
    resumed_validation = FiniteIterator([])
    run(
        restarted_dir, 2, resumed_train, resumed_validation,
        allow_restart=True,
    )
    # The checkpoint closed epoch 0, so the restart continues with epoch 1.
    assert [c for c in resumed_train.calls if c[0] == 'train'] == [('train', 1)]

    def restored_state(path):
        manager = checkpoint_utils.make_training_checkpoint_manager(path)
        latest = manager.latest_step()
        state = manager.restore(
            latest,
            args=ocp.args.Composite(
                training_state=ocp.args.StandardRestore()
            ),
        )['training_state']
        manager.close()
        return latest, state

    uninterrupted_step, uninterrupted_state = restored_state(uninterrupted_dir)
    restarted_step, restarted_state = restored_state(restarted_dir)
    assert uninterrupted_step == restarted_step == 4
    npt.assert_allclose(
        uninterrupted_state['raw_params']['w'],
        restarted_state['raw_params']['w'],
        rtol=0,
        atol=0,
    )
    npt.assert_allclose(
        uninterrupted_state['ema_params']['w'],
        restarted_state['ema_params']['w'],
        rtol=0,
        atol=0,
    )
    uninterrupted_opt_leaves = jax.tree_util.tree_leaves(
        uninterrupted_state['optimizer_state']
    )
    restarted_opt_leaves = jax.tree_util.tree_leaves(
        restarted_state['optimizer_state']
    )
    assert len(uninterrupted_opt_leaves) == len(restarted_opt_leaves)
    for expected, actual in zip(uninterrupted_opt_leaves, restarted_opt_leaves):
        npt.assert_allclose(expected, actual, rtol=0, atol=0)
    assert restarted_state['processed_graphs'] == 4
    assert restarted_state['processed_nodes'] == 40

    # A params-only checkpoint (written before the full training state was
    # saved) continues from its weights with a fresh optimizer.
    legacy_dir = tmp_path / 'legacy'
    legacy = checkpoint_utils.make_checkpoint_manager(legacy_dir)
    assert legacy.save(
        3,
        args=ocp.args.Composite(params=ocp.args.StandardSave({'w': jnp.array(5.0)})),
    )
    legacy.wait_until_finished()
    legacy.close()
    run(legacy_dir, 1, FiniteIterator([1.0]), FiniteIterator([]), allow_restart=True)
    legacy_step, legacy_state = restored_state(legacy_dir)
    assert legacy_step == 4
    npt.assert_allclose(legacy_state['raw_params']['w'], 5.0 - 0.1 * 1.0, rtol=1e-6)


def test_safe_put_honours_worker_cancellation():
    output_queue = queue.Queue(maxsize=1)
    output_queue.put('already full')
    cancelled = threading.Event()
    cancelled.set()

    assert not QCMLDataLoaderSparseParallel._safe_put(
        output_queue, 'must not block', cancelled
    )


def test_parallel_generator_surfaces_worker_process_failure():
    class FailedFuture:
        @staticmethod
        def done():
            return True

        @staticmethod
        def result():
            raise RuntimeError('worker process disappeared')

    class EmptyQueue:
        @staticmethod
        def get(timeout):
            raise queue.Empty

    class CancellationEvent:
        def __init__(self):
            self.cancelled = False

        def set(self):
            self.cancelled = True

    event = CancellationEvent()

    class Manager:
        @staticmethod
        def Queue(maxsize):
            return EmptyQueue()

        @staticmethod
        def Event():
            return event

    class Executor:
        @staticmethod
        def submit(*args, **kwargs):
            return FailedFuture()

    loader = QCMLDataLoaderSparseParallel.__new__(QCMLDataLoaderSparseParallel)
    loader.n_proc = 1
    loader.manager = Manager()
    loader.executor = Executor()
    loader._epoch_size = lambda mode: 1
    for name in (
        'batch_max_num_nodes', 'batch_max_num_edges',
        'batch_max_num_graphs', 'batch_max_num_pairs', 'cutoff',
        'cutoff_lr', 'max_force_filter', 'min_distance_filter',
        'energy_unit', 'length_unit', 'dipole_vec_unit',
    ):
        setattr(loader, name, 1)
    loader.calculate_neighbors_lr = False
    loader.input_folders = ['unused']
    loader.dataset_weights = [1.0]
    loader.num_train = [1]
    loader.num_valid = [1]

    with pytest.raises(RuntimeError, match='worker process disappeared'):
        list(loader._generator('train', 'train', epoch_seed=3))
    assert event.cancelled


def test_minimum_distance_filter_uses_neighbor_search(monkeypatch):
    positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    cell = np.diag([10.0, 10.0, 10.0])

    # The production predicate must not allocate ASE's full N x N matrix.
    monkeypatch.setattr(
        Atoms,
        'get_all_distances',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('all-pairs distance matrix was allocated')
        ),
    )
    assert minimum_distance_is_valid_np(
        positions, cell, [False, False, False], 0.5
    )
    assert not minimum_distance_is_valid_np(
        positions, cell, [True, False, False], 0.5
    )
    assert minimum_distance_is_valid_np(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        cell,
        [False, False, False],
        0.5,
    )


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
