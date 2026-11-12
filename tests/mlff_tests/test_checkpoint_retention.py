"""Training keeps the max_to_keep lowest-loss checkpoints and always the newest one.

orbax's own policy for a best_fn keeps only the best N, so it deleted a newest
checkpoint whose validation loss was not among them, and a restart (which resumes
from latest_step()) went back to an older step and dropped the training after it.
"""

import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest

from so3lr.mlff.utils import checkpoint_utils, training_utils

STEPS_PER_EPOCH = 4


class _Model:
    @staticmethod
    def init(rng, batch):
        return {'w': jnp.array(0.0)}


class _Iterator:
    """TFDS-style iterator that yields the same gradients every epoch."""

    def __init__(self, values):
        self.values = values
        self.train_epochs = []

    def next_epoch(self, split, mode, epoch_index=None, **kwargs):
        if mode == 'train':
            self.train_epochs.append(epoch_index)
            return iter((value, None) for value in self.values)
        return iter([(0.0, None)])


@pytest.fixture
def rising_loss(monkeypatch):
    """Toy training in which every step raises the validation loss: SGD (lr 0.1) on
    gradient 1 moves w by -0.1 per step, and the validation loss is w**2."""
    def training_step_factory(optimizer, loss_fn, log_gradients):
        def step(params, opt_state, batch):
            updates, opt_state = optimizer.update({'w': batch['gradient']}, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, {'loss': jnp.square(params['w'])}
        return step

    monkeypatch.setattr(training_utils, 'make_training_step_fn', training_step_factory)
    monkeypatch.setattr(training_utils, 'make_validation_step_fn',
                        lambda val_fn: lambda params, batch: {'loss': jnp.square(params['w'])})
    monkeypatch.setattr(training_utils.jraph, 'get_number_of_padding_with_graphs_nodes',
                        lambda graph: 0)
    # fit() batches a list of graphs; here every list item is one batch.
    monkeypatch.setattr(training_utils, 'dynamically_batch_with_lr',
                        lambda graphs, **kwargs: ((graph, None) for graph in graphs))


def _common(ckpt_dir, allow_restart):
    return dict(
        model=_Model(), optimizer=optax.sgd(0.1), loss_fn=None,
        graph_to_batch_fn=lambda graph, lr: {'num_of_non_padded_graphs': 1,
                                             'gradient': jnp.asarray(graph)},
        batch_max_num_nodes=10, batch_max_num_edges=20, batch_max_num_graphs=3,
        batch_max_num_pairs=0, ckpt_dir=ckpt_dir, ckpt_manager_options={'max_to_keep': 2},
        eval_every_num_steps=1, allow_restart=allow_restart, use_wandb=False,
    )


def _fit(ckpt_dir, allow_restart=False):
    training_utils.fit(training_data=[1.0] * STEPS_PER_EPOCH, validation_data=[0.0],
                       num_epochs=1, **_common(ckpt_dir, allow_restart))


def _fit_from_iterator(ckpt_dir, num_epochs, allow_restart=False):
    training = _Iterator([1.0] * STEPS_PER_EPOCH)
    training_utils.fit_from_iterator(training_iterator=training, validation_iterator=_Iterator([]),
                                     num_epochs=num_epochs, **_common(ckpt_dir, allow_restart))
    return training.train_epochs


def _steps(ckpt_dir):
    manager = checkpoint_utils.make_checkpoint_manager(ckpt_dir)
    steps = sorted(manager.all_steps())
    manager.close()
    return steps


def _w(ckpt_dir, step=None):
    return float(checkpoint_utils.load_params_from_checkpoint(ckpt_dir, step=step)['w'])


def test_fit_from_iterator_keeps_and_resumes_from_the_newest_checkpoint(tmp_path, rising_loss):
    _fit_from_iterator(tmp_path, num_epochs=1)
    # Losses 0.01, 0.04, 0.09, 0.16: steps 1 and 2 are the best two, step 4 the newest.
    assert _steps(tmp_path) == [1, 2, 4]

    assert _fit_from_iterator(tmp_path, num_epochs=2, allow_restart=True) == [1]
    # The restart continued from step 4 (w = -0.4), not from step 2.
    assert _steps(tmp_path) == [1, 2, 8]
    assert _w(tmp_path, step=8) == pytest.approx(-0.8)


def test_fit_keeps_and_resumes_from_the_newest_checkpoint(tmp_path, rising_loss):
    _fit(tmp_path)
    assert _steps(tmp_path) == [1, 2, 4]

    # fit() restarts from the newest weights and step, not from the best ones.
    _fit(tmp_path, allow_restart=True)
    assert _steps(tmp_path) == [1, 2, 8]
    assert _w(tmp_path, step=8) == pytest.approx(-0.8)


def test_inference_loads_the_lowest_loss_checkpoint(tmp_path, rising_loss):
    _fit(tmp_path)
    assert _w(tmp_path) == pytest.approx(-0.1)
    assert _w(tmp_path, step=4) == pytest.approx(-0.4)


def test_checkpoints_without_a_loss_load_the_latest(tmp_path):
    manager = checkpoint_utils.make_checkpoint_manager(tmp_path)
    for step in (1, 2):
        manager.save(step, args=ocp.args.Composite(
            params=ocp.args.StandardSave({'w': jnp.array(float(step))})))
        manager.wait_until_finished()
    manager.close()
    assert _w(tmp_path) == 2.0


def test_keep_period_still_applies(tmp_path):
    manager = checkpoint_utils.make_checkpoint_manager(
        tmp_path, checkpoint_utils.make_training_checkpoint_options(
            {'max_to_keep': 1, 'keep_period': 2}))
    for step in range(1, 6):
        manager.save(step, args=ocp.args.Composite(
            params=ocp.args.StandardSave({'w': jnp.array(float(step))})),
            metrics={'loss': float(step)})
        manager.wait_until_finished()
    # The best (1), every second step (2, 4) and the newest (5).
    assert sorted(manager.all_steps()) == [1, 2, 4, 5]
    manager.close()
