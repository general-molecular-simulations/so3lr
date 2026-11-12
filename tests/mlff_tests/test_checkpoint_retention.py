"""Training keeps the max_to_keep lowest-loss checkpoints and always the newest one;
inference loads the lowest-loss one. fit_from_iterator's restart from the newest is
the max_to_keep=1 case of test_fit_from_iterator_restart_matches_uninterrupted_updates."""

import jax.numpy as jnp
import optax
import pytest

from so3lr.mlff.utils import checkpoint_utils, training_utils


class _Model:
    @staticmethod
    def init(rng, batch):
        return {'w': jnp.array(0.0)}


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


def _fit(ckpt_dir, allow_restart=False):
    """One epoch of 4 steps, evaluated after every step."""
    training_utils.fit(
        model=_Model(), optimizer=optax.sgd(0.1), loss_fn=None,
        graph_to_batch_fn=lambda graph, lr: {'num_of_non_padded_graphs': 1,
                                             'gradient': jnp.asarray(graph)},
        training_data=[1.0] * 4, validation_data=[0.0], num_epochs=1,
        batch_max_num_nodes=10, batch_max_num_edges=20, batch_max_num_graphs=3,
        batch_max_num_pairs=0, ckpt_dir=ckpt_dir, ckpt_manager_options={'max_to_keep': 2},
        eval_every_num_steps=1, allow_restart=allow_restart, use_wandb=False)


def _steps(ckpt_dir):
    manager = checkpoint_utils.make_checkpoint_manager(ckpt_dir)
    steps = sorted(manager.all_steps())
    manager.close()
    return steps


def _w(ckpt_dir, step=None):
    return float(checkpoint_utils.load_params_from_checkpoint(ckpt_dir, step=step)['w'])


def test_fit_keeps_and_resumes_from_the_newest_checkpoint(tmp_path, rising_loss):
    _fit(tmp_path)
    # Losses 0.01, 0.04, 0.09, 0.16: steps 1 and 2 are the best two, step 4 the newest.
    assert _steps(tmp_path) == [1, 2, 4]
    # Inference loads the lowest-loss checkpoint.
    assert _w(tmp_path) == pytest.approx(-0.1)

    # fit() restarts from the newest weights and step, not from the best ones.
    _fit(tmp_path, allow_restart=True)
    assert _steps(tmp_path) == [1, 2, 8]
    assert _w(tmp_path, step=8) == pytest.approx(-0.8)
