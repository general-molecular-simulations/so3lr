from types import SimpleNamespace

import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest
from so3lr.mlff import training_utils


def test_optimizer():
    optimizer = training_utils.make_optimizer(
        name='adamw',
        learning_rate=1e-3,
        learning_rate_schedule='exponential_decay',
        learning_rate_schedule_args=dict(transition_steps=10, decay_rate=0.7),
        num_of_nans_to_ignore=2
    )

    _ = optimizer.init({'weights': jnp.ones((20, 50))})


@pytest.mark.parametrize("vectorial", [True, False])
def test_graph_mse_loss(vectorial: bool):
    y = jnp.array([1., 0.2, 0.5])
    y_label = jnp.array([0.9, 0.4, 0.])
    if vectorial:
        y = jnp.repeat(y[:, None], repeats=5, axis=-1)
        y_label = jnp.repeat(y_label[:, None], repeats=5, axis=-1)

    graph_mask = jnp.array([True, True, False])

    loss_value = training_utils.graph_mse_loss(
        y=y,
        y_label=y_label,
        batch_segments=None,
        graph_mask=graph_mask,
        scale=jnp.asarray(1.).reshape(-1)
    )
    expected_loss = 1 / 2 * (0.1 ** 2 + 0.2 ** 2)

    npt.assert_allclose(loss_value, expected_loss, atol=1e-6)


@pytest.mark.parametrize("vectorial", [True, False])
def test_graph_mse_loss_nan(vectorial: bool):
    y = jnp.array([1., 0.2, 0.5])
    y_label = jnp.array([0.9, jnp.nan, 0.])
    if vectorial:
        y = jnp.repeat(y[:, None], repeats=5, axis=-1)
        y_label = jnp.repeat(y_label[:, None], repeats=5, axis=-1)

    graph_mask = jnp.array([True, True, False])

    loss_value = training_utils.graph_mse_loss(
        y=y,
        y_label=y_label,
        batch_segments=None,
        graph_mask=graph_mask,
        scale=jnp.asarray(1.).reshape(-1)
    )
    expected_loss = 1 / 1 * (0.1 ** 2)

    npt.assert_allclose(loss_value, expected_loss, atol=1e-6)


@pytest.mark.parametrize("vectorial", [True, False])
def test_node_mse_loss(vectorial: bool):
    y = jnp.array([1., 0.2, -3., 0.5])
    y_label = jnp.array([0.9, 0.4, -2.7, 0.])
    if vectorial:
        y = jnp.repeat(y[:, None], repeats=5, axis=-1)
        y_label = jnp.repeat(y_label[:, None], repeats=5, axis=-1)

    batch_segments = jnp.array([0, 1, 1, 2])
    graph_mask = jnp.array([True, True, False])

    loss_value = training_utils.node_mse_loss(
        y=y,
        y_label=y_label,
        batch_segments=batch_segments,
        graph_mask=graph_mask,
        scale=jnp.asarray(1.).reshape(-1)
    )
    expected_loss = 1/2 * (1/1 * 0.1**2 + 1/2 * (0.2**2 + 0.3**2))

    npt.assert_allclose(loss_value, expected_loss, atol=1e-6)


@pytest.mark.parametrize("vectorial", [True, False])
def test_node_mse_loss_nan(vectorial: bool):
    y = jnp.array([1., 0.2, -3., 0.5])
    y_label = jnp.array([0.9, jnp.nan, jnp.nan, 0.])
    if vectorial:
        y = jnp.repeat(y[:, None], repeats=5, axis=-1)
        y_label = jnp.repeat(y_label[:, None], repeats=5, axis=-1)

    batch_segments = jnp.array([0, 1, 1, 2])
    graph_mask = jnp.array([True, True, False])

    loss_value = training_utils.node_mse_loss(
        y=y,
        y_label=y_label,
        batch_segments=batch_segments,
        graph_mask=graph_mask,
        scale=jnp.asarray(1.).reshape(-1)
    )

    expected_loss = 1 / 1 * (1 / 1 * 0.1 ** 2)

    npt.assert_allclose(loss_value, expected_loss, atol=1e-6)


def test_expand_observables_to_theory_levels():
    """1D energy-head params are tiled to N theory levels; multi-level and
    missing entries are left unchanged."""
    n = 16
    params = {'params': {'observables_0': {
        'energy_offset': jnp.arange(3.0),                       # (3,)
        'atomic_scales': jnp.ones((3,)),                        # (3,)
        'energy_dense_final': {'kernel': jnp.ones((8, 1))},     # (8, 1)
    }}}
    out = training_utils.expand_observables_to_theory_levels(params, num_theory_levels=n)
    obs0 = out['params']['observables_0']
    npt.assert_equal(obs0['energy_offset'].shape, (3, n))
    npt.assert_equal(obs0['atomic_scales'].shape, (3, n))
    npt.assert_equal(obs0['energy_dense_final']['kernel'].shape, (8, n))
    # First column preserves the original 1D values.
    npt.assert_allclose(obs0['energy_offset'][:, 0], jnp.arange(3.0))

    # Already multi-level -> unchanged.
    already = {'params': {'observables_0': {
        'energy_offset': jnp.ones((3, n)),
        'energy_dense_final': {'kernel': jnp.ones((8, n))},
    }}}
    out2 = training_utils.expand_observables_to_theory_levels(already, num_theory_levels=n)
    npt.assert_equal(out2['params']['observables_0']['energy_offset'].shape, (3, n))
    npt.assert_equal(out2['params']['observables_0']['energy_dense_final']['kernel'].shape, (8, n))

    # Missing observables_0 -> returned unchanged.
    empty = {'params': {}}
    assert training_utils.expand_observables_to_theory_levels(empty) is empty


def test_expand_observables_tiles_the_optional_final_bias():
    """`use_final_bias_bool=True` models (the legacy single-level SO3LR) carry a
    'bias' next to the kernel; leaving it 1-wide while the kernel is tiled makes
    the head shapes inconsistent. Bias-free models must stay untouched."""
    n = 4
    with_bias = {'params': {'observables_0': {'energy_dense_final': {
        'kernel': jnp.ones((8, 1)),
        'bias': jnp.full((1,), 0.5),
    }}}}
    obs0 = training_utils.expand_observables_to_theory_levels(
        with_bias, num_theory_levels=n
    )['params']['observables_0']
    npt.assert_equal(obs0['energy_dense_final']['kernel'].shape, (8, n))
    npt.assert_equal(obs0['energy_dense_final']['bias'].shape, (n,))
    npt.assert_allclose(obs0['energy_dense_final']['bias'], jnp.full((n,), 0.5))

    # No bias key (so3lr-tauri-s/m/l) -> nothing added.
    without_bias = {'params': {'observables_0': {'energy_dense_final': {
        'kernel': jnp.ones((8, 1)),
    }}}}
    out = training_utils.expand_observables_to_theory_levels(
        without_bias, num_theory_levels=n
    )
    assert 'bias' not in out['params']['observables_0']['energy_dense_final']

    # Already multi-level bias -> unchanged.
    already = {'params': {'observables_0': {'energy_dense_final': {
        'kernel': jnp.ones((8, n)),
        'bias': jnp.ones((n,)),
    }}}}
    out = training_utils.expand_observables_to_theory_levels(
        already, num_theory_levels=n
    )
    npt.assert_equal(out['params']['observables_0']['energy_dense_final']['bias'].shape, (n,))


def test_log_gradient_values_is_not_overwritten():
    """Per-leaf gradient norms must survive the step function.

    They were written to metrics['grad_norm'], which the unconditional global
    norm assignment then overwrote, making log_gradient_values a pure no-op.
    """
    def loss_fn(params, batch):
        loss = 0.5 * (jnp.sum(params['a'] ** 2) + jnp.sum(params['b'] ** 2))
        return loss, {'loss': loss}

    params = {'a': jnp.array([3., 4.]), 'b': jnp.array([1.])}
    optimizer = optax.sgd(0.1)
    step_fn = training_utils.make_training_step_fn(
        optimizer, loss_fn, log_gradient_values=True)
    _, _, metrics = step_fn(params, optimizer.init(params), {})

    # The gradient of this loss is the params themselves: per-leaf norms 5 and 1.
    npt.assert_allclose(metrics['grad_norm_per_leaf']['a'], 5., rtol=1e-6)
    npt.assert_allclose(metrics['grad_norm_per_leaf']['b'], 1., rtol=1e-6)
    # The global norm still lands under its own key.
    npt.assert_allclose(metrics['grad_norm'], jnp.sqrt(26.), rtol=1e-6)


class _RecordingManager:
    """Stands in for the orbax manager: no existing checkpoint, records saves."""

    def __init__(self):
        self.saved_steps = []

    def latest_step(self):
        return None

    def save(self, step, **kwargs):
        self.saved_steps.append(step)
        return True

    def wait_until_finished(self):
        pass


class _OneBatchIterator:
    """The loader interface fit_from_iterator expects, one batch per epoch."""

    def next_epoch(self, split, mode, epoch_index=None, **kwargs):
        return iter([('graph', {})])


@pytest.mark.parametrize('entry_point', ['fit', 'fit_from_iterator'])
@pytest.mark.parametrize('eval_loss, saved_steps', [(float('nan'), []), (1.0, [1])])
def test_fit_skips_checkpoints_with_non_finite_loss(
        tmp_path, monkeypatch, entry_point, eval_loss, saved_steps):
    """fit() used to save NaN-loss checkpoints; orbax ranks with best_mode='min'
    and a NaN can be retained as 'best', evicting the real best weights.
    Both save through _save_if_finite.
    The finite case shows the stubbed loop does reach the save."""
    manager = _RecordingManager()
    for name in ('make_checkpoint_manager', 'make_training_checkpoint_manager'):
        monkeypatch.setattr(training_utils.checkpoint_utils, name,
                            lambda **kwargs: manager)
    monkeypatch.setattr(training_utils, 'dynamically_batch_with_lr',
                        lambda *args, **kwargs: iter([('graph', {})]))
    monkeypatch.setattr(training_utils.jraph, 'get_number_of_padding_with_graphs_nodes',
                        lambda graph: 0)
    monkeypatch.setattr(
        training_utils, 'make_training_step_fn',
        lambda *args: lambda params, opt_state, batch: (
            params, opt_state, {'loss': jnp.array(1.0)}))
    monkeypatch.setattr(
        training_utils, 'make_validation_step_fn',
        lambda *args: lambda params, batch: {'loss': jnp.array(eval_loss)})

    kwargs = dict(
        model=None, optimizer=SimpleNamespace(init=lambda params: ()), loss_fn=None,
        graph_to_batch_fn=lambda *args: {'num_of_non_padded_graphs': 1},
        batch_max_num_nodes=8, batch_max_num_edges=16,
        batch_max_num_graphs=4, batch_max_num_pairs=0,
        num_epochs=1, params={'w': jnp.array(0.0)}, ckpt_dir=tmp_path,
        eval_every_num_steps=1, use_wandb=False,
    )
    if entry_point == 'fit':
        training_utils.fit(training_data=[('graph', {})],
                           validation_data=[('graph', {})], **kwargs)
    else:
        training_utils.fit_from_iterator(training_iterator=_OneBatchIterator(),
                                         validation_iterator=_OneBatchIterator(),
                                         **kwargs)
    assert manager.saved_steps == saved_steps
