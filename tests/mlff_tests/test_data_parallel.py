"""Data-parallel training (training.data_parallel) on simulated CPU devices.

The device count is fixed when JAX initialises, so every check runs in a fresh
interpreter with ``--xla_force_host_platform_device_count=4``.
"""

import subprocess
import sys
import textwrap

import pytest

from tests.subprocess_env import so3lr_subprocess_env

NUM_DEVICES = 4


def _run_on_simulated_devices(script, tmp_path=None):
    env = so3lr_subprocess_env({
        'XLA_FLAGS': f'--xla_force_host_platform_device_count={NUM_DEVICES}',
        'JAX_PLATFORMS': 'cpu',
    })
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        env=env, capture_output=True, text=True, timeout=900,
        cwd=None if tmp_path is None else str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_data_parallel_step_matches_averaged_gradient_update():
    """One batch per device == one update on the averaged gradient (Adam and Muon);
    the same batch on every device == the single-device training step."""
    out = _run_on_simulated_devices(f"""
        import jax, jax.numpy as jnp, numpy as np, optax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from so3lr.mlff.utils import training_utils as tu

        n = {NUM_DEVICES}
        assert jax.local_device_count() == n, jax.devices()
        mesh = Mesh(np.array(jax.local_devices()), ('devices',))
        repl, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P('devices'))
        rng = np.random.default_rng(0)

        def loss_fn(params, batch):
            h = jnp.tanh(batch['x'] @ params['w1'] + params['b1'])
            pred = h @ params['w2']
            loss = jnp.mean((pred - batch['y']) ** 2)
            return loss, {{'loss': loss, 'mae': jnp.mean(jnp.abs(pred - batch['y']))}}

        params0 = {{'w1': jnp.asarray(rng.normal(size=(6, 16)), jnp.float32),
                    'b1': jnp.zeros(16, jnp.float32),
                    'w2': jnp.asarray(rng.normal(size=(16, 3)), jnp.float32)}}
        def make_batch():
            return {{'x': rng.normal(size=(32, 6)).astype(np.float32),
                     'y': rng.normal(size=(32, 3)).astype(np.float32)}}
        def stack(batches):
            return jax.device_put(jax.tree_util.tree_map(lambda *xs: np.stack(xs), *batches), shard)
        def max_diff(a, b):
            return max(float(jnp.max(jnp.abs(x - y)))
                       for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))

        for name, opt in (('adam', optax.adam(1e-2)), ('muon', optax.contrib.muon(1e-2))):
            dp_step = tu.make_data_parallel_training_step_fn(opt, loss_fn, mesh)
            single_step = tu.make_training_step_fn(opt, loss_fn, False)

            p_ref, s_ref = params0, opt.init(params0)
            p_dp, s_dp = jax.device_put(params0, repl), jax.device_put(opt.init(params0), repl)
            for _ in range(5):
                batches = [make_batch() for _ in range(n)]
                grads = [jax.grad(lambda p, b=b: loss_fn(p, b)[0])(p_ref) for b in batches]
                g = jax.tree_util.tree_map(lambda *xs: sum(xs) / n, *grads)
                u, s_ref = opt.update(g, s_ref, p_ref)
                p_ref = optax.apply_updates(p_ref, u)
                p_dp, s_dp, metrics = dp_step(p_dp, s_dp, stack(batches))
                assert max_diff(p_ref, p_dp) < 1e-5, (name, max_diff(p_ref, p_dp))
            assert p_dp['w1'].sharding.is_fully_replicated
            assert np.isfinite(float(metrics['grad_norm']))

            p1, s1 = params0, opt.init(params0)
            p2, s2 = jax.device_put(params0, repl), jax.device_put(opt.init(params0), repl)
            for _ in range(5):
                b = make_batch()
                p1, s1, m1 = single_step(p1, s1, jax.tree_util.tree_map(jnp.asarray, b))
                p2, s2, m2 = dp_step(p2, s2, stack([b] * n))
                assert max_diff(p1, p2) < 1e-5, (name, max_diff(p1, p2))
                assert abs(float(m1['loss']) - float(m2['loss'])) < 1e-5
                assert abs(float(m1['grad_norm']) - float(m2['grad_norm'])) < 1e-4
        print('OK')
    """)
    assert 'OK' in out


def test_fit_from_iterator_data_parallel_matches_single_device(tmp_path):
    """fit_from_iterator(data_parallel=True) on n batches per step reproduces a
    single-device run on the averaged batches, writes single-device checkpoints,
    resumes from them, and refuses a resume on another device count or an epoch
    with fewer batches than devices."""
    out = _run_on_simulated_devices(f"""
        import functools
        import jax, jax.numpy as jnp, numpy as np, optax
        import orbax.checkpoint as ocp
        from so3lr.mlff.utils import checkpoint_utils, training_utils

        n = {NUM_DEVICES}
        assert jax.local_device_count() == n
        training_utils.jraph.get_number_of_padding_with_graphs_nodes = lambda graph: 0

        class FiniteIterator:
            n_proc = 1
            def __init__(self, values):
                self.values = [np.asarray(v, np.float32) for v in values]
                self.calls = []
            def next_epoch(self, split, mode, epoch_index=None, **kwargs):
                self.calls.append((mode, epoch_index))
                values = self.values if mode == 'train' else [np.zeros(3, np.float32)]
                return iter((v, None) for v in values)

        class Model:
            @staticmethod
            def init(rng, batch):
                return {{'w': jnp.zeros(3, jnp.float32)}}

        # Linear loss: the gradient is the batch vector, so averaging gradients
        # over devices equals training on the averaged batch.
        def loss_fn(params, batch):
            loss = jnp.sum(params['w'] * batch['gradient'])
            return loss, {{'loss': loss}}

        def graph_to_batch(graph, long_range):
            return {{'num_of_non_padded_graphs': 1, 'gradient': np.asarray(graph)}}

        def run(ckpt_dir, values, data_parallel, epochs=1, allow_restart=False):
            it = FiniteIterator(values)
            training_utils.fit_from_iterator(
                model=Model(), optimizer=optax.sgd(0.1, momentum=0.9),
                loss_fn=loss_fn, val_fn=loss_fn, graph_to_batch_fn=graph_to_batch,
                training_iterator=it, validation_iterator=FiniteIterator([]),
                batch_max_num_nodes=10, batch_max_num_edges=20, batch_max_num_graphs=3,
                batch_max_num_pairs=0, num_epochs=epochs, ckpt_dir=ckpt_dir,
                ckpt_manager_options={{'max_to_keep': 10}}, eval_every_num_steps=1,
                allow_restart=allow_restart, use_wandb=False,
                use_ema=True, ema_decay=0.5, data_parallel=data_parallel)
            return it

        def assert_refused(match, *args, **kwargs):
            try:
                run(*args, **kwargs)
            except RuntimeError as exc:
                assert match in str(exc), exc
            else:
                raise AssertionError(f'run{{args}} was not refused')

        # tree.map also checks that the two trees have the same structure
        # (a zip over the leaves passes vacuously when one side is empty).
        assert_trees_equal = functools.partial(
            jax.tree.map, functools.partial(np.testing.assert_allclose, rtol=0, atol=1e-6))

        def restored(path):
            manager = checkpoint_utils.make_training_checkpoint_manager(path)
            step = manager.latest_step()
            state = manager.restore(step, args=ocp.args.Composite(
                training_state=ocp.args.StandardRestore()))['training_state']
            manager.close()
            return step, state

        rng = np.random.default_rng(0)
        values = [rng.normal(size=3) for _ in range(2 * n)]
        averaged = [np.mean(values[i * n:(i + 1) * n], axis=0) for i in range(2)]

        run('dp', values, data_parallel=True)
        run('single', averaged, data_parallel=False)
        step_dp, dp = restored('dp')
        step_single, single = restored('single')
        assert step_dp == step_single == 2, (step_dp, step_single)
        for key in ('raw_params', 'ema_params', 'optimizer_state'):
            assert_trees_equal(dp[key], single[key])

        # Checkpoints hold ordinary single-device arrays.
        params = checkpoint_utils.load_params_from_checkpoint(ckpt_dir='dp')
        assert all(len(x.devices()) == 1 for x in jax.tree_util.tree_leaves(params)
                   if hasattr(x, 'devices'))

        # A partial set of batches at the end of an epoch is dropped, not stepped.
        run('partial', values[:n + 1], data_parallel=True)
        assert restored('partial')[0] == 1
        # With fewer batches than devices no step at all would be taken.
        assert_refused('fewer than', 'tiny', values[:n - 1], data_parallel=True)

        # The device count sets the effective batch: resuming on another count is refused.
        assert_refused('device(s)', 'dp', values, data_parallel=False, epochs=2, allow_restart=True)
        assert_refused('device(s)', 'single', averaged, data_parallel=True, epochs=2,
                       allow_restart=True)

        # Resume the data-parallel run for a second epoch and compare to an
        # uninterrupted two-epoch single-device run on the averaged batches.
        resumed = run('dp', values, data_parallel=True, epochs=2, allow_restart=True)
        assert [c for c in resumed.calls if c[0] == 'train'] == [('train', 1)]
        run('single2', averaged, data_parallel=False, epochs=2)
        step_dp, dp = restored('dp')
        step_single, single = restored('single2')
        assert step_dp == step_single == 4
        assert_trees_equal(dp['raw_params'], single['raw_params'])
        print('OK')
    """, tmp_path=tmp_path)
    assert 'OK' in out
