"""Restart validation must precede publication of checkpoint metadata."""

from types import SimpleNamespace
import json

import jax.numpy as jnp
from ml_collections import config_dict
import pytest

from so3lr.mlff.config import from_config
from so3lr.mlff.utils import training_utils


@pytest.mark.parametrize('tfds', [False, True])
@pytest.mark.parametrize('fine_tune', [False, True])
def test_training_entry_point_keeps_metadata_when_loop_rejects_restart(
        tmp_path, monkeypatch, tfds, fine_tune):
    names = ('hyperparameters.json', 'hyperparameters.yaml', 'data_splits.json',
             'fine_tuning.json')
    for name in names:
        (tmp_path / name).write_text('original checkpoint metadata')
    config = config_dict.ConfigDict({
        'workdir': str(tmp_path),
        'model': {'message_normalization': None},
        'data': {'neighbors_lr_bool': False, 'filepath': '/unused/training.extxyz'},
        'training': {
            'allow_restart': True, 'batch_max_num_nodes': 8,
            'batch_max_num_edges': 16, 'batch_max_num_graphs': 4,
            'batch_max_num_pairs': 0, 'loss_weights': {'energy': 1.0},
            'eval_every_num_steps': 1, 'training_seed': 0, 'model_seed': 0,
            'log_gradient_values': False, 'num_epochs': 1, 'use_wandb': False,
            'use_adaptive_robust_loss': fine_tune,
            'reduce_lr_on_plateau': True, 'reduce_lr_patience': 7,
        },
    })
    def make_loader(**kwargs):
        return object(), tfds

    monkeypatch.setattr(from_config, 'data_loader_from_config', make_loader)

    def prepare(*, pending_metadata, **kwargs):
        pending_metadata['data_splits.json'] = 'changed split metadata'
        return [], [], None

    monkeypatch.setattr(from_config, 'prepare_training_and_validation_data', prepare)
    for name in ('make_optimizer_from_config', 'make_so3krates_sparse_from_config',
                 'get_energy_and_force_fn_sparse', '_make_graph_to_batch_fn'):
        monkeypatch.setattr(from_config, name, lambda *args, **kwargs: object())
    for name in ('make_loss_fn', 'make_val_fn'):
        monkeypatch.setattr(training_utils, name, lambda *args, **kwargs: object())
    if fine_tune:
        source = tmp_path / 'parent_model'
        source.mkdir()
        (source / 'hyperparameters.json').write_text(json.dumps({
            'model': {'num_layers': 1, 'message_normalization': None},
        }))
        monkeypatch.setattr(from_config, 'resolve_finetuning_strategy',
                            lambda *args, **kwargs: (['w'], None))
        monkeypatch.setattr(from_config.checkpoint_utils, 'load_params_from_workdir',
                            lambda *args: {'w': jnp.array(0.0)})
        monkeypatch.setattr(training_utils, 'expand_observables_to_theory_levels',
                            lambda params: params)
        monkeypatch.setattr(training_utils, 'freeze_parameters',
                            lambda optimizer, **kwargs: optimizer)
        monkeypatch.setattr(from_config, 'print_param_shapes', lambda *args: None)
        for name in ('count_params', 'count_trainable_params'):
            monkeypatch.setattr(from_config, name, lambda *args, **kwargs: 1)
    calls = []

    def reject_restart(**kwargs):
        calls.append(kwargs)
        assert callable(kwargs['on_training_ready'])
        for name in names:
            assert (tmp_path / name).read_text() == 'original checkpoint metadata'
        raise RuntimeError('restart rejected')

    monkeypatch.setattr(
        training_utils, 'fit_from_iterator' if tfds else 'fit', reject_restart
    )
    with pytest.raises(RuntimeError, match='restart rejected'):
        if fine_tune:
            from_config.run_fine_tuning(config, source, strategy='all')
        else:
            from_config.run_training(config)
    assert len(calls) == 1
    # Plateau LR reaches the loop when fine-tuning too (it used to be dropped).
    assert calls[0]['reduce_lr_on_plateau'] is True and calls[0]['reduce_lr_patience'] == 7
    for name in names:
        assert (tmp_path / name).read_text() == 'original checkpoint metadata'

    # The same prepared metadata becomes visible after successful validation.
    calls[0]['on_training_ready']()
    assert (tmp_path / 'data_splits.json').read_text() == 'changed split metadata'
    assert 'allow_restart' in (tmp_path / 'hyperparameters.json').read_text()
    if fine_tune:
        assert json.loads((tmp_path / 'fine_tuning.json').read_text()) == {
            'start_from_workdir': str(source), 'strategy': 'all',
        }


@pytest.mark.parametrize('allow_restart', [False, True])
def test_ase_restart_checks_checkpoint_even_with_supplied_parent_params(
        tmp_path, monkeypatch, allow_restart):
    saved_steps = []
    initial_values = []
    ready = []

    class Manager:
        def latest_step(self):
            return 5

        def save(self, step, **kwargs):
            saved_steps.append(step)
            return True

        def wait_until_finished(self):
            pass

    monkeypatch.setattr(training_utils.checkpoint_utils, 'make_checkpoint_manager',
                        lambda **kwargs: Manager())
    monkeypatch.setattr(training_utils.checkpoint_utils, 'load_params_from_checkpoint',
                        lambda **kwargs: {'w': jnp.array(7.0)})
    monkeypatch.setattr(training_utils, 'expand_observables_to_theory_levels', lambda p: p)
    monkeypatch.setattr(training_utils, 'dynamically_batch_with_lr',
                        lambda *args, **kwargs: iter([('graph', {})]))
    monkeypatch.setattr(training_utils.jraph, 'get_number_of_padding_with_graphs_nodes',
                        lambda graph: 0)

    def train(params, opt_state, batch):
        initial_values.append(float(params['w']))
        return {'w': params['w'] - 1}, opt_state, {'loss': jnp.array(1.0)}

    monkeypatch.setattr(training_utils, 'make_training_step_fn', lambda *args: train)
    monkeypatch.setattr(training_utils, 'make_validation_step_fn',
                        lambda *args: lambda params, batch: {'loss': jnp.array(1.0)})
    kwargs = dict(
        model=None, optimizer=SimpleNamespace(init=lambda params: ()), loss_fn=None,
        graph_to_batch_fn=lambda *args: {'num_of_non_padded_graphs': 1},
        training_data=[('graph', {})], validation_data=[('graph', {})],
        batch_max_num_nodes=8, batch_max_num_edges=16,
        batch_max_num_graphs=4, batch_max_num_pairs=0,
        num_epochs=1, params={'w': jnp.array(-99.0)}, ckpt_dir=tmp_path,
        eval_every_num_steps=1, allow_restart=allow_restart, use_wandb=False,
        on_training_ready=lambda: ready.append(True),
    )
    if allow_restart:
        training_utils.fit(**kwargs)
        assert ready == [True]
        assert initial_values == [7.0]
        assert saved_steps == [6]
    else:
        with pytest.raises(RuntimeError, match='allow_restart=True'):
            training_utils.fit(**kwargs)
        assert ready == initial_values == saved_steps == []


def test_tfds_rejected_checkpoint_does_not_publish_metadata(tmp_path, monkeypatch):
    ready = []
    monkeypatch.setattr(
        training_utils.checkpoint_utils, 'make_training_checkpoint_manager',
        lambda **kwargs: SimpleNamespace(latest_step=lambda: 1),
    )
    monkeypatch.setattr(training_utils, 'make_training_step_fn', lambda *args: None)
    monkeypatch.setattr(training_utils, 'make_validation_step_fn', lambda *args: None)

    def refuse_restore(*args, **kwargs):
        raise RuntimeError('incomplete training state')

    monkeypatch.setattr(training_utils, '_restore_tfds_training_state', refuse_restore)
    with pytest.raises(RuntimeError, match='incomplete training state'):
        training_utils.fit_from_iterator(
            model=None, optimizer=None, loss_fn=None, graph_to_batch_fn=None,
            training_iterator=None, validation_iterator=None,
            batch_max_num_nodes=8, batch_max_num_edges=16,
            batch_max_num_graphs=4, batch_max_num_pairs=0,
            num_epochs=1, ckpt_dir=tmp_path, allow_restart=True, use_wandb=False,
            on_training_ready=lambda: ready.append(True),
        )
    assert ready == []
