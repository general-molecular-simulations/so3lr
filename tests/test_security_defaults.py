"""W&B opt-in and pickle-free MD restart files."""

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from so3lr.mlff.config.wandb_utils import initialize_wandb


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize('training', [{}, {'use_wandb': False}, {'use_wandb': 'true'}])
def test_wandb_off_unless_enabled_and_never_imported(monkeypatch, training):
    # A None entry makes any `import wandb` raise.
    monkeypatch.setitem(sys.modules, 'wandb', None)
    assert initialize_wandb({'training': training}) is False


def test_bundled_configs_disable_wandb():
    for name in ('config.yaml', 'finetune.yaml'):
        with open(PROJECT_ROOT / 'so3lr' / 'config' / name) as f:
            assert yaml.safe_load(f)['training']['use_wandb'] is False


def test_wandb_opt_in_logs_config(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: calls.append(kw)))
    config = {'workdir': 'run', 'training': {'use_wandb': True, 'wandb_init_args': {'project': 'p'}}}
    assert initialize_wandb(config) is True
    assert calls == [{
        'config': {'workdir': 'run', 'training': {'use_wandb': True}},
        'project': 'p',
    }]


def test_wandb_opt_in_without_client_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    with pytest.raises(ImportError, match=r'so3lr\[wandb\]'):
        initialize_wandb({'training': {'use_wandb': True}})


def test_md_restart_round_trip_without_pickle(tmp_path):
    from so3lr.cli.so3lr_md import load_state, save_state

    position = np.array([[0.1, 0.2, 0.3]])
    state = SimpleNamespace(position=position, momentum=position + 1, force=position + 2, mass=np.ones((1, 1)))
    path = tmp_path / 'restart.npz'
    save_state(state, np.array([10.0, 11.0, 12.0]), 7, str(path), ensemble='nve', fractional_coordinates=False)

    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            archive[name]  # raises if any field needs pickle

    loaded, box, cycle = load_state(str(path), ensemble='nve', fractional_coordinates=False)
    np.testing.assert_allclose(loaded.position, position)
    np.testing.assert_allclose(box, [10.0, 11.0, 12.0])
    assert int(cycle) == 7
