"""W&B telemetry is opt-in.

MD restart files are pickle-free because load_state reads them with
allow_pickle=False, so every restart round trip (test_md_output_integrity.py,
test_restart_coordinates.py) fails if a field needs pickle.
"""

import pathlib
import sys
from types import SimpleNamespace

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


@pytest.mark.parametrize('init_args, expected', [
    ({'project': 'p'}, {'config': {'workdir': 'run', 'training': {'use_wandb': True}}, 'project': 'p'}),
    # A 'config' entry is merged into the logged config, not passed twice.
    ({'config': {'tag': 'x'}}, {'config': {'workdir': 'run', 'training': {'use_wandb': True}, 'tag': 'x'}}),
])
def test_wandb_opt_in_logs_config(monkeypatch, init_args, expected):
    calls = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: calls.append(kw)))
    config = {'workdir': 'run', 'training': {'use_wandb': True, 'wandb_init_args': init_args}}
    assert initialize_wandb(config) is True
    assert calls == [expected]


def test_wandb_broken_dependency_is_not_reported_as_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'wandb':
            raise ModuleNotFoundError("No module named 'protobuf'", name='protobuf')
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, 'wandb', raising=False)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ModuleNotFoundError, match='protobuf'):
        initialize_wandb({'training': {'use_wandb': True}})


def test_wandb_opt_in_without_client_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    with pytest.raises(ImportError, match=r'so3lr\[wandb\]'):
        initialize_wandb({'training': {'use_wandb': True}})
