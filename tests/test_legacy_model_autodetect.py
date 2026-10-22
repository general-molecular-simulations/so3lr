"""A model directory whose config predates ``legacy_so3lr_bool`` / ``use_final_bias_bool``.

The original v1 config sets neither flag, yet its checkpoint has the v1 heads and a final
energy bias. ``so3lr.model_config.load_model_config`` reads both from the checkpoint when
the config is silent, and every reader of the config (network, element coverage, eval,
fine-tuning, flash) goes through it.
"""
import json
import pathlib
import re
import shutil

import numpy as np
import pytest
from ase import Atoms

from so3lr import So3lrCalculator
from so3lr import model_registry as mr
from so3lr.model_config import load_model_config
from so3lr.mlff.nn.observable.dispersion_ref_data import supported_atomic_numbers_for_model

FLAGS = ('legacy_so3lr_bool', 'use_final_bias_bool')


def _copy_with_model_keys(source, target, drop=(), set_=None):
    shutil.copytree(source, target)
    hyper = target / 'hyperparameters.json'
    config = json.loads(hyper.read_text())
    for key in drop:
        config['model'].pop(key)
    config['model'].update(set_ or {})
    hyper.write_text(json.dumps(config))
    yaml_file = target / 'hyperparameters.yaml'
    if yaml_file.exists():
        text = yaml_file.read_text()
        for key in drop:
            text = re.sub(rf'(?m)^\s*{key}:.*\n', '', text)
        yaml_file.write_text(text)
    return target


@pytest.fixture(scope='module')
def legacy_dir():
    return pathlib.Path(mr.resolve_model('so3lr').dir)


@pytest.fixture(scope='module')
def legacy_without_flags(legacy_dir, tmp_path_factory):
    """The bundled v1 model as the original v1 config wrote it: neither flag set."""
    return _copy_with_model_keys(legacy_dir, tmp_path_factory.mktemp('v1') / 'model', drop=FLAGS)


def _energy(model):
    atoms = Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
    atoms.info['charge'] = 0.0
    atoms.calc = So3lrCalculator(model=str(model), lr_cutoff=12.0, dtype=np.float64)
    return atoms.get_potential_energy()


def test_flags_are_read_from_a_v1_checkpoint(legacy_without_flags):
    model = load_model_config(legacy_without_flags)['model']
    assert model['legacy_so3lr_bool'] is True
    assert model['use_final_bias_bool'] is True


def test_v1_checkpoint_without_flags_matches_the_flagged_model(legacy_dir, legacy_without_flags):
    np.testing.assert_allclose(_energy(legacy_without_flags), _energy(legacy_dir), rtol=0, atol=1e-8)


def test_v1_element_coverage_without_flags(legacy_without_flags):
    """v1 was trained on eight elements; without the flag the v2 set (Z = 1..85) leaked through."""
    assert supported_atomic_numbers_for_model(str(legacy_without_flags)) == supported_atomic_numbers_for_model('so3lr')
    assert 11 not in supported_atomic_numbers_for_model(str(legacy_without_flags))  # Na


@pytest.mark.parametrize('name', ['so3lr-tauri-s', 'so3lr-tauri-m', 'so3lr-tauri-l'])
def test_v2_model_is_not_flipped(name):
    """The bundled v2 configs leave legacy_so3lr_bool out; the checkpoint must not turn it on."""
    model = load_model_config(mr.resolve_model(name).dir)['model']
    assert model['legacy_so3lr_bool'] is False
    assert model['use_final_bias_bool'] is False
    assert 11 in supported_atomic_numbers_for_model(name)


def test_an_explicit_value_is_kept(legacy_dir, tmp_path):
    """Only an absent flag is read from the checkpoint; an explicit one stands."""
    explicit = _copy_with_model_keys(legacy_dir, tmp_path / 'model', set_={'legacy_so3lr_bool': False})
    assert load_model_config(explicit)['model']['legacy_so3lr_bool'] is False


def test_flash_takes_a_v1_directory_without_flags_as_the_legacy_model(legacy_without_flags):
    """Flash runs the legacy model since #88; its check and loader read the config through the
    resolver, so a pre-flag v1 directory is accepted and loads exactly like the bundled one
    (without the resolver it is checked against the v2 architecture and declined). CPU only:
    neither the check nor the loader touches the kernels."""
    import jax
    from so3lr.flash import check, loader

    assert check.model_problem(str(legacy_without_flags)) is None
    params, statics = loader.load_params(str(legacy_without_flags))
    ref_params, ref_statics = loader.load_params('so3lr')
    for got, ref in ((params, ref_params), (statics, ref_statics)):
        assert jax.tree_util.tree_structure(got) == jax.tree_util.tree_structure(ref)
        for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
