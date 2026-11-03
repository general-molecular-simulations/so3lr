"""A model directory whose config predates legacy_so3lr_bool / use_final_bias_bool.

The original v1 config differs from the bundled legacy one only in these two flags, so the
bundled model with both removed is a real pre-flag v1 directory.
"""
import json
import shutil

import jax
import numpy as np
import pytest
from ase import Atoms

from so3lr import So3lrCalculator
from so3lr import model_registry as mr
from so3lr.model_config import load_model_config
from so3lr.mlff.nn.observable.dispersion_ref_data import supported_atomic_numbers_for_model

# float64 calculators need x64 (without it they now raise instead of running in float32)
pytestmark = pytest.mark.usefixtures('x64')


def _copy(target, **model_keys):
    """The bundled legacy model with its config's model keys changed (None removes one)."""
    shutil.copytree(mr.resolve_model('so3lr').dir, target)
    hyper = target / 'hyperparameters.json'
    config = json.loads(hyper.read_text())
    for key, value in model_keys.items():
        if value is None:
            del config['model'][key]
        else:
            config['model'][key] = value
    hyper.write_text(json.dumps(config))
    return target


@pytest.fixture(scope='module')
def v1(tmp_path_factory):
    return _copy(tmp_path_factory.mktemp('v1') / 'model', legacy_so3lr_bool=None, use_final_bias_bool=None)


def _energy(model):
    atoms = Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
    atoms.info['charge'] = 0.0
    atoms.calc = So3lrCalculator(model=str(model), lr_cutoff=12.0, dtype=np.float64)
    return atoms.get_potential_energy()


def test_v1_without_flags_is_the_legacy_model(v1):
    model = load_model_config(v1)['model']
    assert model['legacy_so3lr_bool'] is True and model['use_final_bias_bool'] is True
    assert supported_atomic_numbers_for_model(str(v1)) == supported_atomic_numbers_for_model('so3lr')
    np.testing.assert_allclose(_energy(v1), _energy('so3lr'), rtol=0, atol=1e-8)


@pytest.mark.parametrize('name', ['so3lr-tauri-s', 'so3lr-tauri-m', 'so3lr-tauri-l'])
def test_v2_model_is_not_flipped(name):
    """The bundled v2 configs leave legacy_so3lr_bool out as well."""
    model = load_model_config(mr.resolve_model(name).dir)['model']
    assert model['legacy_so3lr_bool'] is False and model['use_final_bias_bool'] is False


def test_an_explicit_value_is_kept(tmp_path):
    assert load_model_config(_copy(tmp_path / 'model', legacy_so3lr_bool=False))['model']['legacy_so3lr_bool'] is False


def test_flash_takes_v1_without_flags_as_the_legacy_model(v1):
    """CPU only: neither flash's check nor its loader touches the kernels."""
    from so3lr.flash import check, loader
    assert check.model_problem(str(v1)) is None
    got, ref = loader.load_params(str(v1)), loader.load_params('so3lr')
    assert jax.tree_util.tree_structure(got) == jax.tree_util.tree_structure(ref)
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_a_config_rewritten_in_place_is_read_again(tmp_path):
    model = _copy(tmp_path / 'model', cutoff=4.0)
    assert load_model_config(model)['model']['cutoff'] == 4.0
    _copy(tmp_path / 'tmp', cutoff=5.0)
    (tmp_path / 'tmp' / 'hyperparameters.json').replace(model / 'hyperparameters.json')
    assert load_model_config(model)['model']['cutoff'] == 5.0


def test_orbax_checkpoint_flags_without_starting_jax(v1, tmp_path):
    """A training workdir (checkpoints/ only): the flags come from orbax's JSON tree metadata,
    and no JAX backend starts (flash's reserve_gpu resolves the config before JAX)."""
    import pickle
    import subprocess
    import sys
    import orbax.checkpoint as ocp
    from so3lr.mlff.utils.checkpoint_utils import make_checkpoint_manager
    workdir = tmp_path / 'trained'
    workdir.mkdir()
    shutil.copy(v1 / 'hyperparameters.json', workdir)
    with open(v1 / 'params.pkl', 'rb') as fp:
        params = pickle.load(fp)
    manager = make_checkpoint_manager(ckpt_dir=workdir / 'checkpoints')
    manager.save(0, args=ocp.args.Composite(params=ocp.args.StandardSave(params)))
    manager.wait_until_finished()
    manager.close()
    code = ('import sys; from so3lr.model_config import load_model_config; '
            f'm = load_model_config({str(workdir)!r})["model"]; '
            'print(m["legacy_so3lr_bool"], m["use_final_bias_bool"], "jax" in sys.modules)')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ['True', 'True', 'False']
