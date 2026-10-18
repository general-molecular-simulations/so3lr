"""The energy/force wrapper must forward every head, exactly once.

Two defects lived here: the wrapper dropped 'a0_ratios' (the key the current
so3lr-tauri-s/m/l models emit for the ratio head), and it ran a second full
model.apply purely to read the non-energy outputs.
"""
import json
import pathlib
import pickle

import jax.numpy as jnp
import numpy.testing as npt
import pytest
from ml_collections import config_dict

from so3lr.mlff.config.from_config import make_so3krates_sparse_from_config
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from so3lr.mlff.utils import training_utils

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / 'so3lr' / 'models'


def _water_inputs(n=3):
    return dict(
        positions=jnp.array([[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.]]),
        atomic_numbers=jnp.array([8, 1, 1]),
        idx_i=jnp.array([0, 0, 1, 1, 2, 2]), idx_j=jnp.array([1, 2, 0, 2, 0, 1]),
        idx_i_lr=jnp.array([0, 0, 1, 1, 2, 2]), idx_j_lr=jnp.array([1, 2, 0, 2, 0, 1]),
        cell=None, cell_offsets=None, cell_offsets_lr=None,
        batch_segments=jnp.zeros(n, dtype=int),
        node_mask=jnp.ones(n, dtype=bool),
        graph_mask=jnp.ones(1, dtype=bool),
        graph_mask_expanded=jnp.ones((1, 1), dtype=bool),
        total_charge=jnp.zeros(1), num_unpaired_electrons=jnp.zeros(1),
        theory_mask=jnp.eye(16)[5].reshape(1, -1),
    )


def _load(model_name):
    d = MODELS_DIR / model_name
    cfg = config_dict.ConfigDict(json.loads((d / 'hyperparameters.json').read_text()))
    params = training_utils.expand_observables_to_theory_levels(
        pickle.load(open(d / 'params.pkl', 'rb'))
    )
    return make_so3krates_sparse_from_config(cfg), params


@pytest.mark.parametrize('model_name', ['so3lr-tauri-m-2606', 'so3lr'])
def test_every_ratio_head_is_forwarded(model_name):
    """Whatever ratio keys the model emits must survive the wrapper.

    'a0_ratios' used to be dropped, which also made the _RATIO_ALIAS remap in
    make_loss_fn dead code -- it can only rename a key that actually arrives.
    """
    net, params = _load(model_name)
    inputs = _water_inputs()
    emitted = {k for k in net.apply(params, inputs) if 'ratio' in k}
    forwarded = {k for k in get_energy_and_force_fn_sparse(net)(params, **inputs)
                 if 'ratio' in k}
    assert emitted, f'{model_name} emits no ratio head at all'
    assert emitted <= forwarded, f'wrapper dropped {sorted(emitted - forwarded)}'


def test_ratio_target_trains_on_the_current_generation():
    """`loss_weights: {hirshfeld_ratios: ...}` on a model that emits
    'a0_ratios' must work via the alias remap; it used to raise KeyError."""
    net, params = _load('so3lr-tauri-m-2606')
    inputs = _water_inputs()
    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(net),
        weights={'forces': 1.0, 'hirshfeld_ratios': 1.0},
    )
    batch = dict(inputs)
    batch['forces'] = jnp.zeros((3, 3))
    batch['hirshfeld_ratios'] = jnp.ones(3)
    loss, _ = loss_fn(params, batch)
    assert jnp.isfinite(loss).all()


def test_energy_and_force_call_runs_one_forward_pass():
    """The wrapper used to call model.apply a second time just to read the
    non-energy heads, doubling the cost of every training/validation step."""
    net, params = _load('so3lr-tauri-m-2606')
    calls = {'n': 0}
    original = type(net).apply

    def counting_apply(self, *a, **k):
        calls['n'] += 1
        return original(self, *a, **k)

    type(net).apply = counting_apply
    try:
        get_energy_and_force_fn_sparse(net)(params, **_water_inputs())
    finally:
        type(net).apply = original
    assert calls['n'] == 1, f'model.apply ran {calls["n"]} times'


def test_nlh_table_lookup_matches_the_gathered_form():
    """The direct two-index lookup must equal the old gather-then-select, which
    materialised an (E, 92, 3) intermediate per table just to keep one row."""
    from so3lr.mlff.nn.observable.observable_sparse import NLH_AA, NLH_BB

    zi = jnp.array([1, 6, 8, 1, 7])
    zj = jnp.array([8, 1, 1, 6, 8])
    for table in (NLH_AA, NLH_BB):
        t = jnp.asarray(table)
        gathered = jnp.take_along_axis(
            jnp.take(t, zi, axis=0), zj[:, None, None], axis=1
        ).squeeze(1)
        npt.assert_allclose(t[zi, zj], gathered)
