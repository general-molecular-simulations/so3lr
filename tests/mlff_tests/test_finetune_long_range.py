"""Fine-tuning on energy/force-only data without corrupting the long-range heads.

Two mechanisms, both checked here at the model/loss level (the CLI runs live in
tests/test_cli_finetune.py):

* a residual energy readout (model.energy_residual_mlp_widths), trained with
  `--strategy readout_only` on a frozen backbone. Attaching it must not change the
  pretrained model: its last layer starts at zero.
* anchor targets ('anchor_<quantity>', training.long_range_anchor_weights), which
  pull the charges / a0 / C6 ratios towards a frozen copy of the pretrained model,
  and the per-atom 'partial_charges' target they rely on.
"""
import functools
import json
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ml_collections import config_dict

from so3lr.finetuning_strategies import resolve_finetuning_strategy
from so3lr.mlff.config.from_config import (
    _apply_finetune_model_overrides,
    _fill_missing_params,
    make_so3krates_sparse_from_config,
)
from so3lr.mlff.data.dataloader_sparse_ase import ASE_to_jraph
from so3lr.mlff.nn.stacknet.observable_function_sparse import (
    get_energy_and_force_fn_sparse,
    get_long_range_anchor_fn_sparse,
)
from so3lr.mlff.utils import training_utils
from so3lr.mlff.utils.param_utils import count_trainable_params

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / 'so3lr' / 'models'
MODEL = 'so3lr-tauri-s-2606'


def _water_inputs(n=3):
    return dict(
        positions=jnp.array([[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.]]),
        atomic_numbers=jnp.array([8, 1, 1]),
        idx_i=jnp.array([0, 0, 1, 1, 2, 2]), idx_j=jnp.array([1, 2, 0, 2, 0, 1]),
        idx_i_lr=jnp.array([0, 0, 1, 1, 2, 2]), idx_j_lr=jnp.array([1, 2, 0, 2, 0, 1]),
        batch_segments=jnp.zeros(n, dtype=int),
        node_mask=jnp.ones(n, dtype=bool),
        graph_mask=jnp.ones(1, dtype=bool),
        total_charge=jnp.zeros(1), num_unpaired_electrons=jnp.zeros(1),
        theory_mask=jnp.eye(16)[5].reshape(1, -1),
    )


@functools.lru_cache(maxsize=None)
def _pretrained():
    d = MODELS_DIR / MODEL
    hyper = json.loads((d / 'hyperparameters.json').read_text())
    params = training_utils.expand_observables_to_theory_levels(
        pickle.load(open(d / 'params.pkl', 'rb'))
    )
    return hyper, params


def _net(residual_widths=None, residual_inputs=None, intermediates=None):
    hyper, _ = _pretrained()
    cfg = config_dict.ConfigDict(json.loads(json.dumps(hyper)))
    override = config_dict.ConfigDict({
        'energy_residual_mlp_widths': residual_widths,
        'energy_residual_inputs': residual_inputs,
    })
    _apply_finetune_model_overrides(override, cfg.model)
    return make_so3krates_sparse_from_config(cfg, output_intermediate_quantities=intermediates)


def _with_residual(net):
    """Pretrained params plus the fresh residual readout, as run_fine_tuning builds them."""
    _, params = _pretrained()
    fresh = net.init(jax.random.PRNGKey(0), _water_inputs())
    merged, filled = _fill_missing_params(params, fresh)
    return merged, filled


@pytest.mark.parametrize('residual_inputs', [None, 'all_layers'])
def test_residual_readout_leaves_the_pretrained_model_unchanged(residual_inputs):
    """Zero last layer: energy and forces are the pretrained ones (up to the
    floating-point reassociation a different XLA fusion may bring)."""
    base_net = _net()
    _, params = _pretrained()
    net = _net([32, 16], residual_inputs)
    merged, filled = _with_residual(net)

    assert filled and all('energy_residual' in p for p in filled), filled
    base = get_energy_and_force_fn_sparse(base_net)(params, **_water_inputs())
    new = get_energy_and_force_fn_sparse(net)(merged, **_water_inputs())
    np.testing.assert_allclose(np.asarray(new['energy']), np.asarray(base['energy']), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(new['forces']), np.asarray(base['forces']),
                               rtol=1e-5, atol=1e-6)

    # ...but it is not dead: a non-zero last layer moves the energy.
    bumped = jax.tree_util.tree_map(lambda a: a, merged)
    final = bumped['params']['observables_0']['energy_residual_dense_final']
    final['kernel'] = jnp.ones_like(final['kernel'])
    moved = get_energy_and_force_fn_sparse(net)(bumped, **_water_inputs())
    assert not np.allclose(np.asarray(moved['energy']), np.asarray(base['energy']))


def test_all_layers_input_is_wider_than_the_last_layer():
    hyper, _ = _pretrained()
    net = _net([16], 'all_layers')
    merged, _ = _with_residual(net)
    kernel = merged['params']['observables_0']['energy_residual_dense_0']['kernel']
    m = hyper['model']
    assert kernel.shape == ((m['num_layers'] + 1) * m['num_features'], 16)


def test_readout_only_trains_the_readouts_and_freezes_the_long_range_heads():
    hyper, _ = _pretrained()
    net = _net([32, 16])
    merged, _ = _with_residual(net)
    trainable_keys, frozen_keys = resolve_finetuning_strategy(
        'readout_only', hyper['model']['num_layers']
    )
    assert frozen_keys is None
    obs0 = merged['params']['observables_0']
    readouts = {k: v for k, v in obs0.items()
                if k.startswith(('energy_dense', 'energy_residual'))
                or k in ('energy_offset', 'atomic_scales')}
    expected = sum(int(np.prod(np.shape(x))) for x in jax.tree_util.tree_leaves(readouts))
    assert count_trainable_params(merged, trainable_keys=trainable_keys) == expected

    # The charge, a0 and C6 heads, and the whole backbone, are frozen.
    labels = training_utils.make_annotation_fn(trainable_keys)(merged)['params']
    for name in ('electrostatic_energy',):
        assert set(jax.tree_util.tree_leaves(labels['observables_0'][name])) == {'frozen'}
    for name in ('observables_2', 'observables_3', 'layers_0'):
        assert set(jax.tree_util.tree_leaves(labels[name])) == {'frozen'}


def test_a_changed_residual_readout_is_refused():
    """Fine-tuning a fine-tuned model must keep its residual readout's shape."""
    model = config_dict.ConfigDict({'energy_residual_mlp_widths': [64]})
    with pytest.raises(ValueError, match='already has model.energy_residual_mlp_widths'):
        _apply_finetune_model_overrides(
            config_dict.ConfigDict({'energy_residual_mlp_widths': [32]}), model
        )
    with pytest.raises(ValueError, match='energy_residual_mlp_widths is not'):
        _apply_finetune_model_overrides(
            config_dict.ConfigDict({'energy_residual_inputs': 'all_layers'}),
            config_dict.ConfigDict({'num_layers': 2}),
        )


def test_anchor_targets_vanish_at_the_teacher_and_grow_away_from_it():
    """'anchor_<q>' scores the model's <q> against the label put into the batch;
    with the pretrained model's own prediction as label it is exactly zero."""
    net = _net(intermediates=['partial_charges'])
    _, params = _pretrained()
    keys = ['partial_charges', 'a0_ratios', 'c6_ratios', 'dipole_vec']
    teacher = get_long_range_anchor_fn_sparse(net, keys)(params, **_water_inputs())
    assert set(teacher) == set(keys)

    weights = {'forces': 1.0}
    weights.update({training_utils.ANCHOR_PREFIX + k: 1.0 for k in keys})
    loss_fn = training_utils.make_loss_fn(get_energy_and_force_fn_sparse(net), weights=weights)
    batch = dict(_water_inputs())
    batch['forces'] = get_energy_and_force_fn_sparse(net)(params, **_water_inputs())['forces']
    batch.update({training_utils.ANCHOR_PREFIX + k: v for k, v in teacher.items()})
    _, metrics = loss_fn(params, batch)
    for k in keys:
        assert np.asarray(metrics[f'anchor_{k}_mse']).item() == pytest.approx(0.0, abs=1e-12)

    batch['anchor_partial_charges'] = teacher['partial_charges'] + jnp.array([0.1, -0.05, -0.05])
    _, metrics = loss_fn(params, batch)
    assert np.asarray(metrics['anchor_partial_charges_mse']).item() > 1e-3


def test_hirshfeld_spelling_anchors_the_a0_head():
    """The legacy name works for the current generation, as for ordinary targets."""
    net = _net()
    _, params = _pretrained()
    out = get_long_range_anchor_fn_sparse(net, ['hirshfeld_ratios', 'a0_ratios'])(
        params, **_water_inputs())
    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(net), weights={'anchor_hirshfeld_ratios': 1.0})
    batch = dict(_water_inputs(), anchor_hirshfeld_ratios=out['a0_ratios'])
    loss, _ = loss_fn(params, batch)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_partial_charges_are_read_from_extxyz_and_nan_when_absent():
    atoms = Atoms('OH2', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    graph, _ = ASE_to_jraph(atoms, cutoff=5.0, calculate_neighbors_lr=False)
    assert np.isnan(graph.nodes['partial_charges']).all()

    atoms.arrays['partial_charges'] = np.array([-0.8, 0.4, 0.4])
    graph, _ = ASE_to_jraph(atoms, cutoff=5.0, calculate_neighbors_lr=False)
    np.testing.assert_allclose(graph.nodes['partial_charges'], [-0.8, 0.4, 0.4])
