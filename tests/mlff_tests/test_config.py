"""The shipped config.yaml is the template users copy for `so3lr train`.

It is not merged with any defaults, so it must carry every key the training
path reads by attribute (without a fallback). Values are deliberately not
pinned: changing a default is not a bug, dropping a required key is.
"""
from importlib.resources import files

import yaml
from ml_collections import config_dict

# Keys read as `config.<section>.<key>` by make_so3krates_sparse_from_config,
# make_optimizer_from_config and run_training in so3lr/mlff/config/from_config.py.
REQUIRED_KEYS = {
    'data': {
        'filepath', 'energy_unit', 'length_unit', 'dipole_vec_unit', 'shift_mode',
        'energy_shifts', 'split_seed', 'neighbors_lr_bool', 'neighbors_lr_cutoff',
        'avg_num_neighbors',
    },
    'model': {
        'num_layers', 'num_features', 'num_heads', 'num_features_head',
        'radial_basis_fn', 'num_radial_basis_fn', 'cutoff_fn', 'cutoff', 'cutoff_lr',
        'degrees', 'residual_mlp_1', 'residual_mlp_2', 'layer_normalization_1',
        'layer_normalization_2', 'message_normalization', 'qk_non_linearity',
        'activation_fn', 'layers_behave_like_identity_fn_at_init',
        'output_is_zero_at_init', 'input_convention', 'use_charge_embed',
        'use_spin_embed', 'energy_regression_dim', 'energy_activation_fn',
        'energy_learn_atomic_type_scales', 'energy_learn_atomic_type_shifts',
        'electrostatic_energy_bool', 'electrostatic_energy_scale',
        'dispersion_energy_bool', 'dispersion_energy_cutoff_lr_damping',
        'dispersion_energy_scale', 'zbl_repulsion_bool',
    },
    'optimizer': {
        'name', 'optimizer_args', 'learning_rate', 'learning_rate_schedule',
        'learning_rate_schedule_args', 'num_of_nans_to_ignore', 'gradient_clipping',
        'gradient_clipping_args',
    },
    'training': {
        'allow_restart', 'num_epochs', 'num_train', 'num_valid',
        'batch_max_num_nodes', 'batch_max_num_edges', 'batch_max_num_graphs',
        'batch_max_num_pairs', 'eval_every_num_steps', 'loss_weights',
        'model_seed', 'training_seed', 'log_gradient_values',
    },
}


def test_default_config_yaml_has_every_required_key():
    with (files('so3lr.config') / 'config.yaml').open('rb') as stream:
        cfg = config_dict.ConfigDict(yaml.safe_load(stream))

    assert {'workdir', *REQUIRED_KEYS} <= set(cfg.keys())
    missing = {section: sorted(keys - set(cfg[section].keys()))
               for section, keys in REQUIRED_KEYS.items()}
    missing = {section: keys for section, keys in missing.items() if keys}
    assert not missing, f'config.yaml lacks keys the trainer reads: {missing}'
    assert len(cfg.training.loss_weights) > 0
