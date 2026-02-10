from ml_collections import config_dict
import numpy.testing as npt
import pytest
import yaml
from importlib.resources import files


@pytest.mark.parametrize("suffix", ['json', 'yaml'])
def test_default_config(suffix):
    stream = (files('so3lr.mlff.config') / f'config.{suffix}').open('rb')

    cfg = config_dict.ConfigDict(yaml.load(stream=stream, Loader=yaml.FullLoader))

    cfg_model = cfg.model
    cfg_optimizer = cfg.optimizer
    cfg_training = cfg.training
    cfg_data = cfg.data

    npt.assert_equal(cfg.workdir, 'first_experiment')
    npt.assert_equal(cfg_data.filepath, None)
    npt.assert_equal(cfg_data.energy_unit, 'eV')
    npt.assert_equal(cfg_data.length_unit, 'Angstrom')
    npt.assert_equal(cfg_data.shift_mode, None)
    npt.assert_equal(cfg_data.energy_shifts, None)
    npt.assert_equal(cfg_data.split_seed, 0)

    npt.assert_equal(cfg_model.num_layers, 2)
    npt.assert_equal(cfg_model.num_features, 128)
    npt.assert_equal(cfg_model.num_heads, 4)
    npt.assert_equal(cfg_model.num_features_head, 32)
    npt.assert_equal(cfg_model.degrees, [1, 2, 3, 4])
    npt.assert_equal(cfg_model.cutoff, 5)
    npt.assert_equal(cfg_model.cutoff_fn, 'cosine')
    npt.assert_equal(cfg_model.num_radial_basis_fn, 32)
    npt.assert_equal(cfg_model.radial_basis_fn, 'physnet')
    npt.assert_equal(cfg_model.activation_fn, 'silu')
    npt.assert_equal(cfg_model.qk_non_linearity, 'identity')
    npt.assert_equal(cfg_model.residual_mlp_1, False)
    npt.assert_equal(cfg_model.residual_mlp_2, False)
    npt.assert_equal(cfg_model.layer_normalization_1, False)
    npt.assert_equal(cfg_model.layer_normalization_2, False)
    npt.assert_equal(cfg_model.layers_behave_like_identity_fn_at_init, False)
    npt.assert_equal(cfg_model.output_is_zero_at_init, True)
    npt.assert_equal(cfg_model.energy_regression_dim, 128)
    npt.assert_equal(cfg_model.energy_activation_fn, 'identity')
    npt.assert_equal(cfg_model.energy_learn_atomic_type_scales, False)
    npt.assert_equal(cfg_model.energy_learn_atomic_type_shifts, False)
    npt.assert_equal(cfg_model.input_convention, 'positions')

    npt.assert_equal(cfg_optimizer.name, 'adam')
    npt.assert_equal(cfg_optimizer.learning_rate, 1e-3)
    npt.assert_equal(cfg_optimizer.learning_rate_schedule, 'exponential_decay')
    npt.assert_equal(cfg_optimizer.learning_rate_schedule_args['decay_rate'], 0.75)
    npt.assert_equal(cfg_optimizer.learning_rate_schedule_args['transition_steps'], 125_000)
    npt.assert_equal(cfg_optimizer.num_of_nans_to_ignore, 0)
    npt.assert_equal(cfg_optimizer.gradient_clipping, 'identity')
    npt.assert_equal(cfg_optimizer.gradient_clipping_args, None)

    npt.assert_equal(cfg_training.allow_restart, False)
    npt.assert_equal(cfg_training.num_epochs, 100)
    npt.assert_equal(cfg_training.num_train, 950)
    npt.assert_equal(cfg_training.num_valid, 50)
    npt.assert_equal(cfg_training.batch_max_num_nodes, None)
    npt.assert_equal(cfg_training.batch_max_num_edges, None)
    npt.assert_equal(cfg_training.batch_max_num_graphs, 6)
    npt.assert_equal(cfg_training.eval_every_num_steps, 1000)
    npt.assert_equal(cfg_training.loss_weights['energy'], 0.01)
    npt.assert_equal(cfg_training.loss_weights['forces'], 0.99)
    npt.assert_equal(cfg_training.model_seed, 0)
    npt.assert_equal(cfg_training.training_seed, 0)
    npt.assert_equal(cfg_training.log_gradient_values, False)