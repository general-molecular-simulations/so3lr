from ase.units import *
import json
from so3lr.mlff import nn
from so3lr.mlff import training_utils
from so3lr.mlff import evaluation_utils
from so3lr.mlff import data
from so3lr.mlff import jraph_utils
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from ml_collections import config_dict
import numpy as np
from pathlib import Path
from typing import Sequence, Optional
import yaml
import logging
import os
from functools import partial, partialmethod
import jax.numpy as jnp

from ..utils import checkpoint_utils
from ..utils.param_utils import print_param_shapes, count_params

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)

def make_so3krates_sparse_from_config(
        config: config_dict.ConfigDict = None,
        return_representations_bool: bool = False,
        output_intermediate_quantities: Optional[Sequence[str]] = None
):
    """Make a SO3krates model from a config.

    Args:
        config (): The config.
        return_representations_bool (): Create a SO3krates model that only returns the atomic representatios.

    Returns:
        SO3krates flax model.
    """
    model_config = config.model

    return nn.SO3kratesSparse(
        num_layers=model_config.num_layers,
        num_features=model_config.num_features,
        num_heads=model_config.num_heads,
        num_features_head=model_config.num_features_head,
        radial_basis_fn=model_config.radial_basis_fn,
        num_radial_basis_fn=model_config.num_radial_basis_fn,
        cutoff_fn=model_config.cutoff_fn,
        cutoff=model_config.cutoff,
        cutoff_lr=model_config.cutoff_lr,
        degrees=model_config.degrees,
        residual_mlp_1=model_config.residual_mlp_1,
        residual_mlp_2=model_config.residual_mlp_2,
        layer_normalization_1=model_config.layer_normalization_1,
        layer_normalization_2=model_config.layer_normalization_2,
        use_rms_norm=model_config.get('use_rms_norm', False),
        qk_norm=model_config.get('qk_norm', False),
        use_residual_scalars=model_config.get('use_residual_scalars', False),
        message_normalization=config.model.message_normalization,
        avg_num_neighbors=config.data.avg_num_neighbors if config.model.message_normalization == 'avg_num_neighbors' else None,
        qk_non_linearity=model_config.qk_non_linearity,
        activation_fn=model_config.activation_fn,
        layers_behave_like_identity_fn_at_init=model_config.layers_behave_like_identity_fn_at_init,
        output_is_zero_at_init=model_config.output_is_zero_at_init,
        input_convention=model_config.input_convention,
        use_charge_embed=model_config.use_charge_embed,
        use_spin_embed=model_config.use_spin_embed,
        energy_regression_dim=model_config.energy_regression_dim,
        energy_activation_fn=model_config.energy_activation_fn,
        energy_learn_atomic_type_scales=model_config.energy_learn_atomic_type_scales,
        energy_learn_atomic_type_shifts=model_config.energy_learn_atomic_type_shifts,
        electrostatic_energy_bool=model_config.electrostatic_energy_bool,
        electrostatic_energy_kspace_do_ewald_bool = getattr(config, 'electrostatic_energy_kspace_do_ewald_bool', None),
        electrostatic_energy_kspace_interp_nodes = getattr(config, 'electrostatic_energy_kspace_interp_nodes', 4),
        electrostatic_energy_scale=model_config.electrostatic_energy_scale,
        dispersion_energy_bool=model_config.dispersion_energy_bool,
        dispersion_energy_cutoff_lr_damping=model_config.dispersion_energy_cutoff_lr_damping,
        dispersion_energy_scale=model_config.dispersion_energy_scale,
        return_representations_bool=return_representations_bool,
        zbl_repulsion_bool=model_config.zbl_repulsion_bool,
        use_final_bias_bool=model_config.get('use_final_bias_bool', True),
        neighborlist_format_lr=config.neighborlist_format_lr,
        output_intermediate_quantities=output_intermediate_quantities,
    )




def make_optimizer_from_config(config: config_dict.ConfigDict = None):
    """Make optax optimizer from config.

    Args:
        config (): The config.

    Returns:
        optax.Optimizer.
    """
    if config is None:
        return training_utils.make_optimizer()
    else:
        opt_config = config.optimizer
        return training_utils.make_optimizer(
            name=opt_config.name,
            optimizer_args=opt_config.optimizer_args if opt_config.optimizer_args is not None else dict(),
            learning_rate_schedule=opt_config.learning_rate_schedule,
            learning_rate=opt_config.learning_rate,
            learning_rate_schedule_args=opt_config.learning_rate_schedule_args if opt_config.learning_rate_schedule_args is not None else dict(),
            num_of_nans_to_ignore=opt_config.num_of_nans_to_ignore,
            gradient_clipping=opt_config.gradient_clipping,
            gradient_clipping_args=opt_config.gradient_clipping_args if opt_config.gradient_clipping_args is not None else dict()
        )


def run_training(config: config_dict.ConfigDict, model: str = 'so3krates'):
    """Run training given a config.

    Args:
        config (): The config.
        model (): The model to train. Defaults to SO3krates.

    Returns:

    """
    workdir = workdir_from_config(config=config)
    workdir.mkdir(exist_ok=config.training.allow_restart)

    # Update the workdir in config with absolute path.
    config.workdir = str(workdir)

    # Currently, training is always performed with sparse neighborlist_format for long range blocks.
    config.neighborlist_format_lr = config_dict.placeholder(str)
    config.neighborlist_format_lr = 'sparse'

    loader, tf_record_present = data_loader_from_config(
        config=config
    )

    training_data, validation_data, data_stats = prepare_training_and_validation_data(
        config=config,
        loader=loader,
        tf_record_present=tf_record_present
    )

    # If messages are normalized by the average number of neighbors, we need to calculate this quantity from the
    # training data or read it from the config when provided.
    if config.model.message_normalization == 'avg_num_neighbors':
        try:
            config.data.avg_num_neighbors
        except AttributeError:
            logging.warning(
                'Passing a config without data.avg_num_neighbors is deprecated and will raise an error in the future.'
                'Add data.avg_num_neighbors: null to the config to disable the warning. For now, we automatically set '
                'data.avg_num_neighbors: null.'
            )
            config.data.avg_num_neighbors = config_dict.placeholder(float)
            config.data.avg_num_neighbors = None

        if config.data.avg_num_neighbors is not None:
            logging.mlff(
                f'Read average number of neighbors = {config.data.avg_num_neighbors} from config.'
            )
        else:
            logging.mlff('Calculate average number of neighbors ...')
            avg_num_neighbors = np.array(data.transformations.calculate_average_number_of_neighbors(training_data))
            config.data.avg_num_neighbors = np.array(avg_num_neighbors).item()
            logging.mlff('... done.')

    opt = make_optimizer_from_config(config)
    if model == 'so3krates':
        net = make_so3krates_sparse_from_config(config)
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )

    force_fn = get_energy_and_force_fn_sparse(net)
    
    # Robust loss configuration
    use_robust_loss = config.training.get('use_robust_loss', False)
    robust_loss_alpha = config.training.get('robust_loss_alpha', 1.99)
    use_adaptive_robust_loss = config.training.get('use_adaptive_robust_loss', False)

    # Validate: can't use both fixed and adaptive robust loss
    if use_robust_loss and use_adaptive_robust_loss:
        raise ValueError(
            "Cannot use both 'use_robust_loss' (fixed) and 'use_adaptive_robust_loss' (learned). "
            "Please enable only one."
        )

    loss_fn = training_utils.make_loss_fn(
        force_fn,
        weights=config.training.loss_weights,
        use_robust_loss=use_robust_loss,
        robust_loss_alpha=robust_loss_alpha,
        use_adaptive_robust_loss=use_adaptive_robust_loss,
    )

    val_fn = training_utils.make_val_fn( #maybe not needed for val_fn - want to see rmse
        force_fn,
        weights=config.training.loss_weights,
        use_robust_loss=config.training.get('use_robust_loss_validation', use_robust_loss),
        robust_loss_alpha=config.training.get('robust_loss_alpha_validation', robust_loss_alpha),
        use_adaptive_robust_loss=use_adaptive_robust_loss,
    )

    if config.training.batch_max_num_nodes is None:
        assert config.training.batch_max_num_edges is None
        assert config.training.batch_max_num_pairs is None

        if tf_record_present:
            raise ValueError(
                'When reading TFDSDataSet, `max_num_nodes`, `max_num_edges` and `max_num_pairs` can not be auto- '
                'determined. Please set the corresponding values in the config file via '
                'training.batch_max_num_nodes and training.batch_max_num_edges.'
            )

        batch_max_num_nodes = data_stats['max_num_of_nodes'] * (config.training.batch_max_num_graphs - 1) + 1
        batch_max_num_edges = data_stats['max_num_of_edges'] * (config.training.batch_max_num_graphs - 1) + 1

        config.training.batch_max_num_nodes = batch_max_num_nodes
        config.training.batch_max_num_edges = batch_max_num_edges

    if config.training.batch_max_num_pairs is None:
        if config.data.neighbors_lr_bool is True:
            # TODO: This always creates num_pairs to be quadratic in the number of nodes. Add data_stats about max
            #  num_pairs which is important for the case of lr_cutoff smaller than largest separation in the data
            #  as this allows to safe cost.

            if tf_record_present:
                raise ValueError(
                    'When reading TFDSDataSet, `max_num_pairs` can not be auto- '
                    'determined. Please set the corresponding values in the config file via '
                    'training.batch_max_num_nodes and training.batch_max_num_edges.'
                )

            batch_max_num_pairs = data_stats['max_num_of_nodes'] * (data_stats['max_num_of_nodes'] - 1) * (config.training.batch_max_num_graphs - 1) + 1
            # batch_max_num_pairs = config.training.batch_max_num_nodes * (config.training.batch_max_num_nodes - 1) + 1
        else:
            batch_max_num_pairs = 0

        config.training.batch_max_num_pairs = batch_max_num_pairs

    with open(workdir / 'hyperparameters.json', 'w') as fp:
        json.dump(config.to_dict(), fp)

    with open(workdir / "hyperparameters.yaml", "w") as yaml_file:
        yaml.dump(config.to_dict(), yaml_file, default_flow_style=False)

    # Initialize wandb if needed.
    use_wandb = config.training.use_wandb
    if use_wandb is True:
        import wandb
        wandb.init(config=config.to_dict(), **config.training.wandb_init_args)

    # Adaptive robust loss settings
    adaptive_robust_loss_targets = list(config.training.loss_weights.keys()) if use_adaptive_robust_loss else None
    adaptive_robust_loss_init_alpha = config.training.get('adaptive_robust_loss_init_alpha', 1.5)
    adaptive_robust_loss_init_scale = config.training.get('adaptive_robust_loss_init_scale', 1.0)

    logging.mlff('Training is starting!')
    if not tf_record_present:
        training_utils.fit(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
            batch_max_num_edges=config.training.batch_max_num_edges,
            batch_max_num_nodes=config.training.batch_max_num_nodes,
            batch_max_num_graphs=config.training.batch_max_num_graphs,
            batch_max_num_pairs=config.training.batch_max_num_pairs,
            training_data=training_data,
            validation_data=validation_data,
            ckpt_dir=workdir / 'checkpoints',
            eval_every_num_steps=config.training.eval_every_num_steps,
            allow_restart=config.training.allow_restart,
            num_epochs=config.training.num_epochs,
            training_seed=config.training.training_seed,
            model_seed=config.training.model_seed,
            log_gradient_values=config.training.log_gradient_values,
            use_wandb=use_wandb,
            # Adaptive robust loss
            use_adaptive_robust_loss=use_adaptive_robust_loss,
            adaptive_robust_loss_targets=adaptive_robust_loss_targets,
            adaptive_robust_loss_init_alpha=adaptive_robust_loss_init_alpha,
            adaptive_robust_loss_init_scale=adaptive_robust_loss_init_scale
        )
    else:
        training_utils.fit_from_iterator(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
            batch_max_num_edges=config.training.batch_max_num_edges,
            batch_max_num_nodes=config.training.batch_max_num_nodes,
            batch_max_num_graphs=config.training.batch_max_num_graphs,
            batch_max_num_pairs=config.training.batch_max_num_pairs,
            training_iterator=training_data,
            validation_iterator=validation_data,
            ckpt_dir=workdir / 'checkpoints',
            eval_every_num_steps=config.training.eval_every_num_steps,
            allow_restart=config.training.allow_restart,
            training_seed=config.training.training_seed,
            model_seed=config.training.model_seed,
            log_gradient_values=config.training.log_gradient_values,
            num_epochs=config.training.num_epochs,
            use_wandb=use_wandb,
            # Adaptive robust loss
            use_adaptive_robust_loss=use_adaptive_robust_loss,
            adaptive_robust_loss_targets=adaptive_robust_loss_targets,
            adaptive_robust_loss_init_alpha=adaptive_robust_loss_init_alpha,
            adaptive_robust_loss_init_scale=adaptive_robust_loss_init_scale
        )
    logging.mlff('Training has finished!')


def run_evaluation(
        config,
        model: str = 'so3krates',
        num_test: int = None,
        testing_targets: Sequence[str] = None,
        pick_idx: np.ndarray = None,
        on_split: str = None,
        write_batch_metrics_to: str = None
):
    """Run evaluation, given the config and additional args.

    Args:
        config (): The config file.
        model (): The model to evaluate. Defaults to SO3krates.
        num_test (): Number of testing points. If not given, is determined from config using
            num_test = num_data - num_train - num_valid. Note that still the whole data set is loaded. If one wants to
            only load subparts of the data, one has to use pick_idx.
        testing_targets (): Targets used for computing metrics. Defaults to the ones found in
            config.training.loss_weights.
        pick_idx (): Indices to evaluate the model on. Loads only the data at the given indices.
        on_split (): On which split to evaluate (training, validation, test). Only needed for TFDS data set since it does not
            support efficient loading from indices yet.
        write_batch_metrics_to (str): Path to file where metrics per batch should be written to. If not given,
            batch metrics are not written to a file. Note, that the metrics are written per batch, so one-to-one
            correspondence to the original data set can only be achieved when `batch_max_num_nodes = 2` which allows
            one graph per batch, following the `jraph` logic that one graph in used as padding graph.

    Returns:
        The metrics on `testing_targets`.
    """
    energy_unit = energy_unit_from_config(config=config)
    length_unit = length_unit_from_config(config=config)
    dipole_vec_unit = dipole_vec_unit_from_config(config=config)

    data_filepath = config.data.filepath
    data_filepath = Path(data_filepath).expanduser().resolve()

    targets = testing_targets if testing_targets is not None else list(config.training.loss_weights.keys())

    loader, tf_record_present = data_loader_from_config(
        config=config
    )

    if not tf_record_present:
        eval_data, data_stats = loader.load(
            # We need to do the inverse transforms, since in config everything is in ASE default units.
            cutoff=config.model.cutoff / length_unit,
            cutoff_lr=config.data.neighbors_lr_cutoff / length_unit if config.data.neighbors_lr_bool is True else None,
            calculate_neighbors_lr=config.data.neighbors_lr_bool,
            pick_idx=pick_idx
        )
    else:
        training_data, validation_data, test_data = loader.load(
            cutoff=config.model.cutoff / length_unit,
            num_train=config.training.num_train,
            num_valid=config.training.num_valid,
            cutoff_lr=config.data.neighbors_lr_cutoff / length_unit if config.data.neighbors_lr_bool is True else None,
            calculate_neighbors_lr=config.data.neighbors_lr_bool,
            num_test=num_test,
            return_test=True,
        )
        assert on_split is not None
        if on_split == 'training':
            eval_data = training_data
        elif on_split == 'validation':
            eval_data = validation_data
        elif on_split == 'test':
            eval_data = test_data
        elif on_split == 'full':
            raise NotImplementedError(
                'on_split=full is not supported for TFDS data set yet.'
            )
        else:
            raise ValueError(
                f'{on_split} is not a valid split string. Choose one of (training, validation, test, full).'
            )

    if not tf_record_present:
        num_data = len(eval_data)
    else:
        num_data = len([1 for _ in eval_data.as_numpy_iterator()])

    if num_test is not None:
        if num_test > num_data:
            raise RuntimeError(f'num_test = {num_test} > num_data = {num_data} in data set {data_filepath}.')

        # We assume the tensorflow data set is already shuffled.
        if not tf_record_present:
            # Only shuffle when num_test is not None, since one is then evaluating on a subset of the data.
            numpy_rng = np.random.RandomState(0)
            numpy_rng.shuffle(eval_data)

    if config.training.batch_max_num_nodes is None:
        assert config.training.batch_max_num_edges is None

        if tf_record_present:
            raise ValueError(
                'When reading TFDSDataSet, `max_num_nodes`, `max_num_edges` and `max_num_pairs` can not be auto- '
                'determined. Please set the corresponding values in the config file via '
                'training.batch_max_num_nodes and training.batch_max_num_edges.'
            )

        batch_max_num_nodes = data_stats['max_num_of_nodes'] * (config.training.batch_max_num_graphs - 1) + 1
        batch_max_num_edges = data_stats['max_num_of_edges'] * (config.training.batch_max_num_graphs - 1) + 1

        config.training.batch_max_num_nodes = batch_max_num_nodes
        config.training.batch_max_num_edges = batch_max_num_edges

    if config.training.batch_max_num_pairs is None:
        if config.data.neighbors_lr_bool is True:
            if tf_record_present:
                raise ValueError(
                    'When reading TFDSDataSet, `max_num_pairs` can not be auto- '
                    'determined. Please set the corresponding values in the config file via '
                    'training.batch_max_num_nodes and training.batch_max_num_edges.'
                )

            batch_max_num_pairs = data_stats['max_num_of_nodes'] * (data_stats['max_num_of_nodes'] - 1) * (config.training.batch_max_num_graphs - 1) + 1
        else:
            batch_max_num_pairs = 0

        config.training.batch_max_num_pairs = batch_max_num_pairs

    if not tf_record_present:
        testing_data = data.transformations.subtract_atomic_energy_shifts(
            data.transformations.unit_conversion(
                eval_data[:num_test],
                energy_unit=energy_unit,
                length_unit=length_unit,
                dipole_vec_unit=dipole_vec_unit
            ),
            atomic_energy_shifts={int(k): v for (k, v) in config.data.energy_shifts.items()}
        )
    else:
        testing_data = eval_data.map(
            lambda graph: data.transformations.unit_conversion_graph(
                graph,
                energy_unit=energy_unit,
                length_unit=length_unit,
                dipole_vec_unit=dipole_vec_unit
            )
        )

    ckpt_dir = Path(config.workdir) / 'checkpoints'
    ckpt_dir = ckpt_dir.expanduser().resolve()
    logging.mlff(f'Restore parameters from {ckpt_dir} ...')

    params = checkpoint_utils.load_params_from_checkpoint(ckpt_dir=ckpt_dir)


    print("\nParameter shapes:")

    # Modify parameters to handle theory levels
    if 'params' in params and 'observables_0' in params['params']:
        num_theory_levels = 16

        # Modify energy_offset
        if 'energy_offset' in params['params']['observables_0']:
            old_energy_offset = params['params']['observables_0']['energy_offset']

            # Only tile if shape is 1D
            if len(old_energy_offset.shape) == 1:
                new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
                params['params']['observables_0']['energy_offset'] = new_energy_offset

        # Modify atomic_scales
        if 'atomic_scales' in params['params']['observables_0']:
            old_atomic_scales = params['params']['observables_0']['atomic_scales']

            # Only tile if shape is 1D
            if len(old_atomic_scales.shape) == 1:
                new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
                params['params']['observables_0']['atomic_scales'] = new_atomic_scales

        # Modify energy_dense_final
        if 'energy_dense_final' in params['params']['observables_0']:
            old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']

            # Check the shape to determine if tiling is needed
            if old_kernel.shape[1] == 1:
                new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
                params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel

    print("=" * 50)
    print_param_shapes(params)
    print("=" * 50)

    total_params = count_params(params)
    print(f"\nTotal number of parameters: {total_params:,}")

    logging.mlff(f'... done.')

    if model == 'so3krates':
        net = make_so3krates_sparse_from_config(config)
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )
    logging.mlff(f'Evaluate on {data_filepath} for targets {targets}.')

    return evaluation_utils.evaluate(
        model=net,
        params=params,
        graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
        testing_data=testing_data,
        testing_targets=targets,
        batch_max_num_nodes=config.training.batch_max_num_nodes,
        batch_max_num_edges=config.training.batch_max_num_edges,
        batch_max_num_graphs=config.training.batch_max_num_graphs,
        batch_max_num_pairs=config.training.batch_max_num_pairs,
        write_batch_metrics_to=write_batch_metrics_to
    )


def run_fine_tuning(
        config: config_dict.ConfigDict,
        start_from_workdir: str,
        strategy: str,
        model: str = 'so3krates',
):
    """Run training given a config.

        Args:
            config (): The config.
            start_from_workdir (str): The workdir of the model that should be fine tuned.
            strategy (str): Fine tuning strategy.
            model (str): Model to finetune.

        Returns:

    """

    # Currently, fine tuning is always performed with sparse neighborlist_format for long range blocks.
    config.neighborlist_format_lr = config_dict.placeholder(str)
    config.neighborlist_format_lr = 'sparse'

    # Determine the workdir from which to load the model for fine-tuning.
    start_from_workdir = Path(start_from_workdir).expanduser().resolve()
    if not start_from_workdir.exists():
        raise ValueError(
            f'Trying to start fine tuning from {start_from_workdir} but directory does not exist.'
        )
    
    # Load the config in the workdir to obtain the model hyperparameters and other important statistics.
    hyperparams_path = start_from_workdir / 'hyperparameters.json'
    with open(hyperparams_path, mode='r') as fp:
        config_start_from_workdir = config_dict.ConfigDict(json.load(fp=fp))

    # Select a fine-tuning strategy.
    if strategy == 'full':
        # All parameters are re-fined.
        trainable_subset_keys = None
    elif strategy == 'final_mlp':
        # Only the final MLP is refined
        trainable_subset_keys = ['observables_0']
    elif strategy == 'final_mlp_and_hirshfeld':
        # Only the final MLP and the Hirshfeld are refined
        trainable_subset_keys = ['observables_0', 'observables_2']
    elif strategy == 'hirshfeld':
        # Only the Hirshfeld is refined
        trainable_subset_keys = ['observables_2']
    elif strategy == 'last_layer':
        # Only the last MP layer is refined.
        trainable_subset_keys = [f'layers_{config_start_from_workdir.model.num_layers - 1}']
    elif strategy == 'last_layer_and_final_mlp':
        # Only the last layer and the final MLP are refined.
        trainable_subset_keys = [f'layers_{config_start_from_workdir.model.num_layers - 1}', 'observables_0']
    elif strategy == 'first_layer':
        # Only the first layer is refined.
        trainable_subset_keys = ['layers_0']
    elif strategy == 'first_layer_and_last_layer':
        # Only the first and layer MP layer are refined.
        trainable_subset_keys = ['layers_0', f'layers_{config_start_from_workdir.model.num_layers - 1}']
    else:
        raise ValueError(
            f'--strategy {strategy} is unknown. Select one of '
            f'(`full`, '
            f'`final_mlp`, '
            f'`last_layer`, '
            f'`last_layer_and_final_mlp`, '
            f'`first_layer`, '
            f'`first_layer_and_last_layer`)'
        )
    
    # Finalize the config for IO
    config.model = config_start_from_workdir.model
    # For average number of neighbors message normalization, we have to overwrite the average
    # number of neighbors in case a finetuned model will serve as starting point for another finetuning run.
    # Otherwise the second finetuned model, will use the average number of neighbors from the first finetuning run, 
    # not corresponding to the trained message normalization.
    if config_start_from_workdir.model.message_normalization == 'avg_num_neighbors':
        config.data.avg_num_neighbors = config_start_from_workdir.data.avg_num_neighbors

    # Workdir for fine-tuning experiments.
    workdir = workdir_from_config(config=config)
    workdir.mkdir(exist_ok=True)

    # Update the workdir in config with absolute path.
    config.workdir = str(workdir)

    # Save fine-tuning hyper-parameters.
    with open(workdir / 'fine_tuning.json', mode='w') as fp:
        json.dump(
            {
                'start_from_workdir': start_from_workdir.as_posix(),
                'strategy': strategy
            },
            fp=fp
        )

    # Load the parameters from the model for fine-tuning.
    params = checkpoint_utils.load_params_from_workdir(start_from_workdir)

    # Modify parameters to handle theory levels
    if 'params' in params and 'observables_0' in params['params']:
        num_theory_levels = 16 

        # Modify energy_offset
        if 'energy_offset' in params['params']['observables_0']:
            old_energy_offset = params['params']['observables_0']['energy_offset']

            # Only tile if shape is 1D
            if len(old_energy_offset.shape) == 1:
                new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
                params['params']['observables_0']['energy_offset'] = new_energy_offset

        # Modify atomic_scales
        if 'atomic_scales' in params['params']['observables_0']:
            old_atomic_scales = params['params']['observables_0']['atomic_scales']

            # Only tile if shape is 1D
            if len(old_atomic_scales.shape) == 1:
                new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
                params['params']['observables_0']['atomic_scales'] = new_atomic_scales

        # Modify energy_dense_final
        if 'energy_dense_final' in params['params']['observables_0']:
            old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']

            # Check the shape to determine if tiling is needed
            if old_kernel.shape[1] == 1:
                new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
                params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel

    print("=" * 50)
    print_param_shapes(params)
    print("=" * 50)

    total_params = count_params(params)
    print(f"\nTotal number of parameters: {total_params:,}")

    # Get data filepath.
    data_filepath = data_path_from_config(config=config)

    # Create a loader and check if tf records are present.
    loader, tf_record_present = data_loader_from_config(
        config=config
    )

    # Prepare training and validation data and load the data set statistics.
    training_data, validation_data, data_stats = prepare_training_and_validation_data(
        config=config,
        model_config=None,
        loader=loader,
        tf_record_present=tf_record_present
    )

    opt = make_optimizer_from_config(config)

    # Freeze all parameters arrays except the ones that lie under trainable_subset_keys.
    if trainable_subset_keys is not None:
        opt = training_utils.freeze_parameters(
            optimizer=opt,
            trainable_subset_keys=trainable_subset_keys
        )

    if model == 'so3krates':
        net = make_so3krates_sparse_from_config(config_start_from_workdir)
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )

    # Robust loss configuration
    use_robust_loss = config.training.get('use_robust_loss', False)
    robust_loss_alpha = config.training.get('robust_loss_alpha', 1.99)
    use_adaptive_robust_loss = config.training.get('use_adaptive_robust_loss', False)

    # Validate: can't use both fixed and adaptive robust loss
    if use_robust_loss and use_adaptive_robust_loss:
        raise ValueError(
            "Cannot use both 'use_robust_loss' (fixed) and 'use_adaptive_robust_loss' (learned). "
            "Please enable only one."
        )

    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(net),
        weights=config.training.loss_weights,
        use_robust_loss=use_robust_loss,
        robust_loss_alpha=robust_loss_alpha,
        use_adaptive_robust_loss=use_adaptive_robust_loss,
    )

    val_fn = training_utils.make_val_fn(
        get_energy_and_force_fn_sparse(net),
        weights=config.training.loss_weights,
        use_robust_loss=config.training.get('use_robust_loss_validation', use_robust_loss),
        robust_loss_alpha=config.training.get('robust_loss_alpha_validation', robust_loss_alpha),
        use_adaptive_robust_loss=use_adaptive_robust_loss,
    )

    if config.training.batch_max_num_nodes is None:
        assert config.training.batch_max_num_edges is None

        if tf_record_present:
            raise ValueError(
                'When reading TFDSDataSet, `max_num_nodes`, `max_num_edges` and `max_num_pairs` can not be auto- '
                'determined. Please set the corresponding values in the config file via '
                'training.batch_max_num_nodes and training.batch_max_num_edges.'
            )

        batch_max_num_nodes = data_stats['max_num_of_nodes'] * (config.training.batch_max_num_graphs - 1) + 1
        batch_max_num_edges = data_stats['max_num_of_edges'] * (config.training.batch_max_num_graphs - 1) + 1

        config.training.batch_max_num_nodes = batch_max_num_nodes
        config.training.batch_max_num_edges = batch_max_num_edges

    if config.training.batch_max_num_pairs is None:
        if config.data.neighbors_lr_bool is True:
            if tf_record_present:
                raise ValueError(
                    'When reading TFDSDataSet, `max_num_pairs` can not be auto- '
                    'determined. Please set the corresponding values in the config file via '
                    'training.batch_max_num_nodes and training.batch_max_num_edges.'
                )

            batch_max_num_pairs = data_stats['max_num_of_nodes'] * (data_stats['max_num_of_nodes'] - 1) * (config.training.batch_max_num_graphs - 1) + 1
        else:
            batch_max_num_pairs = 0

        config.training.batch_max_num_pairs = batch_max_num_pairs

    with open(workdir / 'hyperparameters.json', 'w') as fp:
        json.dump(config.to_dict(), fp)

    with open(workdir / "hyperparameters.yaml", "w") as yaml_file:
        yaml.dump(config.to_dict(), yaml_file, default_flow_style=False)

    # Initialize wandb if needed.
    use_wandb = config.training.use_wandb
    if use_wandb is True:
        import wandb
        wandb.init(config=config.to_dict(), **config.training.wandb_init_args)

    logging.mlff(
        f'Fine tuning model from {start_from_workdir} on {data_filepath}!'
    )

    # Adaptive robust loss settings for fine-tuning
    adaptive_robust_loss_targets = list(config.training.loss_weights.keys()) if use_adaptive_robust_loss else None
    adaptive_robust_loss_init_alpha = config.training.get('adaptive_robust_loss_init_alpha', 1.5)
    adaptive_robust_loss_init_scale = config.training.get('adaptive_robust_loss_init_scale', 1.0)

    ckpt_dir = workdir / 'checkpoints'
    if Path(ckpt_dir).exists():
        raise ValueError(
            f"Checkpoint directory {ckpt_dir} already exists."
        )

    if tf_record_present is True:
        training_utils.fit_from_iterator(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
            batch_max_num_edges=config.training.batch_max_num_edges,
            batch_max_num_nodes=config.training.batch_max_num_nodes,
            batch_max_num_graphs=config.training.batch_max_num_graphs,
            batch_max_num_pairs=config.training.batch_max_num_pairs,
            training_iterator=training_data,
            validation_iterator=validation_data,
            params=params,
            ckpt_dir=ckpt_dir,
            eval_every_num_steps=config.training.eval_every_num_steps,
            allow_restart=config.training.allow_restart,
            training_seed=config.training.training_seed,
            model_seed=config.training.model_seed,
            log_gradient_values=config.training.log_gradient_values,
            num_epochs=config.training.num_epochs,
            use_wandb=use_wandb,
            # Adaptive robust loss
            use_adaptive_robust_loss=use_adaptive_robust_loss,
            adaptive_robust_loss_targets=adaptive_robust_loss_targets,
            adaptive_robust_loss_init_alpha=adaptive_robust_loss_init_alpha,
            adaptive_robust_loss_init_scale=adaptive_robust_loss_init_scale
        )
    else:
        training_utils.fit(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
            batch_max_num_edges=config.training.batch_max_num_edges,
            batch_max_num_nodes=config.training.batch_max_num_nodes,
            batch_max_num_graphs=config.training.batch_max_num_graphs,
            batch_max_num_pairs=config.training.batch_max_num_pairs,
            training_data=training_data,
            validation_data=validation_data,
            params=params,
            ckpt_dir=ckpt_dir,
            eval_every_num_steps=config.training.eval_every_num_steps,
            allow_restart=config.training.allow_restart,
            num_epochs=config.training.num_epochs,
            training_seed=config.training.training_seed,
            model_seed=config.training.model_seed,
            log_gradient_values=config.training.log_gradient_values,
            use_wandb=use_wandb,
            # Adaptive robust loss
            use_adaptive_robust_loss=use_adaptive_robust_loss,
            adaptive_robust_loss_targets=adaptive_robust_loss_targets,
            adaptive_robust_loss_init_alpha=adaptive_robust_loss_init_alpha,
            adaptive_robust_loss_init_scale=adaptive_robust_loss_init_scale
        )
    logging.mlff('Training has finished!')


def data_loader_from_config(config):
    """Create a data loader from config.
    
    Args:
        config: Configuration object
        
    Returns:
        DataLoader instance and tf_record_present flag
    """
    # Initialize variables
    loader = None
    tf_record_present = False

    # Get dataset paths and weights from config
    if hasattr(config.data, 'datasets'):
        # Multiple datasets case
        dataset_paths = [Path(d['path']).expanduser().resolve() for d in config.data.datasets]
        dataset_weights = [d.get('weight', 1.0) for d in config.data.datasets]
        tf_record_present = all(len([1 for x in os.scandir(p) if Path(x).suffix[:9] == '.tfrecord']) > 0 for p in dataset_paths)
    else:
        # Single dataset case (original functionality)
        data_filepath = data_path_from_config(config=config)

    energy_unit = energy_unit_from_config(config=config)
    length_unit = length_unit_from_config(config=config)

    if tf_record_present:
        loader = data.QCMLDataLoaderSparseParallel(
            config=config,
            input_folders=dataset_paths,
            dataset_weights=dataset_weights,
            length_unit=length_unit,
            energy_unit=energy_unit
        )
    else:
        if data_filepath.is_file():
            if data_filepath.suffix == '.npz':
                loader = data.NpzDataLoaderSparse(input_file=data_filepath)
            elif data_filepath.stem[:5].lower() == 'spice':
                logging.mlff(f'Found SPICE dataset at {data_filepath}.')
                if data_filepath.suffix != '.hdf5':
                    raise ValueError(
                        f'Loader assumes that SPICE is in hdf5 format. Found {data_filepath.suffix} as'
                        f'suffix.')
                loader = data.SpiceDataLoaderSparse(input_file=data_filepath)
            else:
                loader = data.AseDataLoaderSparse(input_file=data_filepath)
        elif data_filepath.is_dir():
            npz_record_present = len([1 for x in os.scandir(data_filepath) if Path(x).suffix == '.npz']) > 0
            if npz_record_present:
                loader = data.NpzDataLoaderSparse(input_folder=data_filepath)
            else:
                loader = data.AseDataLoaderSparse(input_folder=data_filepath)
        else:
            raise ValueError(f"Data path {data_filepath} does not exist or is not accessible")

    if loader is None:
        raise ValueError(f"Could not initialize data loader for paths {dataset_paths}")

    return loader, tf_record_present


def prepare_training_and_validation_data(config, loader, tf_record_present, model_config: Optional = None):
    # Lock the config.
    config = config.lock()
    # If model config is not None, lock it. Otherwise, use model config from the config. 
    # This handles the case where config and model config can be different, i.e. during finetuning or transfer learning.
    if model_config is not None:
        model_config = model_config.lock()
    else:
        model_config = config.model

    workdir = workdir_from_config(config=config)
    data_filepaths = data_path_from_config(config=config)

    # Extract the units from config.
    energy_unit = energy_unit_from_config(config=config)
    length_unit = length_unit_from_config(config=config)
    dipole_vec_unit = dipole_vec_unit_from_config(config=config)

    # Get dataset weights if specified
    if hasattr(config.data, 'datasets'):
        dataset_weights = [d.get('weight', 1.0) for d in config.data.datasets]
    else:
        dataset_weights = [1.0]  # Default weight for single dataset

    num_train = config.training.num_train
    num_valid = config.training.num_valid

    # Get the total number of data points.
    if not tf_record_present:
        num_data = loader.cardinality()
    else:
        num_data = loader.cardinality()
        print(f"Number of points in train tfds split: {num_data}")

    if num_train + num_valid > num_data:
        raise ValueError(
            f"num_train + num_valid = {num_train + num_valid} exceeds the number of data points {num_data}"
            f" in {data_filepaths}."
        )
    if not tf_record_present:
        split_seed = config.data.split_seed
        numpy_rng = np.random.RandomState(split_seed)

        # Choose the data points that are used training (training + validation data).
        all_indices = np.arange(num_data)
        numpy_rng.shuffle(all_indices)
        # We sort the indices after extracting them from the shuffled list, since we iteratively load the data with the
        # data loader.
        training_and_validation_indices = np.sort(all_indices[:(num_train + num_valid)])
        test_indices = np.sort(all_indices[(num_train + num_valid):])

        # Cutoff is in Angstrom, so we have to divide the cutoff by the length unit.
        training_and_validation_data, data_stats = loader.load(
            cutoff=model_config.cutoff / length_unit,
            cutoff_lr=config.data.neighbors_lr_cutoff / length_unit if config.data.neighbors_lr_bool is True else None,
            calculate_neighbors_lr=config.data.neighbors_lr_bool,
            pick_idx=training_and_validation_indices
        )
        # Since the training and validation indices are sorted, the index i at the n-th entry in
        # training_and_validation_indices corresponds to the n-th entry in training_and_validation_data which is
        # the i-th data entry in the loaded data.
        split_indices = np.arange(num_train + num_valid)
        numpy_rng.shuffle(split_indices)
        internal_train_indices = split_indices[:num_train]
        internal_validation_indices = split_indices[num_train:]

        training_data = [training_and_validation_data[i_train] for i_train in internal_train_indices]
        validation_data = [training_and_validation_data[i_val] for i_val in internal_validation_indices]
        del training_and_validation_data

        assert len(internal_train_indices) == num_train
        assert len(internal_validation_indices) == num_valid

        # internal_*_indices only run from [0, num_train+num_valid]. To get their original position in the full data set
        # we collect them from training_and_validation_indices. Since we will load training and validation data as
        # training_and_validation_data[internal_*_indices], we need to make sure that training_and_validation_indices
        # and training_and_validation_data have the same order in the sense of referencing indices. This is achieved by
        # sorting the indices as described above.
        train_indices = training_and_validation_indices[internal_train_indices]
        validation_indices = training_and_validation_indices[internal_validation_indices]
        assert len(train_indices) == num_train
        assert len(validation_indices) == num_valid
        with open(workdir / 'data_splits.json', 'w') as fp:
            j = dict(
                training=train_indices.tolist(),
                validation=validation_indices.tolist(),
                test=test_indices.tolist()
            )
            json.dump(j, fp)
    else:
        # For parallel data loading, we need to ensure batch parameters are set
        if config.training.batch_max_num_nodes is None or config.training.batch_max_num_edges is None:
            raise ValueError(
                'When using QCMLDataLoaderSparseParallel, `batch_max_num_nodes` and `batch_max_num_edges` must be '
                'specified in the config file via training.batch_max_num_nodes and training.batch_max_num_edges.'
            )
            
        # Check batch_max_num_pairs if neighbors_lr_bool is True
        if config.data.neighbors_lr_bool is True and config.training.batch_max_num_pairs is None:
            raise ValueError(
                'When using QCMLDataLoaderSparseParallel with neighbors_lr_bool=True, `batch_max_num_pairs` must be '
                'specified in the config file via training.batch_max_num_pairs.'
            )
        
        # Create the parallel loader with the full config
        # parallel_loader = data.QCMLDataLoaderSparseParallel(
        #     config=config,
        #     input_folders=data_filepaths,
        #     dataset_weights=dataset_weights,
        #     length_unit=length_unit,
        #     energy_unit=energy_unit
        # )
        
        # # Get data for training and validation
        # training_data = parallel_loader
        # validation_data = parallel_loader
        training_data = loader
        validation_data = loader

        data_stats = None
        # Save the splits.
        with open(workdir / 'data_splits.json', 'w') as fp:
            j = dict(
                training='tfds',
                validation='tfds',
                test='tfds',
            )
            json.dump(j, fp)

    # Explicitly unlock the config.
    config = config.unlock()
    if config.data.shift_mode == 'mean':
        config.data.energy_shifts = config_dict.placeholder(dict)
        energy_mean = data.transformations.calculate_energy_mean(training_data) * energy_unit
        num_nodes = data.transformations.calculate_average_number_of_nodes(training_data)
        energy_shifts = {str(a): float(energy_mean / num_nodes) for a in range(119)}
        config.data.energy_shifts = energy_shifts
    elif config.data.shift_mode == 'lse':
        config.data.energy_shifts = config_dict.placeholder(dict)
        config.data.energy_shifts = data.transformations.calculate_lse_energy_shifts(
            training_data, energy_unit=energy_unit
        )
    elif config.data.shift_mode == 'custom':
        if config.data.energy_shifts is None:
            raise ValueError('For config.data.shift_mode == custom config.data.energy_shifts must be given.')
    else:
        config.data.energy_shifts = {str(a): 0. for a in range(119)}
    # And lock again.
    config = config.lock()

    # Print energy shifts
    non_zero_shifts = {int(k): v for k, v in config.data.energy_shifts.items() if abs(v) > 1e-10}
    if non_zero_shifts:
        logging.mlff(f'Energy shifts (shift_mode={config.data.shift_mode}):')
        for z in sorted(non_zero_shifts.keys()):
            logging.mlff(f'  Z={z}: {non_zero_shifts[z]:.6f} eV')
    else:
        logging.mlff('No energy shifts applied.')

    if not tf_record_present:
        training_data = list(data.transformations.subtract_atomic_energy_shifts(
            data.transformations.unit_conversion(
                training_data,
                energy_unit=energy_unit,
                length_unit=length_unit,
                dipole_vec_unit=dipole_vec_unit
            ),
            atomic_energy_shifts={int(k): v for (k, v) in config.data.energy_shifts.items()}
        ))

        validation_data = list(data.transformations.subtract_atomic_energy_shifts(
            data.transformations.unit_conversion(
                validation_data,
                energy_unit=energy_unit,
                length_unit=length_unit,
                dipole_vec_unit=dipole_vec_unit
            ),
            atomic_energy_shifts={int(k): v for (k, v) in config.data.energy_shifts.items()}
        ))
    else:
        if config.data.shift_mode in ['custom', 'mean', 'lse']:
            raise NotImplementedError(
                'For TFDSDataSets, energy shifting is not supported yet.'
            )
        # TODO: Handle unit conversion inside the dataloader
            

    return training_data, validation_data, data_stats


def workdir_from_config(config):
    workdir = config.workdir
    workdir = Path(workdir).expanduser().resolve()
    return workdir


def data_path_from_config(config):
    """Get the data path from the config.
    
    Args:
        config: The configuration object.
        
    Returns:
        List of paths to the input data folders.
    """
    if hasattr(config.data, 'datasets'):
        # Handle multiple datasets
        data_paths = [d['path'] for d in config.data.datasets]
        return data_paths
    else:
        # Handle single dataset (backward compatibility)
        data_path = config.data.filepath
        data_path = Path(data_path).expanduser().resolve()
        return data_path  # Return as list for consistency


def dipole_vec_unit_from_config(config):
    dipole_vec_unit = config.data.dipole_vec_unit

    # We need to define elementary charge, as it is not defined in ase.units.
    e = 1.

    return eval(dipole_vec_unit)


def charge_unit_from_config(config):
    electric_charge_unit = config.data.electric_charge_unit

    # We need to define elementary charge, as it is not defined in ase.units.
    e = 1.

    return eval(electric_charge_unit)

def energy_unit_from_config(config):
    energy_unit = eval(config.data.energy_unit)
    return energy_unit


def length_unit_from_config(config):
    length_unit = eval(config.data.length_unit)
    return length_unit

def check_config(config):
    neighbors_lr_cutoff = config.data.neighbors_lr_cutoff
    neighbors_lr_bool = config.data.neighbors_lr_bool

    if neighbors_lr_bool is True:
        if neighbors_lr_cutoff is None:
            raise ValueError(
                f'If long-range neighbors are calculated, long-range cutoff must be specified in config. '
                f'Received {neighbors_lr_bool=} and {neighbors_lr_cutoff=} from config.'
            )
