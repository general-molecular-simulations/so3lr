from ase.units import *
import functools
import json
from so3lr.mlff import nn
from so3lr.mlff import training_utils
from so3lr.mlff import evaluation_utils
from so3lr.mlff import data
from so3lr.mlff import jraph_utils
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from so3lr.mlff.utils.kspace_utils import setup_kspace_grid
from ml_collections import config_dict
import numpy as np
from pathlib import Path
from typing import Sequence, Optional
import yaml
import logging
import os
import jax.numpy as jnp

from ..utils import checkpoint_utils
from ..utils.param_utils import (
    count_params,
    count_trainable_params,
    print_param_shapes,
    unmatched_param_keys,
)
from ...finetuning_strategies import resolve_finetuning_strategy
from .wandb_utils import initialize_wandb

logger = logging.getLogger("SO3LR")


def _compute_kspace_constants(config, training_data, tf_record_present):
    """Compute k-space grid constants from config and the first periodic cell in training data.

    Returns a dict with k_grid, k_smearing, k_grid_disp, k_smearing_disp as jnp arrays,
    or None if kspace is disabled.
    """
    kspace_method = config.model.get('kspace_electrostatics', None)
    if kspace_method is None:
        return None

    kspace_smearing = config.model.get('kspace_smearing', 2.0)
    kspace_spacing = config.model.get('kspace_spacing', 1.0)

    # Get a cell from the first periodic structure in training data.
    if tf_record_present:
        # For TFDS, peek at the first batch to get the cell. `parallel=False`
        # is required: this loop breaks after one batch, and the parallel
        # pipeline's workers would then block forever on a full queue, holding
        # every slot of the shared ProcessPoolExecutor so the real training
        # epoch never starts (training hangs at 0 steps with no error).
        first_iter = training_data.next_epoch(
            split='train', mode='train', parallel=False
        )
        for graph_batch, _ in first_iter:
            cell = graph_batch.globals.get('cell')
            if cell is not None:
                # cell shape: (num_graphs, 3, 3) — take the first non-zero cell
                for g in range(cell.shape[0]):
                    c = cell[g]
                    if np.any(c != 0):
                        cell_3x3 = jnp.array(c)
                        break
                else:
                    continue
                break
        else:
            # (No close() needed here: the for/else runs only once the
            # generator is exhausted, which already closed it.)
            raise ValueError(
                "kspace_electrostatics is enabled but no periodic cell found in training data."
            )
        # Release the synchronous generator (and its TFDS iterator) now that the
        # cell is known; we deliberately consume only one batch.
        first_iter.close()
    else:
        # For ASE/Npz data, training_data is a list of (graph, lr_data) tuples.
        cell_3x3 = None
        for graph, _ in training_data:
            cell = graph.globals.get('cell')
            if cell is not None and np.any(cell != 0):
                # cell shape: (1, 3, 3) for a single graph
                cell_3x3 = jnp.array(cell[0])
                break
        if cell_3x3 is None:
            raise ValueError(
                "kspace_electrostatics is enabled but no periodic cell found in training data."
            )

    logger.info(f'Computing k-space grid for training with method={kspace_method}, '
                f'smearing={kspace_smearing}, spacing={kspace_spacing}')
    logger.info(f'Using cell: {cell_3x3}')

    k_grid, k_smearing = setup_kspace_grid(
        kspace_electrostatics=kspace_method,
        kspace_smearing=kspace_smearing,
        kspace_spacing=kspace_spacing,
        box=cell_3x3,
    )

    logger.info(f'K-space grid shape: {k_grid.shape}, smearing: {k_smearing}')

    # Electrostatic and dispersion use the same grid and smearing
    return {
        'k_grid': k_grid,
        'k_smearing': k_smearing,
        'k_grid_disp': k_grid,
        'k_smearing_disp': k_smearing,
    }


def _make_graph_to_batch_fn(config, training_data, tf_record_present):
    """Build graph_to_batch_fn, optionally binding kspace constants via functools.partial."""
    kspace_constants = _compute_kspace_constants(config, training_data, tf_record_present)
    if kspace_constants is not None:
        return functools.partial(
            jraph_utils.graph_to_batch_fn,
            kspace_constants=kspace_constants,
        )
    return jraph_utils.graph_to_batch_fn


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
    kspace_method = model_config.get('kspace_electrostatics', None)

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
        charge_embed_total_charge=getattr(config, 'charge_embed_total_charge', None),
        charge_equilibration_total_charge=getattr(config, 'charge_equilibration_total_charge', None),
        use_spin_embed=model_config.use_spin_embed,
        energy_regression_dim=model_config.energy_regression_dim,
        energy_final_mlp_widths=model_config.get('energy_final_mlp_widths', None),
        energy_activation_fn=model_config.energy_activation_fn,
        energy_learn_atomic_type_scales=model_config.energy_learn_atomic_type_scales,
        energy_learn_atomic_type_shifts=model_config.energy_learn_atomic_type_shifts,
        electrostatic_energy_bool=model_config.electrostatic_energy_bool,
        electrostatic_energy_kspace_method=kspace_method or getattr(config, 'electrostatic_energy_kspace_method', 'pme'),
        electrostatic_energy_kspace_interp_nodes=getattr(config, 'electrostatic_energy_kspace_interp_nodes', 4),
        dispersion_energy_kspace_do_ewald_bool=(kspace_method == 'ewald') if kspace_method else getattr(config, 'dispersion_energy_kspace_do_ewald_bool', None),
        dispersion_energy_kspace_interp_nodes=getattr(config, 'dispersion_energy_kspace_interp_nodes', 4),
        electrostatic_energy_scale=model_config.electrostatic_energy_scale,
        dispersion_energy_bool=model_config.dispersion_energy_bool,
        dispersion_energy_cutoff_lr_damping=model_config.dispersion_energy_cutoff_lr_damping,
        dispersion_energy_scale=model_config.dispersion_energy_scale,
        return_representations_bool=return_representations_bool,
        zbl_repulsion_bool=model_config.zbl_repulsion_bool,
        legacy_so3lr_bool=model_config.get('legacy_so3lr_bool', False),
        use_final_bias_bool=model_config.get('use_final_bias_bool', False),
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
            gradient_clipping_args=opt_config.gradient_clipping_args if opt_config.gradient_clipping_args is not None else dict(),
            # Add the injectable LR knob only when plateau scaling is requested.
            reduce_lr_on_plateau=config.training.get('reduce_lr_on_plateau', False)
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
            logger.warning(
                'Passing a config without data.avg_num_neighbors is deprecated and will raise an error in the future.'
                'Add data.avg_num_neighbors: null to the config to disable the warning. For now, we automatically set '
                'data.avg_num_neighbors: null.'
            )
            config.data.avg_num_neighbors = config_dict.placeholder(float)
            config.data.avg_num_neighbors = None

        if config.data.avg_num_neighbors is not None:
            logger.info(
                f'Read average number of neighbors = {config.data.avg_num_neighbors} from config.'
            )
        else:
            logger.info('Calculate average number of neighbors ...')
            avg_num_neighbors = np.array(data.transformations.calculate_average_number_of_neighbors(training_data))
            config.data.avg_num_neighbors = np.array(avg_num_neighbors).item()
            logger.info('... done.')

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
    robust_loss_alpha = config.training.get('robust_loss_alpha', 1.0)
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

    val_fn = training_utils.make_val_fn( #TODO: maybe not needed for val_fn - want to see rmse
        force_fn,
        weights=config.training.loss_weights,
        use_robust_loss=config.training.get('use_robust_loss_validation', use_robust_loss),
        robust_loss_alpha=config.training.get('robust_loss_alpha_validation', robust_loss_alpha),
        use_adaptive_robust_loss=use_adaptive_robust_loss,
    )

    # Optional two-stage loss reweighting: after `stage_two_start_epoch` the
    # training/validation loss is rebuilt with `stage_two_loss_weights`.
    stage_two_start_epoch = config.training.get('stage_two_start_epoch', None)
    stage_two_loss_weights = config.training.get('stage_two_loss_weights', None)
    stage_two_loss_fn = None
    stage_two_val_fn = None
    if stage_two_start_epoch is not None and stage_two_loss_weights is not None:
        stage_two_loss_fn = training_utils.make_loss_fn(
            force_fn,
            weights=stage_two_loss_weights,
            use_robust_loss=use_robust_loss,
            robust_loss_alpha=robust_loss_alpha,
            use_adaptive_robust_loss=use_adaptive_robust_loss,
        )
        stage_two_val_fn = training_utils.make_val_fn(
            force_fn,
            weights=stage_two_loss_weights,
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
            if tf_record_present:
                raise ValueError(
                    'When reading TFDSDataSet, `max_num_pairs` can not be auto- '
                    'determined. Please set the corresponding values in the config file via '
                    'training.batch_max_num_nodes and training.batch_max_num_edges.'
                )

            batch_max_num_pairs = data_stats['max_num_of_pairs'] * (config.training.batch_max_num_graphs - 1) + 1
        else:
            batch_max_num_pairs = 0

        config.training.batch_max_num_pairs = batch_max_num_pairs

    # Echo device and batching info to the (file-backed) SO3LR logger so a
    # training run is self-documenting from its log file alone -- not only from
    # captured SLURM stdout. Runs regardless of whether training was started via
    # `so3lr train`/`so3lr finetune` or by calling the functions directly.
    try:
        import jax as _jax
        _devices = _jax.devices()
        logger.info("=" * 60)
        logger.info(f"JAX backend:               {_jax.default_backend()}")
        logger.info(f"Number of devices:         {len(_devices)}")
        for _i, _d in enumerate(_devices):
            logger.info(f"  Device {_i}:                 {_d.device_kind} ({_d.platform})")
    except Exception as _e:  # pragma: no cover - purely informational
        logger.info(f"Could not query JAX devices: {_e}")
    logger.info("-" * 60)
    logger.info("Training batch settings:")
    logger.info(f"  batch_max_num_graphs:    {config.training.batch_max_num_graphs}")
    logger.info(f"  batch_max_num_nodes:     {config.training.batch_max_num_nodes}")
    logger.info(f"  batch_max_num_edges:     {config.training.batch_max_num_edges}")
    logger.info(f"  batch_max_num_pairs:     {config.training.batch_max_num_pairs}")
    logger.info(f"  num_epochs:              {config.training.num_epochs}")
    logger.info("=" * 60)

    with open(workdir / 'hyperparameters.json', 'w') as fp:
        json.dump(config.to_dict(), fp)

    with open(workdir / "hyperparameters.yaml", "w") as yaml_file:
        yaml.dump(config.to_dict(), yaml_file, default_flow_style=False)

    # W&B is a strict opt-in. The helper returns before importing W&B when the
    # setting is absent/false and redacts the effective config when enabled.
    use_wandb = initialize_wandb(config)

    # Adaptive robust loss settings
    adaptive_robust_loss_targets = list(config.training.loss_weights.keys()) if use_adaptive_robust_loss else None
    adaptive_robust_loss_init_alpha = config.training.get('adaptive_robust_loss_init_alpha', 1.5)
    adaptive_robust_loss_init_scale = config.training.get('adaptive_robust_loss_init_scale', 1.0)

    _graph_to_batch_fn = _make_graph_to_batch_fn(config, training_data, tf_record_present)

    logger.info('Training is starting!')
    if not tf_record_present:
        training_utils.fit(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=_graph_to_batch_fn,
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
            adaptive_robust_loss_init_scale=adaptive_robust_loss_init_scale,
            # Exponential moving average of weights.
            use_ema=config.training.get('use_ema', False),
            ema_decay=config.training.get('ema_decay', 0.999),
            # Early stopping.
            early_stopping_patience=config.training.get('early_stopping_patience', None),
            early_stopping_min_delta=config.training.get('early_stopping_min_delta', 1e-4),
            # Reduce LR on plateau.
            reduce_lr_on_plateau=config.training.get('reduce_lr_on_plateau', False),
            reduce_lr_factor=config.training.get('reduce_lr_factor', 0.75),
            reduce_lr_patience=config.training.get('reduce_lr_patience', 1000),
            reduce_lr_min_scale=config.training.get('reduce_lr_min_scale', 0.0),
            reduce_lr_min_delta=config.training.get('reduce_lr_min_delta', 1e-4),
            reduce_lr_cooldown=config.training.get('reduce_lr_cooldown', 0),
            # Two-stage loss reweighting.
            stage_two_start_epoch=stage_two_start_epoch,
            stage_two_loss_fn=stage_two_loss_fn,
            stage_two_val_fn=stage_two_val_fn
        )
    else:
        training_utils.fit_from_iterator(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=_graph_to_batch_fn,
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
            adaptive_robust_loss_init_scale=adaptive_robust_loss_init_scale,
            # Exponential moving average of weights.
            use_ema=config.training.get('use_ema', False),
            ema_decay=config.training.get('ema_decay', 0.999),
            # Early stopping.
            early_stopping_patience=config.training.get('early_stopping_patience', None),
            early_stopping_min_delta=config.training.get('early_stopping_min_delta', 1e-4),
            # Reduce LR on plateau.
            reduce_lr_on_plateau=config.training.get('reduce_lr_on_plateau', False),
            reduce_lr_factor=config.training.get('reduce_lr_factor', 0.75),
            reduce_lr_patience=config.training.get('reduce_lr_patience', 1000),
            reduce_lr_min_scale=config.training.get('reduce_lr_min_scale', 0.0),
            reduce_lr_min_delta=config.training.get('reduce_lr_min_delta', 1e-4),
            reduce_lr_cooldown=config.training.get('reduce_lr_cooldown', 0),
            # Two-stage loss reweighting.
            stage_two_start_epoch=stage_two_start_epoch,
            stage_two_loss_fn=stage_two_loss_fn,
            stage_two_val_fn=stage_two_val_fn,
            # Dataset info for effective epoch tracking
            dataset_info=training_data.dataset_info() if hasattr(training_data, 'dataset_info') else None,
        )
    logger.info('Training has finished!')


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

            batch_max_num_pairs = data_stats['max_num_of_pairs'] * (config.training.batch_max_num_graphs - 1) + 1
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
    logger.info(f'Restore parameters from {ckpt_dir} ...')

    params = checkpoint_utils.load_params_from_checkpoint(ckpt_dir=ckpt_dir)


    print("\nParameter shapes:")

    # Expand a single-theory-level checkpoint to the multi-head energy readout.
    # Shared with fit()/fit_from_iterator() so the three call sites cannot drift.
    params = training_utils.expand_observables_to_theory_levels(params)

    print("=" * 50)
    print_param_shapes(params)
    print("=" * 50)

    total_params = count_params(params)
    print(f"\nTotal number of parameters: {total_params:,}")

    logger.info(f'... done.')

    if model == 'so3krates':
        net = make_so3krates_sparse_from_config(config)
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )
    logger.info(f'Evaluate on {data_filepath} for targets {targets}.')

    return evaluation_utils.evaluate(
        model=net,
        params=params,
        # Bind the k-space constants exactly as run_training/run_fine_tuning do.
        # Using the bare graph_to_batch_fn left 'k_smearing'/'k_grid' out of the
        # batch, and EnergySparse silently skips its reciprocal-space branch when
        # they are absent -- so evaluating a model trained with
        # kspace_electrostatics reported metrics for a different Hamiltonian than
        # the one that was trained, with no warning.
        graph_to_batch_fn=_make_graph_to_batch_fn(
            config, testing_data, tf_record_present
        ),
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
    # Read once: this object also supplies `config.model` below and is what the
    # network is built from, so a second read later would shadow it with a
    # different (but equal) ConfigDict and silently split the two.
    hyperparams_path = start_from_workdir / 'hyperparameters.json'
    with open(hyperparams_path, mode='r') as fp:
        config_start_from_workdir = config_dict.ConfigDict(json.load(fp=fp))

    # The net is built from `config_start_from_workdir` (see below), so forcing
    # 'sparse' on `config` alone was dead: a checkpoint saved with
    # `neighborlist_format_lr: ordered_sparse` would have been rebuilt with
    # ordered_sparse long-range indexing while the data loader emits plain
    # sparse pairs -- and the finetune workdir would then advertise 'sparse' in
    # its own hyperparameters.json. Override the value the model is actually
    # built with, and add the key when the source config predates it.
    if config_start_from_workdir.get('neighborlist_format_lr', None) != 'sparse':
        with config_start_from_workdir.unlocked():
            config_start_from_workdir.neighborlist_format_lr = 'sparse'

    # Select a fine-tuning strategy. The name -> keys table lives in
    # so3lr.finetuning_strategies so that the CLI's --strategy choices and this
    # dispatch cannot drift apart (they had: `freeze_all_but_final_mlp` was
    # implemented here but rejected by click, and `final_mlp_and_hirshfeld` was
    # missing from the unknown-strategy message).
    #
    # trainable_subset_keys: whitelist (train only these). frozen_subset_keys:
    # blacklist (freeze only these, train the rest). At most one is set.
    #
    # `num_layers` must come from the checkpoint, not from the user's finetune
    # config: the last-layer strategies index the module by name, and an
    # off-by-one selects a `layers_N` that does not exist.
    trainable_subset_keys, frozen_subset_keys = resolve_finetuning_strategy(
        strategy,
        num_layers=config_start_from_workdir.model.num_layers
    )

    # Finalize the config for IO.
    # The fine-tuned model's architecture is fully determined by the
    # checkpoint we are starting from, so any `model:` block in the user's
    # finetune YAML cannot actually take effect — the saved hyperparameters
    # are authoritative. Previously this assignment was silent; now warn
    # loudly so users notice that their model: overrides are being dropped.
    #
    # Capture the values the checkpoint is about to overwrite FIRST: the
    # consistency checks below have to compare the user's request against the
    # checkpoint. Running them after the assignment compared the checkpoint
    # against itself, so the message-normalization guard could never fire and
    # the avg_num_neighbors warning fired on every single run.
    user_message_normalization = None
    user_avg_num_neighbors = config.data.get('avg_num_neighbors', None)
    if 'model' in config and config.model is not None:
        user_model_keys = sorted(config.model.keys())
        user_message_normalization = config.model.get('message_normalization', None)
        if user_model_keys:
            logger.warning(
                'Ignoring user-supplied `model:` block in finetune config; '
                'the checkpoint at %s defines the model architecture. '
                'Dropped keys: %s',
                start_from_workdir,
                user_model_keys,
            )

    # Check that message normalization has not changed from the original model
    # to the fine-tuning model. Fails before any data is loaded.
    message_normalization = config_start_from_workdir.model.message_normalization
    if (user_message_normalization is not None
            and user_message_normalization != message_normalization):
        raise ValueError(
            f'Message normalization must be the same. '
            f'Found {message_normalization} for the original config '
            f'and {user_message_normalization} for the fine tuning config.'
        )

    config.model = config_start_from_workdir.model

    # For average number of neighbors message normalization, we have to overwrite the average
    # number of neighbors in case a finetuned model will serve as starting point for another finetuning run.
    # Otherwise the second finetuned model, will use the average number of neighbors from the first finetuning run, 
    # not corresponding to the trained message normalization.
    if message_normalization == 'avg_num_neighbors':
        if user_avg_num_neighbors is not None:
            logger.warning(
                'Running fine tuning with config.model.message_normalization: avg_num_neighbors does not allow to '
                'reset the avg_num_neighbors in the fine tuning config and must be set to null. It will be loaded from '
                'the config in the workdir that is starting point for the fine tuning.'
            )

        config.data.avg_num_neighbors = config_start_from_workdir.data.avg_num_neighbors
        logger.info(
            f'Read average number of neighbors = {config.data.avg_num_neighbors} from original config at '
            f'{start_from_workdir}.'
        )

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

    # Robust loss configuration. Read before the parameters are counted: the
    # adaptive variant adds learnable parameters to the tree and has to be part
    # of the trainable selection.
    use_robust_loss = config.training.get('use_robust_loss', False)
    robust_loss_alpha = config.training.get('robust_loss_alpha', 1.0)
    use_adaptive_robust_loss = config.training.get('use_adaptive_robust_loss', False)

    # Validate: can't use both fixed and adaptive robust loss
    if use_robust_loss and use_adaptive_robust_loss:
        raise ValueError(
            "Cannot use both 'use_robust_loss' (fixed) and 'use_adaptive_robust_loss' (learned). "
            "Please enable only one."
        )

    adaptive_robust_loss_targets = list(config.training.loss_weights.keys()) if use_adaptive_robust_loss else None
    adaptive_robust_loss_init_alpha = config.training.get('adaptive_robust_loss_init_alpha', 1.5)
    adaptive_robust_loss_init_scale = config.training.get('adaptive_robust_loss_init_scale', 1.0)

    # Load the parameters from the model for fine-tuning.
    params = checkpoint_utils.load_params_from_workdir(start_from_workdir)

    # Expand a single-theory-level checkpoint to the multi-head energy readout.
    # Shared with fit()/fit_from_iterator() so the three call sites cannot drift.
    params = training_utils.expand_observables_to_theory_levels(params)

    # A selection key that matches nothing is never intentional. As a whitelist
    # it contributes no trainable parameters -- and if NO key matches, every
    # parameter is frozen, so training runs to completion, writes checkpoints
    # and exits 0 without having updated anything. Fail instead.
    selection_keys = trainable_subset_keys if trainable_subset_keys is not None else frozen_subset_keys
    if selection_keys:
        available_modules = sorted(params.get('params', params).keys())
        unmatched = unmatched_param_keys(params, selection_keys)
        if unmatched:
            logger.warning(
                'Fine-tuning strategy %r selects %s, which match no parameter in the '
                'checkpoint at %s. Available top-level modules: %s.',
                strategy, unmatched, start_from_workdir, available_modules
            )
        if trainable_subset_keys is not None and len(unmatched) == len(trainable_subset_keys):
            raise ValueError(
                f'Fine-tuning strategy {strategy!r} would freeze every parameter: none of '
                f'{list(trainable_subset_keys)} match a parameter in the checkpoint at '
                f'{start_from_workdir}, so training would run to completion without '
                f'updating anything. Available top-level modules: {available_modules}.'
            )

    if use_adaptive_robust_loss:
        # fit()/fit_from_iterator() only create these when they initialize a
        # fresh model, which fine-tuning never does -- so without this the
        # feature silently did nothing (or reused the pretraining run's alphas
        # under the new target ordering).
        params = training_utils.ensure_adaptive_robust_loss_params(
            params,
            targets=adaptive_robust_loss_targets,
            init_alpha=adaptive_robust_loss_init_alpha,
            init_scale=adaptive_robust_loss_init_scale
        )
        if trainable_subset_keys is not None:
            # A whitelist strategy would otherwise freeze the learnable
            # alpha/scale, leaving 'adaptive' robust loss fixed at its init.
            trainable_subset_keys = list(trainable_subset_keys) + ['adaptive_robust_loss']

    print("=" * 50)
    print_param_shapes(params)
    print("=" * 50)

    total_params = count_params(params)
    trainable_params = count_trainable_params(
        params,
        trainable_keys=trainable_subset_keys,
        frozen_keys=frozen_subset_keys
    )

    print(f"\nTotal number of parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if trainable_params != total_params:
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Fine-tuning strategy: {strategy}")
        if trainable_subset_keys:
            print(f"Trainable modules: {trainable_subset_keys}")
    if frozen_subset_keys:
        print(f"Fine-tuning strategy: {strategy}")
        print(f"Frozen modules: {frozen_subset_keys}")

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

    # Freeze all parameter arrays except those under trainable_subset_keys, or
    # (for blacklist strategies) freeze only those under frozen_subset_keys.
    if trainable_subset_keys is not None:
        opt = training_utils.freeze_parameters(
            optimizer=opt,
            trainable_subset_keys=trainable_subset_keys
        )
    elif frozen_subset_keys is not None:
        opt = training_utils.freeze_parameters(
            optimizer=opt,
            frozen_subset_keys=frozen_subset_keys
        )

    if model == 'so3krates':
        net = make_so3krates_sparse_from_config(config_start_from_workdir)
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
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

            batch_max_num_pairs = data_stats['max_num_of_pairs'] * (config.training.batch_max_num_graphs - 1) + 1
        else:
            batch_max_num_pairs = 0

        config.training.batch_max_num_pairs = batch_max_num_pairs

    # Echo device and batching info to the (file-backed) SO3LR logger so a
    # training run is self-documenting from its log file alone -- not only from
    # captured SLURM stdout. Runs regardless of whether training was started via
    # `so3lr train`/`so3lr finetune` or by calling the functions directly.
    try:
        import jax as _jax
        _devices = _jax.devices()
        logger.info("=" * 60)
        logger.info(f"JAX backend:               {_jax.default_backend()}")
        logger.info(f"Number of devices:         {len(_devices)}")
        for _i, _d in enumerate(_devices):
            logger.info(f"  Device {_i}:                 {_d.device_kind} ({_d.platform})")
    except Exception as _e:  # pragma: no cover - purely informational
        logger.info(f"Could not query JAX devices: {_e}")
    logger.info("-" * 60)
    logger.info("Training batch settings:")
    logger.info(f"  batch_max_num_graphs:    {config.training.batch_max_num_graphs}")
    logger.info(f"  batch_max_num_nodes:     {config.training.batch_max_num_nodes}")
    logger.info(f"  batch_max_num_edges:     {config.training.batch_max_num_edges}")
    logger.info(f"  batch_max_num_pairs:     {config.training.batch_max_num_pairs}")
    logger.info(f"  num_epochs:              {config.training.num_epochs}")
    logger.info("=" * 60)

    with open(workdir / 'hyperparameters.json', 'w') as fp:
        json.dump(config.to_dict(), fp)

    with open(workdir / "hyperparameters.yaml", "w") as yaml_file:
        yaml.dump(config.to_dict(), yaml_file, default_flow_style=False)

    # Fine-tuning follows the same opt-in and redaction path as training.
    use_wandb = initialize_wandb(config)

    logger.info(
        f'Fine tuning model from {start_from_workdir} on {data_filepath}!'
    )

    ckpt_dir = workdir / 'checkpoints'
    if Path(ckpt_dir).exists() and not config.training.allow_restart:
        raise ValueError(
            f"Checkpoint directory {ckpt_dir} already exists. Pass "
            f"`training.allow_restart: true` in the config to resume from "
            f"it, or choose a different --workdir."
        )

    _graph_to_batch_fn = _make_graph_to_batch_fn(config, training_data, tf_record_present)

    if tf_record_present is True:
        training_utils.fit_from_iterator(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=_graph_to_batch_fn,
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
            adaptive_robust_loss_init_scale=adaptive_robust_loss_init_scale,
            # Dataset info for effective epoch tracking
            dataset_info=training_data.dataset_info() if hasattr(training_data, 'dataset_info') else None,
        )
    else:
        training_utils.fit(
            model=net,
            optimizer=opt,
            loss_fn=loss_fn,
            val_fn=val_fn,
            graph_to_batch_fn=_graph_to_batch_fn,
            batch_max_num_edges=config.training.batch_max_num_edges,
            batch_max_num_nodes=config.training.batch_max_num_nodes,
            batch_max_num_graphs=config.training.batch_max_num_graphs,
            batch_max_num_pairs=config.training.batch_max_num_pairs,
            training_data=training_data,
            validation_data=validation_data,
            params=params,
            ckpt_dir=ckpt_dir, #TODO: or ckpt_dir=workdir / 'checkpoints',?
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
    logger.info('Training has finished!')


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
        # Multi-dataset training (config.data.datasets) is implemented only for the TFDS loader.
        # If none/some of the paths contain .tfrecord shards the code below would fall through
        # to the ASE/NPZ branch and crash with UnboundLocalError on data_filepath. Fail loudly.
        if not tf_record_present:
            raise ValueError(
                "config.data.datasets (multi-dataset training) is only supported by the TFDS "
                "dataloader, but no .tfrecord shards were found in one or more of: "
                f"{[str(p) for p in dataset_paths]}. Either convert each dataset to TFDS shards "
                "(see README), or use a single dataset via config.data.filepath."
            )
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
        # config.data.filter.min_distance / max_force are honoured only by the TFDS loader.
        # If the user set them but we are about to instantiate an ASE/NPZ loader, warn loudly
        # so the values are not silently ignored.
        if hasattr(config.data, 'filter'):
            _filt = config.data.filter
            _min_d = _filt.min_distance if hasattr(_filt, 'min_distance') else None
            _max_f = _filt.max_force if hasattr(_filt, 'max_force') else None
            if _min_d is not None or _max_f is not None:
                logger.warning(
                    "config.data.filter.* is honoured only by the TFDS dataloader. "
                    f"You set min_distance={_min_d}, max_force={_max_f}, but the ASE/NPZ "
                    "loader will IGNORE these values -- no structures will be filtered out. "
                    "Convert the dataset to TFDS shards if filtering is required."
                )
        if data_filepath.is_file():
            if data_filepath.suffix == '.npz':
                loader = data.NpzDataLoaderSparse(input_file=data_filepath)
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
        logger.info(f'Energy shifts (shift_mode={config.data.shift_mode}):')
        for z in sorted(non_zero_shifts.keys()):
            logger.info(f'  Z={z}: {non_zero_shifts[z]:.6f} eV')
    else:
        logger.info('No energy shifts applied.')

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

    try:
        return eval(dipole_vec_unit)
    except Exception as exc:
        raise ValueError(
            f"Could not parse unit string {dipole_vec_unit!r} for data.dipole_vec_unit. "
            f"Use ASE.units expressions like 'e * Angstrom' or 'Debye' "
            f"(note the spaces around operators)."
        ) from exc


def charge_unit_from_config(config):
    electric_charge_unit = config.data.electric_charge_unit

    # We need to define elementary charge, as it is not defined in ase.units.
    e = 1.

    try:
        return eval(electric_charge_unit)
    except Exception as exc:
        raise ValueError(
            f"Could not parse unit string {electric_charge_unit!r} for data.electric_charge_unit. "
            f"Use ASE.units expressions like 'e' or '1.0' "
            f"(note the spaces around operators)."
        ) from exc

def energy_unit_from_config(config):
    energy_unit_str = config.data.energy_unit
    try:
        energy_unit = eval(energy_unit_str)
    except Exception as exc:
        raise ValueError(
            f"Could not parse unit string {energy_unit_str!r} for data.energy_unit. "
            f"Use ASE.units expressions like 'kcal / mol' or 'eV' "
            f"(note the spaces around operators)."
        ) from exc
    return energy_unit


def length_unit_from_config(config):
    length_unit_str = config.data.length_unit
    try:
        length_unit = eval(length_unit_str)
    except Exception as exc:
        raise ValueError(
            f"Could not parse unit string {length_unit_str!r} for data.length_unit. "
            f"Use ASE.units expressions like 'Angstrom' or 'Bohr' "
            f"(note the spaces around operators)."
        ) from exc
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
