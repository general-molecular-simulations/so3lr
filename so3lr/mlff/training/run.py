import jax.numpy as jnp
import numpy as np
import jax
import logging
import time
import wandb
import optax

from orbax.checkpoint import (CheckpointManagerOptions,
                              CheckpointManager,
                              PyTreeCheckpointer)

from functools import partial
from typing import (Any, Callable, Dict, Tuple)
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict, unfreeze

from so3lr.mlff.io import save_dict
from so3lr.mlff.io.io import __STEP_PREFIX__

# logging.basicConfig(level=logging.INFO)

Array = Any
StackNet = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]


@partial(jax.jit, static_argnums=2)
def train_step_fn(state: TrainState,
                  batch: Dict,
                  loss_fn: Callable) -> Tuple[TrainState, Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Training step.

        state (TrainState): Flax train state.
        batch (Tuple): Batch of validation data.
        loss_fn (Callable): Loss function.

    Returns: Updated optimizer state and loss for current batch.
    """
    (loss, train_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    train_metrics['gradients_norm'] = optax.global_norm(grads)
    return state, train_metrics, grads


def valid_epoch(state: TrainState,
                ds: DataTupleT,
                metric_fn: LossFn,
                bs: int) -> Tuple[Dict[str, np.array], int]:
    """
    Validation epoch for NN training.

    Args:
        state (TrainState): Flax train state.
        ds (Tuple): Validation data. First entry is input, second is expected output
        metric_fn (Callable): Function that evaluates the model wrt some metric fn
        bs (int): Batch size.

    Returns: Validation metric.
    """
    inputs, targets = ds
    _k = list(inputs.keys())
    n_data = len(inputs[_k[0]])

    steps_per_epoch = n_data // bs
    batch_metrics = []
    idxs = jnp.arange(n_data)
    idxs = idxs[:steps_per_epoch * bs]  # skip incomplete batch
    idxs = idxs.reshape((steps_per_epoch, bs))
    for idx in idxs:
        batch = jax.tree_util.tree_map(lambda y: y[idx, ...], ds)
        # batch = (Dict[str, Array[perm, ...]], Dict[str, Array[perm, ...]])
        # TODO: replace valid_step_fn with metric_fn
        metrics = valid_step_fn(state, batch, metric_fn)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    # batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics]) for k in batch_metrics[0]}
    return epoch_metrics_np, n_data


# TODO: remove jit @valid_step_fn and JIT the obs_fn in metric_fn creator. This will allow to report
@partial(jax.jit, static_argnums=2)
def valid_step_fn(state: TrainState,
                  batch: DataTupleT,
                  metric_fn: Callable) -> Dict[str, jnp.ndarray]:
    """
    Validation step.

    Args:
        state (TrainState): Flax train state.
        batch (Tuple): Batch of validation data.
        metric_fn (Callable): Function that evaluates the model wrt some metric fn

    Returns: Validation metrics on the batch.
    """
    _, metrics = metric_fn(state.valid_params, batch)
    return metrics


def run_training(state: TrainState,
                 loss_fn: LossFn,
                 train_ds: DataTupleT,
                 valid_ds: DataTupleT,
                 train_bs: int,
                 valid_bs: int,
                 metric_fn: LossFn = None,
                 epochs: int = 100,
                 ckpt_dir: str = None,
                 ckpt_manager_options: dict = None,
                 eval_every_t: int = None,
                 log_every_t: int = None,
                 restart_by_nan: bool = True,
                 stop_lr_fn: Callable[[float], bool] = None,
                 stop_metric_fn: Callable[[Dict[str, float]], bool] = None,
                 loss_fn_input: Any = None,
                 seed: int = 0,
                 use_wandb: bool = True
                 ):
    """
    Run training for a NN. The checkpoints are saved, such that _n corresponds to the case of n optimizer updates. By
    doing so, it can be directly compared to number steps that can be specified for the neural tangent function on
    NNGPs where a time step can be specified. Also, checkpoints saved with _0 at the end, correspond to the model at
    initialization which also might be interesting for our case.
    Args:
        state (TrainState): Flax train state.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        train_ds (Tuple): Tuple of training data. First entry is input, second is expected output.
        valid_ds (Tuple): Tuple of validation data. First entry is input, second is expected output.
        train_bs (int): Training batch size.
        valid_bs (int): Validation batch size.
        metric_fn (Callable): Dictionary of functions, which are evaluated on the validation set and logged.
        epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_t (int): Evaluate the metrics every t-th step
        log_every_t (int): Log the training loss every t-th step
        restart_by_nan (bool): Soft restart from last checkpoint when NaNs appear in the gradients.
        stop_lr_fn (Callable): Function that returns True if a certain lr threshold is passed.
        stop_metric_fn (Callable): Function that returns True if a certain threshold for one of the specified metrics
            is passed.
        loss_fn_input (Tuple): Additional input to the loss function, that is not fixed prior to training. E.g. a
            quantity that is associated with each data point and thus depends on the internal splitting of the
            batches.
        seed (int): Random seed.
        use_wandb (bool): Log statistics to WeightsAndBias.
    Returns:
    """
    rng = jax.random.PRNGKey(seed)

    if metric_fn is None:
        metric_fn = loss_fn
        logging.info('No metrics functions defined, default to loss function.')

    tot_time = 0
    best_valid_metrics = None

    # process training data
    train_input, _ = train_ds
    _k_train = list(train_input.keys())
    n_train = len(train_input[_k_train[0]])

    steps_per_epoch = n_train // train_bs

    if eval_every_t is None:
        eval_every_t = steps_per_epoch

    if log_every_t is None:
        log_every_t = 1

    if ckpt_manager_options is None:
        ckpt_manager_options = {'max_to_keep': 1}

    options = CheckpointManagerOptions(best_fn=lambda u: u['loss'], best_mode='min', step_prefix=__STEP_PREFIX__,
                                       **ckpt_manager_options)

    # mngr = CheckpointManager(ckpt_dir, {'state': AsyncCheckpointer(PyTreeCheckpointHandler())}, options=options)
    mngr = CheckpointManager(ckpt_dir, {'state': PyTreeCheckpointer()}, options=options)

    for i in range(1, int(steps_per_epoch * epochs) + 1):
        epoch_start = time.time()

        # create checkpoint before any gradient updates happen which also define the best (at initialization)
        # validation metrics
        if i == 1:
            best_valid_metrics, _ = valid_epoch(state, valid_ds, metric_fn, bs=valid_bs)
            mngr.save(i - 1, {'state': state}, metrics={'loss': best_valid_metrics['loss'].item()})

        step_in_epoch = (i - 1) % steps_per_epoch
        if step_in_epoch == 0:
            rng, input_rng = jax.random.split(rng)
            perms = jax.random.permutation(input_rng, n_train)
            perms = perms[:steps_per_epoch * train_bs]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, train_bs))

        train_batch = jax.tree_util.tree_map(lambda y: y[perms[step_in_epoch], ...], train_ds)
        if loss_fn_input is not None:
            loss_input_batch = jax.tree_util.tree_map(lambda y: y[perms[step_in_epoch], ...], loss_fn_input)
            train_batch = train_batch + loss_input_batch
        train_start = time.time()
        state, train_batch_metrics, grads = train_step_fn(state=state, batch=train_batch, loss_fn=loss_fn)
        train_end = time.time()

        # check for NaN
        train_batch_metrics_np = jax.tree_util.tree_map(lambda x: np.array(x), train_batch_metrics)

        if (np.isnan(train_batch_metrics_np['loss']) or
            np.isinf(train_batch_metrics_np['loss']) or
            np.isnan(train_batch_metrics_np['gradients_norm']) or
            np.isinf(train_batch_metrics_np['gradients_norm'])) and restart_by_nan:

            if np.isnan(train_batch_metrics_np['loss']) or np.isinf(train_batch_metrics_np['loss']):
                logging.warning(f'NaN detected during training in step {i} in the loss function value. Reload the '
                                'last checkpoint.')

            if np.isnan(train_batch_metrics_np['gradients_norm']) or np.isinf(train_batch_metrics_np['gradients_norm']):
                logging.warning(f'NaN detected during training in step {i} in gradient values. Reload the '
                                'last checkpoint.')

            grads_np = jax.tree_util.tree_map(lambda x: np.array(x).tolist(), grads)
            save_dict(ckpt_dir, filename=f'gradients_nan_step_{i}.json', data=unfreeze(grads_np), exists_ok=True)

            def reset_records():
                x = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape), state.params['record'])
                y = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape), state.valid_params['record'])
                return x, y

            state_dict = mngr.restore(mngr.best_step(), items={'state': None})['state']
            try:
                state_dict['params']['record'], state_dict['valid_params']['record'] = reset_records()
            except KeyError:
                pass
            state = state.reset_params(params=FrozenDict(state_dict['params']),
                                       valid_params=FrozenDict(state_dict['valid_params']))
            opt_state = state.tx.init(state.params)
            state = state.reset_opt_state(opt_state=opt_state)

        valid_start, valid_end, n_valid = (0., 0., 1.)
        evaluate = (i % eval_every_t == 0)
        if evaluate:
            valid_start = time.time()
            valid_metrics, n_valid = valid_epoch(state, valid_ds, metric_fn, bs=valid_bs)
            valid_end = time.time()

            valid_batch_metrics_np = jax.device_get(valid_metrics)

            if use_wandb:
                wandb.log({f'Validation {k}': v for (k, v) in valid_batch_metrics_np.items()}, step=i)
            else:
                print("Validation metrics: ")
                print(valid_batch_metrics_np)

            # keep track if metrics improved
            if valid_metrics['loss'] < best_valid_metrics['loss']:
                best_valid_metrics['loss'] = valid_metrics['loss']
                state = state.improved(True)
            else:
                state = state.improved(False)

            mngr.save(i - 1, {'state': state}, metrics={'loss': valid_metrics['loss'].item()})

            # check if one of the metrics meets a defined stopping criteria
            if stop_metric_fn is not None:
                if stop_metric_fn(valid_batch_metrics_np):
                    logging.info(f'Stopping criteria for metrics has been met for step {i}.')
                    exit()

        epoch_end = time.time()

        e_time = epoch_end - epoch_start
        t_time = train_end - train_start
        v_time = (valid_end - valid_start) / (n_valid // valid_bs) if evaluate else None
        # divide by number of validation points since we time the evaluation over whole validation set

        tot_time += e_time
        times = {'Step time (s)': e_time,
                 'Training step time (s)': t_time,
                 'Validation step time (s)': v_time,
                 'Total time (s)': tot_time,
                 'Epoch': int(i // steps_per_epoch)}

        if i % log_every_t == 0:
            # log the training metrics
            if use_wandb:
                wandb.log({'Training {}'.format(k): v for (k, v) in train_batch_metrics_np.items()}, step=i)
            else:
                print('Training metrics: ')
                print(train_batch_metrics_np)

            plateau_decay, _, _ = state.reduce_lr_on_plateau_fn(plateau_count=state.plateau_count,
                                                                plateau_length=jnp.ones(1))
            schedule_decay = state.lr_decay_fn(state.step - 1)
            decay = plateau_decay * schedule_decay

            # TODO: this will fail with opt_states that do not follow the default structure of mlff.
            lr = jax.device_get(state.opt_state[3].hyperparams['step_size'] * decay)

            if use_wandb:
                wandb.log({'Learning rate': abs(lr)}, step=i)

            if stop_lr_fn is not None:
                if stop_lr_fn(abs(lr)):
                    logging.info(f'Stopping criteria for learning rate has been met for step {i}.')
                    exit()

            if (i > 1) and (i % log_every_t == 0):
                if use_wandb:
                    wandb.log(times, step=i)
    # mngr.wait_until_finished()
