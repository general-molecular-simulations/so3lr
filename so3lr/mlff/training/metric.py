import jax.numpy as jnp

from typing import (Callable, Dict, Tuple)

from .loss import mse_loss

DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
BatchImportanceWeights = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]


def create_mae_metric(pad_value: float = None) -> Callable:
    """
    Create metric function for mean absolute error. Padding values can be excluded from the metric calculation.

    Args:
        prediction (Array): The predictions, shape: (...)
        target (Array): The targets, shape: (...)
        pad_value (float): Value of padded values.

    Returns: scalar value, shape: (1)

    """

    def mae(prediction, target):
        p = prediction.reshape(-1)
        t = target.reshape(-1)

        if pad_value is not None:
            not_pad_idx = t != pad_value
        else:
            not_pad_idx = jnp.arange(len(p))

        return jnp.abs(p[not_pad_idx] - t[not_pad_idx]).mean()

    return mae


def create_mse_metric(pad_value: float = None) -> Callable:
    """
    Create metric function for mean squared error. Padding values can be excluded from the metric calculation.

    Args:
        pad_value (float): Value of padded values.

    Returns: scalar value, shape: (1)

    """
    def mse(prediction: jnp.ndarray, target: jnp.ndarray):
        p = prediction.reshape(-1)
        t = target.reshape(-1)

        if pad_value is not None:
            not_pad_idx = t != pad_value
        else:
            not_pad_idx = jnp.arange(len(p))

        return ((p[not_pad_idx] - t[not_pad_idx])**2).mean()

    return mse


def create_rmse_metric(pad_value: float = None) -> Callable:
    """
    Metric function for root mean square error. Padding values can be excluded from the metric calculation.

    Args:
        pad_value (float): Value of padded values.

    Returns: scalar value, shape: (1)

    """
    mse_fn = create_mse_metric(pad_value)

    def rmse(prediction: jnp.ndarray, target: jnp.ndarray):
        return jnp.sqrt(mse_fn(prediction=prediction, target=target))

    return rmse


def get_metric_fn(obs_fn, loss_weights, scales: Dict = None, metric_fns: Dict[str, Callable] = None):

    all_scales = {k: jnp.float32(1.) for (k, v) in loss_weights.items()}
    all_scales.update(scales)

    def scale(k, v):
        return v * all_scales[k]

    def metric_fn(params, batch: DataTupleT):
        inputs, targets = batch
        outputs = obs_fn(params, inputs)
        outputs_original_scales = {k: scale(k, v) for (k, v) in outputs.items()}
        targets_original_scales = {k: scale(k, v)for (k, v) in targets.items()}
        loss = jnp.zeros(1)
        metrics = {}
        for name, target in targets.items():
            _l = mse_loss(y_true=outputs[name], y=targets[name])
            loss += loss_weights[name] * _l
            metrics[f"loss_{name}"] = _l
            if metric_fns is not None:
                for m_name, m_fn in metric_fns.items():
                    metric = m_fn(y_true=outputs_original_scales[name], y=targets_original_scales[name])
                    metrics[f"{m_name.upper()}_{name}"] = metric

        metrics.update({'loss': loss})

        return None, metrics
    return metric_fn
