import jax.numpy as jnp
from typing import (Callable, Dict, Tuple)

from so3lr.mlff.masking.mask import safe_mask
from so3lr.mlff.properties import properties as pn

DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
BatchImportanceWeights = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]


def mse_loss(*, y, y_true): return jnp.mean((y_true - y)**2)


def scaled_safe_masked_mse_loss(y, y_true, scale, msk):
    """

    Args:
        y (): shape: (B,d1, *, dN)
        y_true (): (B,d1, *, dN)
        scale (): (d1, *, dN) or everything broadcast-able to (B, d1, *, dN)
        msk (): shape: (B,*)

    Returns:

    """
    full_mask = ~jnp.isnan(y_true) & msk
    v = safe_mask(full_mask, fn=lambda u: scale * (u - y)**2, operand=y_true)
    den = full_mask.reshape(-1).sum().astype(dtype=v.dtype)
    return safe_mask(den > 0, lambda x: v.reshape(-1).sum() / x, den, 0)


masks = {pn.energy: lambda u: jnp.ones(len(u)).astype(bool)[:, None],
         pn.atomic_energy: lambda u: u[..., None],
         pn.force: lambda u: u[..., None],
         pn.stress: lambda u: jnp.ones(len(u)).astype(bool)[:, None, None],
         pn.partial_charge: lambda u: u[..., None],
         pn.hirshfeld_volume: lambda u: u[..., None],
         pn.hirshfeld_volume_ratio: lambda u: u[..., None]
         }


def get_loss_fn(obs_fn: Callable, weights: Dict, prop_keys: Dict, scales: Dict = None):
    _weights = {prop_keys[k]: v for (k, v) in weights.items()}
    if scales is None:
        _scales = {prop_keys[k]: jnp.ones(1) for (k, _) in weights.items()}
    else:
        _scales = scales

    _masks = {prop_keys[k]: masks[k] for k in weights.keys()}

    # _with_stress = pn.stress in list(weights.keys())

    def loss_fn(params, batch: DataTupleT):
        inputs, targets = batch
        outputs = obs_fn(params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():  # name is the value in prop_keys
            _l = scaled_safe_masked_mse_loss(y=outputs[name],
                                             y_true=targets[name],
                                             scale=_scales[name],
                                             msk=_masks[name](inputs[prop_keys[pn.node_mask]])
                                             )

            loss += _weights[name] * _l
            train_metrics.update({name: _l / _scales[name].mean()})

        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics

    return loss_fn


def get_active_learning_loss_fn(obs_fn, weights):
    def loss_fn(params, batch: BatchImportanceWeights):
        inputs, targets, imp_weights = batch  # shape: (B, ...), (B, ...), (B)
        imp_weights_normalized = jnp.sqrt(imp_weights / imp_weights.mean())  # shape: (B)

        outputs = obs_fn(params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():
            _y = jnp.einsum('b, b... -> b...', imp_weights_normalized, outputs[name])
            _y_true = jnp.einsum('b, b... -> b...', imp_weights_normalized, targets[name])
            _l = mse_loss(y=_y, y_true=_y_true)
            loss += weights[name] * _l
            train_metrics.update({name: _l})
        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics
    return loss_fn
