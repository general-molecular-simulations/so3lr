import jraph
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb

from flax import traverse_util
from flax.core.frozen_dict import unfreeze
from pathlib import Path
from typing import Any, Callable, Dict

from ..masking import mask as gradient_utils
from ..utils import checkpoint_utils
from ..utils.param_utils import print_param_shapes, count_params
from .jraph_utils import dynamically_batch_with_lr

from .robust_loss_jax import distribution as robust_loss_dist
from .robust_loss_jax import general as robust_loss_general

# Global distribution object for adaptive robust loss (handles partition function)
ROBUST_LOSS_DIST = robust_loss_dist.Distribution()


def init_adaptive_robust_loss_params(
    targets: list,
    init_alpha: float = 1.5,
    init_scale: float = 1.0
) -> Dict[str, jnp.ndarray]:
    """Initialize learnable parameters for adaptive robust loss.

    Args:
        targets: List of target names (e.g., ['energy', 'forces'])
        init_alpha: Initial alpha value in [0, 2]. Default 1.5.
        init_scale: Initial scale value (positive). Default 1.0.

    Returns:
        Dictionary with 'alpha_raw' and 'scale_raw' arrays, one value per target.
        These are unconstrained parameters that get transformed during training:
        - alpha = 2 * sigmoid(alpha_raw) -> [0, 2]
        - scale = softplus(scale_raw) + eps -> (0, inf)
    """
    n_targets = len(targets)

    # Inverse transform to get raw values from desired initial values
    # alpha = 2 * sigmoid(alpha_raw) -> alpha_raw = logit(alpha / 2)
    alpha_init_clipped = jnp.clip(init_alpha, 0.01, 1.99)
    alpha_raw_init = jnp.log(alpha_init_clipped / (2.0 - alpha_init_clipped))

    # scale = softplus(scale_raw) + eps -> scale_raw ≈ inverse_softplus(scale - eps)
    # For simplicity, scale_raw = log(exp(scale) - 1) when scale > 0
    scale_raw_init = jnp.log(jnp.expm1(jnp.maximum(init_scale, 0.1)))

    return {
        'adaptive_robust_loss': {
            'alpha_raw': jnp.full((n_targets,), alpha_raw_init),
            'scale_raw': jnp.full((n_targets,), scale_raw_init),
            # 'target_names': targets,  # for reference (not a parameter)
        }
    }


def transform_adaptive_robust_params(alpha_raw: jnp.ndarray, scale_raw: jnp.ndarray):
    """Transform unconstrained parameters to valid alpha and scale values.

    Args:
        alpha_raw: Unconstrained alpha parameters
        scale_raw: Unconstrained scale parameters

    Returns:
        alpha: Constrained to [0, 2]
        scale: Constrained to (0, inf)
    """
    alpha = 2.0 * jax.nn.sigmoid(alpha_raw)  # [0, 2]
    scale = jax.nn.softplus(scale_raw) + 1e-6  # (0, inf)
    return alpha, scale


def set_seeds(seed=0):
    """Set random seeds for reproducibility."""
    try:
        random.seed(seed)
    except NameError:
        import random
        random.seed(seed)

    np.random.seed(seed)


def print_metrics(epoch, eval_metrics):
    formatted_output = f"{epoch}: "
    for key, value in eval_metrics.items():
        if isinstance(value, np.ndarray) and value.size == 1:
            formatted_output += f"{key}={value.item():.4f}, "
        else:
            formatted_output += f"{key}={', '.join(map('{:.4f}'.format, value))}, " if isinstance(value, np.ndarray) else f"{key}={value:.4f}, "
    return formatted_output.rstrip(", ")

def graph_mse_loss(
    y, y_label, batch_segments, graph_mask, scale,
    use_robust_loss: bool = False, robust_loss_alpha: float = 1.99,
    adaptive_alpha: jnp.ndarray = None, adaptive_scale: jnp.ndarray = None,
    atomic_numbers=None, atom_loss_weights=None
):
    """Compute MSE loss for graph-level properties (energy, stress, etc.).

    Args:
        y: Predicted values
        y_label: Target values
        batch_segments: Segment IDs for batching
        graph_mask: Mask for valid graphs
        scale: Loss scale factor
        use_robust_loss: Whether to use robust loss instead of L2
        robust_loss_alpha: Alpha parameter for fixed robust loss (ignored if adaptive)
        adaptive_alpha: Learned alpha parameter for adaptive robust loss
        adaptive_scale: Learned scale parameter for adaptive robust loss
        atomic_numbers: Atomic numbers for per-element weighting
        atom_loss_weights: Per-element loss weights
    """
    assert y.shape == y_label.shape

    # mask of valid entries + broadcast graph_mask to y's trailing dims
    full_mask = (~jnp.isnan(y_label)) & jnp.expand_dims(
        graph_mask, [y.ndim - 1 - o for o in range(0, y.ndim - 1)]
    ).astype(bool)

    # per-graph weights from atomic numbers (default 1.0)
    if (atomic_numbers is not None) and (atom_loss_weights is not None):
        w_table = jnp.asarray(atom_loss_weights, dtype=y.dtype)
        z = jnp.asarray(atomic_numbers).astype(jnp.int32)
        z = jnp.clip(z, 0, w_table.shape[0] - 1)
        node_w = w_table[z]
        valid_node = (z > 0).astype(y.dtype)  # ignore padded nodes Z==0

        # average node weight per graph
        w_sum = jraph.segment_sum(node_w * valid_node, batch_segments, num_segments=len(graph_mask))
        n_sum = jraph.segment_sum(valid_node,          batch_segments, num_segments=len(graph_mask))
        graph_w = w_sum / jnp.maximum(n_sum, 1.0)
    else:
        graph_w = jnp.ones_like(graph_mask, dtype=y.dtype)

    # broadcast weights to y's shape, then form a weighted mask
    expand_axes = [y.ndim - 1 - o for o in range(0, y.ndim - 1)]
    graph_w_b = jnp.expand_dims(graph_w, expand_axes)  # same broadcast as graph_mask above
    weight_mask = graph_w_b * full_mask.astype(y.dtype)

    # denominator = total weighted count of valid elements
    denom = jnp.sum(weight_mask)
    denom = jnp.maximum(denom, jnp.asarray(1e-12, dtype=y.dtype))

    if adaptive_alpha is not None and adaptive_scale is not None:
        # Adaptive robust loss: alpha and scale are learned parameters
        # Use nllfun which includes the partition function for proper gradients w.r.t. alpha
        diff = jnp.where(full_mask, y - y_label, 0)
        per_el = 2 * scale * ROBUST_LOSS_DIST.nllfun(diff, adaptive_alpha, adaptive_scale)
        loss = jnp.sum(per_el * weight_mask) / denom
    elif use_robust_loss:
        # Fixed robust loss: alpha is a hyperparameter
        # Use lossfun directly (no partition function needed since alpha is not learned)
        diff = jnp.where(full_mask, y - y_label, 0)
        per_el = 2 * scale * robust_loss_general.lossfun(diff, robust_loss_alpha, 1.0)
        loss = jnp.sum(per_el * weight_mask) / denom
    else:
        # Standard L2 loss: optax.l2_loss returns 0.5 * (a - b)^2
        a = jnp.where(full_mask, y, 0)
        b = jnp.where(full_mask, y_label, 0)
        per_el = 2 * scale * optax.l2_loss(a, b)
        loss = jnp.sum(per_el * graph_w_b) / denom  # per_el already zeroed by mask

    return loss



def node_mse_loss(
    y, y_label, batch_segments, graph_mask, scale,
    use_robust_loss: bool = False, robust_loss_alpha: float = 1.99,
    adaptive_alpha: jnp.ndarray = None, adaptive_scale: jnp.ndarray = None,
    atomic_numbers=None, atom_loss_weights=None
):
    """Compute MSE loss for node-level properties (forces, etc.).

    Args:
        y: Predicted values
        y_label: Target values
        batch_segments: Segment IDs for batching
        graph_mask: Mask for valid graphs
        scale: Loss scale factor
        use_robust_loss: Whether to use robust loss instead of L2
        robust_loss_alpha: Alpha parameter for fixed robust loss (ignored if adaptive)
        adaptive_alpha: Learned alpha parameter for adaptive robust loss
        adaptive_scale: Learned scale parameter for adaptive robust loss
    """
    assert y.shape == y_label.shape

    num_graphs = graph_mask.sum().astype(y.dtype)  # ()

    if adaptive_alpha is not None and adaptive_scale is not None:
        # Adaptive robust loss: alpha and scale are learned parameters
        diff = y - y_label
        masked_diff = gradient_utils.safe_mask(
            fn=lambda u: u,
            operand=diff,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )

        squared = gradient_utils.safe_mask(
            fn=lambda u: 2 * ROBUST_LOSS_DIST.nllfun(u, adaptive_alpha, adaptive_scale),
            operand=masked_diff,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )
    elif use_robust_loss:
        # Fixed robust loss: alpha is a hyperparameter
        diff = y - y_label
        masked_diff = gradient_utils.safe_mask(
            fn=lambda u: u,
            operand=diff,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )

        squared = gradient_utils.safe_mask(
            fn=lambda u: 2 * robust_loss_general.lossfun(u, robust_loss_alpha, 1.0),
            operand=masked_diff,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )
    else:
        # Regular L2 loss
        squared = gradient_utils.safe_mask(
            fn=lambda u: jnp.square(u),
            operand=y - y_label,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )

    # sum up the losses for node properties along the non-leading dimension. For e.g. scalar node quantities
    # this does not have any effect, but e.g. for vectorial and tensorial node properties one averages over all
    # additional non-leading dimension. E.g. for forces this corresponds to taking mean over x, y, z component.
    node_mean_squared = squared.reshape(len(squared), -1).mean(axis=-1)  # (num_nodes)

    per_graph_mse = jraph.segment_mean(
        data=node_mean_squared,
        segment_ids=batch_segments,
        num_segments=len(graph_mask)
    )  # (num_graphs)

    # Set contributions from padding graphs to zero.
    per_graph_mse = jnp.where(
        graph_mask,
        per_graph_mse,
        jnp.asarray(0., dtype=per_graph_mse.dtype)
    )  # (num_graphs)

    # Create msk that has True when data is present and is false if no data is present, i.e. y_label equals NaN.
    # Note that padding graphs still have zero valued entries.
    data_msk = ~jnp.isnan(
        jax.ops.segment_max(
            data=jnp.max(y_label.reshape(len(y_label), -1), axis=-1),
            segment_ids=batch_segments,
            num_segments=len(graph_mask)
        )  # evaluates to NaN if one entry in the segment is NaN.
    )  # (num_graphs)

    # Set contributions from graphs for which no node labels are present to zero.
    per_graph_mse = jnp.where(
        data_msk,
        per_graph_mse,
        jnp.asarray(0., dtype=per_graph_mse.dtype)
    )  # (num_graphs)

    # Calculate the number of graphs that have no data present.
    num_graphs_no_data = jnp.where(
        data_msk,
        jnp.asarray(0., dtype=per_graph_mse.dtype),
        jnp.asarray(1., dtype=per_graph_mse.dtype),
    ).sum()

    # subtract the number of graphs for which no data is present.
    num_graphs = num_graphs - num_graphs_no_data

    # Calculate mean and scale. Prevent the case of division by zero if no data is present at all.
    mse = scale * jnp.sum(per_graph_mse) / jnp.maximum(num_graphs, 1.)  # ()

    return mse

def graph_mae_loss(
    y, y_label, batch_segments, graph_mask, scale,
    atomic_numbers=None, atom_loss_weights=None  # <- new
):
    assert y.shape == y_label.shape

    full_mask = ~jnp.isnan(y_label) & jnp.expand_dims(
        graph_mask, [y.ndim - 1 - o for o in range(0, y.ndim - 1)]
    )
    # Build per-graph weights from per-atom weights (default = 1.0 if not provided)
    if (atomic_numbers is not None) and (atom_loss_weights is not None):
        # atom_loss_weights is a 1D lookup table where index = atomic number (0 for padding)
        w_table = jnp.asarray(atom_loss_weights, dtype=y.dtype)
        z = jnp.asarray(atomic_numbers).astype(jnp.int32)
        z = jnp.clip(z, 0, w_table.shape[0] - 1)
        node_w = w_table[z]
        valid_node = (z > 0).astype(y.dtype)  # ignore padded nodes where Z==0

        w_sum = jraph.segment_sum(node_w * valid_node, batch_segments, num_segments=len(graph_mask))
        n_sum = jraph.segment_sum(valid_node,          batch_segments, num_segments=len(graph_mask))
        graph_w = w_sum / jnp.maximum(n_sum, 1.0)  # mean weight per graph
    else:
        graph_w = jnp.ones_like(graph_mask, dtype=y.dtype)

    # Broadcast graph weights to the shape of y
    expand_axes = [y.ndim - 1 - o for o in range(0, y.ndim - 1)]
    graph_w_b = jnp.expand_dims(graph_w, expand_axes)

    # Weighted MAE (normalize by weighted mask)
    abs_err = jnp.abs(jnp.where(full_mask, y - y_label, 0))
    weighted_abs = graph_w_b * abs_err
    weighted_mask = graph_w_b * full_mask.astype(y.dtype)

    denominator = jnp.sum(weighted_mask).astype(y.dtype)
    loss = jnp.sum(weighted_abs) / jnp.maximum(denominator, jnp.asarray(1e-12, dtype=y.dtype))
    return loss


def node_mae_loss(y, y_label, batch_segments, graph_mask, scale, **kwargs):
    assert y.shape == y_label.shape

    num_graphs = graph_mask.sum().astype(y.dtype)  # ()

    # Use absolute error for MAE
    abs_error = gradient_utils.safe_mask(
        fn=lambda u: jnp.abs(u),
        operand=y - y_label,
        mask=~jnp.isnan(y_label),
        placeholder=0.
    )

    # sum up the losses for node properties along the non-leading dimension
    node_mean_abs = abs_error.reshape(len(abs_error), -1).mean(axis=-1)  # (num_nodes)

    per_graph_mae = jraph.segment_mean(
        data=node_mean_abs,
        segment_ids=batch_segments,
        num_segments=len(graph_mask)
    )  # (num_graphs)

    # Set contributions from padding graphs to zero.
    per_graph_mae = jnp.where(
        graph_mask,
        per_graph_mae,
        jnp.asarray(0., dtype=per_graph_mae.dtype)
    )  # (num_graphs)

    # Create mask that has True when data is present and is false if no data is present
    data_msk = ~jnp.isnan(
        jax.ops.segment_max(
            data=jnp.max(y_label.reshape(len(y_label), -1), axis=-1),
            segment_ids=batch_segments,
            num_segments=len(graph_mask)
        )  # evaluates to NaN if one entry in the segment is NaN.
    )  # (num_graphs)

    # Set contributions from graphs for which no node labels are present to zero.
    per_graph_mae = jnp.where(
        data_msk,
        per_graph_mae,
        jnp.asarray(0., dtype=per_graph_mae.dtype)
    )  # (num_graphs)

    # Calculate the number of graphs that have no data present.
    num_graphs_no_data = jnp.where(
        data_msk,
        jnp.asarray(0., dtype=per_graph_mae.dtype),
        jnp.asarray(1., dtype=per_graph_mae.dtype),
    ).sum()

    # subtract the number of graphs for which no data is present.
    num_graphs = num_graphs - num_graphs_no_data

    # Calculate mean. Prevent division by zero if no data is present.
    mae = jnp.sum(per_graph_mae) / jnp.maximum(num_graphs, 1.)

    return mae

def graph_mse_loss_per_atom(
    y, y_label, batch_segments, graph_mask, scale,
    use_robust_loss: bool = False, robust_loss_alpha: float = 1.99,
    adaptive_alpha: jnp.ndarray = None, adaptive_scale: jnp.ndarray = None,
    atomic_numbers=None, atom_loss_weights=None
):
    """Compute MSE loss for graph-level energy normalized per atom (E/N).

    Same as graph_mse_loss but divides predictions and labels by the number of
    atoms per graph before computing the loss, so that molecules of different
    sizes contribute equally.
    """
    assert y.shape == y_label.shape

    # Number of atoms per graph
    num_atoms_per_graph = jraph.segment_sum(
        jnp.ones(len(batch_segments), dtype=y.dtype),
        batch_segments, num_segments=len(graph_mask)
    )
    num_atoms_per_graph = jnp.maximum(num_atoms_per_graph, 1.0)

    # Normalize by number of atoms
    # Expand dims to broadcast with y's trailing dimensions (e.g., (N,) -> (N,1) for dipole_vec)
    expand_axes = tuple(range(1, y.ndim))
    num_atoms = jnp.expand_dims(num_atoms_per_graph, axis=expand_axes)
    y = y / num_atoms
    y_label = y_label / num_atoms

    return graph_mse_loss(
        y, y_label, batch_segments, graph_mask, scale,
        use_robust_loss=use_robust_loss, robust_loss_alpha=robust_loss_alpha,
        adaptive_alpha=adaptive_alpha, adaptive_scale=adaptive_scale,
        atomic_numbers=atomic_numbers, atom_loss_weights=atom_loss_weights
    )


def graph_mae_loss_per_atom(
    y, y_label, batch_segments, graph_mask, scale,
    atomic_numbers=None, atom_loss_weights=None
):
    """Compute MAE loss for graph-level energy normalized per atom (E/N).

    Same as graph_mae_loss but divides predictions and labels by the number of
    atoms per graph before computing the loss, so that molecules of different
    sizes contribute equally.
    """
    assert y.shape == y_label.shape

    # Number of atoms per graph
    num_atoms_per_graph = jraph.segment_sum(
        jnp.ones(len(batch_segments), dtype=y.dtype),
        batch_segments, num_segments=len(graph_mask)
    )
    num_atoms_per_graph = jnp.maximum(num_atoms_per_graph, 1.0)

    # Normalize by number of atoms
    # Expand dims to broadcast with y's trailing dimensions (e.g., (N,) -> (N,1) for dipole_vec)
    expand_axes = tuple(range(1, y.ndim))
    num_atoms = jnp.expand_dims(num_atoms_per_graph, axis=expand_axes)
    y = y / num_atoms
    y_label = y_label / num_atoms

    return graph_mae_loss(
        y, y_label, batch_segments, graph_mask, scale,
        atomic_numbers=atomic_numbers, atom_loss_weights=atom_loss_weights
    )


property_to_mae = {
    'energy': graph_mae_loss_per_atom,
    'stress': graph_mae_loss,
    'forces': node_mae_loss,
    'dipole_vec': graph_mae_loss_per_atom,
    'hirshfeld_ratios': node_mae_loss,
    'c6_ratios': node_mae_loss,
}

property_to_loss = {
    'energy': graph_mse_loss_per_atom,
    'stress': graph_mse_loss,
    'forces': node_mse_loss,
    'dipole_vec': graph_mse_loss_per_atom,
    'hirshfeld_ratios': node_mse_loss,
    'c6_ratios': node_mse_loss,
}

def make_loss_fn(
    obs_fn: Callable, weights: Dict, scales: Dict = None,
    use_robust_loss: bool = False, robust_loss_alpha: float = 1.99,
    use_adaptive_robust_loss: bool = False,
    atom_loss_weights: jnp.ndarray = None
):
    """Create a loss function for training.

    Args:
        obs_fn: Observable function that computes predictions from params
        weights: Dictionary of loss weights per target
        scales: Dictionary of scales per target (for normalization)
        use_robust_loss: Use fixed robust loss with specified alpha
        robust_loss_alpha: Alpha parameter for fixed robust loss
        use_adaptive_robust_loss: Use adaptive robust loss where alpha/scale are learned.
            When True, params must contain 'adaptive_robust_loss' key with
            'alpha_raw' and 'scale_raw' arrays (one value per target).
        atom_loss_weights: Per-element loss weights

    Returns:
        loss_fn: Function (params, batch) -> (loss, metrics)
    """
    targets = list(weights.keys())
    _scales = {k: jnp.ones(1) for k in targets} if scales is None else scales
    target_to_idx = {t: i for i, t in enumerate(targets)}

    @jax.jit
    def loss_fn(params, batch: Dict[str, jnp.ndarray]):
        inputs = {k: v for k, v in batch.items() if k not in targets}
        outputs_true = {k: v for k, v in batch.items() if k in targets}
        outputs_predict = obs_fn(params, **inputs)

        # Extract adaptive robust loss parameters if enabled
        adaptive_alphas = None
        adaptive_scales = None
        if use_adaptive_robust_loss and 'adaptive_robust_loss' in params:
            alpha_raw = params['adaptive_robust_loss']['alpha_raw']
            scale_raw = params['adaptive_robust_loss']['scale_raw']
            adaptive_alphas, adaptive_scales = transform_adaptive_robust_params(alpha_raw, scale_raw)

        loss = jnp.zeros(1)
        loss_mae = jnp.zeros(1)
        metrics = {}

        for target in targets:
            target_mae_fn = property_to_mae[target]
            _mae = target_mae_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
                atomic_numbers=inputs.get('atomic_numbers', None),
                atom_loss_weights=atom_loss_weights
            )
            metrics.update({f'{target}_mae': _mae / _scales[target].mean()})
            loss_mae += weights[target] * _mae

            # Get adaptive parameters for this target if enabled
            target_adaptive_alpha = None
            target_adaptive_scale = None
            if adaptive_alphas is not None:
                idx = target_to_idx[target]
                target_adaptive_alpha = adaptive_alphas[idx]
                target_adaptive_scale = adaptive_scales[idx]

            target_loss_fn = property_to_loss[target]
            _l = target_loss_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
                use_robust_loss=use_robust_loss,
                robust_loss_alpha=robust_loss_alpha,
                adaptive_alpha=target_adaptive_alpha,
                adaptive_scale=target_adaptive_scale,
                atomic_numbers=inputs.get('atomic_numbers', None),
                atom_loss_weights=atom_loss_weights
            )

            loss += weights[target] * _l
            metrics.update({f'{target}_mse': _l / _scales[target].mean()})

            # Log adaptive parameters if enabled
            if adaptive_alphas is not None:
                idx = target_to_idx[target]
                metrics.update({
                    f'{target}_robust_alpha': adaptive_alphas[idx],
                    f'{target}_robust_scale': adaptive_scales[idx]
                })

        loss = jnp.reshape(loss, ())
        loss_mae = jnp.reshape(loss_mae, ())
        metrics.update({'loss': loss, 'loss_mae': loss_mae})
        return loss, metrics

    return loss_fn

def make_val_fn(
    obs_fn: Callable, weights: Dict, scales: Dict = None,
    use_robust_loss: bool = False, robust_loss_alpha: float = 1.99,
    use_adaptive_robust_loss: bool = False,
    atom_loss_weights: jnp.ndarray = None
):
    """Create a validation function for evaluation.

    Args:
        obs_fn: Observable function that computes predictions from params
        weights: Dictionary of loss weights per target
        scales: Dictionary of scales per target (for normalization)
        use_robust_loss: Use fixed robust loss with specified alpha
        robust_loss_alpha: Alpha parameter for fixed robust loss
        use_adaptive_robust_loss: Use adaptive robust loss where alpha/scale are learned
        atom_loss_weights: Per-element loss weights

    Returns:
        val_fn: Function (params, batch) -> (loss, metrics)
    """
    targets = list(weights.keys())
    _scales = {k: jnp.ones(1) for k in targets} if scales is None else scales
    target_to_idx = {t: i for i, t in enumerate(targets)}

    @jax.jit
    def val_fn(params, batch: Dict[str, jnp.ndarray]):
        inputs = {k: v for k, v in batch.items() if k not in targets}
        outputs_true = {k: v for k, v in batch.items() if k in targets}
        outputs_predict = obs_fn(params, **inputs)
        metrics = {}

        # Extract adaptive robust loss parameters if enabled
        adaptive_alphas = None
        adaptive_scales = None
        if use_adaptive_robust_loss and 'adaptive_robust_loss' in params:
            alpha_raw = params['adaptive_robust_loss']['alpha_raw']
            scale_raw = params['adaptive_robust_loss']['scale_raw']
            adaptive_alphas, adaptive_scales = transform_adaptive_robust_params(alpha_raw, scale_raw)

        for target in targets:
            target_mae_fn = property_to_mae[target]
            _mae = target_mae_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
                atomic_numbers=inputs.get('atomic_numbers', None),
                atom_loss_weights=atom_loss_weights
            )
            metrics.update({f'{target}_mae': _mae / _scales[target].mean()})

            # Get adaptive parameters for this target if enabled
            target_adaptive_alpha = None
            target_adaptive_scale = None
            if adaptive_alphas is not None:
                idx = target_to_idx[target]
                target_adaptive_alpha = adaptive_alphas[idx]
                target_adaptive_scale = adaptive_scales[idx]

            target_loss_fn = property_to_loss[target]
            _mse = target_loss_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
                use_robust_loss=use_robust_loss,
                robust_loss_alpha=robust_loss_alpha,
                adaptive_alpha=target_adaptive_alpha,
                adaptive_scale=target_adaptive_scale,
                atomic_numbers=inputs.get('atomic_numbers', None),
                atom_loss_weights=atom_loss_weights
            )

            metrics.update({f'{target}_mse': _mse / _scales[target].mean()})

            # Log adaptive parameters if enabled
            if adaptive_alphas is not None:
                idx = target_to_idx[target]
                metrics.update({
                    f'{target}_robust_alpha': adaptive_alphas[idx],
                    f'{target}_robust_scale': adaptive_scales[idx]
                })

        loss = jnp.zeros(1)
        loss_mae = jnp.zeros(1)
        for target in targets:
            loss += weights[target] * metrics[f'{target}_mse'] * _scales[target].mean()
            loss_mae += weights[target] * metrics[f'{target}_mae'] * _scales[target].mean()

        metrics.update({'loss': loss, 'loss_mae': loss_mae})
        return loss, metrics

    return val_fn


def make_training_step_fn(
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        log_gradient_values: bool
):
    """
    Make a training step fn, which takes params, optimizer state, and a batch of data and returns
    new params based on the gradients according to the loss_fn, new optimizer state and metrics.

    Args:
        optimizer (optax.GradientTransformation): Optax optimizer.
        loss_fn (Callable): Loss function.
        log_gradient_values (bool): Log gradient values for each leaf in the params pytree.

    Returns:
        Training step fn.

    """

    @jax.jit
    def training_step_fn(
            params,
            opt_state,
            batch
    ):
        """
        Training step.

        Args:
            params (FrozenDict): Parameter dictionary.
            opt_state: Optax optimizer state.
            batch (Tuple): Batch of validation data.

        Returns:
            Updated state and metrics.

        """
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True
        )(
            params,
            batch
        )

        if log_gradient_values:
            metrics['grad_norm'] = unfreeze(jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1), axis=0), grads))

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params
        )

        params = optax.apply_updates(
            params=params,
            updates=updates
        )

        metrics['grad_norm'] = optax.global_norm(grads)

        return params, opt_state, metrics

    return training_step_fn


def make_validation_step_fn(
        metric_fn: Callable
):
    """
    Make validation step function, which takes params and batch of data as input and returns metrics.

    Args:
        metric_fn (Callable): Function that calculates metrics, given params and batch of data.

    Returns:
        Validation step function.

    """
    @jax.jit
    def validation_step_fn(params, batch) -> Dict[str, jnp.ndarray]:
        """
        Validation step.

        Args:
            params (FrozenDict): Parameters.
            batch (Tuple): Batch of validation data.

        Returns:
            Validation metrics.
        """
        _, metrics = metric_fn(
            params,
            batch
        )

        return metrics

    return validation_step_fn

def fit(
        model,
        optimizer,
        loss_fn,
        graph_to_batch_fn,
        training_data,
        validation_data,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs,
        batch_max_num_pairs,
        params=None,
        val_fn=None,
        num_epochs: int = 100,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart: bool = False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        log_gradient_values: bool = False,
        use_robust_loss_validation: bool = False,
        robust_loss_alpha_validation: float = 1.99,
        # Adaptive robust loss parameters
        use_adaptive_robust_loss: bool = False,
        adaptive_robust_loss_targets: list = None,
        adaptive_robust_loss_init_alpha: float = 1.5,
        adaptive_robust_loss_init_scale: float = 1.0
):
    """
    Fit model.

    Args:
        model: flax module.
        optimizer: optax optimizer.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        graph_to_batch_fn (Callable): Function that takes a batched graph and returns a batch for the loss_fn.
        training_data (Sequence): Sequence of jraph.GraphTuples.
        validation_data (Sequence): Sequence of jraph.GraphTuples.
        batch_max_num_nodes (int): Maximal number of nodes per batch.
        batch_max_num_edges (int): Maximal number of edges per batch.
        batch_max_num_graphs (int): Maximal number of graphs per batch.
        batch_max_num_pairs (int): Maximal number of pairs in long-range indices.
        params: Parameters to start from during training. If not given, either new parameters are initialized randomly
            or loaded from ckpt_dir if the checkpoint already exists and `allow_restart=True`.
        val_fn (Callable, optional): Validation function to use for metrics during validation.
            If None, loss_fn will be used. Defaults to None.
        num_epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_num_steps (int): Evaluate the metrics every num-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias. If true, wandb.init() must be called before call to fit().
        log_gradient_values (bool): Gradient values for each set of weights is logged.
        use_robust_loss_validation (bool): Whether to use robust loss during validation. Defaults to False.
        robust_loss_alpha_validation (float): Alpha parameter for robust loss during validation. Defaults to 1.99.
        use_adaptive_robust_loss (bool): Whether to use adaptive robust loss. Defaults to False.
        adaptive_robust_loss_targets (list): List of target names for adaptive robust loss.
        adaptive_robust_loss_init_alpha (float): Initial alpha for adaptive robust loss. Defaults to 1.5.
        adaptive_robust_loss_init_scale (float): Initial scale for adaptive robust loss. Defaults to 1.0.
    Returns:

    """
    numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(exist_ok=True)

    # Create orbax CheckpointManager.
    if ckpt_manager_options is None:
        ckpt_manager_options = {'max_to_keep': 3}

    options = ocp.CheckpointManagerOptions(
        best_fn=lambda u: u['loss'],
        best_mode='min',
        step_prefix='ckpt',
        **ckpt_manager_options
    )

    ckpt_mngr = checkpoint_utils.make_checkpoint_manager(
        ckpt_dir=ckpt_dir,
        ckpt_mngr_options=options
    )

    training_step_fn = make_training_step_fn(
        optimizer,
        loss_fn,
        log_gradient_values
    )

    # If no validation function is provided, use loss_fn as validation function
    if val_fn is None:
        validation_step_fn = make_validation_step_fn(
            loss_fn
        )
    else:
        validation_step_fn = make_validation_step_fn(
            val_fn
        )


    processed_graphs = 0
    processed_nodes = 0
    step = 0

    opt_state = None
    for epoch in range(num_epochs):
        # Shuffle the training data.
        numpy_rng.shuffle(training_data)
        # Create batched graphs from list of (graph, long_range) tuples.
        iterator_training = dynamically_batch_with_lr(
            iter(training_data),
            n_node=batch_max_num_nodes,
            n_edge=batch_max_num_edges,
            n_graph=batch_max_num_graphs,
            n_pairs=batch_max_num_pairs,
        )

        # Start iteration over batched graphs.
        for graph_batch_training, lr_batch_training in iterator_training:
            batch_training = graph_to_batch_fn(graph_batch_training, lr_batch_training)
            processed_graphs += batch_training['num_of_non_padded_graphs']
            processed_nodes += batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training)
            # Training data is numpy arrays so we now transform them to jax.numpy arrays.
            batch_training = jax.tree_util.tree_map(jnp.array, batch_training)

            # If params are None (in the first step), initialize the parameters or load from existing checkpoint.
            if params is None:
                # Check if checkpoint already exists.
                latest_step = ckpt_mngr.latest_step()
                if latest_step is not None:
                    if allow_restart:
                        # params = ckpt_mngr.restore(
                        #     latest_step,
                        #     args=ocp.args.Composite(params=ocp.args.StandardRestore())
                        # )['params']
                        params = checkpoint_utils.load_params_from_checkpoint(
                            ckpt_dir=ckpt_dir
                        )
                        print(f"Loaded parameters from {ckpt_dir}")
                        print(f"Params keys: {params.keys()}")
                        print("\nParameter shapes:")
                        print("=" * 50)
                        print_param_shapes(params)
                        print("=" * 50)
                        print('This is fit_from_iterator function')
                        # Modify parameters to handle theory levels
                        if 'params' in params and 'observables_0' in params['params']:
                            num_theory_levels = 16
                            # Modify energy_offset
                            if 'energy_offset' in params['params']['observables_0']:
                                print("\nOriginal energy_offset:")
                                print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                print("Values:", params['params']['observables_0']['energy_offset'])
                                old_energy_offset = params['params']['observables_0']['energy_offset']

                                # Only tile if shape is 1D
                                if len(old_energy_offset.shape) == 1:
                                    new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
                                    params['params']['observables_0']['energy_offset'] = new_energy_offset
                                    print("Applied tiling to energy_offset")
                                else:
                                    print("Energy offset already has multiple dimensions, no tiling applied")

                                print("\nNew energy_offset:")
                                print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                print("Values:", params['params']['observables_0']['energy_offset'])

                            # Modify atomic_scales
                            if 'atomic_scales' in params['params']['observables_0']:
                                print("\nOriginal atomic_scales:")
                                print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                print("Values:", params['params']['observables_0']['atomic_scales'])
                                old_atomic_scales = params['params']['observables_0']['atomic_scales']

                                # Only tile if shape is 1D
                                if len(old_atomic_scales.shape) == 1:
                                    new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
                                    params['params']['observables_0']['atomic_scales'] = new_atomic_scales
                                    print("Applied tiling to atomic_scales")
                                else:
                                    print("Atomic scales already has multiple dimensions, no tiling applied")

                                print("\nNew atomic_scales:")
                                print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                print("Values:", params['params']['observables_0']['atomic_scales'])

                            # Modify energy_dense_final
                            if 'energy_dense_final' in params['params']['observables_0']:
                                print("\nOriginal energy_dense_final kernel:")
                                print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])
                                old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']

                                # Check the shape to determine if tiling is needed
                                if old_kernel.shape[1] == 1:
                                    new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
                                    params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel
                                    print("Applied tiling to energy_dense_final kernel")
                                else:
                                    print("Energy dense final kernel already has correct output dimension, no tiling applied")

                                print("\nNew energy_dense_final kernel:")
                                print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])

                            print("\nParameter shapes after modification:")
                            print("=" * 50)
                            print_param_shapes(params)
                            print("=" * 50)
                        step += latest_step

                        print(f'Re-start training from {latest_step}.')
                    else:
                        raise RuntimeError(f'{ckpt_dir} already exists at step {latest_step}. If you want to re-start '
                                           f'training, set `allow_restart=True`.')
                else:
                    params = model.init(jax_rng, batch_training)

                    # Initialize adaptive robust loss parameters if enabled
                    if use_adaptive_robust_loss and adaptive_robust_loss_targets is not None:
                        adaptive_params = init_adaptive_robust_loss_params(
                            targets=adaptive_robust_loss_targets,
                            init_alpha=adaptive_robust_loss_init_alpha,
                            init_scale=adaptive_robust_loss_init_scale
                        )
                        params = {**params, **adaptive_params}
                        print(f"Initialized adaptive robust loss params for targets: {adaptive_robust_loss_targets}")
                        alpha, scale = transform_adaptive_robust_params(
                            adaptive_params['adaptive_robust_loss']['alpha_raw'],
                            adaptive_params['adaptive_robust_loss']['scale_raw']
                        )
                        print(f"  Initial alpha: {alpha}")
                        print(f"  Initial scale: {scale}")

            # If optimizer state is None (in the first step), initialize from the parameter pyTree.
            if opt_state is None:
                opt_state = optimizer.init(params)

            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                wandb.log(
                    {f'train_{k}': v for (k, v) in train_metrics_np.items()},
                    step=step
                )

            # Start validation process.
            if step % eval_every_num_steps == 0:
                iterator_validation = dynamically_batch_with_lr(
                    iter(validation_data),
                    n_node=batch_max_num_nodes,
                    n_edge=batch_max_num_edges,
                    n_graph=batch_max_num_graphs,
                    n_pairs=batch_max_num_pairs,
                )

                # Start iteration over validation batches.
                eval_totals = {}
                eval_counts = {}
                for graph_batch_validation, lr_batch_validation in iterator_validation:
                    batch_validation = graph_to_batch_fn(graph_batch_validation, lr_batch_validation)
                    batch_validation = jax.tree_util.tree_map(jnp.array, batch_validation)

                    eval_out = validation_step_fn(
                        params,
                        batch_validation
                    )

                    for k, v in eval_out.items():
                        eval_totals[k] = eval_totals.get(k, 0.0) + np.asarray(v).item()
                        eval_counts[k] = eval_counts.get(k, 0) + 1

                eval_metrics = {k: eval_totals[k] / eval_counts[k] for k in eval_totals}

                # Convert to dict to log with weights and bias.
                eval_metrics = {
                    f'eval_{k}': float(v) for k, v in eval_metrics.items()
                }

                # Print eval_metrics
                print(print_metrics(f"val_{epoch}_{step}:", eval_metrics))

                # Save checkpoint.
                ckpt_mngr.save(
                    step,
                    args=ocp.args.Composite(params=ocp.args.StandardSave(params)),
                    metrics={
                        'loss': eval_metrics['eval_loss']
                    }
                )

                # Log to weights and bias.
                if use_wandb:
                    wandb.log(
                        eval_metrics,
                        step=step
                    )
            # Finished validation process.

    # Wait until checkpoint manager completes all save operations.
    ckpt_mngr.wait_until_finished()

def fit_from_iterator(
        model,
        optimizer,
        loss_fn,
        graph_to_batch_fn,
        training_iterator,
        validation_iterator,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs,
        batch_max_num_pairs,
        num_epochs,
        params=None,
        val_fn=None,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart: bool = False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        log_gradient_values: bool = False,
        use_robust_loss_validation: bool = False,
        robust_loss_alpha_validation: float = 1.99,
        # Adaptive robust loss parameters
        use_adaptive_robust_loss: bool = False,
        adaptive_robust_loss_targets: list = None,
        adaptive_robust_loss_init_alpha: float = 1.5,
        adaptive_robust_loss_init_scale: float = 1.0
):
    """
    Fit model.

    Args:
        model: flax module.
        optimizer: optax optimizer.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        graph_to_batch_fn (Callable): Function that takes a batched graph and returns a batch for the loss_fn.
        training_iterator (): Iterator yielding jraph.GraphTuples.
        validation_iterator (): Iterator yielding jraph.GraphTuples.
        batch_max_num_nodes (int): Maximal number of nodes per batch.
        batch_max_num_edges (int): Maximal number of edges per batch.
        batch_max_num_graphs (int): Maximal number of graphs per batch.
        batch_max_num_pairs (int): Maximal number of pairs in long-range indices.
        num_epochs (int): Number of epochs to train for.
        params: Parameters to start from during training. If not given, either new parameters are initialized randomly
            or loaded from ckpt_dir if the checkpoint already exists and `allow_restart=True`.
        val_fn (Callable, optional): Validation function to use for metrics during validation.
            If None, loss_fn will be used. Defaults to None.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_num_steps (int): Evaluate the metrics every num-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias. If true, wandb.init() must be called before call to fit().
        log_gradient_values (bool): Gradient values for each set of weights is logged.
        use_robust_loss_validation (bool): Whether to use robust loss during validation. Defaults to False.
        robust_loss_alpha_validation (float): Alpha parameter for robust loss during validation. Defaults to 1.99.
        use_adaptive_robust_loss (bool): Whether to use adaptive robust loss. Defaults to False.
        adaptive_robust_loss_targets (list): List of target names for adaptive robust loss.
        adaptive_robust_loss_init_alpha (float): Initial alpha for adaptive robust loss. Defaults to 1.5.
        adaptive_robust_loss_init_scale (float): Initial scale for adaptive robust loss. Defaults to 1.0.
    Returns:

    """
    del training_seed
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(exist_ok=True)

    # Create orbax CheckpointManager.
    if ckpt_manager_options is None:
        ckpt_manager_options = {'max_to_keep': 3}

    options = ocp.CheckpointManagerOptions(
        best_fn=lambda u: u['loss'],
        best_mode='min',
        step_prefix='ckpt',
        **ckpt_manager_options
    )

    ckpt_mngr = checkpoint_utils.make_checkpoint_manager(
        ckpt_dir=ckpt_dir,
        ckpt_mngr_options=options
    )

    training_step_fn = make_training_step_fn(
        optimizer,
        loss_fn,
        log_gradient_values
    )

    validation_step_fn = make_validation_step_fn(
        val_fn
    )

    processed_graphs = 0
    processed_nodes = 0
    step = 0

    opt_state = None

    for epoch in range(num_epochs):
        if use_wandb:
            wandb.log({"epoch": epoch})
        print(f'Epoch {epoch} of {num_epochs}')
        training_iterator_loop = training_iterator.next_epoch(split='train', mode='train')
        for graph_batch_training, lr_batch_training in training_iterator_loop:
            batch_training = graph_to_batch_fn(graph_batch_training, lr_batch_training)
            processed_graphs += batch_training['num_of_non_padded_graphs']
            processed_nodes += int(batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training))
            # Training data is numpy arrays so we now transform them to jax.numpy arrays.
            batch_training = jax.tree_util.tree_map(jnp.array, batch_training)

            # If params are None (in the first step), initialize the parameters or load from existing checkpoint.
            if params is None:
                # Check if checkpoint already exists.
                latest_step = ckpt_mngr.latest_step()
                if latest_step is not None:
                    if allow_restart:
                        params = checkpoint_utils.load_params_from_checkpoint(
                            ckpt_dir=ckpt_dir
                        )
                        print(f"Loaded parameters from {ckpt_dir}")
                        print(f"Params keys: {params.keys()}")
                        print("\nParameter shapes:")
                        print("=" * 50)
                        print_param_shapes(params)
                        print("=" * 50)
                        print('This is fit_from_iterator function')
                        # Modify parameters to handle theory levels
                        if 'params' in params and 'observables_0' in params['params']:
                            num_theory_levels = 16
                            old_energy_offset = params['params']['observables_0']['energy_offset']
                            print('CHECK THE SHAPE ', old_energy_offset.shape, len(old_energy_offset.shape))
                            if len(old_energy_offset.shape) == 1:
                                print('len(old_energy_offset.shape) == 1, so tiling the shape')
                                # Modify energy_offset
                                if 'energy_offset' in params['params']['observables_0']:
                                    print("\nOriginal energy_offset:")
                                    print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                    print("Values:", params['params']['observables_0']['energy_offset'])

                                    # Only tile if shape is 1D
                                    if len(old_energy_offset.shape) == 1:
                                        new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
                                        params['params']['observables_0']['energy_offset'] = new_energy_offset
                                        print("Applied tiling to energy_offset")
                                    else:
                                        print("Energy offset already has multiple dimensions, no tiling applied")

                                    print("\nNew energy_offset:")
                                    print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                    print("Values:", params['params']['observables_0']['energy_offset'])

                                # Modify atomic_scales
                                if 'atomic_scales' in params['params']['observables_0']:
                                    print("\nOriginal atomic_scales:")
                                    print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                    print("Values:", params['params']['observables_0']['atomic_scales'])
                                    old_atomic_scales = params['params']['observables_0']['atomic_scales']

                                    # Only tile if shape is 1D
                                    if len(old_atomic_scales.shape) == 1:
                                        new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
                                        params['params']['observables_0']['atomic_scales'] = new_atomic_scales
                                        print("Applied tiling to atomic_scales")
                                    else:
                                        print("Atomic scales already has multiple dimensions, no tiling applied")

                                    print("\nNew atomic_scales:")
                                    print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                    print("Values:", params['params']['observables_0']['atomic_scales'])

                                # Modify energy_dense_final
                                if 'energy_dense_final' in params['params']['observables_0']:
                                    print("\nOriginal energy_dense_final kernel:")
                                    print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                    print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])
                                    old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']

                                    # Check the shape to determine if tiling is needed
                                    if old_kernel.shape[1] == 1:
                                        new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
                                        params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel
                                        print("Applied tiling to energy_dense_final kernel")
                                    else:
                                        print("Energy dense final kernel already has correct output dimension, no tiling applied")

                                    print("\nNew energy_dense_final kernel:")
                                    print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                    print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])
                            else:
                                print('NO TILING IS NEEDED, len(old_energy_offset.shape) != 1 ', old_energy_offset.shape, len(old_energy_offset.shape))

                            print("\nParameter shapes after modification:")
                            print("=" * 50)
                            print_param_shapes(params)
                            print("=" * 50)

                        step += latest_step
                        print(f'Re-start training from {latest_step}.')
                    else:
                        raise RuntimeError(f'{ckpt_dir} already exists at step {latest_step}. If you want to re-start '
                                           f'training, set `allow_restart=True`.')
                else:
                    print(f'Initialize new parameters.')
                    params = model.init(jax_rng, batch_training)

                    # Initialize adaptive robust loss parameters if enabled
                    if use_adaptive_robust_loss and adaptive_robust_loss_targets is not None:
                        adaptive_params = init_adaptive_robust_loss_params(
                            targets=adaptive_robust_loss_targets,
                            init_alpha=adaptive_robust_loss_init_alpha,
                            init_scale=adaptive_robust_loss_init_scale
                        )
                        params = {**params, **adaptive_params}
                        print(f"Initialized adaptive robust loss params for targets: {adaptive_robust_loss_targets}")
                        alpha, scale = transform_adaptive_robust_params(
                            adaptive_params['adaptive_robust_loss']['alpha_raw'],
                            adaptive_params['adaptive_robust_loss']['scale_raw']
                        )
                        print(f"  Initial alpha: {alpha}")
                        print(f"  Initial scale: {scale}")

            # If optimizer state is None (in the first step), initialize from the parameter pyTree.
            if opt_state is None:
                opt_state = optimizer.init(params)
            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                wandb.log(
                    {f'train_{k}': v for (k, v) in train_metrics_np.items()},
                    step=step
                )

            # Start validation process.
            if step % eval_every_num_steps == 0:
                # Start iteration over validation batches.
                eval_totals = {}
                eval_counts = {}
                validation_iterator_loop = validation_iterator.next_epoch(split='train', mode='validation')
                for graph_batch_validation, lr_batch_validation in validation_iterator_loop:
                    batch_validation = graph_to_batch_fn(graph_batch_validation, lr_batch_validation)
                    batch_validation = jax.tree_util.tree_map(jnp.array, batch_validation)

                    eval_out = validation_step_fn(
                        params,
                        batch_validation
                    )

                    for k, v in eval_out.items():
                        eval_totals[k] = eval_totals.get(k, 0.0) + np.asarray(v).item()
                        eval_counts[k] = eval_counts.get(k, 0) + 1

                eval_metrics = {k: eval_totals[k] / eval_counts[k] for k in eval_totals}

                # Convert to dict to log with weights and bias.
                eval_metrics = {
                    f'eval_{k}': float(v) for k, v in eval_metrics.items()
                }

                print(print_metrics(f"val_{epoch}_{step}:", eval_metrics))
                eval_loss = eval_metrics['eval_loss']

                # Only save if loss is finite
                if jnp.isfinite(eval_loss):
                    ckpt_mngr.save(
                        step,
                        args=ocp.args.Composite(params=ocp.args.StandardSave(params)),
                        metrics={'loss': eval_loss}
                    )
                else:
                    print(f"Skipping checkpoint at step {step} because eval_loss={eval_loss}")

                # Log to weights and bias.
                if use_wandb:
                    wandb.log(
                        eval_metrics,
                        step=step
                    )
            # Finished validation process.

    # Wait until checkpoint manager completes all save operations.
    ckpt_mngr.wait_until_finished()
def make_optimizer(
        name: str = 'adam',
        optimizer_args: Dict = dict(),
        learning_rate: float = 1e-3,
        learning_rate_schedule: str = 'constant_schedule',
        learning_rate_schedule_args: Dict = dict(),
        gradient_clipping: str = 'identity',
        gradient_clipping_args: Dict = dict(),
        num_of_nans_to_ignore: int = 0
):
    """Make optax optimizer.

    Args:
        name (str): Name of the optimizer. Defaults to the Adam optimizer.
            Supports standard optax optimizers (adam, adamw, sgd, etc.) and
            contrib optimizers like 'muon' from optax.contrib.
        optimizer_args (dict): Arguments passed to the optimizer.
            For muon, useful args include:
            - ns_steps (int): Newton-Schulz iterations, default 5
            - beta (float): Momentum decay, default 0.95
            - nesterov (bool): Use Nesterov momentum, default True
            - adaptive (bool): Scale updates by dual norm, default False
        learning_rate (float): Learning rate.
        learning_rate_schedule (str): Learning rate schedule. Defaults to no schedule, meaning learning rate is
            held constant.
        learning_rate_schedule_args (dict): Arguments for the learning rate schedule.
        num_of_nans_to_ignore (int): Number of times NaNs are ignored during in the gradient step. Defaults to 0.
        gradient_clipping (str): Gradient clipping to apply.
        gradient_clipping_args (dict): Arguments to the gradient clipping to apply.
    Returns:
        optax.GradientTransformation

    """
    lr_schedule = getattr(
        optax,
        learning_rate_schedule
    )

    lr_schedule = lr_schedule(
        learning_rate,
        **learning_rate_schedule_args
    )

    # Check if optimizer is in optax.contrib (like muon)
    if hasattr(optax.contrib, name):
        opt = getattr(optax.contrib, name)
    elif hasattr(optax, name):
        opt = getattr(optax, name)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Not found in optax or optax.contrib.")

    opt = opt(
        lr_schedule,
        **optimizer_args
    )

    clip_transform = getattr(
        optax,
        gradient_clipping
    )

    clip_transform = clip_transform(
        **gradient_clipping_args
    )

    return optax.chain(
        clip_transform,
        optax.zero_nans(),
        opt
    )


def freeze_parameters(optimizer, trainable_subset_keys):
    """Freeze parameters by giving keys for trainable subsets. Thus, all parameters that are NOT in
    `trainable_subset_keys` are frozen.

    Args:
        optimizer (): optax.GradientTransformation.
        trainable_subset_keys (Sequence): Keys which belong to entries in the PyTree that are trainable. Note that
        for a pyTree like {'a': {'b': *, 'c': *}, 'd': *} and trainable_subset_keys = ['a'] one gets the following
        {'a': {'b': 'trainable', 'c': 'trainable'}, 'd': 'frozen'}. If 'c' and 'd' should be trainable one has to
        pass trainable_subset_keys = ['c', 'd'].

    Returns:

    """

    return optax.multi_transform(
        {'trainable': optimizer, 'frozen': zero_grads()},
        param_labels=make_annotation_fn(trainable_subset_keys)
    )


def make_annotation_fn(keys):
    return lambda params: traverse_util.path_aware_map(
        lambda path, v: 'trainable' if len(set(keys) & set(path)) > 0 else 'frozen', params
    )


def zero_grads():

    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)
