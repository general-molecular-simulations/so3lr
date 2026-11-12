import jraph
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from flax import traverse_util
from flax.core.frozen_dict import unfreeze
from pathlib import Path
from typing import Callable, Dict

from ..masking import mask as gradient_utils
from ..utils import checkpoint_utils
from ..utils.param_utils import (
    MAX_NUM_THEORY_LEVELS,
    path_matches_key,
)
from .jraph_utils import dynamically_batch_with_lr

from .robust_loss_jax import distribution as robust_loss_dist
from .robust_loss_jax import general as robust_loss_general

import logging
logger = logging.getLogger("SO3LR")


def _wandb_log(*args, **kwargs):
    """Import and call W&B only from an explicitly enabled logging branch."""
    import wandb
    return wandb.log(*args, **kwargs)

# Global distribution object for adaptive robust loss (handles partition function)
ROBUST_LOSS_DIST = robust_loss_dist.Distribution()

_TFDS_TRAINING_STATE_VERSION = 1
# Checkpoints written before the fingerprint was dropped also carry a
# ``restart_fingerprint`` leaf; the restore target is built from the saved
# tree, so that extra leaf is restored and ignored.  ``num_devices`` is
# optional for the same reason: checkpoints written before it was saved lack it.
_TFDS_TRAINING_STATE_KEYS = frozenset({
    'version', 'checkpoint_step',
    'raw_params', 'optimizer_state', 'ema_params', 'jax_rng',
    'epoch', 'batches_consumed', 'processed_graphs', 'processed_nodes',
    'best_eval_loss', 'evals_without_improvement', 'stop_training',
    'plateau_best_loss', 'plateau_wait', 'plateau_cooldown_counter',
    'plateau_scale', 'plateau_active', 'reduce_lr_enabled', 'stage_two_active',
    'header_printed', 'use_ema',
})


def _make_tfds_training_state(
        *, step, raw_params, optimizer_state, ema_params,
        jax_rng, epoch, batches_consumed, processed_graphs, processed_nodes,
        best_eval_loss, evals_without_improvement, stop_training,
        plateau_best_loss, plateau_wait, plateau_cooldown_counter,
        plateau_scale, plateau_active, reduce_lr_enabled, stage_two_active,
        header_printed, use_ema, num_devices=1):
    """Build the versioned, Orbax-compatible restart PyTree."""
    return {
        'version': _TFDS_TRAINING_STATE_VERSION,
        'checkpoint_step': int(step),
        'raw_params': raw_params,
        'optimizer_state': optimizer_state,
        'ema_params': ema_params,
        'jax_rng': jax_rng,
        'epoch': int(epoch),
        'batches_consumed': int(batches_consumed),
        'processed_graphs': int(np.asarray(jax.device_get(processed_graphs)).item()),
        'processed_nodes': int(np.asarray(jax.device_get(processed_nodes)).item()),
        'best_eval_loss': float(best_eval_loss),
        'evals_without_improvement': int(evals_without_improvement),
        'stop_training': bool(stop_training),
        'plateau_best_loss': float(plateau_best_loss),
        'plateau_wait': int(plateau_wait),
        'plateau_cooldown_counter': int(plateau_cooldown_counter),
        'plateau_scale': float(plateau_scale),
        'plateau_active': bool(plateau_active),
        'reduce_lr_enabled': bool(reduce_lr_enabled),
        'stage_two_active': bool(stage_two_active),
        'header_printed': bool(header_printed),
        'use_ema': bool(use_ema),
        'num_devices': int(num_devices),
    }


def _validate_tfds_training_state(state, step):
    """Reject a training state this version cannot resume from."""
    missing = sorted(_TFDS_TRAINING_STATE_KEYS.difference(state or {}))
    if missing:
        raise RuntimeError(
            f'Checkpoint step {step} has incomplete training state; missing {missing}.'
        )
    if state['version'] != _TFDS_TRAINING_STATE_VERSION:
        raise RuntimeError(
            f'Checkpoint step {step} uses training-state version '
            f'{state["version"]!r}; expected {_TFDS_TRAINING_STATE_VERSION}.'
        )
    if state['raw_params'] is None or state['optimizer_state'] is None:
        raise RuntimeError(
            f'Checkpoint step {step} lacks raw parameters or optimizer state.'
        )


def _restore_tfds_training_state(
        checkpoint_manager, *, step, optimizer):
    """Restore training state while preserving Optax namedtuple structure.

    Returns ``None`` for a params-only checkpoint (written before the full
    training state was saved); the caller then continues from its weights.
    """
    step_dir = Path(checkpoint_manager.directory) / f'ckpt_{step}'
    if not (step_dir / 'training_state').exists():
        return None
    untyped = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            training_state=ocp.args.StandardRestore()
        ),
    )['training_state']

    _validate_tfds_training_state(untyped, step)

    # Target-free Orbax restore turns Optax namedtuples into lists/dicts.  A
    # fresh optimizer state supplies the exact container structure, then Orbax
    # fills it with the saved values.  A changed optimizer structure therefore
    # fails closed rather than producing a subtly invalid continuation.
    restore_target = dict(untyped)
    restore_target['optimizer_state'] = optimizer.init(untyped['raw_params'])
    try:
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                training_state=ocp.args.StandardRestore(restore_target)
            ),
        )['training_state']
    except Exception as exc:
        raise RuntimeError(
            f'Checkpoint step {step} training state is incompatible with the '
            'current optimizer or parameter structure; refusing an inexact restart.'
        ) from exc
    return restored


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


def ensure_adaptive_robust_loss_params(
    params: Dict,
    targets: list,
    init_alpha: float = 1.5,
    init_scale: float = 1.0
) -> Dict:
    """Attach adaptive robust loss parameters to an already-loaded ``params`` tree.

    ``fit`` and ``fit_from_iterator`` only create these on the branch that
    *initializes* a fresh model (``params is None``), which fine-tuning never
    takes -- it always passes the restored checkpoint. Without this, enabling
    ``use_adaptive_robust_loss`` while fine-tuning either silently did nothing
    (checkpoint has no 'adaptive_robust_loss' key, so ``make_loss_fn`` falls
    through to the plain loss) or, worse, reused the pretraining run's alphas
    indexed by the *new* target order -- so `forces` could be scaled by whatever
    alpha the pretraining had learned for its first target.

    Existing entries are reused only when their length still matches
    ``targets``; otherwise they are re-initialized, because the per-target index
    is positional and a different target list makes the old values meaningless.
    The checkpoint stores no target *names*, so a same-length but different (or
    reordered) target list is indistinguishable from a genuine continuation and
    is reused as-is -- the log line below states the assumption so such a run is
    at least diagnosable.

    Returns a new dict; ``params`` is not mutated.
    """
    existing = params.get('adaptive_robust_loss')
    if existing is not None and existing['alpha_raw'].shape[0] == len(targets):
        logger.info(
            'Reusing the checkpoint\'s %d adaptive robust loss parameter(s), '
            'assuming they correspond positionally to the targets %s. If the '
            'run that wrote the checkpoint trained a different target list of '
            'the same length, each target inherits the wrong alpha/scale.',
            len(targets), targets
        )
        return params

    if existing is not None:
        logger.warning(
            'Discarding adaptive robust loss parameters from the checkpoint: they '
            'hold %d target(s) but this run trains %d (%s). The values are indexed '
            'positionally, so keeping them would apply the wrong alpha/scale to '
            'each target.',
            existing['alpha_raw'].shape[0], len(targets), targets
        )

    adaptive_params = init_adaptive_robust_loss_params(
        targets=targets,
        init_alpha=init_alpha,
        init_scale=init_scale
    )
    logger.info(
        'Initialized adaptive robust loss params for targets: %s', targets
    )
    return {**params, **adaptive_params}


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


def _display_metric_keys(eval_metrics):
    """Metric keys shown in the printed table. Hide the adaptive-robust-loss
    internal params (``*_robust_alpha`` / ``*_robust_scale``) -- they are still
    computed and logged to wandb, just not printed in the per-eval table."""
    return [k for k in eval_metrics if 'robust' not in k]


def print_metrics_header(eval_metrics):
    """Print table header row. Call once before training loop."""
    cols = ['epoch', 'step'] + [k.removeprefix('eval_') for k in _display_metric_keys(eval_metrics)]
    header = ' | '.join(f'{c:>14s}' for c in cols)
    sep = '-+-'.join('-' * 14 for _ in cols)
    return f"\n{header}\n{sep}"


def print_metrics_row(epoch, step, eval_metrics):
    """Print one table row for a validation checkpoint."""
    vals = [f'{epoch:>14d}', f'{step:>14d}'] + [
        f'{eval_metrics[k]:>14.4f}' for k in _display_metric_keys(eval_metrics)]
    return ' | '.join(vals)

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
    'a0_ratios': node_mae_loss,
    'c6_ratios': node_mae_loss,
    'partial_charges': node_mae_loss,
}

property_to_loss = {
    'energy': graph_mse_loss_per_atom,
    'stress': graph_mse_loss,
    'forces': node_mse_loss,
    'dipole_vec': graph_mse_loss_per_atom,
    'hirshfeld_ratios': node_mse_loss,
    'a0_ratios': node_mse_loss,
    'c6_ratios': node_mse_loss,
    'partial_charges': node_mse_loss,
}

# Ratio-head alias: legacy SO3LR emits 'hirshfeld_ratios'; the new so3lr-tauri-s/m/l
# models emit 'a0_ratios' (per-atom static-polarizability ratios). Loaded data
# is always keyed 'hirshfeld_ratios'. We accept either spelling in loss_weights
# and remap at the loss-fn entry point so users do not see a KeyError.
_RATIO_ALIAS = {'hirshfeld_ratios': 'a0_ratios', 'a0_ratios': 'hirshfeld_ratios'}

# Anchor targets ('anchor_<quantity>'): <quantity> as predicted by the model being
# fine-tuned, labelled by a frozen copy of the pretrained model. The labels are put
# into each batch under the target name by run_fine_tuning, and the loss is the
# one of <quantity>. Used to keep the long-range heads (partial charges, a0/C6
# ratios) at their pretrained predictions when the fine-tuning data has no labels
# for them.
ANCHOR_PREFIX = 'anchor_'


def _base_target(target: str) -> str:
    return target[len(ANCHOR_PREFIX):] if target.startswith(ANCHOR_PREFIX) else target


def _prediction_key(target: str, outputs_predict: Dict) -> str:
    """The model-output key a target is scored on (trace-time only)."""
    key = _base_target(target)
    if key not in outputs_predict and _RATIO_ALIAS.get(key) in outputs_predict:
        key = _RATIO_ALIAS[key]
    return key

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

    # One-time announcement if a ratio target is present; the actual rename
    # happens at JIT-trace time inside loss_fn below.
    for _t in targets:
        if _t in _RATIO_ALIAS:
            logger.info(
                f"[loss] ratio target '{_t}' is alias-aware: model output "
                f"'{_RATIO_ALIAS[_t]}' (if present) will be remapped to '{_t}'."
            )
            break

    @jax.jit
    def loss_fn(params, batch: Dict[str, jnp.ndarray]):
        inputs = {k: v for k, v in batch.items() if k not in targets}
        outputs_true = {k: v for k, v in batch.items() if k in targets}
        outputs_predict = obs_fn(params, **inputs)

        # Ratio-head alias ('a0_ratios' vs 'hirshfeld_ratios'): predictions are
        # resolved per target by _prediction_key; labels are renamed here.
        # Trace-time only: the dict keys are Python-level.
        for _t in targets:
            _alias = _RATIO_ALIAS.get(_t)
            if _alias is None:
                continue
            if _t not in outputs_true and _alias in outputs_true:
                outputs_true[_t] = outputs_true.pop(_alias)

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
            base_target = _base_target(target)
            y_pred = outputs_predict[_prediction_key(target, outputs_predict)]
            target_mae_fn = property_to_mae[base_target]
            _mae = target_mae_fn(
                y=y_pred,
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

            target_loss_fn = property_to_loss[base_target]
            _l = target_loss_fn(
                y=y_pred,
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

        # Mirror the ratio-head label rename used in make_loss_fn.
        for _t in targets:
            _alias = _RATIO_ALIAS.get(_t)
            if _alias is None:
                continue
            if _t not in outputs_true and _alias in outputs_true:
                outputs_true[_t] = outputs_true.pop(_alias)

        metrics = {}

        # Extract adaptive robust loss parameters if enabled
        adaptive_alphas = None
        adaptive_scales = None
        if use_adaptive_robust_loss and 'adaptive_robust_loss' in params:
            alpha_raw = params['adaptive_robust_loss']['alpha_raw']
            scale_raw = params['adaptive_robust_loss']['scale_raw']
            adaptive_alphas, adaptive_scales = transform_adaptive_robust_params(alpha_raw, scale_raw)

        for target in targets:
            base_target = _base_target(target)
            y_pred = outputs_predict[_prediction_key(target, outputs_predict)]
            target_mae_fn = property_to_mae[base_target]
            _mae = target_mae_fn(
                y=y_pred,
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

            target_loss_fn = property_to_loss[base_target]
            _mse = target_loss_fn(
                y=y_pred,
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
        return _training_step(optimizer, loss_fn, log_gradient_values, params, opt_state, batch)

    return training_step_fn


def _training_step(optimizer, loss_fn, log_gradient_values, params, opt_state, batch, reduce=None):
    """Loss, gradient and optimizer update shared by the single-device and the
    data-parallel training steps; `reduce` (e.g. a pmean over devices) is applied
    to the gradients and metrics before the update."""
    (loss, metrics), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True
    )(
        params,
        batch
    )
    if reduce is not None:
        grads, metrics = reduce(grads), reduce(metrics)

    # Per-leaf gradient norms go under their own key: writing them to
    # 'grad_norm' meant the unconditional global-norm assignment below
    # overwrote them every step, making log_gradient_values a no-op.
    if log_gradient_values:
        metrics['grad_norm_per_leaf'] = unfreeze(
            jax.tree_util.tree_map(
                lambda x: jnp.linalg.norm(x.reshape(-1), axis=0), grads
            )
        )

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


def make_data_parallel_training_step_fn(
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        mesh,
        log_gradient_values: bool = False,
):
    """
    Make a data-parallel training step over the devices of `mesh` (one mesh axis, 'devices').

    params and opt_state are replicated on every device; the batch pytree carries a leading
    device axis that is sharded over the mesh, i.e. one padded batch per device. Each device
    computes the gradient of `loss_fn` on its own batch, gradients and metrics are averaged
    across devices (pmean), and every device applies the same optimizer update, so the
    replicas stay identical. With one batch per device this equals a single-device step on
    the averaged gradient.

    Args:
        optimizer (optax.GradientTransformation): Optax optimizer.
        loss_fn (Callable): Loss function, as for make_training_step_fn.
        mesh (jax.sharding.Mesh): One-dimensional mesh with axis name 'devices'.
        log_gradient_values (bool): Log gradient norms for each leaf in the params pytree.

    Returns:
        Training step fn with the same signature and outputs as make_training_step_fn.
    """
    from jax.sharding import PartitionSpec

    def local_step(params, opt_state, batch):
        batch = jax.tree_util.tree_map(lambda x: x[0], batch)  # this device's batch
        return _training_step(optimizer, loss_fn, log_gradient_values, params, opt_state, batch,
                              reduce=lambda tree: jax.lax.pmean(tree, 'devices'))

    # check_vma=False keeps each device's gradient local, so the explicit pmean is the
    # only cross-device reduction (with the check on, differentiating with respect to
    # the replicated params already sums the gradients over devices).
    return jax.jit(jax.shard_map(
        local_step,
        mesh=mesh,
        in_specs=(PartitionSpec(), PartitionSpec(), PartitionSpec('devices')),
        out_specs=(PartitionSpec(), PartitionSpec(), PartitionSpec()),
        check_vma=False,
    ))


def _batch_groups(iterator, n):
    """Yield (index of the last batch, list of n consecutive batches) from `iterator`;
    a partial group left at the end is dropped. n=1 gives one batch per group."""
    group = []
    for index, item in enumerate(iterator):
        group.append(item)
        if len(group) == n:
            yield index, group
            group = []


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

def expand_observables_to_theory_levels(
        params,
        num_theory_levels: int = MAX_NUM_THEORY_LEVELS
):
    """Expand a restored single-theory-level checkpoint to ``num_theory_levels``.

    Tiles the per-atom energy-head parameters (``energy_offset``,
    ``atomic_scales`` and the ``energy_dense_final`` kernel plus its optional
    ``bias``) of ``observables_0`` from one level of theory to
    ``num_theory_levels`` levels. Parameters that are already multi-level are
    left unchanged. Mutates and returns ``params``.
    """
    if 'params' not in params or 'observables_0' not in params['params']:
        return params
    obs0 = params['params']['observables_0']

    if 'energy_offset' in obs0 and jnp.ndim(obs0['energy_offset']) == 1:
        obs0['energy_offset'] = jnp.tile(obs0['energy_offset'][:, None], (1, num_theory_levels))
        logger.debug('Tiled energy_offset to %d theory levels.', num_theory_levels)

    if 'atomic_scales' in obs0 and jnp.ndim(obs0['atomic_scales']) == 1:
        obs0['atomic_scales'] = jnp.tile(obs0['atomic_scales'][:, None], (1, num_theory_levels))
        logger.debug('Tiled atomic_scales to %d theory levels.', num_theory_levels)

    if 'energy_dense_final' in obs0:
        if obs0['energy_dense_final']['kernel'].shape[1] == 1:
            kernel = obs0['energy_dense_final']['kernel']
            obs0['energy_dense_final']['kernel'] = jnp.tile(kernel, (1, num_theory_levels))
            logger.debug('Tiled energy_dense_final kernel to %d theory levels.', num_theory_levels)

        # Tile the optional final bias too (use_final_bias_bool=True base models,
        # e.g. the single-level legacy SO3LR). Mirrors potential.py so finetuning
        # a 1-level model into num_theory_levels works; bias-free models
        # (so3lr-tauri-s/m/l) have no 'bias' key and are left untouched.
        bias = obs0['energy_dense_final'].get('bias')
        if bias is not None and bias.shape[-1] == 1:
            obs0['energy_dense_final']['bias'] = jnp.tile(bias, (num_theory_levels,))
            logger.debug('Tiled energy_dense_final bias to %d theory levels.', num_theory_levels)

    return params


def _has_plateau_lr_knob(opt_state) -> bool:
    """True if ``opt_state`` exposes an injectable ``plateau_lr_scale`` knob
    (i.e. the optimizer was built with ``reduce_lr_on_plateau=True``)."""
    return optax.tree_utils.tree_get(opt_state, 'plateau_lr_scale') is not None


def _reduce_lr_on_plateau_step(
        opt_state, eval_loss, *, best_loss, wait, cooldown_counter, scale,
        factor, patience, min_scale, min_delta, cooldown, step):
    """One Reduce-LR-on-plateau bookkeeping step (Keras-style semantics).

    Tracks the best validation loss seen and the number of consecutive
    evaluations without improvement; once that reaches ``patience`` the
    multiplicative ``plateau_lr_scale`` in ``opt_state`` is shrunk by ``factor``
    (floored at ``min_scale``) out-of-jit via ``optax.tree_utils.tree_set``,
    then a ``cooldown`` of evaluations is observed before counting again.

    Returns the updated ``(opt_state, best_loss, wait, cooldown_counter, scale)``.
    """
    if cooldown_counter > 0:
        cooldown_counter -= 1
        wait = 0

    if eval_loss < best_loss - min_delta:
        best_loss = eval_loss
        wait = 0
    elif cooldown_counter == 0:
        wait += 1
        if wait >= patience:
            new_scale = max(scale * factor, min_scale)
            if new_scale < scale - 1e-12:
                opt_state = optax.tree_utils.tree_set(opt_state, plateau_lr_scale=new_scale)
                logger.info(
                    f"Reduce-LR-on-plateau: validation loss stalled for {patience} "
                    f"evals at step {step}; scaling LR x{scale:.4g} -> x{new_scale:.4g}.")
                scale = new_scale
                cooldown_counter = cooldown
            else:
                logger.info(
                    f"Reduce-LR-on-plateau: LR scale already at floor "
                    f"(x{min_scale:.4g}); not reducing further at step {step}.")
            wait = 0

    return opt_state, best_loss, wait, cooldown_counter, scale


def _validation_metrics(validation_step_fn, params, batches, graph_to_batch_fn):
    """Mean of each validation metric over the batches, keyed 'eval_<name>'."""
    totals, n = {}, 0
    for graph_batch, lr_batch in batches:
        batch = jax.tree_util.tree_map(jnp.array, graph_to_batch_fn(graph_batch, lr_batch))
        for k, v in validation_step_fn(params, batch).items():
            totals[k] = totals.get(k, 0.0) + np.asarray(v).item()
        n += 1
    return {f'eval_{k}': float(v / n) for k, v in totals.items()}


def _log_and_checkpoint(ckpt_mngr, step, epoch, eval_params, eval_metrics, print_header, use_wandb):
    """Log the eval metrics (the table header first if asked) and save eval_params
    when the loss is finite: orbax ranks checkpoints with best_fn/best_mode='min',
    and a NaN comparison can leave a diverged step in the max_to_keep window as
    'best', evicting the genuinely best weights."""
    if print_header:
        logger.info(print_metrics_header(eval_metrics))
    logger.info(print_metrics_row(epoch, step, eval_metrics))
    eval_loss = eval_metrics['eval_loss']
    if jnp.isfinite(eval_loss):
        ckpt_mngr.save(
            step,
            args=ocp.args.Composite(params=ocp.args.StandardSave(eval_params)),
            metrics={'loss': eval_loss}
        )
    else:
        logger.warning(f"Skipping checkpoint at step {step} because eval_loss={eval_loss}")
    if use_wandb:
        _wandb_log(eval_metrics, step=step)


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
        use_wandb: bool = False,
        log_gradient_values: bool = False,
        use_robust_loss_validation: bool = False,
        robust_loss_alpha_validation: float = 1.99,
        # Adaptive robust loss parameters
        use_adaptive_robust_loss: bool = False,
        adaptive_robust_loss_targets: list = None,
        adaptive_robust_loss_init_alpha: float = 1.5,
        adaptive_robust_loss_init_scale: float = 1.0,
        # Exponential moving average of weights. When enabled, the
        # raw params keep training while a shadow EMA copy is used for the
        # validation metrics and saved in the checkpoint -- this is the deployed
        # ("averaged") model and is usually more stable/accurate.
        use_ema: bool = False,
        ema_decay: float = 0.999,
        # Early stopping: halt training when the validation loss has not improved
        # by more than ``early_stopping_min_delta`` for ``early_stopping_patience``
        # consecutive evaluations. ``None`` disables early stopping (default).
        early_stopping_patience: int = None,
        early_stopping_min_delta: float = 0.0,
        # Reduce-LR-on-plateau: when the validation loss has not improved by more
        # than ``reduce_lr_min_delta`` for ``reduce_lr_patience`` consecutive
        # evaluations, multiply the learning rate by ``reduce_lr_factor`` (down to
        # a floor of ``reduce_lr_min_scale`` times the base LR). Requires the
        # optimizer to have been built with ``reduce_lr_on_plateau=True`` so the
        # injectable ``plateau_lr_scale`` knob exists in ``opt_state``.
        reduce_lr_on_plateau: bool = False,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 10,
        reduce_lr_min_scale: float = 1e-2,
        reduce_lr_min_delta: float = 0.0,
        reduce_lr_cooldown: int = 0,
        # Two-stage loss reweighting: once ``epoch >= stage_two_start_epoch`` swap
        # the training (and validation) loss to ``stage_two_loss_fn`` /
        # ``stage_two_val_fn`` and reset the best-loss / early-stopping / plateau
        # trackers (the loss scale changes between stages). ``None`` disables.
        stage_two_start_epoch: int = None,
        stage_two_loss_fn=None,
        stage_two_val_fn=None,
        on_training_ready=None,
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
        params: Initial parameters for a new run. When a target checkpoint exists
            and `allow_restart=True`, its weights take precedence over these parameters.
        val_fn (Callable, optional): Validation function to use for metrics during validation.
            If None, loss_fn will be used. Defaults to None.
        num_epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options. Training keeps the
            ``max_to_keep`` (default 3) lowest-loss checkpoints and always the newest one.
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
        on_training_ready (Callable, optional): Called after checkpoint validation
            and restoration, before optimizer updates, to publish run metadata.
    Returns:

    """
    numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(exist_ok=True)

    # Create orbax CheckpointManager.
    ckpt_mngr = checkpoint_utils.make_checkpoint_manager(
        ckpt_dir=ckpt_dir,
        ckpt_mngr_options=checkpoint_utils.make_training_checkpoint_options(ckpt_manager_options)
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
    header_printed = False

    # Fine-tuning supplies initial parameters, but an existing target
    # checkpoint takes precedence for a requested restart. Validate this
    # before publishing new metadata or entering the optimizer loop.
    latest_step = ckpt_mngr.latest_step()
    if latest_step is not None:
        if not allow_restart:
            raise RuntimeError(
                f'{ckpt_dir} already exists at step {latest_step}. If you want '
                'to re-start training, set `allow_restart=True`.'
            )
        params = checkpoint_utils.load_params_from_checkpoint(ckpt_dir=ckpt_dir, step=latest_step)
        params = expand_observables_to_theory_levels(params)
        step = latest_step
        logger.info('Loaded parameters from %s at step %s.', ckpt_dir, latest_step)

    if on_training_ready is not None:
        on_training_ready()

    opt_state = None
    # Shadow EMA copy of the parameters. Updated after every
    # optimizer step; used for validation + checkpoints when ``use_ema`` is set.
    ema_params = None
    if use_ema:
        @jax.jit
        def _ema_update_fn(ema_p, p):
            return jax.tree_util.tree_map(
                lambda e, x: ema_decay * e + (1.0 - ema_decay) * x, ema_p, p
            )
        logger.info(f"Using EMA of weights for validation/checkpoints (decay={ema_decay}).")

    # Early-stopping bookkeeping (best validation loss seen + stall counter).
    best_eval_loss = float('inf')
    evals_without_improvement = 0
    stop_training = False
    if early_stopping_patience is not None:
        logger.info(f"Early stopping enabled (patience={early_stopping_patience} evals, "
                    f"min_delta={early_stopping_min_delta}).")

    # Reduce-LR-on-plateau bookkeeping (best loss seen + stall/cooldown counters
    # + current multiplicative scale). ``_plateau_active`` is finalised once the
    # optimizer state exists and the injectable knob is confirmed present.
    plateau_best_loss = float('inf')
    plateau_wait = 0
    plateau_cooldown_counter = 0
    plateau_scale = 1.0
    _plateau_active = False
    if reduce_lr_on_plateau:
        logger.info(f"Reduce-LR-on-plateau enabled (factor={reduce_lr_factor}, "
                    f"patience={reduce_lr_patience} evals, min_scale={reduce_lr_min_scale}, "
                    f"min_delta={reduce_lr_min_delta}, cooldown={reduce_lr_cooldown}).")

    # Two-stage loss reweighting bookkeeping; the swap happens at the epoch
    # boundary >= ``stage_two_start_epoch``.
    _stage_two_active = False
    if stage_two_start_epoch is not None and stage_two_loss_fn is not None:
        logger.info(f"Two-stage loss reweighting enabled (switch at epoch "
                    f"{stage_two_start_epoch}).")

    def validation_batches():
        return dynamically_batch_with_lr(
            iter(validation_data),
            n_node=batch_max_num_nodes,
            n_edge=batch_max_num_edges,
            n_graph=batch_max_num_graphs,
            n_pairs=batch_max_num_pairs,
        )

    start_step = step
    for epoch in range(num_epochs):
        # Switch to the stage-two loss at the configured epoch boundary.
        if (stage_two_start_epoch is not None and stage_two_loss_fn is not None
                and not _stage_two_active and epoch >= stage_two_start_epoch):
            logger.info(f"Two-stage loss reweighting: switching to the stage-two "
                        f"loss at epoch {epoch} (step {step}); resetting best-loss, "
                        f"early-stopping and plateau trackers.")
            training_step_fn = make_training_step_fn(
                optimizer, stage_two_loss_fn, log_gradient_values)
            validation_step_fn = make_validation_step_fn(
                stage_two_val_fn if stage_two_val_fn is not None else stage_two_loss_fn)
            # Loss scale changes between stages -> reset stall/plateau trackers.
            best_eval_loss = float('inf')
            evals_without_improvement = 0
            plateau_best_loss = float('inf')
            plateau_wait = 0
            plateau_cooldown_counter = 0
            header_printed = False
            _stage_two_active = True

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

            # Initialize only when neither caller weights nor a restart were supplied.
            if params is None:
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

            # Initialize the EMA shadow copy from the first available params.
            if use_ema and ema_params is None:
                ema_params = params

            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            # Confirm (once) that the injectable LR knob exists before relying on it.
            if reduce_lr_on_plateau and not _plateau_active:
                if _has_plateau_lr_knob(opt_state):
                    _plateau_active = True
                else:
                    logger.warning(
                        "reduce_lr_on_plateau=True but the optimizer has no "
                        "'plateau_lr_scale' knob -- build the optimizer with "
                        "reduce_lr_on_plateau=True. Disabling plateau LR scaling.")
                    reduce_lr_on_plateau = False

            params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            if use_ema:
                ema_params = _ema_update_fn(ema_params, params)
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                _wandb_log(
                    {f'train_{k}': v for (k, v) in train_metrics_np.items()},
                    step=step
                )

            # Start validation process.
            if step % eval_every_num_steps == 0:
                # Validate the deployed weights (the EMA copy when EMA is on).
                eval_params = ema_params if use_ema else params
                eval_metrics = _validation_metrics(
                    validation_step_fn, eval_params, validation_batches(), graph_to_batch_fn)
                _log_and_checkpoint(ckpt_mngr, step, epoch, eval_params, eval_metrics,
                                    not header_printed, use_wandb)
                header_printed = True

                # Early stopping on the validation loss.
                if early_stopping_patience is not None:
                    _eval_loss = eval_metrics['eval_loss']
                    if _eval_loss < best_eval_loss - early_stopping_min_delta:
                        best_eval_loss = _eval_loss
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1
                        if evals_without_improvement >= early_stopping_patience:
                            logger.info(
                                f"Early stopping at step {step}: validation loss has not "
                                f"improved by >{early_stopping_min_delta} for "
                                f"{early_stopping_patience} evaluations "
                                f"(best={best_eval_loss:.6f}).")
                            stop_training = True

                # Reduce the learning rate when the validation loss plateaus.
                if _plateau_active:
                    (opt_state, plateau_best_loss, plateau_wait,
                     plateau_cooldown_counter, plateau_scale) = _reduce_lr_on_plateau_step(
                        opt_state, eval_metrics['eval_loss'],
                        best_loss=plateau_best_loss, wait=plateau_wait,
                        cooldown_counter=plateau_cooldown_counter, scale=plateau_scale,
                        factor=reduce_lr_factor, patience=reduce_lr_patience,
                        min_scale=reduce_lr_min_scale, min_delta=reduce_lr_min_delta,
                        cooldown=reduce_lr_cooldown, step=step)
            # Finished validation process.

            if stop_training:
                break  # exit the batch loop

        if stop_training:
            break  # exit the epoch loop

    # Validate and save the final weights too when this call trained past the
    # last multiple of eval_every_num_steps.
    if step > start_step and step % eval_every_num_steps != 0:
        eval_params = ema_params if use_ema else params
        _log_and_checkpoint(
            ckpt_mngr, step, epoch, eval_params,
            _validation_metrics(validation_step_fn, eval_params, validation_batches(), graph_to_batch_fn),
            not header_printed, use_wandb)

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
        use_wandb: bool = False,
        log_gradient_values: bool = False,
        use_robust_loss_validation: bool = False,
        robust_loss_alpha_validation: float = 1.99,
        # Adaptive robust loss parameters
        use_adaptive_robust_loss: bool = False,
        adaptive_robust_loss_targets: list = None,
        adaptive_robust_loss_init_alpha: float = 1.5,
        adaptive_robust_loss_init_scale: float = 1.0,
        dataset_info: dict = None,
        # Exponential moving average of weights; see fit().
        use_ema: bool = False,
        ema_decay: float = 0.999,
        # Early stopping; see fit().
        early_stopping_patience: int = None,
        early_stopping_min_delta: float = 0.0,
        # Reduce-LR-on-plateau; see fit().
        reduce_lr_on_plateau: bool = False,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 10,
        reduce_lr_min_scale: float = 1e-2,
        reduce_lr_min_delta: float = 0.0,
        reduce_lr_cooldown: int = 0,
        # Two-stage loss reweighting; see fit().
        stage_two_start_epoch: int = None,
        stage_two_loss_fn=None,
        stage_two_val_fn=None,
        on_training_ready=None,
        data_parallel: bool = False,
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
        ckpt_manager_options (dict): Checkpoint manager options; see fit().
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
        dataset_info (dict): Per-dataset info from dataloader.dataset_info() with keys
            'names', 'sizes', 'weights'. Used to log effective epochs per dataset.
        on_training_ready (Callable, optional): Called after checkpoint validation
            and restoration, before optimizer updates, to publish run metadata.
        data_parallel (bool): Train data-parallel on all local devices. Every optimizer step
            then consumes one padded batch per device and averages the gradients, so the
            effective batch -- and the number of batches per step -- grows with the device
            count (eval_every_num_steps still counts optimizer steps). Parameters, optimizer
            state and EMA are replicated on every device; validation and checkpoints use
            single-device copies, so checkpoints are identical in format to single-device
            ones. A restart must use the same device count, and every epoch needs at least
            one batch per device. Defaults to False.
    Returns:

    """
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    num_devices = jax.local_device_count() if data_parallel else 1
    if data_parallel and num_devices == 1:
        logger.warning('data_parallel=True but only one local device is visible; '
                       'training on a single device.')
    if num_devices > 1:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
        mesh = Mesh(np.array(jax.local_devices()), ('devices',))
        replicated = NamedSharding(mesh, PartitionSpec())
        device_sharded = NamedSharding(mesh, PartitionSpec('devices'))
        logger.info(f'Data-parallel training on {num_devices} devices: {num_devices} batches '
                    f'per optimizer step, gradients averaged across devices.')

    def _make_training_step(selected_loss_fn):
        if num_devices > 1:
            return make_data_parallel_training_step_fn(
                optimizer, selected_loss_fn, mesh, log_gradient_values
            )
        return make_training_step_fn(optimizer, selected_loss_fn, log_gradient_values)

    def _single_device(tree):
        # Validation and checkpoints use one device's copy of the replicated state.
        return jax.device_put(tree, jax.local_devices()[0]) if num_devices > 1 else tree

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(exist_ok=True)

    # Create orbax CheckpointManager.
    ckpt_mngr = checkpoint_utils.make_training_checkpoint_manager(
        ckpt_dir=ckpt_dir,
        ckpt_mngr_options=checkpoint_utils.make_training_checkpoint_options(ckpt_manager_options)
    )

    training_step_fn = _make_training_step(loss_fn)

    validation_step_fn = make_validation_step_fn(
        val_fn
    )

    processed_graphs = 0
    processed_nodes = 0
    step = 0
    header_printed = False
    start_epoch = 0
    opt_state = None
    # Shadow EMA copy of the parameters; see fit().
    ema_params = None
    if use_ema:
        @jax.jit
        def _ema_update_fn(ema_p, p):
            return jax.tree_util.tree_map(
                lambda e, x: ema_decay * e + (1.0 - ema_decay) * x, ema_p, p
            )
        logger.info(f"Using EMA of weights for validation/checkpoints (decay={ema_decay}).")

    # Early-stopping bookkeeping; see fit().
    best_eval_loss = float('inf')
    evals_without_improvement = 0
    stop_training = False
    if early_stopping_patience is not None:
        logger.info(f"Early stopping enabled (patience={early_stopping_patience} evals, "
                    f"min_delta={early_stopping_min_delta}).")

    # Reduce-LR-on-plateau bookkeeping; see fit().
    plateau_best_loss = float('inf')
    plateau_wait = 0
    plateau_cooldown_counter = 0
    plateau_scale = 1.0
    _plateau_active = False
    if reduce_lr_on_plateau:
        logger.info(f"Reduce-LR-on-plateau enabled (factor={reduce_lr_factor}, "
                    f"patience={reduce_lr_patience} evals, min_scale={reduce_lr_min_scale}, "
                    f"min_delta={reduce_lr_min_delta}, cooldown={reduce_lr_cooldown}).")

    # Two-stage loss reweighting bookkeeping; see fit().
    _stage_two_active = False
    if stage_two_start_epoch is not None and stage_two_loss_fn is not None:
        logger.info(f"Two-stage loss reweighting enabled (switch at epoch "
                    f"{stage_two_start_epoch}).")

    # Restore the raw training weights (not the EMA/deployed copy), optimizer
    # and control state.  Params-only checkpoints from older versions continue
    # from their weights with a fresh optimizer.
    latest_step = ckpt_mngr.latest_step()
    training_state = None
    if latest_step is not None:
        if not allow_restart:
            raise RuntimeError(
                f'{ckpt_dir} already exists at step {latest_step}. If you want '
                f'to re-start training, set `allow_restart=True`.'
            )
        training_state = _restore_tfds_training_state(
            ckpt_mngr,
            step=latest_step,
            optimizer=optimizer,
        )
        if training_state is None:
            logger.warning(
                'Checkpoint step %s has no training state (written by an older '
                'version). Continuing from its parameters with a fresh optimizer '
                'and epoch counter.', latest_step,
            )
            params = expand_observables_to_theory_levels(
                checkpoint_utils.load_params_from_checkpoint(ckpt_dir=ckpt_dir, step=latest_step)
            )
            step = int(latest_step)
    if training_state is not None:
        params = training_state['raw_params']
        opt_state = training_state['optimizer_state']
        ema_params = training_state['ema_params']
        jax_rng = training_state['jax_rng']
        step = int(training_state['checkpoint_step'])
        # The device count sets the effective batch, so a different count would
        # silently change the optimisation the learning-rate schedule was set for.
        saved_devices = training_state.get('num_devices')
        if saved_devices is not None and int(saved_devices) != num_devices:
            raise RuntimeError(
                f'Checkpoint step {latest_step} was written by a run on {int(saved_devices)} '
                f'device(s), but this run trains on {num_devices}; the effective batch would '
                'change. Resume on the same number of devices.'
            )
        # Continue with the next epoch rather than replaying the batches this
        # one already consumed: the replay runs the full input pipeline and
        # costs about as long as training on them did (hours to days).
        start_epoch = int(training_state['epoch'])
        if int(training_state['batches_consumed']) > 0:
            start_epoch += 1
        processed_graphs = int(training_state['processed_graphs'])
        processed_nodes = int(training_state['processed_nodes'])
        best_eval_loss = float(training_state['best_eval_loss'])
        evals_without_improvement = int(
            training_state['evals_without_improvement']
        )
        stop_training = bool(training_state['stop_training'])
        plateau_best_loss = float(training_state['plateau_best_loss'])
        plateau_wait = int(training_state['plateau_wait'])
        plateau_cooldown_counter = int(
            training_state['plateau_cooldown_counter']
        )
        plateau_scale = float(training_state['plateau_scale'])
        _plateau_active = bool(training_state['plateau_active'])
        reduce_lr_on_plateau = bool(training_state['reduce_lr_enabled'])
        _stage_two_active = bool(training_state['stage_two_active'])
        header_printed = bool(training_state['header_printed'])

        if _stage_two_active:
            if stage_two_loss_fn is None:
                raise RuntimeError(
                    'Checkpoint is in stage two but no stage_two_loss_fn was provided.'
                )
            training_step_fn = _make_training_step(stage_two_loss_fn)
            validation_step_fn = make_validation_step_fn(
                stage_two_val_fn
                if stage_two_val_fn is not None else stage_two_loss_fn
            )

        logger.info(
            'Restored TFDS training state from %s at step %s.',
            ckpt_dir, latest_step,
        )
        print(
            f'Re-start training from step {latest_step} at epoch {start_epoch}.'
        )

    if on_training_ready is not None:
        on_training_ready()

    for epoch in range(start_epoch, num_epochs):
        if stop_training:
            logger.info(
                'Checkpoint step %s already reached its early-stopping condition; '
                'no further optimizer updates will be applied.', step
            )
            break
        # Switch to the stage-two loss at the configured epoch boundary.
        if (stage_two_start_epoch is not None and stage_two_loss_fn is not None
                and not _stage_two_active and epoch >= stage_two_start_epoch):
            logger.info(f"Two-stage loss reweighting: switching to the stage-two "
                        f"loss at epoch {epoch} (step {step}); resetting best-loss, "
                        f"early-stopping and plateau trackers.")
            training_step_fn = _make_training_step(stage_two_loss_fn)
            validation_step_fn = make_validation_step_fn(
                stage_two_val_fn if stage_two_val_fn is not None else stage_two_loss_fn)
            best_eval_loss = float('inf')
            evals_without_improvement = 0
            plateau_best_loss = float('inf')
            plateau_wait = 0
            plateau_cooldown_counter = 0
            header_printed = False
            _stage_two_active = True

        if use_wandb:
            _wandb_log({"epoch": epoch})
        print(f'Epoch {epoch} of {num_epochs}')
        training_iterator_loop = training_iterator.next_epoch(
            split='train',
            mode='train',
            epoch_index=epoch,
        )
        # One optimizer step per group of num_devices batches (one per device); with data
        # parallelism a partial group at the end of an epoch is dropped.
        epoch_start_step = step
        for batch_index, group in _batch_groups(training_iterator_loop, num_devices):
            batches_consumed = batch_index + 1
            batches = []
            for graph_batch_training, lr_batch_training in group:
                batch_training = graph_to_batch_fn(graph_batch_training, lr_batch_training)
                processed_graphs += batch_training['num_of_non_padded_graphs']
                processed_nodes += int(batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training))
                batches.append(batch_training)
            if num_devices > 1:
                # Stack along a leading device axis (on device: graph_to_batch_fn is jitted,
                # so no host round trip) and give each device its own batch.
                batch_training = jax.device_put(
                    jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *batches), device_sharded
                )
            else:
                # Training data is numpy arrays so we now transform them to jax.numpy arrays.
                batch_training = jax.tree_util.tree_map(jnp.array, batches[0])

            # If params are None (in the first step), initialize the parameters or load from existing checkpoint.
            if params is None:
                print(f'Initialize new parameters.')
                params = model.init(jax_rng, jax.tree_util.tree_map(jnp.array, batches[0]))

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
            # Initialize the EMA shadow copy from the first available params.
            if use_ema and ema_params is None:
                ema_params = params
            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            # Confirm (once) that the injectable LR knob exists before relying on it.
            if reduce_lr_on_plateau and not _plateau_active:
                if _has_plateau_lr_knob(opt_state):
                    _plateau_active = True
                else:
                    logger.warning(
                        "reduce_lr_on_plateau=True but the optimizer has no "
                        "'plateau_lr_scale' knob -- build the optimizer with "
                        "reduce_lr_on_plateau=True. Disabling plateau LR scaling.")
                    reduce_lr_on_plateau = False

            if num_devices > 1:
                # Replicate the state; a no-op once it is placed, so initialisation, a restart
                # and host-side optimizer-state changes (reduce-LR) need no bookkeeping.
                params, opt_state, ema_params = jax.device_put(
                    (params, opt_state, ema_params), replicated
                )
            params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            if use_ema:
                ema_params = _ema_update_fn(ema_params, params)
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                log_dict = {f'train_{k}': v for (k, v) in train_metrics_np.items()}
                log_dict['processed_graphs'] = processed_graphs
                log_dict['processed_nodes'] = processed_nodes
                # Log effective epochs per dataset
                if dataset_info is not None:
                    for name, size, weight in zip(
                        dataset_info['names'], dataset_info['sizes'], dataset_info['weights']
                    ):
                        if size > 0:
                            eff_epoch = processed_graphs * weight / size
                            log_dict[f'effective_epoch/{name}'] = eff_epoch
                _wandb_log(log_dict, step=step)

            # Start validation process.
            if step % eval_every_num_steps == 0:
                # Use the EMA (deployed) weights for validation when enabled.
                eval_params = _single_device(ema_params if use_ema else params)
                # Start iteration over validation batches.
                eval_totals = {}
                eval_counts = {}
                validation_iterator_loop = validation_iterator.next_epoch(split='train', mode='validation')
                for graph_batch_validation, lr_batch_validation in validation_iterator_loop:
                    batch_validation = graph_to_batch_fn(graph_batch_validation, lr_batch_validation)
                    batch_validation = jax.tree_util.tree_map(jnp.array, batch_validation)

                    eval_out = validation_step_fn(
                        eval_params,
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

                if not header_printed:
                    logger.info(print_metrics_header(eval_metrics))
                    header_printed = True
                logger.info(print_metrics_row(epoch, step, eval_metrics))
                eval_loss = eval_metrics['eval_loss']

                # Log to weights and bias.
                if use_wandb:
                    _wandb_log(
                        eval_metrics,
                        step=step
                    )

                # Early stopping on the validation loss.
                if early_stopping_patience is not None:
                    if eval_loss < best_eval_loss - early_stopping_min_delta:
                        best_eval_loss = eval_loss
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1
                        if evals_without_improvement >= early_stopping_patience:
                            logger.info(
                                f"Early stopping at step {step}: validation loss has not "
                                f"improved by >{early_stopping_min_delta} for "
                                f"{early_stopping_patience} evaluations "
                                f"(best={best_eval_loss:.6f}).")
                            stop_training = True

                # Reduce the learning rate when the validation loss plateaus.
                if _plateau_active:
                    (opt_state, plateau_best_loss, plateau_wait,
                     plateau_cooldown_counter, plateau_scale) = _reduce_lr_on_plateau_step(
                        opt_state, eval_metrics['eval_loss'],
                        best_loss=plateau_best_loss, wait=plateau_wait,
                        cooldown_counter=plateau_cooldown_counter, scale=plateau_scale,
                        factor=reduce_lr_factor, patience=reduce_lr_patience,
                        min_scale=reduce_lr_min_scale, min_delta=reduce_lr_min_delta,
                        cooldown=reduce_lr_cooldown, step=step)

                # Validation can update early-stop and plateau/optimizer state.
                # Save only after those transitions so the restored next update
                # sees exactly the same control state as uninterrupted training.
                if jnp.isfinite(eval_loss):
                    raw_params, optimizer_state, saved_ema_params = _single_device(
                        (params, opt_state, ema_params)
                    )
                    training_state = _make_tfds_training_state(
                        step=step,
                        raw_params=raw_params,
                        optimizer_state=optimizer_state,
                        ema_params=saved_ema_params,
                        jax_rng=jax_rng,
                        epoch=epoch,
                        batches_consumed=batches_consumed,
                        processed_graphs=processed_graphs,
                        processed_nodes=processed_nodes,
                        best_eval_loss=best_eval_loss,
                        evals_without_improvement=evals_without_improvement,
                        stop_training=stop_training,
                        plateau_best_loss=plateau_best_loss,
                        plateau_wait=plateau_wait,
                        plateau_cooldown_counter=plateau_cooldown_counter,
                        plateau_scale=plateau_scale,
                        plateau_active=_plateau_active,
                        reduce_lr_enabled=reduce_lr_on_plateau,
                        stage_two_active=_stage_two_active,
                        header_printed=header_printed,
                        use_ema=use_ema,
                        num_devices=num_devices,
                    )
                    checkpoint_started = ckpt_mngr.save(
                        step,
                        args=ocp.args.Composite(
                            params=ocp.args.StandardSave(eval_params),
                            training_state=ocp.args.StandardSave(training_state),
                        ),
                        metrics={'loss': eval_loss}
                    )
                    if checkpoint_started:
                        ckpt_mngr.wait_until_finished()
                else:
                    print(
                        f"Skipping checkpoint at step {step} because "
                        f"eval_loss={eval_loss}"
                    )
            # Finished validation process.

            if stop_training:
                close_training_iterator = getattr(
                    training_iterator_loop, 'close', None
                )
                if close_training_iterator is not None:
                    close_training_iterator()
                break  # exit the batch loop

        if step == epoch_start_step and num_devices > 1:
            raise RuntimeError(
                f'Epoch {epoch} has fewer than {num_devices} training batches, so data-parallel '
                f'training on {num_devices} devices takes no optimizer step. Use fewer devices, '
                'a smaller batch or data_parallel: false.'
            )
        if stop_training:
            break  # exit the epoch loop

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
        num_of_nans_to_ignore: int = 0,
        reduce_lr_on_plateau: bool = False
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
        reduce_lr_on_plateau (bool): If True, append an injectable multiplicative
            learning-rate scale (hyperparameter ``plateau_lr_scale``, initialised
            to 1.0) to the optimizer chain. ``fit`` mutates this knob between
            steps to shrink the effective learning rate when the validation loss
            plateaus. The training step itself stays jitted; only ``opt_state`` is
            updated out-of-jit via ``optax.tree_utils.tree_set``. Defaults to False.
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

    transforms = [
        clip_transform,
        optax.zero_nans(),
        opt
    ]

    if reduce_lr_on_plateau:
        # Injectable multiplicative LR scale, driven out-of-jit by ``fit``.
        # Named via the factory argument so the hyperparameter key is
        # ``plateau_lr_scale`` (looked up with optax.tree_utils.tree_get/set).
        def _plateau_scale_factory(plateau_lr_scale):
            return optax.scale(plateau_lr_scale)

        transforms.append(
            optax.inject_hyperparams(_plateau_scale_factory)(plateau_lr_scale=1.0)
        )

    return optax.chain(*transforms)


def freeze_parameters(optimizer, trainable_subset_keys=None, frozen_subset_keys=None):
    """Freeze parameters by giving keys for trainable or frozen subsets.

    Exactly one of ``trainable_subset_keys`` (whitelist: only these are trained,
    everything else frozen) or ``frozen_subset_keys`` (blacklist: only these are
    frozen, everything else trained) should be provided.

    Args:
        optimizer (): optax.GradientTransformation.
        trainable_subset_keys (Sequence): Keys which belong to entries in the PyTree that are trainable. Note that
        for a pyTree like {'a': {'b': *, 'c': *}, 'd': *} and trainable_subset_keys = ['a'] one gets the following
        {'a': {'b': 'trainable', 'c': 'trainable'}, 'd': 'frozen'}. If 'c' and 'd' should be trainable one has to
        pass trainable_subset_keys = ['c', 'd'].
        frozen_subset_keys (Sequence): Keys whose entries are frozen; all other
        parameters remain trainable (the inverse of ``trainable_subset_keys``).

    Returns:

    """
    if frozen_subset_keys is not None:
        label_fn = make_freeze_annotation_fn(frozen_subset_keys)
    else:
        label_fn = make_annotation_fn(trainable_subset_keys)

    return optax.multi_transform(
        {'trainable': optimizer, 'frozen': zero_grads()},
        param_labels=label_fn
    )


def make_annotation_fn(keys):
    return lambda params: traverse_util.path_aware_map(
        lambda path, v: 'trainable' if any(path_matches_key(path, k) for k in keys) else 'frozen', params
    )


def make_freeze_annotation_fn(keys):
    """Inverse of make_annotation_fn: parameters whose path matches any of
    ``keys`` are 'frozen', everything else is 'trainable'."""
    return lambda params: traverse_util.path_aware_map(
        lambda path, v: 'frozen' if any(path_matches_key(path, k) for k in keys) else 'trainable', params
    )


def zero_grads():

    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)
