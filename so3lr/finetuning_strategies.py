"""Fine-tuning strategies: the single source of truth for ``--strategy``.

Both the CLI (``so3lr finetune --strategy``, which needs the list of names at
*import* time to build its ``click.Choice``) and
``mlff.config.from_config.run_fine_tuning`` (which needs the parameter keys)
resolve through this table. Keeping them together is deliberate: the two lists
had drifted, so ``freeze_all_but_final_mlp`` was rejected by click before it
could ever reach the dispatch that implements it, and
``final_mlp_and_hirshfeld`` was missing from the unknown-strategy error message.

This module must stay import-cheap -- ``so3lr/cli/so3lr_cli.py`` imports it at
module scope and ``so3lr --help`` must not pull in JAX.

Keys use the matching grammar of ``mlff.utils.param_utils.path_matches_key``:

* ``'observables_0'``                -- matches any single component of a path
* ``'observables_0/energy_offset'``  -- matches a contiguous run of components
* ``'observables_0/energy_dense*'``  -- a trailing ``*`` is a prefix match
"""

from typing import Callable, Dict, List, Optional, Tuple

__all__ = [
    'FINETUNING_STRATEGIES',
    'FINETUNING_STRATEGY_NAMES',
    'REMOVED_FINETUNING_STRATEGIES',
    'resolve_finetuning_strategy',
]

# The per-atom energy readout MLP: the regression layer, any deeper
# 'energy_dense_hidden_*' layers and the final projection, plus the atomic
# scales/offsets.
_FINAL_MLP_KEYS = [
    'observables_0/atomic_scales',
    'observables_0/energy_dense*',
    'observables_0/energy_offset',
]

def _last_layer(num_layers: int) -> str:
    """Name of the last message-passing block of an ``num_layers``-deep model."""
    return f'layers_{num_layers - 1}'


# Strategy name -> callable(num_layers) -> (trainable_subset_keys,
# frozen_subset_keys). At most one of the two is non-None:
# ``trainable_subset_keys`` is a whitelist (train only these, freeze the rest),
# ``frozen_subset_keys`` a blacklist (freeze only these, train the rest). See
# ``training_utils.freeze_parameters``.
#
# Insertion order is the order shown in ``--help`` and in the error message for
# an unknown strategy, so keep related entries adjacent.
FINETUNING_STRATEGIES: Dict[str, Callable[[int], Tuple[Optional[List[str]], Optional[List[str]]]]] = {
    # Everything is refined.
    'full': lambda n: (None, None),
    # Freeze EVERYTHING (the SO3krates backbone and the partial-charge /
    # Hirshfeld / C6 heads) and refine ONLY the per-atom energy readout MLP.
    # This keeps the physically-meaningful charge & ratio predictions untouched.
    'final_mlp': lambda n: (list(_FINAL_MLP_KEYS), None),
    'freeze_all_but_final_mlp': lambda n: (list(_FINAL_MLP_KEYS), None),
    # Final MLP + the Hirshfeld head.
    'final_mlp_and_hirshfeld': lambda n: (['observables_0', 'observables_2'], None),
    # Only the Hirshfeld head.
    'hirshfeld': lambda n: (['observables_2'], None),
    # Only the last message-passing block.
    'last_layer': lambda n: ([_last_layer(n)], None),
    # Last message-passing block + the whole energy observable.
    'last_layer_and_final_mlp': lambda n: ([_last_layer(n), 'observables_0'], None),
    # Only the first message-passing block.
    'first_layer': lambda n: (['layers_0'], None),
    # First and last message-passing blocks.
    'first_layer_and_last_layer': lambda n: (['layers_0', _last_layer(n)], None),
    # Whole-observable selections.
    'obs_0': lambda n: (['observables_0'], None),
    'obs_023': lambda n: (['observables_0', 'observables_2', 'observables_3'], None),
    'obs_23': lambda n: (['observables_2', 'observables_3'], None),
    'obs_3': lambda n: (['observables_3'], None),
    # Last message-passing block + the energy readout MLP (incl. deeper head layers).
    'll_fmlp': lambda n: ([_last_layer(n)] + list(_FINAL_MLP_KEYS), None),
    'll_obs_023': lambda n: (
        [_last_layer(n), 'observables_0', 'observables_2', 'observables_3'], None
    ),
    # Freeze the backbone and the long-range heads; train the energy readout and
    # the residual readout (model.energy_residual_mlp_widths in the finetune
    # config). Charges and a0/C6 ratios stay exactly the pretrained ones.
    'readout_only': lambda n: (list(_FINAL_MLP_KEYS) + ['observables_0/energy_residual*'], None),
}

FINETUNING_STRATEGY_NAMES: Tuple[str, ...] = tuple(FINETUNING_STRATEGIES)

# Removed strategy name -> what to use instead. `freeze_long_range` (alias `forces_only`)
# froze the long-range heads, but they read the message-passing features, which kept
# training, so the predicted charges and a0/C6 ratios drifted anyway.
_KEEP_LONG_RANGE = (
    'it did not keep the long-range terms: the frozen heads read features that keep '
    'training. To keep them on energy/force-only data, use --strategy readout_only with '
    'a residual readout (model.energy_residual_mlp_widths), or --strategy full with '
    'training.long_range_anchor_weights; see so3lr/config/finetune.yaml.'
)
REMOVED_FINETUNING_STRATEGIES: Dict[str, str] = {
    'freeze_long_range': f'--strategy freeze_long_range was removed: {_KEEP_LONG_RANGE}',
    'forces_only': f'--strategy forces_only was removed: {_KEEP_LONG_RANGE}',
}


def resolve_finetuning_strategy(
        strategy: str,
        num_layers: int
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Resolve a strategy name into parameter-selection keys.

    Args:
        strategy: One of :data:`FINETUNING_STRATEGY_NAMES`.
        num_layers: ``num_layers`` of the *pretrained* model being fine tuned;
            used to name the last message-passing block. Read it from the
            checkpoint's hyperparameters, not from the user's finetune config --
            an off-by-one here selects a module that does not exist.

    Returns:
        ``(trainable_subset_keys, frozen_subset_keys)``; at most one is non-None.

    Raises:
        ValueError: If ``strategy`` is not a known strategy name, or a removed one
            (the message says what replaces it).
    """
    if strategy in REMOVED_FINETUNING_STRATEGIES:
        raise ValueError(REMOVED_FINETUNING_STRATEGIES[strategy])
    try:
        spec = FINETUNING_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f'--strategy {strategy} is unknown. Select one of '
            f'({", ".join(f"`{name}`" for name in FINETUNING_STRATEGY_NAMES)}).'
        ) from None

    return spec(num_layers)
