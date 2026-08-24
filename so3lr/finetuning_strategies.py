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

# The long-range heads: partial charges (electrostatics), Hirshfeld/a0 ratios
# and C6 ratios. The partial charges live under
# ``observables_0/electrostatic_energy`` -- no shipped checkpoint has an
# ``observables_1``, so freezing that name was a silent no-op that left the
# charges training on data carrying no electrostatics labels.
_LONG_RANGE_KEYS = [
    'observables_0/electrostatic_energy',
    'observables_2',
    'observables_3',
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
    # Forces-only fine-tuning: train the short-range representation + energy
    # head, but FREEZE the long-range heads so the (unlabelled) electrostatics
    # and dispersion keep their physically-meaningful pretrained predictions.
    'freeze_long_range': lambda n: (None, list(_LONG_RANGE_KEYS)),
    'forces_only': lambda n: (None, list(_LONG_RANGE_KEYS)),
}

FINETUNING_STRATEGY_NAMES: Tuple[str, ...] = tuple(FINETUNING_STRATEGIES)


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
        ValueError: If ``strategy`` is not a known strategy name.
    """
    try:
        spec = FINETUNING_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f'--strategy {strategy} is unknown. Select one of '
            f'({", ".join(f"`{name}`" for name in FINETUNING_STRATEGY_NAMES)}).'
        ) from None

    return spec(num_layers)
