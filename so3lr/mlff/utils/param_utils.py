from typing import Iterator, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

# Number of levels of theory the multi-head energy readout is sized for. The
# data loaders emit a one-hot `theory_mask` of this width and the energy-head
# parameters are tiled to match, so the value has to agree everywhere; it used
# to be spelled out as a bare `16` in a dozen places.
MAX_NUM_THEORY_LEVELS = 16


def print_param_shapes(params, prefix='', emit=print):
    """Write the parameter tree's shapes, one line each, through emit (print or a logger)."""
    if isinstance(params, dict):
        for key, value in params.items():
            if isinstance(value, (dict, jnp.ndarray, np.ndarray)):
                if isinstance(value, (jnp.ndarray, np.ndarray)):
                    emit(f"{prefix}{key}: {value.shape}")
                else:
                    emit(f"{prefix}{key}:")
                    print_param_shapes(value, prefix + '  ', emit)


def count_params(params_dict):
    total = 0
    if isinstance(params_dict, dict):
        for key, value in params_dict.items():
            if isinstance(value, jnp.ndarray):
                total += value.size
            elif isinstance(value, dict):
                total += count_params(value)
            elif isinstance(value, np.ndarray):
                total += value.size
    return total


def path_matches_key(path: Sequence[str], key: str) -> bool:
    """Return True if a parameter ``path`` (tuple of nested module/leaf names)
    matches a selection ``key``.

    Supported key forms:
      * Plain component, e.g. ``'observables_0'`` -- matches if it equals ANY
        single component of ``path`` (the historical behaviour).
      * Hierarchical, e.g. ``'observables_0/energy_dense_final'`` -- matches if
        its '/'-separated parts appear as a CONTIGUOUS subsequence of ``path``.
        (The previous ``set(keys) & set(path)`` test silently matched nothing
        for slashed keys, so strategies like ``final_mlp`` froze everything.)
      * Trailing ``*`` on any component for prefix matching, e.g.
        ``'observables_0/energy_dense*'`` matches ``energy_dense_regression``,
        ``energy_dense_hidden_0``, ``energy_dense_final``, ...

    Lives here rather than in ``training_utils`` so that parameter *counting*
    and the optimizer's *labelling* cannot disagree about what a strategy
    selects (``training_utils`` imports this module, not the other way round).
    """
    def _component_matches(component, pattern):
        if pattern.endswith('*'):
            return component.startswith(pattern[:-1])
        return component == pattern

    parts = key.split('/')
    n = len(parts)
    if n == 1:
        return any(_component_matches(c, parts[0]) for c in path)
    for i in range(len(path) - n + 1):
        if all(_component_matches(path[i + j], parts[j]) for j in range(n)):
            return True
    return False


def iter_param_leaves(params_dict, prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], object]]:
    """Yield ``(path, array)`` for every array leaf of a nested parameter dict.

    Paths are full (they keep the leading ``'params'`` component) so they match
    what ``flax.traverse_util.path_aware_map`` hands to the optimizer's label
    function.
    """
    if not isinstance(params_dict, dict):
        return
    for key, value in params_dict.items():
        if isinstance(value, dict):
            yield from iter_param_leaves(value, prefix + (key,))
        elif isinstance(value, (jnp.ndarray, np.ndarray)):
            yield prefix + (key,), value


def count_trainable_params(
        params_dict,
        trainable_keys: Optional[Sequence[str]] = None,
        frozen_keys: Optional[Sequence[str]] = None
) -> int:
    """Count the parameters that will actually receive gradient updates.

    Mirrors ``training_utils.freeze_parameters``: ``trainable_keys`` is a
    whitelist (only matching parameters train) and ``frozen_keys`` a blacklist
    (only matching parameters are frozen). Both use the same
    :func:`path_matches_key` semantics as the optimizer's label function -- the
    previous flat ``key in params_dict`` test reported 0 trainable parameters
    for every hierarchical ('a/b') or prefix ('a/b*') strategy and ignored
    ``frozen_keys`` entirely, so the printed counts contradicted what was being
    trained.
    """
    if trainable_keys is None and frozen_keys is None:
        return count_params(params_dict)

    total = 0
    for path, value in iter_param_leaves(params_dict):
        if trainable_keys is not None:
            is_trainable = any(path_matches_key(path, k) for k in trainable_keys)
        else:
            is_trainable = not any(path_matches_key(path, k) for k in frozen_keys)
        if is_trainable:
            total += value.size
    return total


def unmatched_param_keys(params_dict, keys: Sequence[str]) -> List[str]:
    """Return the subset of ``keys`` that matches no parameter in ``params_dict``.

    A selection key that matches nothing is never intentional: as a whitelist it
    silently contributes no trainable parameters, and as a blacklist it freezes
    nothing. Callers use this to fail (or warn) instead of running a fine-tuning
    job that quietly updates the wrong set of weights.
    """
    paths = [path for path, _ in iter_param_leaves(params_dict)]
    return [k for k in keys if not any(path_matches_key(p, k) for p in paths)]
