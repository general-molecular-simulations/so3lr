from jax.ops import segment_sum
import jax.numpy as jnp
from typing import (Callable, Sequence)
from functools import partial
import itertools as it

from so3lr.mlff.masking.mask import safe_mask


def make_degree_sum_fn(degrees: Sequence[int]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Make function that calculates the sum across the orders for each degree l in an SPHC vector.

    Args:
        degrees (List): List of degrees l.

    Returns: function that takes vector of `shape=m_tot` as input and returns vector of `shape=n_l` which corresponds
        to the sum across each degree l.

    """
    segment_ids = jnp.array([y for y in it.chain(*[[n] * (2 * degrees[n] + 1) for n in range(len(degrees))])])
    num_segments = len(degrees)
    degree_sum = partial(segment_sum, segment_ids=segment_ids, num_segments=num_segments)
    return degree_sum


def make_degree_norm_fn(degrees: Sequence[int]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Make function that calculates the norm for each degree l on an SPHC vector.

    Args:
        degrees (List): Sequence of degrees l.

    Returns: function that takes vector of `shape=m_tot` as input and returns vector of `shape=n_l` which corresponds
        to the norm for each degree l.

    """
    per_degree_sum = make_degree_sum_fn(degrees)

    def fn(_x):
        _y = per_degree_sum(_x**2)
        return safe_mask(_y > 0, jnp.sqrt, _y, 0)

    return fn


def to_scalar_feature_inner_product(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate scalar feature representation from equivariant features.

    Args:
        v1 (Array): First equivariant feature, shape: (...,3,F)
        v2 (Array): Second equivariant feature, shape: (...,3,F)

    Returns: Scalar feature, shape: (...,F)

    """
    return jnp.einsum('...mi, ...mi -> ...i', v1, v2)


def to_scalar_feature_norm(v1: jnp.ndarray, axis=0) -> jnp.ndarray:
    """

    Args:
        v1 (Array): Equivariant feature, shape: (...,F)
        axis (Array): Along which dimension to take the norm

    Returns:

    """

    x = jnp.sum(v1 ** 2, axis=axis, keepdims=True)

    return jnp.squeeze(safe_mask(x, jnp.sqrt, x), axis)
