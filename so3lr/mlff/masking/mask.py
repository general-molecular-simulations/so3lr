# The JAX MD-derived safe_mask portion is Copyright 2019 Google LLC and
# licensed under Apache-2.0 (see LICENSE-APACHE-2.0). Modified for SO3LR with
# local typing/documentation; the additional masking helpers are original code.
import jax.numpy as jnp
from typing import Callable


def safe_mask(mask, fn: Callable, operand: jnp.ndarray, placeholder: float = 0.) -> jnp.ndarray:
    """
    Safe mask which ensures that gradients flow nicely. See also
    https://github.com/google/jax-md/blob/b4bce7ab9b37b6b9b2d0a5f02c143aeeb4e2a560/jax_md/util.py#L67

    Args:
        mask (array_like): Array of booleans.
        fn (Callable): The function to apply at entries where mask=True.
        operand (array_like): The values to apply fn to.
        placeholder (int): The values to fill in if mask=False.

    Returns: New array with values either being the output of fn or the placeholder value.

    """
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)


def safe_scale(x: jnp.ndarray, scale: jnp.ndarray, placeholder: float = 0):
    """
    Autograd safe scaling tensor of x with scale.

    Args:
        x (Array): Tensor to scale, shape: (...)
        scale (Array): Tensor by which x is scaled, shape: (1) or same as x
        placeholder (float): Value to put, when scale equals zero

    Returns: Scaled tensor.

    """
    scale_fn = lambda inputs: scale * inputs
    return safe_mask(mask=scale != 0, fn=scale_fn, operand=x, placeholder=placeholder)


def safe_norm(x: jnp.ndarray, axis: int = 0, placeholder: float = 0.):
    y = jnp.sum(jnp.square(x), axis=axis)
    return safe_mask(mask=y > 0, fn=jnp.sqrt, operand=y, placeholder=placeholder)
