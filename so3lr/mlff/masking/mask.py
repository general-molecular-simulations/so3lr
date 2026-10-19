import jax.numpy as jnp
from typing import Callable


def safe_mask(mask, fn: Callable, operand: jnp.ndarray, placeholder: float = 0.) -> jnp.ndarray:
    """
    fn(operand) where mask is True, placeholder elsewhere, with finite gradients.

    fn only sees operand where the mask holds (0 elsewhere), so masked-out entries cannot
    put a NaN or inf into the gradient: the "double where" of the JAX FAQ
    (https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where).

    Args:
        mask (array_like): Array of booleans.
        fn (Callable): The function to apply at entries where mask=True.
        operand (array_like): The values to apply fn to.
        placeholder (float): The value where mask=False.

    Returns: fn(operand) where mask is True, else placeholder.
    """
    inside = jnp.where(mask, operand, jnp.zeros_like(operand))
    return jnp.where(mask, fn(inside), placeholder)


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
