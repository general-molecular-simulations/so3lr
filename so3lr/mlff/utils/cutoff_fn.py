import jax.numpy as jnp
from so3lr.mlff.masking.mask import safe_mask


def phys(r: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """
    Cutoff function used in PhysNet.

    Args:
        r (array): Distances, shape: (...)
        r_cut (float): Cutoff distance

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """
    cut_fn = lambda x: jnp.ones_like(x) - 6 * (x / r_cut) ** 5 + 15 * (x / r_cut) ** 4 - 10 * (x / r_cut) ** 3
    return safe_mask(r < r_cut, cut_fn, r, 0)


def cosine(r: jnp.ndarray, r_cut: float):
    """
    Behler-style cosine cutoff function.

    Args:
        r (array): Distances, shape: (...)
        r_cut (float): Cutoff distance

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """
    cut_fn = lambda x: 0.5 * (jnp.cos(x * jnp.pi / r_cut) + 1.0)
    return safe_mask(r < r_cut, cut_fn, r, 0)


def polynomial(r: jnp.ndarray, r_cut: float, p: int) -> jnp.ndarray:
    """
    Polynomial cutoff function.

    Args:
        r (array): Distances, shape: (...)
        r_cut (float): Cutoff distance
        p (int): Maximal order of the polynomial

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """
    pol_fn = lambda x: 1 - (1/2) * (p+1)*(p+2) * (x/r_cut)**p + p*(p+2)*(x/r_cut)**(p+1)-(1/2)*p*(p+1)*(x/r_cut)**(p+2)
    return safe_mask(r < r_cut, pol_fn, r, 0)


def exponential(r: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """
    Exponential cutoff function used in SpookyNet.

    Args:
        r (Array): Distances, shape: (...)
        r_cut (float): Cutoff distance

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """
    f_cut = lambda x: jnp.exp(-x ** 2 / ((r_cut - x) * (r_cut + x)))
    return safe_mask(mask=r < r_cut, fn=f_cut, operand=r, placeholder=0)
