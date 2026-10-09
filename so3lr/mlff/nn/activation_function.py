import jax.numpy as jnp


def softplus_inverse(x):
    return x + jnp.log(-jnp.expm1(-x))
