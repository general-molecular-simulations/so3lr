import jax
import jax.numpy as jnp
import logging

# l = 0
_Y00 = lambda x, y, z: 1/2 * jnp.sqrt(1/jnp.pi)  # in shape: (...) / out shape: (...)

# l = 1
_Y1_1 = lambda x, y, z: jnp.sqrt(3/(4*jnp.pi)) * y  # in shape: (...) / out shape: (...)
_Y10 = lambda x, y, z: jnp.sqrt(3/(4*jnp.pi)) * z  # in shape: (...) / out_shape: (...)
_Y11 = lambda x, y, z: jnp.sqrt(3/(4*jnp.pi)) * x  # in shape: (...) / out_shape: (...)

# l = 2
_Y2_2 = lambda x, y, z: 1/2 * jnp.sqrt(15/jnp.pi) * x * y
_Y2_1 = lambda x, y, z: 1/2 * jnp.sqrt(15/jnp.pi) * y * z
_Y20 = lambda x, y, z: 1/4 * jnp.sqrt(5/jnp.pi) * (3*z**2 - 1)
_Y21 = lambda x, y, z: 1/2 * jnp.sqrt(15/jnp.pi) * x * z
_Y22 = lambda x, y, z: 1/4 * jnp.sqrt(15/jnp.pi) * (x**2 - y**2)

# l = 3
_Y3_3 = lambda x, y, z: 1/4 * jnp.sqrt(35 / (2*jnp.pi)) * y * (3*x**2 - y**2)
_Y3_2 = lambda x, y, z: 1/2 * jnp.sqrt(105 / jnp.pi) * x * y * z
_Y3_1 = lambda x, y, z: 1/4 * jnp.sqrt(21 / (2*jnp.pi)) * y * (5*z**2 - 1)
_Y30 = lambda x, y, z: 1/4 * jnp.sqrt(7/jnp.pi) * (5*z**3 - 3*z)
_Y31 = lambda x, y, z: 1/4 * jnp.sqrt(21 / (2*jnp.pi)) * x * (5*z**2 - 1)
_Y32 = lambda x, y, z: 1/4 * jnp.sqrt(105 / jnp.pi) * (x**2 - y**2) * z
_Y33 = lambda x, y, z: 1/4 * jnp.sqrt(35 / (2*jnp.pi)) * x * (x**2 - 3*y**2)

# l = 4
_Y4_4 = lambda x, y, z: 3/4 * jnp.sqrt(35 / jnp.pi) * x * y * (x**2 - y**2)
_Y4_3 = lambda x, y, z: 3/4 * jnp.sqrt(35 / (2*jnp.pi)) * y * (3*x**2 - y**2) * z
_Y4_2 = lambda x, y, z: 3/4 * jnp.sqrt(5 / jnp.pi) * x * y * (7*z**2 - 1)
_Y4_1 = lambda x, y, z: 3/4 * jnp.sqrt(5 / (2*jnp.pi)) * y * (7*z**3 - 3*z)
_Y40 = lambda x, y, z: 3/16 * jnp.sqrt(1 / jnp.pi) * (35*z**4 - 30*z**2 + 3)
_Y41 = lambda x, y, z: 3/4 * jnp.sqrt(5 / (2*jnp.pi)) * x * (7*z**3 - 3*z)
_Y42 = lambda x, y, z: 3/8 * jnp.sqrt(5 / jnp.pi) * (x**2 - y**2) * (7*z**2 - 1)
_Y43 = lambda x, y, z: 3/4 * jnp.sqrt(35 / (2*jnp.pi)) * x * (x**2 - 3*y**2) * z
_Y44 = lambda x, y, z: 3/16 * jnp.sqrt(35 / jnp.pi) * (x**2 * (x**2 - 3*y**2) - y**2 * (3*x**2 - y**2))


@jax.jit
def fn_Y0(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Spherical harmonics expansion of order l=0. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l+1) = (...,1)

    """
    return jnp.ones_like(x)*_Y00(x, y, z)


@jax.jit
def fn_Y1(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Spherical harmonics expansion of order l=1. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,3)

    """
    return jnp.concatenate([_Y1_1(x, y, z), _Y10(x, y, z), _Y11(x, y, z)], axis=-1)


@jax.jit
def fn_Y2(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Spherical harmonics expansion of order l=2. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,5)

    """
    return jnp.concatenate([_Y2_2(x, y, z),
                            _Y2_1(x, y, z),
                            _Y20(x, y, z),
                            _Y21(x, y, z),
                            _Y22(x, y, z)], axis=-1)


@jax.jit
def fn_Y3(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Spherical harmonics expansion of order l=3. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,7)

    """
    return jnp.concatenate([_Y3_3(x, y, z),
                            _Y3_2(x, y, z),
                            _Y3_1(x, y, z),
                            _Y30(x, y, z),
                            _Y31(x, y, z),
                            _Y32(x, y, z),
                            _Y33(x, y, z)], axis=-1)


@jax.jit
def fn_Y4(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Spherical harmonics expansion of order l=4. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,9)

    """
    return jnp.concatenate([_Y4_4(x, y, z),
                            _Y4_3(x, y, z),
                            _Y4_2(x, y, z),
                            _Y4_1(x, y, z),
                            _Y40(x, y, z),
                            _Y41(x, y, z),
                            _Y42(x, y, z),
                            _Y43(x, y, z),
                            _Y44(x, y, z)], axis=-1)


def init_sph_fn(l: int):
    if l == 0:
        return lambda rij: fn_Y0(*jnp.split(rij, indices_or_sections=3, axis=-1))
    elif l == 1:
        return lambda rij: fn_Y1(*jnp.split(rij, indices_or_sections=3, axis=-1))
    elif l == 2:
        return lambda rij: fn_Y2(*jnp.split(rij, indices_or_sections=3, axis=-1))
    elif l == 3:
        return lambda rij: fn_Y3(*jnp.split(rij, indices_or_sections=3, axis=-1))
    elif l == 4:
        return lambda rij: fn_Y4(*jnp.split(rij, indices_or_sections=3, axis=-1))
    else:
        logging.error('Spherical harmonics are only defined up to order l = 4.')
        raise NotImplementedError
