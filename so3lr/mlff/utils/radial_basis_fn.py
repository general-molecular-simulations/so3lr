import flax.linen as nn
from jax import numpy as jnp
from so3lr.mlff.masking.mask import safe_mask
import numpy as np
from typing import Any


PlaceHolder = Any


bernstein = lambda **kwargs: BernsteinBasis(**kwargs)
fourier = lambda **kwargs: FourierBasis(**kwargs)
gaussian = lambda **kwargs: Gaussian(**kwargs)
physnet = lambda **kwargs: PhysNetBasis(**kwargs)


class BernsteinBasis(nn.Module):
    """Bernstein basis functions.

    Attributes:
        n_rbf: Number of basis functions.
        r_cut: Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0,
            as boundary condition when solving the Helmholtz equation.
        gamma: Scaling parameter.
    """
    n_rbf: int
    r_cut: float = None
    gamma_init: float = 0.9448630629184640

    def setup(self) -> None:
        b = jnp.stack(list(map(lambda x: log_binomial_coefficient(self.n_rbf - 1, x), np.arange(self.n_rbf))))
        # shape: (K)
        k = jnp.arange(self.n_rbf)  # shape: (K)
        k_rev = jnp.arange(self.n_rbf)[::-1]  # shape: (K)

        def log_bernstein_polynomial(x):
            """

            Args:
                x (Array): Array, shape: (...,1)

            Returns:

            """
            k_x = k * jnp.log(jnp.where(k != 0., x, 1.))
            kk_x = k_rev * jnp.log(jnp.where(k_rev != 0., 1 - x, 1.))

            return b + k_x + kk_x

        self.log_bernstein_polynomial = log_bernstein_polynomial
        self.gamma = jnp.float32(self.gamma_init)

    @nn.compact
    def __call__(self, r_ij) -> jnp.ndarray:
        """

        Args:
            r_ij (Array): shape: (...,1)

        Returns:

        """
        exp_r = jnp.exp(-self.gamma * r_ij)  # shape: (...,1)

        # the cases exp_r=0 and exp_r=1 are the boundaries which give inf for the log_polynomial. They correspond
        # to the extreme cases of r_ij = 0 and r_ij = +/- inf, which are only possible in theory but might appear
        # e.g. during initialization or due to padding.
        return safe_mask(mask=((exp_r != 0) & (exp_r != 1)),
                         fn=lambda y: jnp.exp(self.log_bernstein_polynomial(y)),
                         operand=exp_r,
                         placeholder=0)


class Gaussian(nn.Module):
    """Gaussian basis functions.

    Attributes:
        n_rbf: Number of basis functions.
        r_cut: Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0,
            as boundary condition when solving the Helmholtz equation.
    """
    n_rbf: int
    r_cut: float

    @nn.compact
    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances in the RBF basis, used in SchNet
        (see https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)

        Args:
            r (Array): Distances, shape: (...,1)

        Returns: The expanded distances, shape: (...,n,n,L)
        """
        widths = self.param('widths', _init_widths(self.r_0, self.r_cut, self.n_rbf), ())
        centers = self.param('centers', _init_centers(self.r_0, self.r_cut, self.n_rbf), ())
        return jnp.exp(-0.5 / (widths**2) * (r - centers) ** 2)


class PhysNetBasis(nn.Module):
    """Basis used in PhysNet.

    Attributes:
        n_rbf: Number of basis functions.
        r_cut: Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0,
            as boundary condition when solving the Helmholtz equation.
    """
    n_rbf: int
    r_cut: float

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances in the basis used in PhysNet (see https://arxiv.org/abs/1902.08408)

        Args:
            r (Array): Distances, shape: (...,1)

        Returns: The expanded distances, shape: (...,n,n,L)
        """
        r_cut = jnp.asarray(self.r_cut, dtype=r.dtype)
        offsets = jnp.linspace(jnp.exp(-r_cut), jnp.asarray(1, dtype=r.dtype), self.n_rbf)  # shape: (n_rbf)
        coefficient = ((2 / self.n_rbf) * (1 - jnp.exp(-r_cut))) ** (-2)
        return jnp.exp(-abs(coefficient) * (jnp.exp(-r) - offsets) ** 2)


class FourierBasis(nn.Module):
    """Fourier basis.

    Attributes:
        n_rbf: Number of basis functions.
        r_cut: Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0,
            as boundary condition when solving the Helmholtz equation.
    """
    n_rbf: int
    r_cut: float

    def setup(self):
        self.offsets = jnp.arange(0, self.n_rbf, 1)  # shape: (n_rbf)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf)

        Args:
                r (Array): Distances, shape: (...,1)

        Returns: The expanded distances, shape: (...,n,n,L)
        """

        f = lambda x: jnp.sin(jnp.pi / self.r_cut * self.offsets * x) / x
        return safe_mask(r != 0, f, r, 0)


def _init_centers(min,
                  max,
                  n,
                  dtype=jnp.float32
                  ):
    """
    Function to initialize the centers of some radial basis function using jnp.linspace(min,max,n).

    Args:
        min (float):
        max (float):
        n (int):
        dtype ():

    Returns: Vector of linearly spaced centers, shape: (n)

    """

    from jax._src import dtypes

    def init(key,
             shape,
             dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return jnp.linspace(min, max, n, dtype=dtype)

    return init


def _init_widths(min,
                 max,
                 n,
                 dtype=jnp.float32
                 ):
    """
    Function to initialize the centers of some radial basis function using jnp.linspace(min,max,n).

    Args:
        min (float):
        max (float):
        n (int):
        dtype ():

    Returns: Vector of linearly spaced centers, shape: (n)

    """
    from jax._src import dtypes
    centers = jnp.linspace(min, max, n, dtype=dtype)

    def init(key,
             shape,
             dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)

        return (jnp.abs(centers[0] - centers[1]) * jnp.ones_like(centers)).astype(dtype=dtype)

    return init


def log_factorial(n):
    """
    Logarithm of factorial for n.
    Note: Function makes use of the FORTRAN behavior of jax.numpy.sum() that gives 0 over an empty array.
    In that way, log_factorial returns 0 instead of -inf for n=0. Useful for implementing
    factorial as exp(log_factorial) as this gives automatically the correct behavior factorial(0) = 1.

    Args:
        n (int):

    Returns:

    """
    return np.sum(np.log(np.arange(1, n + 1)))


def factorial(n):
    return np.exp(log_factorial(n))


def log_binomial_coefficient(n, k):
    n_factorial = log_factorial(n)
    k_factorial = log_factorial(k)
    n_k_factorial = log_factorial(n - k)
    return n_factorial - (k_factorial + n_k_factorial)


def binomial_coefficient(n, k):
    return np.exp(log_binomial_coefficient(n, k))


