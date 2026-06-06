"""Unit tests for the radial basis functions in so3lr.mlff.utils.radial_basis_fn."""
import jax
import jax.numpy as jnp
import pytest

from so3lr.mlff.utils.radial_basis_fn import (
    Gaussian,
    PhysNetBasis,
    FourierBasis,
)


def test_gaussian_basis_finite_shape():
    """Gaussian builds its [0, r_cut] grid and returns finite output.

    Regression: Gaussian.__call__ previously referenced an undeclared
    ``self.r_0`` (the class only declares n_rbf / r_cut), so any
    instantiation+call raised AttributeError before this test could assert.
    """
    n_rbf = 8
    m = Gaussian(n_rbf=n_rbf, r_cut=5.0)
    r = jnp.array([[0.5], [1.0], [2.0]])  # (3, 1)
    params = m.init(jax.random.PRNGKey(0), r)
    out = m.apply(params, r)
    assert out.shape == (r.shape[0], n_rbf)
    assert bool(jnp.all(jnp.isfinite(out)))
    # Grid params built with the right shapes (proves no r_0 dependency).
    assert params['params']['centers'].shape == (n_rbf,)
    assert params['params']['widths'].shape == (n_rbf,)


@pytest.mark.parametrize('basis_cls', [Gaussian, PhysNetBasis, FourierBasis])
def test_radial_basis_dispatch_finite(basis_cls):
    """Each (n_rbf, r_cut)-style radial basis returns finite (N, n_rbf) output.

    Guards the embed_sparse dispatch contract (getattr(radial_basis_fn, name)
    constructed with n_rbf=..., r_cut=...).
    """
    n_rbf = 16
    m = basis_cls(n_rbf=n_rbf, r_cut=5.0)
    r = jnp.array([[0.5], [1.0], [2.0], [4.5]])  # (4, 1)
    params = m.init(jax.random.PRNGKey(0), r)
    out = m.apply(params, r)
    assert out.shape == (r.shape[0], n_rbf)
    assert bool(jnp.all(jnp.isfinite(out)))
