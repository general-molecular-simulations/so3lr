import pytest

import jax
import jax.numpy as jnp

import numpy.testing as npt

from so3lr.mlff.nn import GeometryEmbedSparse


degrees = [0, 1, 2]

atomic_numbers = jnp.array([1, 11, 31])

positions = jnp.array([
    [0., 0., 0.],
    [0., -2., 0.],
    [1., 0.5, 0.],
])

idx_i = jnp.array([0, 0, 1, 2])
idx_j = jnp.array([1, 2, 0, 0])
idx_i_lr = jnp.array([0, 0, 1, 1, 2, 2])
idx_j_lr = jnp.array([1, 2, 0, 2, 0, 1])

inputs = dict(positions=positions,
              atomic_numbers=atomic_numbers,
              idx_i=idx_i,
              idx_j=idx_j,
              cell=None,
              cell_offset=None)


@pytest.mark.parametrize("cutoff", [1.5, 2.5])
def test_apply(cutoff: float):
    geometry_embed = GeometryEmbedSparse(degrees=degrees,
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='exponential',
                                         cutoff=cutoff,
                                         input_convention='positions',
                                         prop_keys=None)

    params = geometry_embed.init(
        jax.random.PRNGKey(0),
        inputs
    )

    output = geometry_embed.apply(params, inputs)

    npt.assert_equal(output.get('ylm_ij').shape, (4, 9))
    npt.assert_equal(output.get('rbf_ij').shape, (4, 16))
    npt.assert_allclose(output.get('d_ij'), jnp.array([2., jnp.sqrt(1.25), 2., jnp.sqrt(1.25)]))
    npt.assert_equal(output.get('d_ij_lr'), None)
    npt.assert_allclose(
        output.get('r_ij'),
        jnp.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 2.0, 0.0],
                [-1.0, -0.5, 0.0],
            ]
        )
    )
    npt.assert_allclose(
        output.get('unit_r_ij'),
        jnp.array(
            [
                [0.0, -1.0, 0.0],
                [1.0/jnp.sqrt(1.25), 0.5/jnp.sqrt(1.25), 0.0],
                [0.0, 1.0, 0.0],
                [-1.0/jnp.sqrt(1.25), -0.5/jnp.sqrt(1.25), 0.0],
            ]
        )
    )

    if cutoff == 2.5:
        npt.assert_allclose(output.get('cut') > 0., jnp.array([True, True, True, True]))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(output.get('cut') > 0., jnp.array([False, True, False, True]))
    elif cutoff == 1.5:
        npt.assert_allclose(output.get('cut') > 0., jnp.array([False, True, False, True]))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(output.get('cut') > 0., jnp.array([True, True, True, True]))
    else:
        raise RuntimeError('Invalid test argument.')


def test_apply_with_long_range():
    geometry_embed = GeometryEmbedSparse(degrees=degrees,
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='exponential',
                                         cutoff=10.,
                                         input_convention='positions',
                                         prop_keys=None)

    params = geometry_embed.init(
        jax.random.PRNGKey(0),
        inputs
    )

    # A copy: updating the module-level dict would hand the long-range indices
    # to every test that runs after this one.
    inputs_lr = dict(inputs, idx_i_lr=idx_i_lr, idx_j_lr=idx_j_lr)

    output = geometry_embed.apply(params, inputs_lr)

    npt.assert_equal(output.get('ylm_ij').shape, (4, 9))
    npt.assert_equal(output.get('rbf_ij').shape, (4, 16))
    npt.assert_allclose(
        output.get('d_ij'),
        jnp.array([2., jnp.sqrt(1.25), 2., jnp.sqrt(1.25)])
    )
    npt.assert_allclose(
        output.get('r_ij'),
        jnp.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 2.0, 0.0],
                [-1.0, -0.5, 0.0],
            ]
        )
    )
    npt.assert_allclose(
        output.get('unit_r_ij'),
        jnp.array(
            [
                [0.0, -1.0, 0.0],
                [1.0/jnp.sqrt(1.25), 0.5/jnp.sqrt(1.25), 0.0],
                [0.0, 1.0, 0.0],
                [-1.0/jnp.sqrt(1.25), -0.5/jnp.sqrt(1.25), 0.0],
            ]
        )
    )
    npt.assert_allclose(
        output.get('d_ij_lr'),
        jnp.array([2., jnp.sqrt(1.25), 2., jnp.sqrt(2.5**2 + 1), jnp.sqrt(1.25), jnp.sqrt(2.5**2 + 1)])
    )


def test_apply_with_long_range_pbc_non_orthogonal_cell():
    """Long-range PBC distances use r_ij + cell_offset_lr @ cell, cell vectors as rows.

    The offsets point along b and c of a triclinic cell, so the off-diagonal
    entries enter the result: reading the cell column-wise (the transposed
    lattice) gives 3.0 and 5.85 A instead of 2.0 and 4.5 A, and keeping only its
    diagonal gives 2.83 and 5.85 A.
    """
    cell = jnp.array([
        [10., 0., 0.],   # a
        [2., 10., 0.],   # b
        [1., 1., 10.],   # c
    ])
    positions_triclinic = jnp.array([
        [1., 1., 1.],
        [3., 9., 1.],
        [5., 5., 9.5],
    ])
    # 0 -> 1: [2, 8, 0] - b = [0, -2, 0]           -> 2.0 A
    # 0 -> 2: [4, 4, 8.5] - c = [3, 3, -1.5]       -> 4.5 A
    # and the reverse pairs with the opposite offsets.
    idx_i_lr = jnp.array([0, 1, 0, 2])
    idx_j_lr = jnp.array([1, 0, 2, 0])
    cell_offset_lr = jnp.array([[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])

    geometry_embed = GeometryEmbedSparse(
        degrees=degrees,
        radial_basis_fn='bernstein',
        num_radial_basis_fn=16,
        cutoff_fn='exponential',
        cutoff=15.,
        input_convention='positions',
        prop_keys=None
    )

    inputs_triclinic = dict(
        positions=positions_triclinic,
        atomic_numbers=jnp.array([6, 6, 6]),
        # Short-range pairs inside the cell (no offset).
        idx_i=jnp.array([0, 2]),
        idx_j=jnp.array([2, 0]),
        cell=cell,
        cell_offset=jnp.array([[0, 0, 0], [0, 0, 0]]),
        idx_i_lr=idx_i_lr,
        idx_j_lr=idx_j_lr,
        cell_offset_lr=cell_offset_lr,
    )

    params = geometry_embed.init(jax.random.PRNGKey(0), inputs_triclinic)
    output = geometry_embed.apply(params, inputs_triclinic)

    npt.assert_allclose(
        output.get('d_ij_lr'),
        jnp.array([2.0, 2.0, 4.5, 4.5]),
        atol=1e-5
    )


def test_cutoff_is_zero_on_padding_self_loops():
    """jraph pads a batch with self-loop edges (d = 0); they must carry no weight.

    phys(0) = 1, and with ~10^5 such edges on one padding node the second-order
    parameter gradient of a force loss overflowed float32 to NaN, which
    optax.zero_nans silently turned into zero updates.
    """
    geometry_embed = GeometryEmbedSparse(degrees=degrees,
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='phys',
                                         cutoff=5.,
                                         input_convention='positions',
                                         prop_keys=None)
    # Two real pairs, then two self-loops on atom 2.
    graph = dict(positions=positions, atomic_numbers=atomic_numbers,
                 idx_i=jnp.array([0, 0, 2, 2]), idx_j=jnp.array([1, 2, 2, 2]),
                 cell=None, cell_offset=None)
    params = geometry_embed.init(jax.random.PRNGKey(0), graph)
    cut = geometry_embed.apply(params, graph)['cut']
    assert bool(jnp.all(cut[:2] > 0))
    npt.assert_array_equal(cut[2:], 0.)
