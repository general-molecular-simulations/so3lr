import pytest

import jax
import jax.numpy as jnp

import numpy.testing as npt

from so3lr.mlff.nn import GeometryEmbedSparse
from so3lr.mlff.geometric import add_cell_offsets_sparse


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


def test_init():
    geometry_embed = GeometryEmbedSparse(degrees=degrees,
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='exponential',
                                         cutoff=5.,
                                         input_convention='positions',
                                         prop_keys=None)

    _ = geometry_embed.init(
        jax.random.PRNGKey(0),
        inputs
    )


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

    inputs.update(
        dict(
            idx_i_lr=idx_i_lr,
            idx_j_lr=idx_j_lr
        )
    )

    output = geometry_embed.apply(params, inputs)

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


# =============================================================================
# PBC Test Data
# =============================================================================

# Simple cubic cell (10 Å)
cell_matrix_pbc = jnp.array([
    [10., 0., 0.],
    [0., 10., 0.],
    [0., 0., 10.]
])

# 3 atoms - positions chosen so PBC effects are testable
# Atom 0 at (1, 1, 1)
# Atom 1 at (9, 1, 1): raw distance to atom 0 is 8 Å, but PBC distance is 2 Å
# Atom 2 at (5, 5, 5)
positions_pbc = jnp.array([
    [1., 1., 1.],
    [9., 1., 1.],
    [5., 5., 5.],
])

atomic_numbers_pbc = jnp.array([6, 6, 6])  # Carbon atoms

# Short-range pairs (direct, no offset needed - within same cell)
idx_i_sr_pbc = jnp.array([0, 2])
idx_j_sr_pbc = jnp.array([2, 0])
num_sr_pairs_pbc = 2
# Cell repeated for each short-range pair
cell_sr_pbc = jnp.repeat(cell_matrix_pbc[None], num_sr_pairs_pbc, axis=0)  # (2, 3, 3)
cell_offset_sr_pbc = jnp.array([[0, 0, 0], [0, 0, 0]])  # (2, 3)

# Long-range pairs crossing PBC
# Pair 0: atom 0 -> atom 1
#   Raw r = positions[1] - positions[0] = [8, 0, 0]
#   With offset [-1, 0, 0]: r = [8, 0, 0] + [-1, 0, 0] @ cell = [8, 0, 0] + [-10, 0, 0] = [-2, 0, 0]
#   Distance = 2 Å (correct PBC distance)
# Pair 1: atom 1 -> atom 0
#   Raw r = positions[0] - positions[1] = [-8, 0, 0]
#   With offset [1, 0, 0]: r = [-8, 0, 0] + [1, 0, 0] @ cell = [-8, 0, 0] + [10, 0, 0] = [2, 0, 0]
#   Distance = 2 Å (correct PBC distance)
idx_i_lr_pbc = jnp.array([0, 1])
idx_j_lr_pbc = jnp.array([1, 0])
num_lr_pairs_pbc = 2
cell_lr_pbc = jnp.repeat(cell_matrix_pbc[None], num_lr_pairs_pbc, axis=0)  # (2, 3, 3)
cell_offset_lr_pbc = jnp.array([[-1, 0, 0], [1, 0, 0]])  # (2, 3)

# Expected distances with PBC: 2.0 Å (not 8.0 Å which would be the raw distance)
expected_d_ij_lr_pbc = jnp.array([2.0, 2.0])


def test_add_cell_offsets_sparse_lr():
    """
    Test add_cell_offsets_sparse function directly for long-range pairs with PBC.

    This tests the core calculation: r_ij_corrected = r_ij + cell_offsets @ cell
    """
    # Raw distance vectors (before PBC correction)
    r_ij_raw = jnp.array([
        [8., 0., 0.],   # atom 0 -> atom 1: positions[1] - positions[0]
        [-8., 0., 0.],  # atom 1 -> atom 0: positions[0] - positions[1]
    ])

    # Apply cell offsets
    r_ij_corrected = add_cell_offsets_sparse(
        r_ij=r_ij_raw,
        cell=cell_lr_pbc,
        cell_offsets=cell_offset_lr_pbc
    )

    # Expected corrected vectors
    expected_r_ij = jnp.array([
        [-2., 0., 0.],  # [8,0,0] + [-1,0,0]@cell = [8,0,0] + [-10,0,0] = [-2,0,0]
        [2., 0., 0.],   # [-8,0,0] + [1,0,0]@cell = [-8,0,0] + [10,0,0] = [2,0,0]
    ])

    npt.assert_allclose(r_ij_corrected, expected_r_ij, atol=1e-6)

    # Verify distances are 2.0 Å (not 8.0 Å)
    d_ij = jnp.linalg.norm(r_ij_corrected, axis=-1)
    npt.assert_allclose(d_ij, jnp.array([2.0, 2.0]), atol=1e-6)


def test_apply_with_long_range_and_pbc():
    """
    Test GeometryEmbedSparse with long-range indices and periodic boundary conditions.

    This verifies that the long-range distance calculation correctly applies PBC
    via cell_lr and cell_offset_lr.
    """
    geometry_embed = GeometryEmbedSparse(
        degrees=degrees,
        radial_basis_fn='bernstein',
        num_radial_basis_fn=16,
        cutoff_fn='exponential',
        cutoff=15.,  # Large cutoff to include all pairs
        input_convention='positions',
        prop_keys=None
    )

    # Build inputs with PBC data
    inputs_pbc = dict(
        positions=positions_pbc,
        atomic_numbers=atomic_numbers_pbc,
        idx_i=idx_i_sr_pbc,
        idx_j=idx_j_sr_pbc,
        cell=cell_sr_pbc,
        cell_offset=cell_offset_sr_pbc,
        idx_i_lr=idx_i_lr_pbc,
        idx_j_lr=idx_j_lr_pbc,
        cell_lr=cell_lr_pbc,
        cell_offset_lr=cell_offset_lr_pbc,
    )

    params = geometry_embed.init(jax.random.PRNGKey(0), inputs_pbc)
    output = geometry_embed.apply(params, inputs_pbc)

    # Critical assertion: LR distances should be 2.0 Å (with PBC), NOT 8.0 Å (raw)
    npt.assert_allclose(
        output.get('d_ij_lr'),
        expected_d_ij_lr_pbc,
        atol=1e-6,
        err_msg="Long-range distances with PBC should be 2.0 Å, not 8.0 Å"
    )

    # This should FAIL if PBC is not applied:
    with npt.assert_raises(AssertionError):
        npt.assert_allclose(
            output.get('d_ij_lr'),
            jnp.array([8.0, 8.0]),  # Raw distances without PBC
            atol=1e-6
        )


def test_apply_with_long_range_pbc_non_orthogonal_cell():
    """
    Test GeometryEmbedSparse with long-range indices and PBC using a non-orthogonal cell.

    This tests a triclinic cell to ensure the offset calculation works for general cells.
    """
    # Non-orthogonal (triclinic) cell
    cell_matrix_triclinic = jnp.array([
        [10., 0., 0.],
        [2., 10., 0.],
        [1., 1., 10.]
    ])

    # Positions
    positions_triclinic = jnp.array([
        [1., 1., 1.],
        [9., 1., 1.],
        [5., 5., 5.],
    ])

    # Long-range pairs
    idx_i_lr_tri = jnp.array([0, 1])
    idx_j_lr_tri = jnp.array([1, 0])
    num_lr_pairs_tri = 2
    cell_lr_tri = jnp.repeat(cell_matrix_triclinic[None], num_lr_pairs_tri, axis=0)
    cell_offset_lr_tri = jnp.array([[-1, 0, 0], [1, 0, 0]])

    # Short-range pairs (no offset)
    idx_i_sr_tri = jnp.array([0, 2])
    idx_j_sr_tri = jnp.array([2, 0])
    num_sr_pairs_tri = 2
    cell_sr_tri = jnp.repeat(cell_matrix_triclinic[None], num_sr_pairs_tri, axis=0)
    cell_offset_sr_tri = jnp.array([[0, 0, 0], [0, 0, 0]])

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
        atomic_numbers=atomic_numbers_pbc,
        idx_i=idx_i_sr_tri,
        idx_j=idx_j_sr_tri,
        cell=cell_sr_tri,
        cell_offset=cell_offset_sr_tri,
        idx_i_lr=idx_i_lr_tri,
        idx_j_lr=idx_j_lr_tri,
        cell_lr=cell_lr_tri,
        cell_offset_lr=cell_offset_lr_tri,
    )

    params = geometry_embed.init(jax.random.PRNGKey(0), inputs_triclinic)
    output = geometry_embed.apply(params, inputs_triclinic)

    # Calculate expected distances manually:
    # Raw r_ij for pair 0: [9-1, 1-1, 1-1] = [8, 0, 0]
    # Offset [-1, 0, 0] @ cell_matrix_triclinic = [-10, 0, 0]
    # Corrected r_ij: [8, 0, 0] + [-10, 0, 0] = [-2, 0, 0]
    # Distance: 2.0
    expected_d_lr_0 = 2.0

    # Raw r_ij for pair 1: [1-9, 1-1, 1-1] = [-8, 0, 0]
    # Offset [1, 0, 0] @ cell_matrix_triclinic = [10, 0, 0]
    # Corrected r_ij: [-8, 0, 0] + [10, 0, 0] = [2, 0, 0]
    # Distance: 2.0
    expected_d_lr_1 = 2.0

    npt.assert_allclose(
        output.get('d_ij_lr'),
        jnp.array([expected_d_lr_0, expected_d_lr_1]),
        atol=1e-6
    )
