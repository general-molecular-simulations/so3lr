import pytest

import jax
import jax.numpy as jnp

import numpy.testing as npt

from so3lr.mlff.nn.layer import SO3kratesLayerSparse
from so3lr.mlff.nn import GeometryEmbedSparse

rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 2)

num_nodes = 7
num_edges = 10
num_features = 64

x = jnp.ones((num_nodes, num_features))
ev = jnp.ones((num_nodes, 8))
rbf_ij = jnp.ones((num_edges, 15))
ylm_ij = jnp.ones((num_edges, 8))
cut = jnp.ones((num_edges,))
idx_i = jax.random.randint(rng1, shape=(num_edges,), minval=0, maxval=num_nodes)
idx_j = jax.random.randint(rng2, shape=(num_edges,), minval=0, maxval=num_nodes)


def test_init():
    so3krates_layer = SO3kratesLayerSparse(
        degrees=[1, 2],
        use_spherical_filter=True,
        num_heads=2,
        num_features_head=16,
        qk_non_linearity=jax.nn.softplus,
        residual_mlp_1=True,
        residual_mlp_2=True,
        layer_normalization_1=True,
        layer_normalization_2=True,
        activation_fn=jax.nn.softplus,
        behave_like_identity_fn_at_init=True
    )

    _ = so3krates_layer.init(
        jax.random.PRNGKey(0),
        x=x,
        ev=ev,
        rbf_ij=rbf_ij,
        ylm_ij=ylm_ij,
        cut=cut,
        idx_i=idx_i,
        idx_j=idx_j,
    )


@pytest.mark.parametrize("behave_like_identity_fn_at_init", [True, False])
def test_apply(behave_like_identity_fn_at_init):
    so3krates_layer = SO3kratesLayerSparse(
        degrees=[1, 2],
        use_spherical_filter=True,
        num_heads=2,
        num_features_head=16,
        qk_non_linearity=jax.nn.softplus,
        residual_mlp_1=True,
        residual_mlp_2=True,
        layer_normalization_1=False,
        layer_normalization_2=False,
        activation_fn=jax.nn.softplus,
        behave_like_identity_fn_at_init=behave_like_identity_fn_at_init
    )

    params = so3krates_layer.init(
        jax.random.PRNGKey(0),
        x=x,
        ev=ev,
        rbf_ij=rbf_ij,
        ylm_ij=ylm_ij,
        cut=cut,
        idx_i=idx_i,
        idx_j=idx_j,
    )

    output = so3krates_layer.apply(
        params,
        x=x,
        ev=ev,
        rbf_ij=rbf_ij,
        ylm_ij=ylm_ij,
        cut=cut,
        idx_i=idx_i,
        idx_j=idx_j
    )

    npt.assert_equal(output.get('x').shape, x.shape)
    npt.assert_equal(output.get('ev').shape, ev.shape)

    if behave_like_identity_fn_at_init:
        npt.assert_allclose(output.get('x'), x)
    else:
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(output.get('x'), x)


def test_translation_invariance():
    x = jnp.ones((3, 64))
    positions = jnp.array([
        [0., 0., 0.],
        [0., -2., 0.],
        [1., 0.5, 0.],
    ])
    idx_i = jnp.array([0, 0, 1, 2])
    idx_j = jnp.array([1, 2, 0, 0])

    geometry_embed = GeometryEmbedSparse(degrees=[1, 2],
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='exponential',
                                         cutoff=2.5,
                                         input_convention='positions',
                                         prop_keys=None)

    geometry_embed_inputs = dict(
        positions=positions,
        idx_i=idx_i,
        idx_j=idx_j,
        cell=None,
        cell_offset=None
    )

    geometry_embed_inputs_translated = dict(
        positions=positions + jnp.array([2.5, 1.0, -0.7])[None],
        idx_i=idx_i,
        idx_j=idx_j,
        cell=None,
        cell_offset=None
    )

    params = geometry_embed.init(
        jax.random.PRNGKey(0),
        geometry_embed_inputs
    )

    geometry_embed_output = geometry_embed.apply(
        params,
        geometry_embed_inputs
    )

    geometry_embed_output_translated = geometry_embed.apply(
        params,
        geometry_embed_inputs_translated
    )

    so3krates_layer = SO3kratesLayerSparse(
        degrees=[1, 2],
        use_spherical_filter=True,
        num_heads=2,
        num_features_head=16,
        qk_non_linearity=jax.nn.softplus,
        residual_mlp_1=True,
        residual_mlp_2=True,
        layer_normalization_1=False,
        layer_normalization_2=False,
        activation_fn=jax.nn.softplus,
        behave_like_identity_fn_at_init=False
    )

    ev = jax.ops.segment_sum(
        geometry_embed_output.get('ylm_ij'),
        segment_ids=idx_i,
        num_segments=len(x)
    )

    ev_translated = jax.ops.segment_sum(
        geometry_embed_output_translated.get('ylm_ij'),
        segment_ids=idx_i,
        num_segments=len(x)
    )

    so3k_params = so3krates_layer.init(
        jax.random.PRNGKey(0),
        x=x,
        ev=ev,
        rbf_ij=geometry_embed_output.get('rbf_ij'),
        ylm_ij=geometry_embed_output.get('ylm_ij'),
        cut=geometry_embed_output.get('cut'),
        idx_i=idx_i,
        idx_j=idx_j,
    )

    output = so3krates_layer.apply(
        so3k_params,
        x=x,
        ev=ev,
        rbf_ij=geometry_embed_output.get('rbf_ij'),
        ylm_ij=geometry_embed_output.get('ylm_ij'),
        cut=geometry_embed_output.get('cut'),
        idx_i=idx_i,
        idx_j=idx_j,
    )

    output_translated = so3krates_layer.apply(
        so3k_params,
        x=x,
        ev=ev_translated,
        rbf_ij=geometry_embed_output_translated.get('rbf_ij'),
        ylm_ij=geometry_embed_output_translated.get('ylm_ij'),
        cut=geometry_embed_output_translated.get('cut'),
        idx_i=idx_i,
        idx_j=idx_j,
    )

    npt.assert_allclose(output_translated.get('x'), output.get('x'))
    npt.assert_allclose(output_translated.get('ev'), output.get('ev'))


def test_rotation_equivariance():
    from so3lr.mlff.geometric import get_rotation_matrix

    x = jnp.ones((3, 64))
    positions = jnp.array([
        [0., 0., 0.],
        [0., -2., 0.],
        [1., 0.5, 0.],
    ])
    idx_i = jnp.array([0, 0, 1, 2])
    idx_j = jnp.array([1, 2, 0, 0])

    geometry_embed = GeometryEmbedSparse(degrees=[1, 2],
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='exponential',
                                         cutoff=2.5,
                                         input_convention='positions',
                                         prop_keys=None)

    geometry_embed_inputs = dict(
        positions=positions,
        idx_i=idx_i,
        idx_j=idx_j,
        cell=None,
        cell_offset=None
    )
    rot = get_rotation_matrix(euler_axes='xyz', angles=[87, 14, 156], degrees=True)
    geometry_embed_inputs_rotated = dict(
        positions=positions@rot,
        idx_i=idx_i,
        idx_j=idx_j,
        cell=None,
        cell_offset=None
    )

    params = geometry_embed.init(
        jax.random.PRNGKey(0),
        geometry_embed_inputs
    )

    geometry_embed_output = geometry_embed.apply(
        params,
        geometry_embed_inputs
    )

    geometry_embed_output_rotated = geometry_embed.apply(
        params,
        geometry_embed_inputs_rotated
    )

    so3krates_layer = SO3kratesLayerSparse(
        degrees=[1, 2],
        use_spherical_filter=True,
        num_heads=2,
        num_features_head=16,
        qk_non_linearity=jax.nn.softplus,
        residual_mlp_1=True,
        residual_mlp_2=True,
        layer_normalization_1=False,
        layer_normalization_2=False,
        activation_fn=jax.nn.softplus,
        behave_like_identity_fn_at_init=False
    )

    ev = jax.ops.segment_sum(
        geometry_embed_output.get('ylm_ij'),
        segment_ids=idx_i,
        num_segments=len(x)
    )

    ev_rotated = jax.ops.segment_sum(
        geometry_embed_output_rotated.get('ylm_ij'),
        segment_ids=idx_i,
        num_segments=len(x)
    )

    so3k_params = so3krates_layer.init(
        jax.random.PRNGKey(0),
        x=x,
        ev=ev,
        rbf_ij=geometry_embed_output.get('rbf_ij'),
        ylm_ij=geometry_embed_output.get('ylm_ij'),
        cut=geometry_embed_output.get('cut'),
        idx_i=idx_i,
        idx_j=idx_j,
    )

    output = so3krates_layer.apply(
        so3k_params,
        x=x,
        ev=ev,
        rbf_ij=geometry_embed_output.get('rbf_ij'),
        ylm_ij=geometry_embed_output.get('ylm_ij'),
        cut=geometry_embed_output.get('cut'),
        idx_i=idx_i,
        idx_j=idx_j,
    )

    output_rotated = so3krates_layer.apply(
        so3k_params,
        x=x,
        ev=ev_rotated,
        rbf_ij=geometry_embed_output_rotated.get('rbf_ij'),
        ylm_ij=geometry_embed_output_rotated.get('ylm_ij'),
        cut=geometry_embed_output_rotated.get('cut'),
        idx_i=idx_i,
        idx_j=idx_j,
    )

    npt.assert_allclose(output_rotated.get('x'), output.get('x'), atol=1e-5)
    with npt.assert_raises(AssertionError):
        npt.assert_allclose(output_rotated.get('ev'), output.get('ev'), atol=1e-5)

    P = jnp.array([2, 0, 1])
    npt.assert_allclose(
        output_rotated.get('ev')[:, :3][:, P]@rot.T,
        output.get('ev')[:, :3][:, P],
        atol=1e-5
    )
