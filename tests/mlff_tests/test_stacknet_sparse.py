import numpy.testing as npt

import jax
import jax.numpy as jnp
import jraph

from so3lr.mlff.nn.stacknet import StackNetSparse
from so3lr.mlff.nn.embed import GeometryEmbedSparse, AtomTypeEmbedSparse
from so3lr.mlff.nn.layer import SO3kratesLayerSparse
from so3lr.mlff.nn.observable import EnergySparse

import pytest

# Symmetry and batching of this toy network (energy and forces) are covered by
# test_observable_fn.py, which builds the same StackNetSparse architecture (at 32
# features instead of 64).


def graph_to_input(graph: jraph.GraphsTuple):
    return dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=None,
        cell_offset=None,
        batch_segments=jnp.zeros((graph.n_node.item(),), dtype=jnp.int32),
        graph_mask=jnp.ones((1,)).astype(jnp.bool_),
        node_mask=jnp.ones((graph.n_node.item(),)).astype(jnp.bool_),
        theory_mask=jnp.ones((1, 1)).astype(jnp.bool_)
    )


graph3 = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array(
            [
                [0., 1., 2.],
                [1., 1., 1.],
                [2., 2., 2.],
                [0., 0., 0.],
            ]
        ),
        atomic_numbers=jnp.array([1, 2, 3, 4]),
        forces=jnp.ones((4, 3))*3,
    ),
    receivers=jnp.array([0, 0, 1, 1, 2, 3]),
    senders=jnp.array([1, 3, 0, 2, 1, 0]),
    globals=dict(energy=jnp.array([3])),
    edges=None,
    n_node=jnp.array([4]),
    n_edge=jnp.array([6]),
)

num_features = 64
num_layers = 2


@pytest.mark.parametrize("output_is_zero_at_init", [True, False])
def test_apply(output_is_zero_at_init: bool):
    atom_type_embed = AtomTypeEmbedSparse(
        num_features=num_features,
        prop_keys=None
    )
    geometry_embed = GeometryEmbedSparse(
        degrees=[1, 2],
        radial_basis_fn='bernstein',
        num_radial_basis_fn=16,
        cutoff_fn='exponential',
        cutoff=2.5,
        input_convention='positions',
        prop_keys=None
    )

    layers = [SO3kratesLayerSparse(
        degrees=[1, 2],
        use_spherical_filter=i > 0,
        num_heads=2,
        num_features_head=8,
        qk_non_linearity=jax.nn.softplus,
        residual_mlp_1=True,
        residual_mlp_2=True,
        layer_normalization_1=False,
        layer_normalization_2=False,
        activation_fn=jax.nn.softplus,
        behave_like_identity_fn_at_init=False
    ) for i in range(num_layers)]

    energy = EnergySparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init
    )

    stacknet_sparse = StackNetSparse(
        geometry_embeddings=[geometry_embed],
        feature_embeddings=[atom_type_embed],
        layers=layers,
        observables=[energy],
        prop_keys=None
    )

    params = stacknet_sparse.init(jax.random.PRNGKey(0), graph_to_input(graph3))

    out = stacknet_sparse.apply(params, graph_to_input(graph3))

    npt.assert_equal(out.get('energy').shape, (1, ))

    if output_is_zero_at_init:
        npt.assert_allclose(out.get('energy'), jnp.zeros(1, ))
    else:
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out.get('energy'), jnp.zeros(1, ))
