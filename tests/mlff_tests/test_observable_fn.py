import numpy.testing as npt
import pytest
import jax
import jax.numpy as jnp
import jraph
from so3lr.mlff.geometric import get_rotation_matrix
from so3lr.mlff.nn.stacknet import StackNetSparse
from so3lr.mlff.nn.embed import GeometryEmbedSparse, AtomTypeEmbedSparse
from so3lr.mlff.nn.layer import SO3kratesLayerSparse
from so3lr.mlff.nn.observable import EnergySparse
from so3lr.mlff.nn.stacknet import get_energy_and_force_fn_sparse
from so3lr.mlff.utils import jraph_utils
from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr


def graph_to_input(graph: jraph.GraphsTuple, long_range: jraph.GraphsTuple = None):
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
        theory_mask=jnp.ones((1, 1)).astype(jnp.bool_),
        idx_i_lr=long_range.receivers if long_range is not None else jnp.array([]),
        idx_j_lr=long_range.senders if long_range is not None else jnp.array([]),
    )


def batched_graph_to_input(graph: jraph.GraphsTuple, long_range: jraph.GraphsTuple = None):
    inputs = dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=None,
        cell_offset=None,
        idx_i_lr=long_range.receivers if long_range is not None else jnp.array([]),
        idx_j_lr=long_range.senders if long_range is not None else jnp.array([]),
    )
    batch_info = jraph_utils.batch_info_fn(graph)
    inputs.update(
        dict(
            batch_segments=batch_info.get('batch_segments'),
            node_mask=batch_info.get('node_mask'),
            graph_mask=batch_info.get('graph_mask'),
            theory_mask=jnp.ones((1, 1)).astype(jnp.bool_)
        )
    )
    return inputs


graph1 = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array(
            [
                [0., 0., 0.],
                [0., -2., 0.],
                [1., 0.5, 0.]
            ]
        ),
        forces=jnp.ones((3, 3)),
        atomic_numbers=jnp.array([1, 2, 8])
    ),
    receivers=jnp.array([0, 0, 1, 2]),
    senders=jnp.array([1, 2, 0, 0]),
    globals=dict(energy=jnp.array([1])),
    edges=None,
    n_node=jnp.array([3]),
    n_edge=jnp.array([4]),
)
lr1 = jraph.GraphsTuple(
    nodes=None, edges=None,
    senders=jnp.array([], dtype=jnp.int64),
    receivers=jnp.array([], dtype=jnp.int64),
    n_node=jnp.array([3]),
    n_edge=jnp.array([0]),
    globals=None,
)

graph2 = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array(
            [
                [1., 1., 1.],
                [2., 2., 2.],
            ]
        ),
        atomic_numbers=jnp.array([2, 3]),
        forces=jnp.ones((2, 3))*2,
    ),
    receivers=jnp.array([0, 1]),
    senders=jnp.array([1, 0]),
    globals=dict(energy=jnp.array([2])),
    edges=None,
    n_node=jnp.array([2]),
    n_edge=jnp.array([2]),
)
lr2 = jraph.GraphsTuple(
    nodes=None, edges=None,
    senders=jnp.array([], dtype=jnp.int64),
    receivers=jnp.array([], dtype=jnp.int64),
    n_node=jnp.array([2]),
    n_edge=jnp.array([0]),
    globals=None,
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
lr3 = jraph.GraphsTuple(
    nodes=None, edges=None,
    senders=jnp.array([], dtype=jnp.int64),
    receivers=jnp.array([], dtype=jnp.int64),
    n_node=jnp.array([4]),
    n_edge=jnp.array([0]),
    globals=None,
)

# Define the StackNetSparse model.
num_features = 32
num_layers = 2

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
    output_is_zero_at_init=False
)

stacknet_sparse = StackNetSparse(
    geometry_embeddings=[geometry_embed],
    feature_embeddings=[atom_type_embed],
    layers=layers,
    observables=[energy],
    prop_keys=None
)

energy_and_force_fn = get_energy_and_force_fn_sparse(stacknet_sparse)
params = stacknet_sparse.init(jax.random.PRNGKey(0), graph_to_input(graph3, lr3))


def test_init():
    out = energy_and_force_fn(params, **graph_to_input(graph3, lr3))

    npt.assert_equal(out.get('energy').shape, (1,))
    npt.assert_equal(out.get('forces').shape, (4, 3))


def test_translation_symmetry():
    inputs = graph_to_input(graph3, lr3)

    inputs_translated = inputs.copy()
    inputs_translated['positions'] = inputs_translated['positions'] + jnp.array([2.5, 1.0, -0.7])[None]

    npt.assert_allclose(
        inputs_translated.get('positions') - inputs.get('positions'),
        jnp.array(
            [
                [2.5, 1.0, -0.7],
                [2.5, 1.0, -0.7],
                [2.5, 1.0, -0.7],
                [2.5, 1.0, -0.7]
            ]
        )
    )

    out = energy_and_force_fn(params, **inputs)
    out_translated = energy_and_force_fn(params, **inputs_translated)

    npt.assert_equal(out.get('energy').shape, (1,))
    npt.assert_equal(out_translated.get('energy').shape, (1,))
    npt.assert_allclose(out_translated.get('energy'), out.get('energy'), atol=1e-5)
    npt.assert_allclose(out_translated.get('forces'), out.get('forces'), atol=1e-5)


def test_rotation_symmetry():
    inputs = graph_to_input(graph3, lr3)

    rot = get_rotation_matrix(euler_axes='xyz', angles=[87, 14, 156], degrees=True)
    inputs_rotated = inputs.copy()
    inputs_rotated['positions'] = inputs_rotated['positions'] @ rot

    npt.assert_allclose(
        inputs_rotated.get('positions') @ rot.T,
        inputs.get('positions'),
        atol=1e-5
    )

    out = energy_and_force_fn(params, **inputs)
    out_rotated = energy_and_force_fn(params, **inputs_rotated)

    npt.assert_equal(out.get('energy').shape, (1,))
    npt.assert_equal(out_rotated.get('energy').shape, (1,))
    npt.assert_allclose(out_rotated.get('energy'), out.get('energy'), atol=1e-5)
    npt.assert_allclose(out_rotated.get('forces'), out.get('forces')@rot, atol=1e-5)
    with npt.assert_raises(AssertionError):
        npt.assert_allclose(out_rotated.get('forces'), out.get('forces'), atol=1e-5)


@pytest.mark.parametrize("max_num_graphs", [2, 3, 4])
def test_batching(max_num_graphs: int):
    max_num_nodes = 19
    max_num_edges = 25
    batched_graphs = dynamically_batch_with_lr(
        iter([(graph1, lr1), (graph2, lr2), (graph3, lr3)]),
        n_node=max_num_nodes,
        n_edge=max_num_edges,
        n_graph=max_num_graphs,
        n_pairs=100
    )

    inputs1 = graph_to_input(graph1, lr1)
    inputs2 = graph_to_input(graph2, lr2)
    inputs3 = graph_to_input(graph3, lr3)

    out1 = energy_and_force_fn(params, **inputs1)
    out2 = energy_and_force_fn(params, **inputs2)
    out3 = energy_and_force_fn(params, **inputs3)

    if max_num_graphs == 2:
        # First graph.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g, lr)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        out = energy_and_force_fn(params, **batched_input)

        npt.assert_allclose(out.get('energy')[1:], jnp.zeros((1,)))
        npt.assert_allclose(out1.get('energy'), out.get('energy')[:1], atol=1e-5)
        npt.assert_allclose(out.get('forces')[3:], jnp.zeros((16, 3)))
        npt.assert_allclose(out1.get('forces'), out.get('forces')[:3], atol=1e-5)

        # Make sure the forces are not trivially zero.
        npt.assert_equal(out.get('forces')[:3].shape, (3, 3))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out.get('forces')[:3], jnp.zeros((3, 3)))

        # Second graph.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g, lr)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        out = energy_and_force_fn(params, **batched_input)

        npt.assert_allclose(out.get('energy')[1:], jnp.zeros((1,)))
        npt.assert_allclose(out2.get('energy'), out.get('energy')[:1], atol=1e-5)
        npt.assert_allclose(out.get('forces')[2:], jnp.zeros((17, 3)))
        npt.assert_allclose(out2.get('forces'), out.get('forces')[:2], atol=1e-5)

        # Make sure the forces are not trivially zero.
        npt.assert_equal(out.get('forces')[:2].shape, (2, 3))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out.get('forces')[:2], jnp.zeros((2, 3)))

        # Third graph.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g, lr)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        out = energy_and_force_fn(params, **batched_input)

        npt.assert_allclose(out.get('energy')[1:], jnp.zeros((1,)))
        npt.assert_allclose(out3.get('energy'), out.get('energy')[:1], atol=1e-5)
        npt.assert_allclose(out.get('forces')[4:], jnp.zeros((15, 3)))
        npt.assert_allclose(out3.get('forces'), out.get('forces')[:4], atol=1e-5)

        # Make sure the forces are not trivially zero.
        npt.assert_equal(out.get('forces')[:4].shape, (4, 3))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out.get('forces')[:4], jnp.zeros((4, 3)))

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 3:
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g, lr)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        out = energy_and_force_fn(params, **batched_input)

        npt.assert_allclose(out.get('energy')[2:], jnp.zeros((1,)))
        npt.assert_allclose(out1.get('energy'), out.get('energy')[:1], atol=1e-5)
        npt.assert_allclose(out2.get('energy'), out.get('energy')[1:2], atol=1e-5)
        npt.assert_allclose(out.get('forces')[5:], jnp.zeros((14, 3)))
        npt.assert_allclose(out1.get('forces'), out.get('forces')[:3], atol=1e-5)
        npt.assert_allclose(out2.get('forces'), out.get('forces')[3:5], atol=1e-5)

        # Make sure the forces are not trivially zero.
        npt.assert_equal(out.get('forces')[:3].shape, (3, 3))
        npt.assert_equal(out.get('forces')[3:5].shape, (2, 3))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out.get('forces')[:3], jnp.zeros((3, 3)))
            npt.assert_allclose(out.get('forces')[3:5], jnp.zeros((2, 3)))

        # Second graph batch.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g, lr)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        out = energy_and_force_fn(params, **batched_input)

        npt.assert_allclose(out.get('energy')[1:], jnp.zeros((2,)))
        npt.assert_allclose(out3.get('energy'), out.get('energy')[:1], atol=1e-5)
        npt.assert_allclose(out.get('forces')[4:], jnp.zeros((15, 3)))
        npt.assert_allclose(out3.get('forces'), out.get('forces')[:4], atol=1e-5)

        # Make sure the forces are not trivially zero.
        npt.assert_equal(out.get('forces')[:4].shape, (4, 3))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out.get('forces')[:4], jnp.zeros((4, 3)))

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 4:
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g, lr)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        out = energy_and_force_fn(params, **batched_input)

        npt.assert_allclose(out.get('energy')[3:], jnp.zeros((1,)))
        npt.assert_allclose(out1.get('energy'), out.get('energy')[:1], atol=1e-5)
        npt.assert_allclose(out2.get('energy'), out.get('energy')[1:2], atol=1e-5)
        npt.assert_allclose(out3.get('energy'), out.get('energy')[2:3], atol=1e-5)

        npt.assert_allclose(out.get('forces')[9:], jnp.zeros((10, 3)))
        npt.assert_allclose(out1.get('forces'), out.get('forces')[:3], atol=1e-5)
        npt.assert_allclose(out2.get('forces'), out.get('forces')[3:5], atol=1e-5)
        npt.assert_allclose(out3.get('forces'), out.get('forces')[5:9], atol=1e-5)

        # Make sure the forces are not trivially zero.
        npt.assert_equal(out.get('forces')[:3].shape, (3, 3))
        npt.assert_equal(out.get('forces')[3:5].shape, (2, 3))
        npt.assert_equal(out.get('forces')[5:9].shape, (4, 3))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out.get('forces')[:3], jnp.zeros((3, 3)))
            npt.assert_allclose(out.get('forces')[3:5], jnp.zeros((2, 3)))
            npt.assert_allclose(out.get('forces')[5:9], jnp.zeros((4, 3)))

        # Check that tests fail for corrupted batch.
        batched_input['batch_segments'] = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
        out_corrupted = stacknet_sparse.apply(params, batched_input)
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(out1.get('energy'), out_corrupted.get('energy')[:1], atol=1e-5)
            npt.assert_allclose(out2.get('energy'), out_corrupted.get('energy')[1:2], atol=1e-5)
            npt.assert_allclose(out3.get('energy'), out_corrupted.get('energy')[2:3], atol=1e-5)

            npt.assert_allclose(out1.get('forces'), out_corrupted.get('forces')[:3], atol=1e-5)
            npt.assert_allclose(out2.get('forces'), out_corrupted.get('forces')[3:5], atol=1e-5)
            npt.assert_allclose(out3.get('forces'), out_corrupted.get('forces')[5:9], atol=1e-5)

        # There is only a single batch.
        with npt.assert_raises(StopIteration):
            next(batched_graphs)


