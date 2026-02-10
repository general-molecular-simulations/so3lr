import numpy.testing as npt

import jax
import jax.numpy as jnp
import jraph

from so3lr.mlff.nn.embed import SpinEmbedSparse, ChargeEmbedSparse
from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr
from so3lr.mlff.utils import jraph_utils

import pytest


def graph_to_input(graph: jraph.GraphsTuple):
    return dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        total_charge=graph.globals.get('total_charge'),
        num_unpaired_electrons=graph.globals.get('num_unpaired_electrons'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=None,
        cell_offset=None,
        batch_segments=jnp.zeros((graph.n_node.item(),), dtype=jnp.int32),
        graph_mask=jnp.ones((1,)).astype(jnp.bool_),
        node_mask=jnp.ones((graph.n_node.item(),)).astype(jnp.bool_)
    )


def batched_graph_to_input(graph: jraph.GraphsTuple):
    inputs = dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        total_charge=graph.globals.get('total_charge'),
        num_unpaired_electrons=graph.globals.get('num_unpaired_electrons'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=None,
        cell_offset=None,
    )
    batch_info = jraph_utils.batch_info_fn(graph)
    inputs.update(batch_info)
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
    globals=dict(
        energy=jnp.array([1]),
        total_charge=jnp.array([0]),
        num_unpaired_electrons=jnp.array([0])
    ),
    edges=None,
    n_node=jnp.array([3]),
    n_edge=jnp.array([4])
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
    globals=dict(
        energy=jnp.array([2]),
        total_charge=jnp.array([1]),
        num_unpaired_electrons=jnp.array([1])
    ),
    edges=None,
    n_node=jnp.array([2]),
    n_edge=jnp.array([2])
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
    globals=dict(
        energy=jnp.array([3]),
        total_charge=jnp.array([-1]),
        num_unpaired_electrons=jnp.array([2])
    ),
    edges=None,
    n_node=jnp.array([4]),
    n_edge=jnp.array([6])
)
lr3 = jraph.GraphsTuple(
    nodes=None, edges=None,
    senders=jnp.array([], dtype=jnp.int64),
    receivers=jnp.array([], dtype=jnp.int64),
    n_node=jnp.array([4]),
    n_edge=jnp.array([0]),
    globals=None,
)

num_features = 64
num_layers = 2


def test_init():
    charge_embed = ChargeEmbedSparse(
        num_features=num_features,
        prop_keys=None
    )
    spin_embed = SpinEmbedSparse(
        num_features=num_features,
        prop_keys=None
    )

    # Q=0, S=0
    _ = charge_embed.init(jax.random.PRNGKey(0), graph_to_input(graph1))
    _ = spin_embed.init(jax.random.PRNGKey(0), graph_to_input(graph1))

    # Q=1, S=1
    _ = charge_embed.init(jax.random.PRNGKey(0), graph_to_input(graph2))
    _ = spin_embed.init(jax.random.PRNGKey(0), graph_to_input(graph2))

    # Q=-1, S=2
    _ = charge_embed.init(jax.random.PRNGKey(0), graph_to_input(graph3))
    _ = spin_embed.init(jax.random.PRNGKey(0), graph_to_input(graph3))


def test_apply():
    charge_embed = ChargeEmbedSparse(
        num_features=num_features,
        activation_fn='identity',
        prop_keys=None
    )
    spin_embed = SpinEmbedSparse(
        num_features=num_features,
        activation_fn='identity',
        prop_keys=None
    )

    params_Q = charge_embed.init(jax.random.PRNGKey(0), graph_to_input(graph1))
    params_S = spin_embed.init(jax.random.PRNGKey(0), graph_to_input(graph1))

    # Q=0, S=0
    out_Q = charge_embed.apply(params_Q, graph_to_input(graph1))
    out_S = spin_embed.apply(params_S, graph_to_input(graph1))

    npt.assert_equal(out_Q.shape, (3, num_features))
    npt.assert_equal(out_S.shape, (3, num_features))

    npt.assert_allclose(out_Q, jnp.zeros((3, num_features)))
    npt.assert_allclose(out_S, jnp.zeros((3, num_features)))

    # Q=1, S=1
    out_Q = charge_embed.apply(params_Q, graph_to_input(graph2))
    out_S = spin_embed.apply(params_S, graph_to_input(graph2))

    npt.assert_equal(out_Q.shape, (2, num_features))
    npt.assert_equal(out_S.shape, (2, num_features))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(out_Q, jnp.zeros((2, num_features)))
        npt.assert_allclose(out_S, jnp.zeros((2, num_features)))

    # Q=-1, S=2
    out_Q = charge_embed.apply(params_Q, graph_to_input(graph3))
    out_S = spin_embed.apply(params_S, graph_to_input(graph3))

    npt.assert_equal(out_Q.shape, (4, num_features))
    npt.assert_equal(out_S.shape, (4, num_features))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(out_Q, jnp.zeros((4, num_features)))
        npt.assert_allclose(out_S, jnp.zeros((4, num_features)))

    # For no non-linear activation, embeddings are linear in the total_charge and number of unpaired electrons (for
    # the same structure).
    inputs_doubled = graph_to_input(graph3)
    inputs_doubled['total_charge'] = inputs_doubled['total_charge'] * 2
    inputs_doubled['num_unpaired_electrons'] = inputs_doubled['num_unpaired_electrons'] * 2

    out_Q2 = charge_embed.apply(params_Q, inputs_doubled)
    out_S2 = spin_embed.apply(params_S, inputs_doubled)

    npt.assert_allclose(2*out_Q, out_Q2)
    npt.assert_allclose(2*out_S, out_S2)


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

    charge_embed = ChargeEmbedSparse(
        num_features=num_features,
        activation_fn='identity',
        prop_keys=None
    )
    spin_embed = SpinEmbedSparse(
        num_features=num_features,
        activation_fn='identity',
        prop_keys=None
    )

    params_Q = charge_embed.init(jax.random.PRNGKey(0), graph_to_input(graph1))
    params_S = spin_embed.init(jax.random.PRNGKey(0), graph_to_input(graph1))

    inputs1 = graph_to_input(graph1)
    inputs2 = graph_to_input(graph2)
    inputs3 = graph_to_input(graph3)

    outQ1 = charge_embed.apply(params_Q, inputs1)
    outQ2 = charge_embed.apply(params_Q, inputs2)
    outQ3 = charge_embed.apply(params_Q, inputs3)

    outS1 = spin_embed.apply(params_S, inputs1)
    outS2 = spin_embed.apply(params_S, inputs2)
    outS3 = spin_embed.apply(params_S, inputs3)

    if max_num_graphs == 2:
        # First graph.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        outQ = charge_embed.apply(params_Q, batched_input)
        outS = spin_embed.apply(params_S, batched_input)

        npt.assert_allclose(outQ1, outQ[:3], atol=1e-5)
        npt.assert_allclose(outS1, outS[:3], atol=1e-5)

        # Second graph.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        outQ = charge_embed.apply(params_Q, batched_input)
        outS = spin_embed.apply(params_S, batched_input)

        npt.assert_allclose(outQ2, outQ[:2], atol=1e-5)
        npt.assert_allclose(outS2, outS[:2], atol=1e-5)

        # Third graph.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        outQ = charge_embed.apply(params_Q, batched_input)
        outS = spin_embed.apply(params_S, batched_input)

        npt.assert_allclose(outQ3, outQ[:4], atol=1e-5)
        npt.assert_allclose(outS3, outS[:4], atol=1e-5)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 3:
        # First graph batch.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        outQ = charge_embed.apply(params_Q, batched_input)
        outS = spin_embed.apply(params_S, batched_input)

        npt.assert_allclose(outQ1, outQ[:3], atol=1e-5)
        npt.assert_allclose(outS1, outS[:3], atol=1e-5)

        npt.assert_allclose(outQ2, outQ[3:5], atol=1e-5)
        npt.assert_allclose(outS2, outS[3:5], atol=1e-5)

        # Second graph batch.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        outQ = charge_embed.apply(params_Q, batched_input)
        outS = spin_embed.apply(params_S, batched_input)

        npt.assert_allclose(outQ3, outQ[:4], atol=1e-5)
        npt.assert_allclose(outS3, outS[:4], atol=1e-5)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 4:
        # First graph batch.
        g, lr = next(batched_graphs)
        batched_input = batched_graph_to_input(g)
        batched_input = jax.tree_util.tree_map(jnp.array, batched_input)
        outQ = charge_embed.apply(params_Q, batched_input)
        outS = spin_embed.apply(params_S, batched_input)

        npt.assert_allclose(outQ1, outQ[:3], atol=1e-5)
        npt.assert_allclose(outS1, outS[:3], atol=1e-5)
        npt.assert_allclose(outQ2, outQ[3:5], atol=1e-5)
        npt.assert_allclose(outS2, outS[3:5], atol=1e-5)
        npt.assert_allclose(outQ3, outQ[5:9], atol=1e-5)
        npt.assert_allclose(outS3, outS[5:9], atol=1e-5)

        # Check that tests fail for corrupted batch.
        batched_input['batch_segments'] = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
        outQ_corrupted = charge_embed.apply(params_Q, batched_input)
        outS_corrupted = spin_embed.apply(params_S, batched_input)
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(outQ1, outQ_corrupted[:3], atol=1e-5)
            npt.assert_allclose(outQ2, outQ_corrupted[3:5], atol=1e-5)
            npt.assert_allclose(outQ3, outQ_corrupted[5:9], atol=1e-5)

            npt.assert_allclose(outS1, outS_corrupted[:3], atol=1e-5)
            npt.assert_allclose(outS2, outS_corrupted[3:5], atol=1e-5)
            npt.assert_allclose(outS3, outS_corrupted[5:9], atol=1e-5)

        # There is only a single batch.
        with npt.assert_raises(StopIteration):
            next(batched_graphs)
