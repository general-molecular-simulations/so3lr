import pytest

import jax.numpy as jnp
import numpy.testing as npt

import jraph

from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr
from so3lr.mlff.utils import jraph_utils

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
    senders=jnp.array([], dtype=jnp.int32),
    receivers=jnp.array([], dtype=jnp.int32),
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
    senders=jnp.array([], dtype=jnp.int32),
    receivers=jnp.array([], dtype=jnp.int32),
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
    senders=jnp.array([], dtype=jnp.int32),
    receivers=jnp.array([], dtype=jnp.int32),
    n_node=jnp.array([4]),
    n_edge=jnp.array([0]),
    globals=None,
)


@pytest.mark.parametrize("max_num_graphs", [2, 3, 4])
def test_batch_info_fn(max_num_graphs):
    max_num_nodes = 11
    max_num_edges = 15
    batched_graphs = dynamically_batch_with_lr(
        iter([(graph1, lr1), (graph2, lr2), (graph3, lr3)]),
        n_node=max_num_nodes,
        n_edge=max_num_edges,
        n_graph=max_num_graphs,
        n_pairs=100,
    )

    if max_num_graphs == 2:
        g, lr = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        g, lr = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        g, lr = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 3:
        g, lr = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 2)

        g, lr = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 4:
        g, lr = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 3)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)


# Test graphs with actual long-range neighbors
graph_with_lr = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        forces=jnp.ones((3, 3)),
    ),
    receivers=jnp.array([0, 1]),
    senders=jnp.array([1, 2]),
    globals=dict(energy=jnp.array([1])),
    edges=None,
    n_node=jnp.array([3]),
    n_edge=jnp.array([2]),
)

# Long-range graph with actual neighbors (all pairs beyond short-range cutoff)
lr_with_neighbors = jraph.GraphsTuple(
    nodes=None, edges=None,
    senders=jnp.array([0, 1, 2]),  # 3 long-range pairs
    receivers=jnp.array([2, 0, 1]),
    n_node=jnp.array([3]),
    n_edge=jnp.array([3]),  # 3 pairs
    globals=None,
)


def test_lr_batching_preserves_indices():
    """Test that long-range indices are correctly batched and padded."""
    batched = list(dynamically_batch_with_lr(
        iter([(graph_with_lr, lr_with_neighbors)]),
        n_node=10,
        n_edge=10,
        n_graph=2,
        n_pairs=10,
    ))
    
    assert len(batched) == 1
    g, lr = batched[0]
    
    # Long-range graph should have the original 3 pairs
    # Check that the receivers/senders are preserved (first 3 values)
    npt.assert_array_equal(lr.receivers[:3], jnp.array([2, 0, 1]))
    npt.assert_array_equal(lr.senders[:3], jnp.array([0, 1, 2]))
    npt.assert_equal(lr.n_edge[0], 3)  # First graph has 3 pairs


def test_lr_n_pairs_budget_constraint():
    """Test that n_pairs budget constraint is respected."""
    # Create multiple graphs that would fit in one batch by n_node/n_edge
    # but should be split due to n_pairs constraint
    data = [(graph_with_lr, lr_with_neighbors) for _ in range(3)]
    
    # n_pairs=5 means only 1 graph (with 3 pairs) + padding fits
    batches = list(dynamically_batch_with_lr(
        iter(data),
        n_node=20,  # Enough for all graphs
        n_edge=20,  # Enough for all graphs
        n_graph=4,  # Enough for all graphs
        n_pairs=5,  # Only allows 1 graph (3 pairs) + some padding
    ))
    
    # Should have 3 batches, one graph per batch, due to n_pairs constraint
    assert len(batches) == 3
    
    for g, lr in batches:
        # Each batch should have exactly 1 non-padding graph
        num_graphs = jnp.sum(jraph.get_graph_padding_mask(g))
        npt.assert_equal(num_graphs.item(), 1)


def test_lr_multiple_graphs_batch():
    """Test batching multiple graphs with long-range neighbors."""
    graph2_with_lr = jraph.GraphsTuple(
        nodes=dict(
            positions=jnp.array([[3., 0., 0.], [4., 0., 0.]]),
            forces=jnp.ones((2, 3)) * 2,
        ),
        receivers=jnp.array([0]),
        senders=jnp.array([1]),
        globals=dict(energy=jnp.array([2])),
        edges=None,
        n_node=jnp.array([2]),
        n_edge=jnp.array([1]),
    )
    lr2 = jraph.GraphsTuple(
        nodes=None, edges=None,
        senders=jnp.array([0, 1]),  # 2 pairs
        receivers=jnp.array([1, 0]),
        n_node=jnp.array([2]),
        n_edge=jnp.array([2]),
        globals=None,
    )
    
    batches = list(dynamically_batch_with_lr(
        iter([(graph_with_lr, lr_with_neighbors), (graph2_with_lr, lr2)]),
        n_node=10,
        n_edge=10,
        n_graph=3,
        n_pairs=10,  # Enough for both: 3 + 2 = 5 pairs
    ))
    
    assert len(batches) == 1
    g, lr = batches[0]
    
    # Should have 2 non-padding graphs
    num_graphs = jnp.sum(jraph.get_graph_padding_mask(g))
    npt.assert_equal(num_graphs.item(), 2)
    
    # Long-range graph should have 5 pairs total (3 + 2)
    total_lr_pairs = sum(lr.n_edge[:2])  # First two graphs
    npt.assert_equal(total_lr_pairs, 5)


def test_lr_empty_raises_on_single_graph_too_large():
    """Test that a single graph that doesn't fit raises RuntimeError."""
    # n_pairs=2 but lr_with_neighbors has 3 pairs - single graph won't fit
    with npt.assert_raises(RuntimeError):
        list(dynamically_batch_with_lr(
            iter([(graph_with_lr, lr_with_neighbors)]),
            n_node=10,
            n_edge=10,
            n_graph=2,
            n_pairs=2,  # Too small - single graph has 3 pairs
        ))

