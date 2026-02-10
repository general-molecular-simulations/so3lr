import jax
import jax.numpy as jnp
import jraph
import numpy as np



def batch_info_fn(batched_graph: jraph.GraphsTuple):
    """
    Collect batching information from batched `jraph.GraphsTuple`.
    Args:
        batched_graph (jraph.GraphsTuple): Batched `jraph.GraphsTuple`

    Returns: Dictionary with `node_mask`, `graph_mask` and `batch_segments`.

    Raises:
        RuntimeError: If a non-batched `jraph.GraphsTuple` is passed as input.

    """
    node_mask = jraph.get_node_padding_mask(batched_graph)
    graph_mask = jraph.get_graph_padding_mask(batched_graph)

    num_of_non_padded_graphs = len(graph_mask) - jraph.get_number_of_padding_with_graphs_graphs(batched_graph)
    if len(graph_mask) == 1:
        raise RuntimeError('Only batched `jraph.GraphsTuple` should be passed to `batch_info_fn`.')

    batch_segments = jnp.repeat(
        jnp.arange(len(graph_mask)),
        repeats=batched_graph.n_node,
        total_repeat_length=len(node_mask)
    )

    return dict(
        node_mask=node_mask,
        graph_mask=graph_mask,
        batch_segments=batch_segments,
        num_of_non_padded_graphs=num_of_non_padded_graphs,
    )


@jax.jit
def graph_to_batch_fn(graph: jraph.GraphsTuple, long_range: jraph.GraphsTuple):
    batch = dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        num_unpaired_electrons=graph.globals.get('num_unpaired_electrons'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=graph.edges.get('cell'),
        cell_offset=graph.edges.get('cell_offset'),
        energy=graph.globals.get('energy'),
        forces=graph.nodes.get('forces'),
        stress=graph.globals.get('stress'),
        total_charge=graph.globals.get('total_charge'),
        dipole_vec=graph.globals.get('dipole_vec'),
        hirshfeld_ratios=graph.nodes.get('hirshfeld_ratios'),
        idx_i_lr=long_range.receivers,
        idx_j_lr=long_range.senders,
        cell_lr=long_range.edges.get('cell') if long_range.edges is not None else None,
        cell_offset_lr=long_range.edges.get('cell_offsets') if long_range.edges is not None else None,
        theory_level=graph.globals.get('theory_level'),
        theory_mask=graph.globals.get('theory_mask'),
        residue_charge=graph.globals.get('residue_charge'),
        residue_segments=graph.globals.get('residue_segments'),
    )
    batch_info = batch_info_fn(graph)
    batch.update(batch_info)
    return batch


def graph_to_inputs_fn(graph: jraph.GraphsTuple, long_range: jraph.GraphsTuple):
    """
    Wrapper around graph_to_batch_fn.

    Args:
        graph: Batched jraph.GraphsTuple.
        long_range: Batched jraph.GraphsTuple for long-range neighbors.

    Returns:
        Dictionary representation of the graph.

    """
    return graph_to_batch_fn(graph, long_range)


def dynamically_batch_with_lr(
    data_iterator,
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_pairs: int,
):
    """Batch graphs + long-range graphs respecting both budgets.

    This wraps jraph's batching logic to simultaneously track both
    short-range (n_node/n_edge) and long-range (n_pairs) budgets when
    deciding how many graphs fit in one batch.

    Args:
        data_iterator: Iterator yielding (graph, long_range_graph) tuples,
            where both are jraph.GraphsTuple objects.
        n_node: Maximum number of nodes in a batch (including padding graph).
        n_edge: Maximum number of edges in a batch.
        n_graph: Maximum number of graphs in a batch (including padding graph).
        n_pairs: Maximum number of long-range pairs in a batch.

    Yields:
        (padded_graph, padded_long_range_graph) tuples with fixed sizes.
    """
    accumulated = []
    acc_nodes = 0
    acc_edges = 0
    acc_graphs = 0
    acc_pairs = 0

    for graph, lr in data_iterator:
        elem_nodes = int(np.sum(graph.n_node))
        elem_edges = int(np.sum(graph.n_edge))
        elem_pairs = int(np.sum(lr.n_edge))  # n_edge in lr graph = n_pairs
        elem_graphs = len(graph.n_node)

        if not accumulated:
            accumulated = [(graph, lr)]
            acc_nodes = elem_nodes
            acc_edges = elem_edges
            acc_graphs = elem_graphs
            acc_pairs = elem_pairs
            continue

        # Check if adding this element would exceed any budget.
        if (
            acc_graphs + elem_graphs > n_graph - 1
            or acc_nodes + elem_nodes > n_node - 1
            or acc_edges + elem_edges > n_edge
            or acc_pairs + elem_pairs > n_pairs
        ):
            # Yield current batch.
            yield _finalize_batch(
                accumulated, n_node, n_edge, n_graph, n_pairs
            )
            accumulated = [(graph, lr)]
            acc_nodes = elem_nodes
            acc_edges = elem_edges
            acc_graphs = elem_graphs
            acc_pairs = elem_pairs
        else:
            accumulated.append((graph, lr))
            acc_nodes += elem_nodes
            acc_edges += elem_edges
            acc_graphs += elem_graphs
            acc_pairs += elem_pairs

    if accumulated:
        yield _finalize_batch(accumulated, n_node, n_edge, n_graph, n_pairs)


def _finalize_batch(items, n_node, n_edge, n_graph, n_pairs):
    """Batch and pad both graph and long-range graph structures."""
    graphs = [g for g, _ in items]
    lrs = [lr for _, lr in items]

    # Batch main graphs using standard jraph
    batched_graph = jraph.batch(graphs)
    batched_graph = jraph.pad_with_graphs(batched_graph, n_node, n_edge, n_graph)

    # Batch long-range graphs using standard jraph
    batched_lr = jraph.batch(lrs)
    # Pad with n_pairs as n_edge budget for the long-range graph
    # Use n_node=n_node since lr.n_node mirrors the main graph
    batched_lr = jraph.pad_with_graphs(batched_lr, n_node, n_pairs, n_graph)

    return batched_graph, batched_lr
