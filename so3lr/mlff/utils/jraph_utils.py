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
def graph_to_batch_fn(graph: jraph.GraphsTuple, lr_batch: dict, kspace_constants: dict = None):
    batch = dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        num_unpaired_electrons=graph.globals.get('num_unpaired_electrons'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=graph.globals.get('cell'),
        pbc=graph.globals.get('pbc'),
        cell_offset=graph.edges.get('cell_offset'),
        energy=graph.globals.get('energy'),
        forces=graph.nodes.get('forces'),
        stress=graph.globals.get('stress'),
        total_charge=graph.globals.get('total_charge'),
        dipole_vec=graph.globals.get('dipole_vec'),
        hirshfeld_ratios=graph.nodes.get('hirshfeld_ratios'),
        c6_ratios=graph.nodes.get('c6_ratios'),
        idx_i_lr=lr_batch['idx_i_lr'],
        idx_j_lr=lr_batch['idx_j_lr'],
        cell_offset_lr=lr_batch.get('cell_offsets_lr'),
        theory_level=graph.globals.get('theory_level'),
        theory_mask=graph.globals.get('theory_mask'),
        residue_charge=graph.globals.get('residue_charge'),
        residue_segments=graph.globals.get('residue_segments'),
    )
    if kspace_constants is not None:
        batch['k_grid'] = kspace_constants['k_grid']
        batch['k_smearing'] = kspace_constants['k_smearing']
        batch['k_grid_disp'] = kspace_constants['k_grid_disp']
        batch['k_smearing_disp'] = kspace_constants['k_smearing_disp']
    batch_info = batch_info_fn(graph)
    batch.update(batch_info)
    return batch


def dynamically_batch_with_lr(
    data_iterator,
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_pairs: int,
):
    """Batch graphs + long-range data respecting both budgets.

    Args:
        data_iterator: Iterator yielding (graph, lr_data) tuples,
            where graph is a jraph.GraphsTuple and lr_data is a dict
            with keys 'idx_i_lr', 'idx_j_lr', 'cell_offsets_lr', 'n_pairs'.
        n_node: Maximum number of nodes in a batch (including padding graph).
        n_edge: Maximum number of edges in a batch.
        n_graph: Maximum number of graphs in a batch (including padding graph).
        n_pairs: Maximum number of long-range pairs in a batch.

    Yields:
        (padded_graph, lr_batch_dict) tuples with fixed sizes.
    """
    # Fast path: no LR budget — use native jraph batching.
    if n_pairs == 0:
        empty_lr = {
            'idx_i_lr': np.array([], dtype=np.int32),
            'idx_j_lr': np.array([], dtype=np.int32),
            'cell_offsets_lr': None,
        }
        # Keep this as a generator.  TFDS training iterators may be large and
        # used to be infinite; eagerly materializing them delayed the first
        # batch until exhaustion and could consume all available memory.
        graphs_only = (g for g, _ in data_iterator)
        for batched in jraph.dynamically_batch(
            iter(graphs_only), n_node=n_node, n_edge=n_edge, n_graph=n_graph
        ):
            yield batched, empty_lr
        return

    accumulated = []
    acc_nodes = 0
    acc_edges = 0
    acc_graphs = 0
    acc_pairs = 0

    for graph, lr_data in data_iterator:
        elem_nodes = int(np.sum(graph.n_node))
        elem_edges = int(np.sum(graph.n_edge))
        elem_pairs = int(lr_data['n_pairs'][0])
        elem_graphs = len(graph.n_node)

        if not accumulated:
            accumulated = [(graph, lr_data)]
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
            yield _finalize_batch(
                accumulated, n_node, n_edge, n_graph, n_pairs
            )
            accumulated = [(graph, lr_data)]
            acc_nodes = elem_nodes
            acc_edges = elem_edges
            acc_graphs = elem_graphs
            acc_pairs = elem_pairs
        else:
            accumulated.append((graph, lr_data))
            acc_nodes += elem_nodes
            acc_edges += elem_edges
            acc_graphs += elem_graphs
            acc_pairs += elem_pairs

    if accumulated:
        yield _finalize_batch(accumulated, n_node, n_edge, n_graph, n_pairs)


def _finalize_batch(items, n_node, n_edge, n_graph, n_pairs):
    """Batch main graph and manually concatenate/offset/pad LR indices."""
    graphs = [g for g, _ in items]
    lr_datas = [lr for _, lr in items]

    # Batch main graph (single batch + pad)
    batched_graph = jraph.batch_np(graphs)  # numpy batch: jraph.batch uses jnp -> slow per-op under jax 0.10 in CPU worker
    batched_graph = jraph.pad_with_graphs(batched_graph, n_node, n_edge, n_graph)

    # Manually concatenate + offset LR indices
    node_cumsum = np.cumsum([0] + [int(np.sum(g.n_node)) for g in graphs[:-1]])
    lr_i_parts = [lr['idx_i_lr'] + offset for lr, offset in zip(lr_datas, node_cumsum)]
    lr_j_parts = [lr['idx_j_lr'] + offset for lr, offset in zip(lr_datas, node_cumsum)]

    all_lr_i = np.concatenate(lr_i_parts) if lr_i_parts else np.array([], dtype=np.int32)
    all_lr_j = np.concatenate(lr_j_parts) if lr_j_parts else np.array([], dtype=np.int32)

    if len(all_lr_i) > n_pairs:
        raise RuntimeError(
            f"Batch has {len(all_lr_i)} LR pairs but n_pairs budget is {n_pairs}. "
            f"Increase batch_max_num_pairs."
        )

    # Pad to fixed n_pairs length, pointing to the padding node (n_node - 1)
    # so that padded pairs have zero contribution (same as jraph.pad_with_graphs)
    pad_len = n_pairs - len(all_lr_i)
    pad_idx = n_node - 1
    padded_lr_i = np.pad(all_lr_i, (0, pad_len), constant_values=pad_idx).astype(np.int32)
    padded_lr_j = np.pad(all_lr_j, (0, pad_len), constant_values=pad_idx).astype(np.int32)

    # Handle cell_offsets_lr if present
    if lr_datas[0].get('cell_offsets_lr') is not None:
        lr_offsets_parts = [lr['cell_offsets_lr'] for lr in lr_datas]
        all_offsets = np.concatenate(lr_offsets_parts)
        padded_offsets = np.pad(all_offsets, ((0, pad_len), (0, 0)), constant_values=0)
    else:
        padded_offsets = None

    lr_batch = {
        'idx_i_lr': padded_lr_i,
        'idx_j_lr': padded_lr_j,
        'cell_offsets_lr': padded_offsets,
    }
    return batched_graph, lr_batch
