import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from ase import Atoms
from typing import Any, Iterable, List, Mapping, Union
from jraph._src import graph as gn_graph


ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


# TODO: move this to mlff jraph_utils at some point
def jraph_to_ase_atoms(graph):
    """ Convert graph to ase.atoms object. """

    cell = graph.edges.get('cell')
    pbc = graph.edges.get('pbc')

    positions = graph.nodes['positions']
    numbers = graph.nodes['atomic_numbers']

    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc
    )

    atoms.info['charge'] = int(graph.globals["total_charge"])
    atoms.info['energy'] = float(f'{float(graph.globals["energy"]):.6g}')
    atoms.info['energy_so3lr'] = float(f'{float(graph.globals["energy_so3lr"]):.6g}')
    atoms.arrays['forces'] = graph.nodes['forces']
    atoms.arrays['forces_so3lr'] = graph.nodes['forces_so3lr']
    atoms.arrays['hirshfeld_ratios'] = graph.nodes['hirshfeld_ratios']
    atoms.arrays['hirshfeld_ratios_so3lr'] = graph.nodes['hirshfeld_ratios_so3lr']
    atoms.info['dipole_vec'] = np.array([float(f'{x:.6g}') for x in graph.globals['dipole_vec'].flatten()])
    atoms.info['dipole_vec_so3lr'] = np.array([float(f'{x:.6g}') for x in graph.globals['dipole_vec_so3lr'].flatten()])

    return atoms


def unbatch(graph: gn_graph.GraphsTuple) -> List[gn_graph.GraphsTuple]:
    """Returns a list of graphs given a batched graph.

    This function does not support jax.jit, because the shape of the output
    is data-dependent!

    Args:
        graph: the batched graph, which will be unbatched into a list of graphs.
    """
    return _unbatch(graph, np_=jnp)


def unbatch_np(graph: gn_graph.GraphsTuple) -> List[gn_graph.GraphsTuple]:
    """Numpy implementation of `unbatch`. See `unbatch` for more details."""
    return _unbatch(graph, np_=np)


def _unbatch(graph: gn_graph.GraphsTuple, np_) -> List[gn_graph.GraphsTuple]:
    """Returns a list of graphs given a batched graph."""

    def _map_split(nest, indices_or_sections):
        """Splits leaf nodes of nests and returns a list of nests."""
        if isinstance(indices_or_sections, int):
            n_lists = indices_or_sections
        else:
            n_lists = len(indices_or_sections) + 1
        concat = lambda field: np_.split(field, indices_or_sections)
        nest_of_lists = tree.tree_map(concat, nest)
        # pylint: disable=cell-var-from-loop
        list_of_nests = [
            tree.tree_map(lambda _, x: x[i], nest, nest_of_lists)
            for i in range(n_lists)
        ]
        return list_of_nests

    all_n_node = graph.n_node[:, None]
    all_n_edge = graph.n_edge[:, None]
    all_n_pair = graph.n_pairs[:, None]
    node_offsets = np_.cumsum(graph.n_node[:-1])
    all_nodes = _map_split(graph.nodes, node_offsets)
    edge_offsets = np_.cumsum(graph.n_edge[:-1])
    pairs_offsets = np_.cumsum(graph.n_pairs[:-1])
    all_edges = _map_split(graph.edges, edge_offsets)
    all_globals = _map_split(graph.globals, len(graph.n_node))
    all_senders = np_.split(graph.senders, edge_offsets)
    all_receivers = np_.split(graph.receivers, edge_offsets)
    all_idx_i_lr = np_.split(graph.idx_i_lr, pairs_offsets)
    all_idx_j_lr = np_.split(graph.idx_j_lr, pairs_offsets)

    # Corrects offset in the sender and receiver arrays, caused by splitting the
    # nodes array.
    n_graphs = graph.n_node.shape[0]
    for graph_index in np_.arange(n_graphs)[1:]:
        all_senders[graph_index] -= node_offsets[graph_index - 1]
        all_receivers[graph_index] -= node_offsets[graph_index - 1]
        all_idx_i_lr[graph_index] -= node_offsets[graph_index - 1]
        all_idx_j_lr[graph_index] -= node_offsets[graph_index - 1]

    return [
        gn_graph.GraphsTuple._make(elements)
        for elements in zip(all_nodes, all_edges, all_receivers, all_senders,
                            all_globals, all_n_node, all_n_edge, all_n_pair, all_idx_i_lr, all_idx_j_lr)
    ]
