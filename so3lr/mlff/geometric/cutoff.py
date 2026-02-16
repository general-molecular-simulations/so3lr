import jax.numpy as jnp
from typing import Any


Array = Any


def add_cell_offsets(r_ij: jnp.ndarray, cell: jnp.ndarray, cell_offsets: jnp.ndarray):
    """
    Add offsets to distance vectors given a cell and cell offsets. Cell vectors are assumed to be row-wise.
    Args:
        r_ij (Array): Distance vectors, shape: (num_pairs, 3)
        cell (Array): Unit cell matrix, shape: (3, 3). Unit cell vectors are assumed to be row-wise.
        cell_offsets (Array): Offsets for each pairwise distance, shape: (num_pairs, 3).
    Returns:
    """
    offsets = jnp.einsum('...i, ij -> ...j', cell_offsets, cell)
    return r_ij + offsets


def add_cell_offsets_sparse(
    r_ij: jnp.ndarray,
    cell: jnp.ndarray,
    cell_offsets: jnp.ndarray,
    batch_segments: jnp.ndarray = None,
    idx_i: jnp.ndarray = None,
):
    """
    Add offsets to distance vectors given a cell and cell offsets. Cell vectors are assumed to be row-wise.
    Args:
        r_ij (Array): Distance vectors, shape: (num_pairs, 3)
        cell (Array): Unit cell matrix. Either per-graph (n_graphs, 3, 3) or per-pair (num_pairs, 3, 3).
            A single (3, 3) matrix is also accepted for a single graph.
        cell_offsets (Array): Offsets for each pairwise distance, shape: (num_pairs, 3).
        batch_segments (Array): Graph index for each atom, shape: (num_atoms,). Required when cell
            is per-graph (n_graphs, 3, 3) to map edges to their graph's cell.
        idx_i (Array): Index of centering atom for each pair, shape: (num_pairs,). Required together
            with batch_segments.
    Returns:
        Array: Corrected distance vectors, shape: (num_pairs, 3)
    """
    if cell.ndim == 2:
        # Single (3, 3) cell matrix — same cell for all pairs
        offsets = jnp.einsum('Pi, ij -> Pj', cell_offsets, cell)
    elif cell.ndim == 3 and batch_segments is not None and idx_i is not None:
        # Per-graph cell (n_graphs, 3, 3) — look up each pair's cell via batch_segments
        per_pair_cell = cell[batch_segments[idx_i]]  # (num_pairs, 3, 3)
        offsets = jnp.einsum('Pi, Pij -> Pj', cell_offsets, per_pair_cell)
    else:
        # Legacy per-pair cell (num_pairs, 3, 3)
        offsets = jnp.einsum('Pi, Pij -> Pj', cell_offsets, cell)
    return r_ij + offsets


def pbc_diff(x: jnp.ndarray, cell: jnp.ndarray):
    """
    Clamp differences of vectors to super cell.

    Args:
        x (Array): vectors, shape: (3)
        cell (Array): matrix containing lattice vectors as rows, shape: (3, 3)

    Returns: clamped distance vectors, shape: (3)

    """
    c = jnp.linalg.solve(cell.T, x).T
    c -= jnp.floor(c + 0.5)
    return c@cell

