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


def add_cell_offsets_sparse(r_ij: jnp.ndarray, cell: jnp.ndarray, cell_offsets: jnp.ndarray):
    """
    Add offsets to distance vectors given a cell and cell offsets. Cell vectors are assumed to be row-wise.
    Args:
        r_ij (Array): Distance vectors, shape: (num_pairs, 3)
        cell (Array): Unit cell matrix, shape: (num_pairs, 3, 3). Unit cell vectors are assumed to be row-wise.
        cell_offsets (Array): Offsets for each pairwise distance, shape: (num_pairs, 3).
    Returns:
    """
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

