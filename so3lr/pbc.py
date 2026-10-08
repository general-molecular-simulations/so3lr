"""Boundary-condition validation shared by public SO3LR entry points."""

from __future__ import annotations

import numpy as np


def is_fully_periodic(pbc, *, context: str) -> bool:
    """Return whether all axes are periodic and reject mixed-axis PBC.

    Current JAX-MD neighbor and reciprocal-space implementations assume either
    vacuum or three-dimensional periodicity. Silently promoting a slab or wire
    to 3-D periodicity gives inconsistent real- and reciprocal-space physics,
    so public entry points fail before model execution instead.
    """
    flags = np.asarray(pbc, dtype=bool).reshape(-1)
    if flags.shape != (3,):
        raise ValueError(
            f"{context} expected three PBC flags, received shape {flags.shape}.")

    if np.any(flags) and not np.all(flags):
        raise ValueError(
            f"{context} does not support mixed-axis periodic boundary conditions "
            f"(received {flags.tolist()}). Use pbc=False for a finite system or "
            f"pbc=True in all three axes for a 3-D periodic system."
        )
    return bool(np.all(flags))


def lattice_rows(cell):
    """Lattice vectors as rows (ASE, HDF5, jaxpme) from a column-convention cell.

    glp's ``System.cell`` is ``atoms.cell.T`` and a jax-md box maps fractional
    to Cartesian as ``box @ s``, so both hold the lattice vectors as columns.
    Scalars and diagonal vectors have no ambiguity and are returned unchanged;
    batched ``(n, 3, 3)`` cells are transposed per cell.
    """
    if cell is None or getattr(cell, 'ndim', 0) < 2:
        return cell
    import jax.numpy as jnp
    return jnp.swapaxes(cell, -1, -2)
