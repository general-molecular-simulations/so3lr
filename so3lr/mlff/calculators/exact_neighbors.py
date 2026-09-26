"""Exact periodic neighbour lists for the ASE calculator (any cell size or shape).

glp's neighbour list, like jax-md's, keeps one image per atom pair (minimum-image
convention). That is only correct when every perpendicular cell height exceeds
twice the cutoff; SO3LR's 12 Ang long-range cutoff therefore silently drops pairs
in any cell narrower than 24 Ang. For a 6.5 Ang CoMOF-74 cell the result is off
by 20 meV/atom and 0.5 eV/Ang.

This module lists every image explicitly: ``neighbour_list('ijSd')`` (matscipy,
or ASE as a fallback) returns each pair (i, j, S) with integer lattice shift S
such that r_ij = R_j - R_i + S @ cell lies within the cutoff. A small cell simply
yields many shifts per pair, including an atom and its own image (i == j, S != 0).
Arrays are padded to a fixed capacity so jit does not retrace every step; padded
entries point at index N, the convention glp uses, so the model masks them alike.
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import numpy as np

from so3lr.mlff.indexing.indices import pad_index_list, pad_shift

try:  # matscipy's C implementation is ~25x faster than ASE's pure-numpy one
    from matscipy.neighbours import neighbour_list as _neighbour_list
    _SLOW_FALLBACK = False
except ImportError:  # pragma: no cover - exercised only without matscipy
    from ase.neighborlist import neighbor_list as _neighbour_list
    _SLOW_FALLBACK = True


class ExactNeighbors(NamedTuple):
    """Explicit-image pair lists (a jax pytree). Shifts are integer lattice vectors."""
    centers: np.ndarray    # (P,)  int32, padded with N
    others: np.ndarray     # (P,)  int32, padded with N
    shifts: np.ndarray     # (P, 3) int32, zero where padded
    idx_i_lr: np.ndarray   # (P_lr,)
    idx_j_lr: np.ndarray   # (P_lr,)
    shifts_lr: np.ndarray  # (P_lr, 3)


def _padded(i, j, S, cap, n):
    return (pad_index_list(i, cap, pad_value=n), pad_index_list(j, cap, pad_value=n),
            pad_shift(S, cap))


def _capacity(n_pairs, previous, multiplier):
    """Reuse the previous capacity while it fits, else grow with headroom (one retrace)."""
    if previous is not None and n_pairs <= previous:
        return previous
    return max(8, int(np.ceil(n_pairs * multiplier)))


def build(atoms, cutoff: float, lr_cutoff: Optional[float],
          prev: Optional[ExactNeighbors] = None, multiplier: float = 1.25) -> ExactNeighbors:
    """All pairs within ``cutoff`` and ``lr_cutoff``, every periodic image included.

    One neighbour search at the larger cutoff; the short-range list is its subset
    with d < cutoff. ``lr_cutoff=None`` means a local model (long-range lists equal
    the short-range ones). ``prev`` supplies the capacities to reuse, and is ignored
    when the atom count changed so a small system does not inherit a large padding.
    """
    global _SLOW_FALLBACK
    if _SLOW_FALLBACK:  # warn once, on first use rather than on import
        logging.getLogger("SO3LR").warning(
            'matscipy not installed: the exact periodic neighbour list falls back to '
            'ASE, which is ~25x slower. `pip install matscipy` for small-cell runs.')
        _SLOW_FALLBACK = False
    n = len(atoms)
    i, j, S, d = _neighbour_list('ijSd', atoms, float(max(cutoff, lr_cutoff or 0.0)))
    i, j, S = i.astype(np.int32), j.astype(np.int32), S.astype(np.int32)
    sr = d < cutoff
    cap = _capacity(int(sr.sum()), None if prev is None else len(prev.centers), multiplier)
    lr = (i, j, S) if lr_cutoff is not None else (i[sr], j[sr], S[sr])
    cap_lr = _capacity(len(lr[0]), None if prev is None else len(prev.idx_i_lr), multiplier)
    return ExactNeighbors(*_padded(i[sr], j[sr], S[sr], cap, n), *_padded(*lr, cap_lr, n))


def cell_heights(cell) -> np.ndarray:
    """Perpendicular distance between opposite faces, per lattice direction (rows)."""
    return 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(cell, float)).T, axis=1)


def needs_exact(cell, cutoff: float, lr_cutoff: Optional[float], skin: float = 0.0) -> bool:
    """True when glp's neighbour list would be wrong for this periodic cell.

    glp keeps one image per pair and bins with a cell list, which it switches on
    for all three axes as soon as ONE height exceeds 3 * rc (so3lr always requests
    it). It is therefore exact only for an orthorhombic cell whose heights all
    exceed 2 * rc (minimum image) and are either all below 3 * rc (no cell list)
    or all above it (>= 3 bins per axis). Anything else takes the exact path.
    """
    cell = np.asarray(cell, float)
    # Off-diagonals at file/round-off level (e.g. 1e-7 A from a CIF) are not a
    # skewed cell; anything larger is, and glp's cell list bins it wrongly.
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1e-6):
        return True
    rc = max(cutoff, lr_cutoff or 0.0) + skin
    h = cell_heights(cell)
    cell_list_on = bool(np.any(h > 3.0 * rc))
    return bool(h.min() <= 2.0 * rc or (cell_list_on and h.min() <= 3.0 * rc))
