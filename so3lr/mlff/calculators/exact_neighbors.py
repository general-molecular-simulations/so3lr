"""Exact periodic neighbour lists for the ASE calculator (any cell size or shape).

glp's neighbour list, like jax-md's, keeps one image per atom pair (minimum-image
convention). That is only correct when every perpendicular cell height exceeds
twice the cutoff; SO3LR's 12 Ang long-range cutoff therefore silently drops pairs
in any cell narrower than 24 Ang, and on a skewed cell a fractional-rounding
minimum image can pick the wrong image even inside that bound. For a 6.5 Ang
CoMOF-74 cell the result is off by 20 meV/atom and 0.5 eV/Ang.

This module lists every image explicitly: ``neighbor_list('ijS')`` from matscipy
(or ASE as a fallback) returns each pair (i, j, S) with integer lattice shift S
such that r_ij = R_j - R_i + S @ cell lies within the cutoff. A small cell simply
yields many shifts per pair, and i == j with S != 0 (an atom and its own image) is
included. The arrays are padded to a fixed capacity so jit does not retrace every
step; padded entries point at index N, the same convention glp uses, so the model
masks them identically.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:  # matscipy is ~10x faster than ASE's pure-numpy implementation
    from matscipy.neighbours import neighbour_list as _nl
    BACKEND = 'matscipy'
except ImportError:  # pragma: no cover - exercised only without matscipy
    from ase.neighborlist import neighbor_list as _nl
    BACKEND = 'ase'


class ExactNeighbors(NamedTuple):
    centers: np.ndarray    # (P,)  int32, padded with N
    others: np.ndarray     # (P,)  int32, padded with N
    shifts: np.ndarray     # (P, 3) float, integer lattice shifts, 0 when padded
    idx_i_lr: np.ndarray   # (P_lr,)
    idx_j_lr: np.ndarray   # (P_lr,)
    shifts_lr: np.ndarray  # (P_lr, 3)
    n_pairs: int
    n_pairs_lr: int


def _pairs(atoms, cutoff):
    i, j, S = _nl('ijS', atoms, float(cutoff))
    return i.astype(np.int32), j.astype(np.int32), S.astype(np.float64)


def _pad(i, j, S, cap, n):
    k = len(i)
    ii = np.full(cap, n, np.int32); jj = np.full(cap, n, np.int32)
    ss = np.zeros((cap, 3), np.float64)
    ii[:k], jj[:k], ss[:k] = i, j, S
    return ii, jj, ss


def capacity(n_pairs, multiplier):
    """Next capacity: headroom so small geometry changes do not force a retrace."""
    return max(8, int(np.ceil(n_pairs * multiplier)))


def build(atoms, cutoff, lr_cutoff, cap=None, cap_lr=None, multiplier=1.25):
    """All pairs within ``cutoff`` and ``lr_cutoff``, every periodic image included.

    Returns (ExactNeighbors, cap, cap_lr). ``cap``/``cap_lr`` are reused while the
    pair count fits and grown (forcing one retrace) when it does not.
    """
    n = len(atoms)
    i, j, S = _pairs(atoms, cutoff)
    if lr_cutoff is not None and lr_cutoff > 0:
        il, jl, Sl = _pairs(atoms, lr_cutoff)
    else:  # local model: long-range indices equal the local ones
        il, jl, Sl = i, j, S
    if cap is None or len(i) > cap:
        cap = capacity(len(i), multiplier)
    if cap_lr is None or len(il) > cap_lr:
        cap_lr = capacity(len(il), multiplier)
    c, o, s = _pad(i, j, S, cap, n)
    cl, ol, sl = _pad(il, jl, Sl, cap_lr, n)
    return ExactNeighbors(c, o, s, cl, ol, sl, len(i), len(il)), cap, cap_lr


def needs_exact(cell, cutoff, lr_cutoff, skin=0.0, pbc=(True, True, True)):
    """True when a minimum-image neighbour list would be wrong for this cell.

    A minimum-image list is exact only if the cell is orthorhombic and every
    height exceeds 2 * (cutoff + skin) for the largest cutoff in use.
    """
    if not np.any(pbc):
        return False
    cell = np.asarray(cell, float)
    vol = abs(np.linalg.det(cell))
    heights = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3]))
               for k in range(3)]
    rc = max(cutoff, lr_cutoff if (lr_cutoff is not None and lr_cutoff < 1e5) else 0.0) + skin
    orthorhombic = np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-8)
    return (not orthorhombic) or (min(heights) <= 2.0 * rc)
