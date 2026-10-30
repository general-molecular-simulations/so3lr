"""The ASE calculator must not compute on an incomplete glp neighbour list.

The default (glp) backend uses a cell list whose per-cell capacity is fixed at allocation. If atoms
later crowd into one cell beyond that capacity (relaxations, MD near ions), glp must flag it and the
calculator must reallocate; otherwise pairs are silently dropped. No model is needed: the
calculator's glp neighbour list is compared with ASE's after squeezing waters together.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.neighborlist import neighbor_list as ase_neighbor_list
from glp import atoms_to_system

from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse

WATER = Path(__file__).parent / 'test_data' / 'water_64.xyz'
CUTOFF, LR_CUTOFF = 2.5, 3.0     # short cutoffs give a 4x4x4 cell list on the 12.4 A box


@pytest.fixture(autouse=True)
def _on_cpu():
    """glp builds the list eagerly from hundreds of small ops. On a GPU each one compiles
    separately, which can take hours, so run this neighbour-list logic on the CPU."""
    with jax.default_device(jax.devices('cpu')[0]):
        yield


def squeeze(atoms, n_waters):
    """Pull the n_waters molecules nearest to water 0 halfway towards it (molecules kept whole)."""
    b = atoms.copy()
    O = np.where(b.numbers == 8)[0]                # atoms are ordered O, H, H per water
    near = O[np.argsort(b.get_distances(O[0], O, mic=True))[1:n_waters + 1]]
    for o in near:
        b.positions[o:o + 3] -= 0.5 * b.get_distance(O[0], o, mic=True, vector=True)
    return b


def glp_calculator():
    """Just the state _glp_neighbors needs. The pair buffer gets ample headroom so that only the
    cell list can overflow (pair-buffer overflow was already handled before the fix)."""
    calc = object.__new__(AseCalculatorSparse)
    Calculator.__init__(calc)
    calc.neighbors = calc.spatial_partitioning = None
    calc.cutoff, calc.lr_cutoff, calc.skin = CUTOFF, LR_CUTOFF, 0.0
    calc.capacity_multiplier, calc.buffer_size_multiplier = 3.0, 1.25
    return calc


def cell_of(atoms):
    return jnp.asarray(np.array(atoms.get_cell())).T   # glp takes lattice vectors as columns


def glp_pairs(calc, atoms):
    """Short- and long-range (i, j) pairs of the calculator's glp list, padding removed."""
    nb = calc._glp_neighbors(atoms_to_system(atoms), cell_of(atoms))
    n = len(atoms)

    def pairs(i, j):
        return {p for p in zip(np.asarray(i).tolist(), np.asarray(j).tolist()) if p[0] != n}
    return pairs(nb.centers, nb.others), pairs(nb.idx_i_lr, nb.idx_j_lr)


def exact_pairs(atoms, cutoff):
    return set(zip(*ase_neighbor_list('ij', atoms, cutoff)))


def test_overfilled_cell_is_reallocated():
    box = read(WATER)
    box.wrap()
    calc = glp_calculator()
    glp_pairs(calc, box)                           # allocated on the original box
    allocated = calc.spatial_partitioning

    squeezed = squeeze(box, 8)                     # one cell now exceeds its allocated capacity
    update = calc.spatial_partitioning.update_fn(atoms_to_system(squeezed).R, calc.neighbors,
                                                 new_cell=cell_of(squeezed))
    assert update.cell_list.reallocate, 'glp did not flag the over-full cell list'
    sr, lr = glp_pairs(calc, squeezed)
    assert calc.spatial_partitioning is not allocated, 'the over-full cell list was not reallocated'
    assert sr == exact_pairs(squeezed, CUTOFF)
    assert lr == exact_pairs(squeezed, LR_CUTOFF)
