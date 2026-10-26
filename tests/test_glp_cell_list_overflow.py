"""The ASE calculator must not compute on an incomplete glp neighbour list.

In large periodic boxes the default (glp) backend uses a cell list whose per-cell capacity is
fixed at allocation. If atoms later crowd into one cell beyond that capacity (relaxations, MD near
ions), glp must flag it and the calculator must reallocate; otherwise pairs are silently dropped
and forces/charges are wrong. Compares the default backend, allocated on the original box, with
the exact backend after squeezing waters together.
"""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from so3lr import So3lrCalculator

WATER = Path(__file__).parent / 'test_data' / 'water_64.xyz'
KW = dict(model='so3lr-tauri-s-2606', lr_cutoff=6.0, dtype=np.float64, has_aux=True, observables=['partial_charges'])


def squeeze(atoms, n_waters):
    """Pull the n_waters molecules nearest to water 0 halfway towards it (molecules kept whole)."""
    b = atoms.copy()
    O = np.where(b.numbers == 8)[0]                # atoms are ordered O, H, H per water
    near = O[np.argsort(b.get_distances(O[0], O, mic=True))[1:n_waters + 1]]
    for o in near:
        b.positions[o:o + 3] -= 0.5 * b.get_distance(O[0], o, mic=True, vector=True)
    return b


@pytest.mark.slow
def test_overfilled_cell_matches_exact_backend(x64):
    box = read(WATER).repeat((2, 2, 2))          # 512 waters: the glp cell list is used
    box.wrap()
    box.info['charge'] = 0
    calc = So3lrCalculator(**KW)                 # default backend, allocated on the original box
    box.calc = calc
    box.get_potential_energy()

    b = squeeze(box, 25)                         # one cell now exceeds its allocated capacity
    b.calc = calc
    f_glp = b.get_forces()
    q_glp = calc.results['aux']['partial_charges'].ravel()

    b.calc = So3lrCalculator(**KW, neighbor_backend='exact')
    f_exact = b.get_forces()
    q_exact = b.calc.results['aux']['partial_charges'].ravel()

    np.testing.assert_allclose(q_glp, q_exact, atol=1e-8)
    np.testing.assert_allclose(f_glp, f_exact, atol=1e-6)
