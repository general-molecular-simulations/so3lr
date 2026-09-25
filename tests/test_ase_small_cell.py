"""Periodic cells narrower than twice the cutoff, and non-orthogonal cells.

A periodic system is extensive: energy per atom, forces and stress must not
depend on how many copies of the unit cell are simulated. glp's minimum-image
neighbour list breaks this for any cell height below 2 x cutoff, and the PME
solver was handed the transposed lattice for non-orthogonal cells. These tests
pin both fixes on a small triclinic water cell (heights ~6-8 A against a 12 A
long-range cutoff), with and without PME.
"""
import numpy as np
import numpy.testing as npt
import pytest
from ase import Atoms

LR = 12.0
PME = dict(kspace_electrostatics='pme', kspace_smearing=4.0, kspace_spacing=0.5)


def _small_triclinic_water():
    """Three water molecules in a skewed ~7 A cell (angles 80/95/100 deg)."""
    from ase.geometry import cellpar_to_cell
    cell = cellpar_to_cell([7.2, 7.8, 8.1, 80.0, 95.0, 100.0])
    mol = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    frac = np.array([[0.15, 0.20, 0.25], [0.55, 0.60, 0.30], [0.35, 0.75, 0.75]])
    pos = np.concatenate([frac[k] @ cell + mol for k in range(3)])
    return Atoms('OH2OH2OH2', positions=pos, cell=cell, pbc=True)


def _calc(backend, kspace):
    from so3lr import So3lrCalculator
    return So3lrCalculator(model='so3lr-s', dtype=np.float64, lr_cutoff=LR,
                           calculate_stress=True, neighbor_backend=backend,
                           **(PME if kspace else {}))


def _efs(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()


def _supercell(atoms):
    """Replicate until every height exceeds 2 * (LR + 1 A): minimum image valid."""
    cell = atoms.get_cell()
    vol = abs(np.linalg.det(cell))
    h = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    reps = [int(np.ceil(2 * (LR + 1.0) / x)) for x in h]
    return atoms.repeat(reps)


@pytest.mark.parametrize('kspace', [False, True])
def test_unit_cell_matches_supercell(kspace):
    unit = _small_triclinic_water()
    sup = _supercell(unit)
    e_u, f_u, s_u = _efs(unit, _calc('exact', kspace))
    e_s, f_s, s_s = _efs(sup, _calc('exact', kspace))
    npt.assert_allclose(e_u / len(unit), e_s / len(sup), rtol=0, atol=1e-7)
    npt.assert_allclose(f_u, f_s[:len(unit)], rtol=0, atol=1e-5)
    npt.assert_allclose(s_u, s_s, rtol=0, atol=1e-7)


@pytest.mark.parametrize('kspace', [False, True])
def test_exact_matches_glp_where_minimum_image_is_valid(kspace):
    sup = _supercell(_small_triclinic_water())
    e_x, f_x, s_x = _efs(sup, _calc('exact', kspace))
    e_g, f_g, s_g = _efs(sup, _calc('glp', kspace))
    npt.assert_allclose(e_x, e_g, rtol=1e-10)
    npt.assert_allclose(f_x, f_g, rtol=0, atol=1e-8)
    npt.assert_allclose(s_x, s_g, rtol=0, atol=1e-9)


def test_exact_forces_and_stress_match_finite_differences():
    unit = _small_triclinic_water()
    calc = _calc('exact', True)
    e0, f0, s0 = _efs(unit, calc)
    h = 1e-4
    for i, k in [(0, 0), (4, 1), (8, 2)]:
        p, m = unit.copy(), unit.copy()
        p.positions[i, k] += h
        m.positions[i, k] -= h
        num = -(_efs(p, calc)[0] - _efs(m, calc)[0]) / (2 * h)
        npt.assert_allclose(num, f0[i, k], rtol=0, atol=1e-5)
    hs = 1e-5
    eps = np.zeros((3, 3)); eps[0, 0] = hs
    vals = []
    for sgn in (1, -1):
        a = unit.copy()
        a.set_cell(unit.get_cell() @ (np.eye(3) + sgn * eps), scale_atoms=True)
        vals.append(_efs(a, calc)[0])
    npt.assert_allclose((vals[0] - vals[1]) / (2 * hs) / unit.get_volume(), s0[0],
                        rtol=0, atol=1e-7)


def test_auto_selects_exact_for_small_or_skewed_cells():
    from so3lr.mlff.calculators.exact_neighbors import needs_exact
    small = _small_triclinic_water()
    assert needs_exact(small.get_cell(), 4.5, LR)
    big_ortho = np.diag([30.0, 30.0, 30.0])
    assert not needs_exact(big_ortho, 4.5, LR)
    skewed_big = _supercell(small).get_cell()
    assert needs_exact(skewed_big, 4.5, LR)  # non-orthogonal: glp's rounding MIC not guaranteed
