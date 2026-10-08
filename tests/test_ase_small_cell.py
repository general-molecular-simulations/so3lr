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
from ase.geometry import cellpar_to_cell

from so3lr.mlff.calculators.exact_neighbors import cell_heights, needs_exact

LR = 12.0
PME = dict(kspace_electrostatics='pme', kspace_smearing=4.0, kspace_spacing=0.5)


def _small_triclinic_water(angles=(80.0, 95.0, 100.0)):
    """Three water molecules in a ~7 A cell, skewed unless angles are all 90."""
    cell = cellpar_to_cell([7.2, 7.8, 8.1, *angles])
    mol = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    frac = np.array([[0.15, 0.20, 0.25], [0.55, 0.60, 0.30], [0.35, 0.75, 0.75]])
    pos = np.concatenate([frac[k] @ cell + mol for k in range(3)])
    return Atoms('OH2OH2OH2', positions=pos, cell=cell, pbc=True)


def _supercell(atoms):
    """Replicate until every height exceeds 2 * (LR + 1 A): minimum image valid."""
    reps = [int(np.ceil(2 * (LR + 1.0) / h)) for h in cell_heights(atoms.get_cell())]
    return atoms.repeat(reps)


def _calc(backend, kspace, **kw):
    from so3lr import So3lrCalculator
    return So3lrCalculator(model='so3lr-s', dtype=np.float64, lr_cutoff=LR,
                           neighbor_backend=backend, **(PME if kspace else {}), **kw)


def _efs(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()


def _central_difference(energy_of, h):
    return (energy_of(+h) - energy_of(-h)) / (2 * h)


@pytest.fixture(scope='module', params=[False, True], ids=['no_kspace', 'pme'])
def kspace(request):
    return request.param


@pytest.fixture(scope='module')
def exact_supercell(kspace):
    """Exact-backend results on the valid-minimum-image supercell (shared, costly)."""
    sup = _supercell(_small_triclinic_water())
    return sup, _efs(sup, _calc('exact', kspace, calculate_stress=True))


def test_unit_cell_matches_supercell(kspace, exact_supercell):
    unit = _small_triclinic_water()
    sup, (e_s, f_s, s_s) = exact_supercell
    e_u, f_u, s_u = _efs(unit, _calc('exact', kspace, calculate_stress=True))
    npt.assert_allclose(e_u / len(unit), e_s / len(sup), rtol=0, atol=1e-7)
    npt.assert_allclose(f_u, f_s[:len(unit)], rtol=0, atol=1e-5)
    npt.assert_allclose(s_u, s_s, rtol=0, atol=1e-7)


def test_exact_matches_glp_where_auto_keeps_glp(kspace):
    """On the cells 'auto' still sends to glp, both backends must agree exactly:
    orthorhombic, heights between 2 and 3 x cutoff (minimum image, no cell list)."""
    sup = _supercell(_small_triclinic_water(angles=(90.0, 90.0, 90.0)))
    assert not needs_exact(sup.get_cell(), 4.5, LR)
    e_x, f_x, s_x = _efs(sup, _calc('exact', kspace, calculate_stress=True))
    e_g, f_g, s_g = _efs(sup, _calc('glp', kspace, calculate_stress=True))
    npt.assert_allclose(e_x, e_g, rtol=1e-10)
    npt.assert_allclose(f_x, f_g, rtol=0, atol=1e-8)
    npt.assert_allclose(s_x, s_g, rtol=0, atol=1e-9)


def test_exact_forces_and_stress_match_finite_differences():
    unit = _small_triclinic_water()
    calc = _calc('exact', True, calculate_stress=True)
    _, f0, s0 = _efs(unit, calc)

    def displaced(i, k):
        def energy(h):
            a = unit.copy(); a.positions[i, k] += h
            return _efs(a, calc)[0]
        return energy

    for i, k in [(0, 0), (4, 1), (8, 2)]:
        npt.assert_allclose(-_central_difference(displaced(i, k), 1e-4), f0[i, k],
                            rtol=0, atol=1e-5)

    def strained(h):
        a = unit.copy()
        a.set_cell(unit.get_cell() @ np.diag([1 + h, 1, 1]), scale_atoms=True)
        return _efs(a, calc)[0]

    npt.assert_allclose(_central_difference(strained, 1e-5) / unit.get_volume(), s0[0],
                        rtol=0, atol=1e-7)


def test_exact_hessian_on_small_cell():
    """Every calculate_fn variant runs on explicit images, the hessian included."""
    unit = _small_triclinic_water()
    n = len(unit)
    atoms = unit.copy()
    atoms.calc = _calc('exact', True, calculate_hessian=True)
    atoms.get_potential_energy()
    hess = np.asarray(atoms.calc.results['hessian']).reshape(3 * n, 3 * n)
    npt.assert_allclose(hess, hess.T, rtol=0, atol=1e-6)

    f_calc = _calc('exact', True)

    def forces(h):
        a = unit.copy(); a.positions[0, 0] += h
        a.calc = f_calc
        return a.get_forces().ravel()

    npt.assert_allclose(hess[:, 0], -_central_difference(forces, 1e-4), rtol=0, atol=1e-4)


def test_auto_selects_exact_when_glp_would_be_wrong():
    small = _small_triclinic_water()
    assert needs_exact(small.get_cell(), 4.5, LR)                       # height < 2 rc
    assert needs_exact(_supercell(small).get_cell(), 4.5, LR)           # non-orthogonal
    assert not needs_exact(np.diag([30.0, 30.0, 30.0]), 4.5, LR)        # MIC, no cell list
    assert not needs_exact(np.diag([40.0, 40.0, 40.0]), 4.5, LR)        # cell list, >=3 bins
    assert needs_exact(np.diag([26.0, 26.0, 80.0]), 4.5, LR)            # glp cell list, <3 bins
    assert not needs_exact(np.diag([30.0, 30.0, 30.0]), 4.5, None)      # local model
    noisy = np.diag([30.0, 30.0, 30.0]); noisy[0, 1] = 1e-7
    assert not needs_exact(noisy, 4.5, LR)                              # CIF round-off

