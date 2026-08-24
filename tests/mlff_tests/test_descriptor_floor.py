"""The descriptor never sees a separation the model was not trained on.

Below its closest trained approach the network extrapolates catastrophically,
and the edge of validity is per element pair: on so3lr-l a compressed S-S pair
returns NaN energies and forces outright over 0.44-0.60 A, while H-H is fine all
the way down. Measured breakdown distances land at 0.45-0.66x the distance where
the tabulated NLH repulsion reaches 1 eV -- which is why the floor is scaled off
that table rather than off a single global constant.

Only the radial basis is floored. The angular part comes from the unit vector,
and the repulsion / electrostatics / dispersion keep the true distance, so they
still diverge at real contact and carry the short-range wall.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from ase.data import atomic_numbers

from so3lr.mlff.nn.embed.embed_sparse import (
    PAIR_FLOOR_SCALE,
    PAIR_FLOOR_WIDTH,
    floor_descriptor_distance,
    nlh_reference_distance,
)

# equilibrium bond lengths [A]; the floor must be inert at and above these
BONDS = {('H', 'H'): 0.741, ('C', 'H'): 1.09, ('N', 'H'): 1.01, ('O', 'H'): 0.96,
         ('C', 'C'): 1.54, ('C', 'O'): 1.43, ('C', 'N'): 1.47, ('O', 'O'): 1.21,
         ('S', 'S'): 1.89, ('C', 'S'): 1.82, ('C', 'Cl'): 1.77, ('S', 'H'): 1.34}


def _floor(e1, e2, d):
    z = jnp.array([atomic_numbers[e1], atomic_numbers[e2]])
    return floor_descriptor_distance(jnp.atleast_1d(jnp.asarray(d, float)), z,
                                     jnp.array([0]), jnp.array([1]))


def _handover(e1, e2):
    t = np.asarray(nlh_reference_distance())
    z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
    return PAIR_FLOOR_SCALE * t[min(z1, z2) - 1, max(z1, z2) - 1] + PAIR_FLOOR_WIDTH


@pytest.mark.parametrize('pair,bond', list(BONDS.items()))
def test_inert_at_and_above_every_real_bond(pair, bond):
    """Ordinary chemistry must be bit-for-bit untouched, so the handover has to
    sit below the equilibrium bond length of every pair the model sees."""
    e1, e2 = pair
    assert _handover(e1, e2) < bond, (
        f'{e1}-{e2}: handover {_handover(e1, e2):.3f} A is at or above the '
        f'{bond} A bond, so the floor would perturb normal geometries')
    d = jnp.array([bond, bond + 0.2, 1.5 * bond, 4.0])
    npt.assert_array_equal(np.asarray(_floor(e1, e2, d)), np.asarray(d))


def test_scale_is_pair_specific_and_ordered_by_size():
    """A single global floor cannot serve both: S-S needs ~4x the reach of H-H."""
    t = np.asarray(nlh_reference_distance())
    def r(e1, e2):
        z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
        return t[min(z1, z2) - 1, max(z1, z2) - 1]
    assert r('H', 'H') < r('O', 'H') < r('C', 'C') < r('S', 'S')
    assert r('S', 'S') / r('H', 'H') > 3.0


def test_reference_distance_matches_the_tabulated_repulsion():
    """nlh_reference_distance must really solve E_NLH(r) = PAIR_FLOOR_ENERGY."""
    from so3lr.mlff.nn.embed.embed_sparse import PAIR_FLOOR_ENERGY
    from so3lr.mlff.nn.observable.dispersion_ref_data import NLH_AA, NLH_BB

    ke = 14.399645351950548
    t = np.asarray(nlh_reference_distance())
    for e1, e2 in [('O', 'H'), ('C', 'C'), ('S', 'S')]:
        z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
        i, j = min(z1, z2) - 1, max(z1, z2) - 1
        r = t[i, j]
        e = ke * z1 * z2 / r * np.sum(NLH_AA[i, j] * np.exp(-NLH_BB[i, j] * r))
        npt.assert_allclose(e, PAIR_FLOOR_ENERGY, rtol=1e-3)


def test_floors_below_the_pair_scale():
    for e1, e2 in [('H', 'H'), ('O', 'H'), ('S', 'S')]:
        out = np.asarray(_floor(e1, e2, jnp.array([0.0, 0.05, 0.2, 0.4])))
        floor = _handover(e1, e2) - PAIR_FLOOR_WIDTH
        assert out.min() >= floor - 1e-6
        assert out.max() <= floor + PAIR_FLOOR_WIDTH


def test_monotone_and_finite_gradient_including_zero():
    """A non-monotone floor would invent forces; a NaN at d=0 would poison MD."""
    z = jnp.array([8, 1])
    ii, jj = jnp.array([0]), jnp.array([1])
    d = jnp.linspace(0.0, 2.0, 400)
    out = np.asarray(floor_descriptor_distance(d, z, ii, jj))
    assert np.all(np.diff(out) >= -1e-7)
    g = jax.grad(lambda x: floor_descriptor_distance(x, z, ii, jj).sum())
    assert np.isfinite(np.asarray(g(jnp.array([0.0, 0.5, 1.2])))).all()


def test_clash_stays_finite_and_repulsive_end_to_end():
    from ase import Atoms

    from so3lr import So3lrCalculator

    water = Atoms('OHH', positions=[[0., 0., 0.], [0.13, 0., 0.], [-0.24, 0.93, 0.]])
    water.calc = So3lrCalculator(model='so3lr-s')
    e, f = water.get_potential_energy(), water.get_forces()
    assert np.isfinite(e) and np.isfinite(f).all()
    assert e > 0, 'a 0.13 A contact must cost energy, not release it'
    assert np.abs(f).max() < 1e5
