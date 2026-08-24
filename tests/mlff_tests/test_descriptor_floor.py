"""The descriptor never sees a separation where the learned heads have failed.

The model was trained with the existing repulsion and its switching function, so
the floor must not touch anything the model is still interpolating. It therefore
engages only where the pair's OWN tabulated NLH repulsion already exceeds
``PAIR_FLOOR_ENERGY`` -- one physically meaningful knob, no per-pair tuning.

Both sides of that choice are pinned here:

  * it REACHES far enough -- past the outermost severe failure measured for each
    pair on the worst of the four shipped checkpoints (NaN, learned energy over
    100 eV, or forces over 1e3 eV/A);
  * it STAYS SHORT enough -- below every equilibrium bond length, and below the
    closest approach observed in the 300-673 K / 1-250 bar NPT water series.

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
    PAIR_FLOOR_ENERGY,
    PAIR_FLOOR_WIDTH,
    floor_descriptor_distance,
    nlh_reference_distance,
)

# Equilibrium bond lengths [A]. The floor must be exactly inert at and above
# each of these.
BONDS = {('H', 'H'): 0.741, ('C', 'H'): 1.09, ('N', 'H'): 1.01, ('O', 'H'): 0.96,
         ('C', 'C'): 1.54, ('C', 'O'): 1.43, ('C', 'N'): 1.47, ('O', 'O'): 1.21,
         ('S', 'S'): 1.89, ('C', 'S'): 1.82, ('C', 'Cl'): 1.77, ('S', 'H'): 1.34}

# Outermost SEVERE failure per pair, over all four shipped checkpoints, measured
# with no floor at 0.01 A resolution: non-finite output, |learned energy| > 100
# eV, or |force| > 1e3 eV/A. The floor has to reach past every one of these.
SEVERE_EDGE = {('H', 'H'): 0.28, ('C', 'H'): 0.68, ('N', 'H'): 0.63,
               ('O', 'H'): 0.60, ('F', 'H'): 0.51, ('S', 'H'): 0.56,
               ('Cl', 'H'): 0.62, ('P', 'H'): 0.61, ('C', 'C'): 0.77,
               ('C', 'N'): 0.76, ('C', 'O'): 0.80, ('N', 'N'): 0.48,
               ('N', 'O'): 0.48, ('O', 'O'): 0.59, ('C', 'S'): 0.80,
               ('S', 'S'): 1.05, ('C', 'Cl'): 0.89, ('P', 'O'): 0.62}

# Closest approach anywhere in the 300-673 K / 1-250 bar NPT water series.
WATER_CLOSEST = {('O', 'H'): 0.863, ('H', 'H'): 1.239, ('O', 'O'): 2.180}


def _rmin(e1, e2):
    """The value the descriptor distance is clamped to for this pair."""
    t = np.asarray(nlh_reference_distance())
    z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
    return float(t[min(z1, z2) - 1, max(z1, z2) - 1])


def _handover(e1, e2):
    """At and above this the floor is exactly the identity."""
    return _rmin(e1, e2) + PAIR_FLOOR_WIDTH


def _floor(e1, e2, d):
    z = jnp.array([atomic_numbers[e1], atomic_numbers[e2]])
    return floor_descriptor_distance(jnp.atleast_1d(jnp.asarray(d, float)), z,
                                     jnp.array([0]), jnp.array([1]))


@pytest.mark.parametrize('pair,bond', list(BONDS.items()))
def test_inert_at_and_above_every_real_bond(pair, bond):
    """Ordinary chemistry must be bit-for-bit untouched."""
    e1, e2 = pair
    assert _handover(e1, e2) < bond, (
        f'{e1}-{e2}: handover {_handover(e1, e2):.3f} A is at or above the '
        f'{bond} A bond, so the floor would perturb normal geometries')
    d = jnp.array([bond, bond + 0.2, 1.5 * bond, 4.0])
    npt.assert_array_equal(np.asarray(_floor(e1, e2, d)), np.asarray(d))


@pytest.mark.parametrize('pair,closest', list(WATER_CLOSEST.items()))
def test_inert_at_the_closest_approach_real_simulations_reach(pair, closest):
    """The binding constraint is not the equilibrium bond but the shortest
    contact hot compressed water actually samples. O-H reaches 0.863 A at
    673 K / 250 bar; the handover has to stay clear of it with room to spare."""
    e1, e2 = pair
    h = _handover(e1, e2)
    assert h < closest, (
        f'{e1}-{e2}: handover {h:.3f} A is at or beyond the {closest} A closest '
        f'approach seen in equilibrated water -- the floor would be active in a '
        f'regime the model was trained on')
    # Compare in the array's own dtype: a float32 input against the float64
    # Python literal differs by 2e-8 for reasons that have nothing to do with
    # the floor.
    d = jnp.array([closest])
    npt.assert_array_equal(np.asarray(_floor(e1, e2, d)), np.asarray(d))


@pytest.mark.parametrize('pair,edge', list(SEVERE_EDGE.items()))
def test_floor_reaches_past_the_measured_severe_failure(pair, edge):
    """The clamp value has to sit outside the broken band, not merely near it.
    Failures are not monotone in distance -- so3lr-l is sane for S-S at 0.2 A,
    where 18 keV of repulsion swamps everything, and catastrophic from 0.45 to
    1.05 A -- so clamping to a distance inside the band would just move the
    descriptor from one bad place to another."""
    e1, e2 = pair
    assert _rmin(e1, e2) > edge, (
        f'{e1}-{e2}: clamp {_rmin(e1, e2):.3f} A is inside the broken band '
        f'(outermost severe failure at {edge} A)')


def test_the_energy_threshold_is_the_only_knob_and_it_is_thermally_unreachable():
    """PAIR_FLOOR_ENERGY is the whole calibration. At 3 eV the handover costs
    ~35 kT at 1000 K in a single pair coordinate, which is what makes 'the model
    is extrapolating here' a statement about the input rather than a guess."""
    assert PAIR_FLOOR_ENERGY == 3.0
    kT_1000K = 0.0861733  # eV
    assert PAIR_FLOOR_ENERGY / kT_1000K > 30


def test_scale_is_pair_specific_and_ordered_by_size():
    """A single global floor cannot serve both: S-S needs ~4x the reach of H-H."""
    assert _rmin('H', 'H') < _rmin('O', 'H') < _rmin('C', 'C') < _rmin('S', 'S')
    # At H-H's value S-S would be unprotected; at S-S's value H-H would be
    # floored above its own bond length.
    assert _rmin('S', 'S') / _rmin('H', 'H') > 2.0


def test_reference_distance_matches_the_repulsion_the_model_computes():
    """The ruler must track the SHIPPED repulsion, switching function included.

    NLHRepulsionSparse multiplies by switching_fn(d, 0, 1.5), which drives the
    term to exactly zero above 1.5 A. Omitting it makes the repulsion look like
    it reaches 2 A for S-S when the model has already switched it off, and every
    floor derived from it comes out too far.
    """
    from so3lr.mlff.nn.observable.dispersion_ref_data import NLH_AA, NLH_BB
    from so3lr.mlff.nn.observable.observable_sparse import switching_fn

    ke = 14.399645351950548
    for e1, e2 in [('O', 'H'), ('C', 'C'), ('S', 'S')]:
        z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
        i, j = min(z1, z2) - 1, max(z1, z2) - 1
        r = _rmin(e1, e2)
        y = float(np.sum(NLH_AA[i, j] * np.exp(-NLH_BB[i, j] * r)))
        e = float(switching_fn(r, 0.0, 1.5)) * ke * z1 * z2 / r * y
        npt.assert_allclose(e, PAIR_FLOOR_ENERGY, rtol=1e-2)
    # and every reference distance must sit inside the repulsion's own range
    assert np.asarray(nlh_reference_distance()).max() < 1.5


def test_clamps_to_rmin_and_never_below():
    for e1, e2 in [('H', 'H'), ('O', 'H'), ('S', 'S')]:
        out = np.asarray(_floor(e1, e2, jnp.array([0.0, 0.05, 0.2, 0.4])))
        rmin = _rmin(e1, e2)
        assert out.min() >= rmin - 1e-6
        assert out.max() <= rmin + PAIR_FLOOR_WIDTH


def test_monotone_and_finite_gradient_including_zero():
    """A non-monotone floor would invent forces; a NaN at d=0 would poison MD."""
    z = jnp.array([8, 1])
    ii, jj = jnp.array([0]), jnp.array([1])
    d = jnp.linspace(0.0, 2.0, 400)
    out = np.asarray(floor_descriptor_distance(d, z, ii, jj))
    assert np.all(np.diff(out) >= -1e-7)
    g = jax.grad(lambda x: floor_descriptor_distance(x, z, ii, jj).sum())
    assert np.isfinite(np.asarray(g(jnp.array([0.0, 0.5, 1.2])))).all()


def test_reference_table_is_symmetric_in_the_pair():
    """Indexed [min(Z), max(Z)], so O-H and H-O must give the same length."""
    assert _rmin('O', 'H') == _rmin('H', 'O')
    assert _rmin('C', 'S') == _rmin('S', 'C')


def test_clash_stays_finite_and_repulsive_end_to_end():
    from ase import Atoms

    from so3lr import So3lrCalculator

    water = Atoms('OHH', positions=[[0., 0., 0.], [0.13, 0., 0.], [-0.24, 0.93, 0.]])
    water.calc = So3lrCalculator(model='so3lr-s')
    e, f = water.get_potential_energy(), water.get_forces()
    assert np.isfinite(e) and np.isfinite(f).all()
    assert e > 0, 'a 0.13 A contact must cost energy, not release it'
    assert np.abs(f).max() < 1e5
