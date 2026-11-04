"""The descriptor never sees a separation where the learned heads have diverged.

The model was trained with the existing repulsion and its switching function, so
the floor must not touch anything the model is still interpolating. It therefore
engages only where the pair's OWN tabulated NLH repulsion already exceeds
``PAIR_FLOOR_ENERGY`` -- one physically meaningful knob, no per-pair tuning.

Both bounds are measured (the measurements and their consequences are in the
comment block next to ``PAIR_FLOOR_ENERGY`` in embed_sparse.py):

  ONSET  the outermost separation where the UN-floored network diverges
         (|F| > 1e3 eV/A or NaN) on dimer scans of the three v2 checkpoints.
         ``rmin`` must reach past it.
  REAL   the closest contact per pair across all 73 labelled test sets (the
         evaluation behind the SO3LR-2 paper's test-set errors). ``rmin`` must
         stay below it, or the floor overrides trained physics.

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

# (onset, closest real contact) per pair [A].
MEASURED = {
    ('B', 'H'): (0.600, 0.703), ('Br', 'H'): (0.580, 0.965),
    ('C', 'C'): (0.500, 0.986), ('C', 'H'): (0.680, 0.691),
    ('C', 'N'): (0.760, 0.951), ('C', 'O'): (0.800, 0.934),
    ('Cl', 'Cl'): (0.860, 1.415), ('Cl', 'H'): (0.620, 0.885),
    ('F', 'F'): (0.600, 1.014), ('F', 'H'): (0.500, 0.658),
    ('Ge', 'H'): (0.720, 0.952), ('H', 'H'): (0.280, 0.407),
    ('Li', 'F'): (0.480, 0.975), ('N', 'H'): (0.620, 0.668),
    ('N', 'N'): (0.480, 0.970), ('N', 'O'): (0.460, 0.922),
    ('O', 'H'): (0.600, 0.666), ('O', 'O'): (0.580, 0.959),
    ('P', 'O'): (0.680, 1.175), ('P', 'P'): (1.300, 1.733),
    ('P', 'S'): (1.260, 1.534), ('S', 'F'): (0.600, 1.217),
    ('S', 'H'): (0.560, 0.891), ('S', 'O'): (0.580, 1.150),
    ('S', 'S'): (1.020, 1.467), ('Si', 'H'): (0.600, 0.915),
}

# Pairs whose bare-dimer onset rmin cannot reach -- see the comment block in
# embed_sparse.py for why only P-S is a real problem. C-H is a choice, not a
# necessity: 4.4-4.6 eV would put rmin past its 0.680 A onset and still below
# every closest real contact, but only 3, 4 and 5 eV were evaluated, and 5 eV is
# what the paper used.
UNPROTECTED = {('P', 'P'), ('P', 'S'), ('C', 'H')}

# Outermost severe failure (NaN, |E_nn| > 100 eV or |F| > 1e3 eV/A) per pair
# from the first scan: 18 pairs x 4 checkpoints, legacy so3lr included, 0.01 A
# steps. Broader than MEASURED in models and failure criterion.
SEVERE_EDGE = {('H', 'H'): 0.28, ('C', 'H'): 0.68, ('N', 'H'): 0.63,
               ('O', 'H'): 0.60, ('F', 'H'): 0.51, ('S', 'H'): 0.56,
               ('Cl', 'H'): 0.62, ('P', 'H'): 0.61, ('C', 'C'): 0.77,
               ('C', 'N'): 0.76, ('C', 'O'): 0.80, ('N', 'N'): 0.48,
               ('N', 'O'): 0.48, ('O', 'O'): 0.59, ('C', 'S'): 0.80,
               ('S', 'S'): 1.05, ('C', 'Cl'): 0.89, ('P', 'O'): 0.62}

# Closest approach anywhere in the 300-673 K / 1-250 bar NPT water series.
WATER_CLOSEST = {('O', 'H'): 0.863, ('H', 'H'): 1.239, ('O', 'O'): 2.180}

# Pairs whose closest real contact lies inside the handover shoulder
# (rmin < real < rmin + PAIR_FLOOR_WIDTH). The floor moves those distances by
# under 0.005 A, but its slope there is 0.65-0.95, so the network's radial
# derivative on those contacts is scaled down by up to 35%. That shoulder is the
# whole cost of the width.
SHOULDER = {('B', 'H'), ('C', 'H'), ('F', 'H'), ('N', 'H'), ('O', 'H')}

# Equilibrium bond lengths [A]. The floor must be exactly inert at and above
# each of these.
BONDS = {('H', 'H'): 0.741, ('C', 'H'): 1.09, ('N', 'H'): 1.01, ('O', 'H'): 0.96,
         ('C', 'C'): 1.54, ('C', 'O'): 1.43, ('C', 'N'): 1.47, ('O', 'O'): 1.21,
         ('S', 'S'): 1.89, ('C', 'S'): 1.82, ('C', 'Cl'): 1.77, ('S', 'H'): 1.34}


def _rmin(e1, e2, energy=PAIR_FLOOR_ENERGY):
    """The value the descriptor distance is clamped to for this pair."""
    t = np.asarray(nlh_reference_distance(energy))
    z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
    return float(t[min(z1, z2) - 1, max(z1, z2) - 1])


def _handover(e1, e2):
    """At and above this the floor is exactly the identity."""
    return _rmin(e1, e2) + PAIR_FLOOR_WIDTH


def _floor(e1, e2, d):
    z = jnp.array([atomic_numbers[e1], atomic_numbers[e2]])
    return floor_descriptor_distance(jnp.atleast_1d(jnp.asarray(d, float)), z,
                                     jnp.array([0]), jnp.array([1]))


def test_inert_at_and_above_every_real_bond():
    """Ordinary chemistry must be bit-for-bit untouched."""
    bad = []
    for (e1, e2), bond in BONDS.items():
        handover = _handover(e1, e2)
        if handover >= bond:
            bad.append(f'{e1}-{e2}: handover {handover:.3f} >= bond {bond}')
            continue
        d = jnp.array([bond, bond + 0.2, 1.5 * bond, 4.0])
        if not np.array_equal(np.asarray(_floor(e1, e2, d)), np.asarray(d)):
            bad.append(f'{e1}-{e2}: not the identity at/above its {bond} A bond')
    assert not bad, 'floor perturbs normal geometries:\n  ' + '\n  '.join(bad)


def test_inert_at_the_closest_approach_real_simulations_reach():
    """The binding constraint for MD is not the equilibrium bond but the
    shortest contact hot compressed water actually samples: O-H reaches 0.863 A
    at 673 K / 250 bar. Compared in the array's own dtype -- a float32 input
    against a float64 literal differs by 2e-8 for reasons unrelated to the floor."""
    for (e1, e2), closest in WATER_CLOSEST.items():
        assert _handover(e1, e2) < closest, (e1, e2)
        d = jnp.array([closest])
        npt.assert_array_equal(np.asarray(_floor(e1, e2, d)), np.asarray(d))


def test_does_not_intrude_on_labelled_data():
    """The binding constraint, and the reason this is 5 eV rather than 3.

    Below the closest contact the labelled data actually contains, the model is
    interpolating; clamping there overrides trained physics and measurably
    worsens forces. rmin must sit below that contact for EVERY pair, with no
    exceptions, so every violation is reported at once rather than one per run.
    The handover shoulder above rmin does reach a few X-H contacts; that is
    pinned separately in test_width_adds_intrusion_and_no_protection.
    """
    bad = [f'{e1}-{e2}: clamp {_rmin(e1, e2):.3f} >= closest real contact {real:.3f}'
           for (e1, e2), (_, real) in MEASURED.items() if _rmin(e1, e2) >= real]
    assert not bad, ('floor is active on trained data for:\n  '
                     + '\n  '.join(bad))


def test_reaches_past_the_measured_failure_onset():
    """The clamp has to sit outside the diverging band, not merely near it.

    Failures are not monotone in distance -- so3lr-tauri-l is sane for S-S at 0.2 A,
    where 18 keV of repulsion swamps everything, and catastrophic further out --
    so clamping to a distance inside the band just moves the descriptor from one
    bad place to another. The pairs it cannot reach must be exactly the
    documented ones, so a constant change cannot quietly add a fourth and fixing
    one shows up as a failure rather than going unnoticed.
    """
    unreached = {pair for pair, (onset, _) in MEASURED.items() if _rmin(*pair) <= onset}
    assert unreached == UNPROTECTED, (
        f'newly unprotected: {sorted(unreached - UNPROTECTED)}; '
        f'now protected (remove from UNPROTECTED): {sorted(UNPROTECTED - unreached)}')


def test_reaches_past_the_four_checkpoint_scan():
    """The floor also runs in legacy so3lr, so it must clear that scan's edges
    too. C-H is the documented gap."""
    bad = [f'{e1}-{e2}: clamp {_rmin(e1, e2):.3f} <= edge {edge}'
           for (e1, e2), edge in SEVERE_EDGE.items()
           if (e1, e2) != ('C', 'H') and _rmin(e1, e2) <= edge]
    assert not bad, 'floor lands inside a severe band for:\n  ' + '\n  '.join(bad)


@pytest.mark.parametrize('pair', [('P', 'P'), ('P', 'S')])
def test_phosphorus_pairs_are_beyond_any_usable_global_energy(pair):
    """Even at 1 eV -- five times more intrusive than the shipped setting, and
    enough to floor real contacts across the board -- rmin still falls short;
    reaching the onsets would take ~0.13 eV (P-S) and ~0.03 eV (P-P). These need
    a per-pair floor or a stronger P repulsion."""
    onset = MEASURED[pair][0]
    at_1ev = _rmin(*pair, energy=1.0)
    assert at_1ev < onset, (
        f'{pair}: rmin reaches {at_1ev:.3f} A already at 1 eV, clearing the '
        f'{onset} A onset -- a global energy could protect it after all')


def test_the_energy_threshold_is_the_only_knob_and_it_is_thermally_unreachable():
    """PAIR_FLOOR_ENERGY is the whole calibration. Reaching rmin costs 5 eV,
    ~58 kT at 1000 K, in a single pair coordinate; the handover starts 0.05 A
    further out, where the repulsion is 2.4-3.5 eV (28-40 kT). Either way 'the
    model is extrapolating here' is a statement about the input, not a guess."""
    assert PAIR_FLOOR_ENERGY == 5.0
    kT_1000K = 0.0861733  # eV
    assert PAIR_FLOOR_ENERGY / kT_1000K > 50


def test_width_adds_intrusion_and_no_protection():
    """Protection is set by rmin alone; the width only widens the shoulder that
    reaches into real data. Kept at 0.05 because that is what the 73-set
    evaluation was run with, but it is pure cost -- pinned so the trade-off
    stays visible, and the shoulder must stay confined to the documented pairs."""
    assert PAIR_FLOOR_WIDTH == 0.05
    shoulder = {pair for pair, (_, real) in MEASURED.items() if _handover(*pair) > real}
    assert shoulder == SHOULDER
    for pair in SHOULDER:
        real = MEASURED[pair][1]
        assert float(_floor(*pair, real)[0]) - real < 0.005, pair


def test_scale_is_pair_specific_and_ordered_by_size():
    """A single global floor cannot serve both: S-S needs ~3x the reach of H-H."""
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


def test_floor_is_symmetric_in_the_pair():
    """The table is indexed [min(Z), max(Z)], so an O-H edge and an H-O edge
    must be floored identically. This goes through floor_descriptor_distance:
    _rmin sorts the pair itself and so could not notice a missing min/max."""
    d = jnp.array([0.0, 0.2, 0.5])
    for e1, e2 in [('O', 'H'), ('C', 'S')]:
        forward = np.asarray(_floor(e1, e2, d))
        reverse = np.asarray(_floor(e2, e1, d))
        # The floor is active at d = 0, so the comparison is not identity vs identity.
        assert forward[0] > 0, (e1, e2)
        npt.assert_array_equal(forward, reverse)


def test_clash_stays_finite_and_repulsive_end_to_end():
    from ase import Atoms

    from so3lr import So3lrCalculator

    water = Atoms('OHH', positions=[[0., 0., 0.], [0.13, 0., 0.], [-0.24, 0.93, 0.]])
    water.calc = So3lrCalculator(model='so3lr-tauri-s')
    e, f = water.get_potential_energy(), water.get_forces()
    assert np.isfinite(e) and np.isfinite(f).all()
    assert e > 0, 'a 0.13 A contact must cost energy, not release it'
    assert np.abs(f).max() < 1e5
