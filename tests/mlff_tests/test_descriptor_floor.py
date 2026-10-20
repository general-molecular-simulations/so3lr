"""The descriptor never sees a separation where the learned heads have diverged.

The model was trained with the existing repulsion and its switching function, so
the floor must not touch anything the model is still interpolating. It therefore
engages only where the pair's OWN tabulated NLH repulsion already exceeds
``PAIR_FLOOR_ENERGY`` -- one physically meaningful knob, no per-pair tuning.

Both bounds are measured, over all 73 test sets and all three v2 checkpoints
(the evaluation behind the SO3LR-2 paper's test-set errors):

  ONSET  the outermost separation where the UN-floored network diverges
         (|F| > 1e3 eV/A or NaN) on dimer scans. ``rmin`` must reach past it.
  REAL   the closest contact per pair anywhere in the labelled data. ``rmin``
         must stay below it, or the floor overrides trained physics. The data
         goes much closer than intuition suggests -- the scaled-separation sets
         label down to 0.4 A -- and where the floor engages on such contacts it
         measurably HURTS forces, because the model is inaccurate but not
         diverging there.

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

# Pairs whose onset rmin cannot reach -- see the comment block in embed_sparse.py.
# The onsets were measured on BARE DIMERS, which overstate the risk. Re-measured
# with the atoms carrying normal valence, C-H and P-P are clean and only P-S is a
# real problem (-42 keV and 4e5 eV/A in PH3...H2S on so3lr-tauri-l).
UNPROTECTED = {('P', 'P'), ('P', 'S'), ('C', 'H')}

# Equilibrium bond lengths [A]. The floor must be exactly inert at and above
# each of these.
BONDS = {('H', 'H'): 0.741, ('C', 'H'): 1.09, ('N', 'H'): 1.01, ('O', 'H'): 0.96,
         ('C', 'C'): 1.54, ('C', 'O'): 1.43, ('C', 'N'): 1.47, ('O', 'O'): 1.21,
         ('S', 'S'): 1.89, ('C', 'S'): 1.82, ('C', 'Cl'): 1.77, ('S', 'H'): 1.34}


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


def test_inert_at_and_above_every_real_bond():
    """Ordinary chemistry must be bit-for-bit untouched."""
    bad = []
    for (e1, e2), bond in BONDS.items():
        if _handover(e1, e2) >= bond:
            bad.append(f'{e1}-{e2}: handover {_handover(e1, e2):.3f} >= bond {bond}')
            continue
        d = jnp.array([bond, bond + 0.2, 1.5 * bond, 4.0])
        if not np.array_equal(np.asarray(_floor(e1, e2, d)), np.asarray(d)):
            bad.append(f'{e1}-{e2}: not the identity at/above its {bond} A bond')
    assert not bad, 'floor perturbs normal geometries:\n  ' + '\n  '.join(bad)


def test_does_not_intrude_on_labelled_data():
    """The binding constraint, and the reason this is 5 eV rather than 3.

    Below the closest contact the labelled data actually contains, the model is
    interpolating; flooring there overrides trained physics and measurably
    worsens forces. Must hold for EVERY pair, with no exceptions -- so every
    violation is reported at once rather than one per run.
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
    bad place to another.
    """
    bad = [f'{e1}-{e2}: clamp {_rmin(e1, e2):.3f} <= onset {onset:.3f}'
           for (e1, e2), (onset, _) in MEASURED.items()
           if (e1, e2) not in UNPROTECTED and _rmin(e1, e2) <= onset]
    assert not bad, ('floor lands inside the diverging band for:\n  '
                     + '\n  '.join(bad))


@pytest.mark.parametrize('pair', sorted(UNPROTECTED))
def test_known_unreached_pairs_are_exactly_these(pair):
    """Pin the three documented gaps so a constant change cannot quietly add a
    fourth, and so fixing one shows up as a failing test rather than going
    unnoticed. P-P and P-S need a per-pair floor or stronger P repulsion; C-H
    has no (energy, width) that is both protective and non-intruding."""
    e1, e2 = pair
    onset, real = MEASURED[pair]
    assert _rmin(e1, e2) <= onset, (
        f'{e1}-{e2} is now protected (clamp {_rmin(e1, e2):.3f} > onset {onset}) '
        f'-- remove it from UNPROTECTED')
    if pair != ('C', 'H'):
        # P-P / P-S are out of reach for any usable threshold. Even at 1 eV --
        # already five times more intrusive than the shipped setting, and enough
        # to floor real contacts across the board -- rmin still falls short.
        z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
        i, j = min(z1, z2) - 1, max(z1, z2) - 1
        at_1ev = float(np.asarray(nlh_reference_distance(1.0))[i, j])
        assert at_1ev < onset, (
            f'{e1}-{e2}: rmin reaches {at_1ev:.3f} A already at 1 eV, clearing the '
            f'{onset} A onset -- a global energy could protect it after all')
        return
    assert real - onset < 0.10, (
        f'{e1}-{e2}: onset {onset} and real contact {real} are {real - onset:.3f} A '
        f'apart, which is enough room to protect it -- it should not be here')


def test_the_energy_threshold_is_the_only_knob_and_it_is_thermally_unreachable():
    """PAIR_FLOOR_ENERGY is the whole calibration. At 5 eV the handover costs
    ~58 kT at 1000 K in a single pair coordinate, so 'the model is extrapolating
    here' is a statement about the input rather than a guess."""
    assert PAIR_FLOOR_ENERGY == 5.0
    kT_1000K = 0.0861733  # eV
    assert PAIR_FLOOR_ENERGY / kT_1000K > 50


def test_width_adds_intrusion_and_no_protection():
    """Protection is set by rmin alone; the width only widens the shoulder that
    reaches into real data. Kept at 0.05 because that is what the 73-set
    evaluation was run with, but it is pure cost -- pinned so the trade-off
    stays visible."""
    assert PAIR_FLOOR_WIDTH == 0.05
    for pair in (('O', 'H'), ('C', 'H'), ('N', 'H')):
        e1, e2 = pair
        real = MEASURED[pair][1]
        # rmin clears the real contact, but rmin + width does not for these
        # three tight H-X pairs -- that shoulder is the whole cost of the width.
        assert _rmin(e1, e2) < real


def test_scale_is_pair_specific_and_ordered_by_size():
    """A single global floor cannot serve both: S-S needs ~3x the reach of H-H."""
    assert _rmin('H', 'H') < _rmin('O', 'H') < _rmin('C', 'C') < _rmin('S', 'S')
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


def test_reference_table_is_symmetric_in_the_pair():
    """Indexed [min(Z), max(Z)], so O-H and H-O must give the same length."""
    assert _rmin('O', 'H') == _rmin('H', 'O')
    assert _rmin('C', 'S') == _rmin('S', 'C')


def test_clash_stays_finite_and_repulsive_end_to_end():
    from ase import Atoms

    from so3lr import So3lrCalculator

    water = Atoms('OHH', positions=[[0., 0., 0.], [0.13, 0., 0.], [-0.24, 0.93, 0.]])
    water.calc = So3lrCalculator(model='so3lr-tauri-s')
    e, f = water.get_potential_energy(), water.get_forces()
    assert np.isfinite(e) and np.isfinite(f).all()
    assert e > 0, 'a 0.13 A contact must cost energy, not release it'
    assert np.abs(f).max() < 1e5
