"""The descriptor never sees a distance the model was not trained on.

The learned heads extrapolate catastrophically at a steric clash: on so3lr-m a
0.2 A contact in a 42-atom peptide puts >100 e on atoms a full bond away (their
own geometry is normal -- message passing carries the damage), and the resulting
Coulomb term reaches 2.5e4 eV against 200 eV of NLH repulsion, so the total
force field points the wrong way.

Flooring the distance handed to the radial basis removes the cause: the network
is only ever asked about separations inside its training range
(``data.filter.min_distance = 0.5``), while the repulsion and the electrostatics
keep the true distance and still diverge at real contact. Measured on that
peptide at 0.2 A: fmax 3.8e6 -> 1.8e3 eV/A, second-shell charge 125 e -> 0.39 e,
electrostatics 2.5e4 -> -3.8 eV, and NLH goes from 0% to ~100% of dE/dd.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from so3lr.mlff.nn.embed.embed_sparse import (
    DESCRIPTOR_FLOOR_OFF,
    DESCRIPTOR_MIN_DISTANCE,
    floor_descriptor_distance,
)


def test_exactly_inert_at_and_above_the_handover():
    """Every real bond is at or above the handover (shortest is H-H, 0.74 A),
    so the floor must be bit-for-bit the identity there -- not merely close."""
    d = jnp.array([DESCRIPTOR_FLOOR_OFF, 0.74, 0.96, 1.5, 4.5, 12.0])
    npt.assert_array_equal(np.asarray(floor_descriptor_distance(d)), np.asarray(d))


def test_floors_below_the_training_minimum():
    d = jnp.array([0.0, 0.05, 0.13, 0.2, 0.35])
    out = np.asarray(floor_descriptor_distance(d))
    assert out.min() >= DESCRIPTOR_MIN_DISTANCE
    # and it stays near the floor rather than shooting past it
    assert out.max() <= DESCRIPTOR_MIN_DISTANCE + 0.05


def test_monotone_and_smooth():
    """A non-monotone floor would invent spurious forces, and a kink would make
    the energy non-differentiable somewhere reachable."""
    d = jnp.linspace(0.01, 2.0, 500)
    out = np.asarray(floor_descriptor_distance(d))
    assert np.all(np.diff(out) >= -1e-7)
    # derivative approaches 1 continuously from below at the handover
    g = jax.grad(lambda x: floor_descriptor_distance(x).sum())
    near = np.asarray(g(jnp.array([DESCRIPTOR_FLOOR_OFF - 1e-3,
                                   DESCRIPTOR_FLOOR_OFF + 1e-3])))
    npt.assert_allclose(near, [1.0, 1.0], atol=2e-3)


def test_gradient_is_finite_at_zero_distance():
    """Two atoms at exactly the same position must not produce NaN forces."""
    g = jax.grad(lambda x: floor_descriptor_distance(x).sum())
    assert np.isfinite(np.asarray(g(jnp.array([0.0, 0.0, 0.5])))).all()


def test_floor_is_below_every_bond_in_the_bundled_example():
    """Guards the choice of DESCRIPTOR_FLOOR_OFF: if it ever crept above a real
    bond length, the floor would start perturbing ordinary chemistry."""
    from ase.io import read
    import pathlib

    xyz = (pathlib.Path(__file__).resolve().parents[2]
           / 'examples' / 'data' / 'ala15_folded.xyz')
    if not xyz.exists():
        pytest.skip('bundled example structure not available')
    atoms = read(xyz, index=0)
    pos = atoms.get_positions()
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    shortest = d[~np.eye(len(atoms), dtype=bool)].min()
    assert shortest > DESCRIPTOR_FLOOR_OFF, (
        f'shortest bond {shortest:.3f} A is inside the floor handover '
        f'{DESCRIPTOR_FLOOR_OFF} A'
    )


def test_clash_forces_stay_bounded_end_to_end():
    """The property the floor exists for: a hard clash must stay finite and
    repulsion-dominated rather than collapsing into a Coulomb sink."""
    from ase import Atoms

    from so3lr import So3lrCalculator

    water = Atoms('OHH', positions=[[0., 0., 0.], [0.13, 0., 0.], [-0.24, 0.93, 0.]])
    water.calc = So3lrCalculator(model='so3lr-s')
    e = water.get_potential_energy()
    f = water.get_forces()
    assert np.isfinite(e) and np.isfinite(f).all()
    assert e > 0, 'a 0.13 A contact must cost energy, not release it'
    assert np.abs(f).max() < 1e5
