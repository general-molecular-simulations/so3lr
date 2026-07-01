"""Tests for the cylindrical / funnel restraint (so3lr.cli.so3lr_md).

Three tiers:
  * Tier 1 - mechanical correctness of the energy term (force = -grad E, exact flat
    bottom, C1 continuity, mass-weighted COM, minimum image, general axis, selectivity).
  * Tier 2 - settings parsing/validation and a self-checking regression.
  * Tier 3 - thermodynamic correctness: a Brownian-dynamics sampler driven by the
    implemented energy must reproduce the analytical Boltzmann distributions
    (radial and axial). Marked ``slow``.

The energy term is tested in isolation (there is no base potential), which is exactly
how it must behave inside SO3LR's ``energy_fn`` wrapper.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from so3lr.cli.so3lr_md import cylindrical_restraint_energy, parse_cylinder_settings


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def ref_energy(R, idx, w, axis_hat, origin, radius, k_r,
               axial_min=None, axial_max=None, k_z=None, box=None, periodic=False):
    """Independent NumPy reference implementation of the restraint energy."""
    R = np.asarray(R, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    axis_hat = np.asarray(axis_hat, dtype=float)
    origin = np.asarray(origin, dtype=float)
    com = np.sum(w * R[idx], axis=0)
    p = com - origin
    if periodic and box is not None:
        box = np.asarray(box, dtype=float)
        diag = np.diag(box) if box.ndim == 2 else box
        p = p - diag * np.round(p / diag)
    s = float(np.dot(p, axis_hat))
    r_perp = float(np.linalg.norm(p - s * axis_hat))
    kz = k_r if k_z is None else k_z
    e = 0.5 * k_r * max(r_perp - radius, 0.0) ** 2
    if axial_min is not None:
        e += 0.5 * kz * max(axial_min - s, 0.0) ** 2
    if axial_max is not None:
        e += 0.5 * kz * max(s - axial_max, 0.0) ** 2
    return e


def make_system(seed=0, n_atoms=6):
    rng = np.random.default_rng(seed)
    R = rng.uniform(-3.0, 3.0, size=(n_atoms, 3))
    idx = np.array([1, 3, 4])
    masses = np.array([12.0, 1.0, 16.0])          # unequal masses
    w = (masses / masses.sum())[:, None]
    return R, idx, jnp.asarray(w)


AXIS = _unit([0.0, 0.0, 1.0])
ORIGIN = np.array([0.0, 0.0, 0.0])
RADIUS = 2.0
KR = 3.0


def E_jax(R, idx, w, **kw):
    return float(cylindrical_restraint_energy(
        jnp.asarray(R), jnp.asarray(idx), jnp.asarray(w), **kw))


# --------------------------------------------------------------------------- #
# Tier 1 - mechanical correctness
# --------------------------------------------------------------------------- #
class TestForceMatchesGradient:
    @pytest.mark.parametrize("shift,label", [
        (np.array([5.0, 0.0, 0.0]), "outside_radius"),
        (np.array([0.0, 0.0, -9.0]), "below_axial_min"),
        (np.array([0.0, 0.0, 9.0]), "above_axial_max"),
        (np.array([5.0, 0.0, 9.0]), "corner"),
    ])
    def test_finite_difference(self, shift, label):
        R, idx, w = make_system()
        # push the whole guest so its COM lands in the target region
        R = R.copy()
        R[idx] += shift
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                  radius=RADIUS, k_r=KR, axial_min=-4.0, axial_max=4.0, k_z=2.0)

        def scalar(Rj):
            return cylindrical_restraint_energy(Rj, jnp.asarray(idx), w, **kw)

        analytic = -np.asarray(jax.grad(scalar)(jnp.asarray(R)))

        eps = 1e-6
        numeric = np.zeros_like(R)
        for a in range(R.shape[0]):
            for c in range(3):
                Rp = R.copy(); Rp[a, c] += eps
                Rm = R.copy(); Rm[a, c] -= eps
                ep = scalar(jnp.asarray(Rp)); em = scalar(jnp.asarray(Rm))
                numeric[a, c] = -float(ep - em) / (2 * eps)
        npt.assert_allclose(analytic, numeric, atol=1e-6, rtol=1e-5,
                            err_msg=f"force mismatch in region {label}")
        # only the restrained atoms feel a force
        others = [i for i in range(R.shape[0]) if i not in idx]
        npt.assert_allclose(analytic[others], 0.0, atol=1e-12)


class TestFlatBottom:
    def test_zero_energy_and_force_inside(self):
        R, idx, w = make_system()
        # place COM exactly at origin, inside radius and within axial band
        R = R.copy()
        com = np.sum(np.asarray(w) * R[idx], axis=0)
        R[idx] -= com  # COM -> 0
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                  radius=RADIUS, k_r=KR, axial_min=-4.0, axial_max=4.0, k_z=2.0)

        def scalar(Rj):
            return cylindrical_restraint_energy(Rj, jnp.asarray(idx), w, **kw)

        assert float(scalar(jnp.asarray(R))) == 0.0
        g = np.asarray(jax.grad(scalar)(jnp.asarray(R)))
        npt.assert_array_equal(g, np.zeros_like(g))


class TestContinuityAtWall:
    def test_force_vanishes_approaching_wall(self):
        # single atom guest on the x-axis; move radially through the wall
        idx = np.array([0])
        w = jnp.ones((1, 1))
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                  radius=RADIUS, k_r=KR)

        def force_mag(r):
            R = jnp.asarray([[r, 0.0, 0.0]])
            g = jax.grad(lambda Rj: cylindrical_restraint_energy(Rj, jnp.asarray(idx), w, **kw))(R)
            return float(np.linalg.norm(np.asarray(g)))

        # inside: zero; just outside: ~ k_r*dr -> 0 as dr->0 (C1 continuity)
        assert force_mag(RADIUS - 0.1) == 0.0
        for dr in (0.1, 0.01, 0.001):
            npt.assert_allclose(force_mag(RADIUS + dr), KR * dr, rtol=1e-6, atol=1e-9)


class TestCenterOfMass:
    def test_unequal_masses_match_reference(self):
        R, idx, w = make_system()
        R = R.copy(); R[idx] += np.array([6.0, 0.0, 0.0])  # push outside
        got = E_jax(R, idx, w, axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                    radius=RADIUS, k_r=KR)
        exp = ref_energy(R, idx, np.asarray(w), AXIS, ORIGIN, RADIUS, KR)
        npt.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)

    def test_single_atom_reduces_to_atom_position(self):
        R, _, _ = make_system()
        idx = np.array([2])
        w = jnp.ones((1, 1))
        got = E_jax(R, idx, w, axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                    radius=0.5, k_r=KR)
        # reference uses the raw atom position as the COM
        exp = ref_energy(R, idx, np.ones((1, 1)), AXIS, ORIGIN, 0.5, KR)
        npt.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)

    def test_equal_masses_reduce_to_geometric_mean(self):
        R, _, _ = make_system()
        idx = np.array([0, 2, 5])
        w = jnp.full((3, 1), 1.0 / 3.0)
        com_helper = np.sum(np.asarray(w) * R[idx], axis=0)
        npt.assert_allclose(com_helper, R[idx].mean(axis=0), rtol=1e-12)


class TestMinimumImage:
    def test_invariance_under_lattice_translation(self):
        box = np.array([10.0, 10.0, 10.0])
        R, idx, w = make_system()
        R = R.copy(); R[idx] += np.array([4.5, 0.0, 0.0])  # off-axis, near half-box
        origin = np.array([0.0, 0.0, 0.0])
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(origin),
                  radius=RADIUS, k_r=KR, box=jnp.asarray(box), periodic=True)
        e0 = E_jax(R, idx, w, **kw)
        # translate the guest by one lattice vector -> same energy under min image
        R2 = R.copy(); R2[idx] += np.array([10.0, 0.0, 0.0])
        e1 = E_jax(R2, idx, w, **kw)
        npt.assert_allclose(e0, e1, rtol=1e-10, atol=1e-10)

    def test_matches_reference_with_wrap(self):
        box = np.array([10.0, 10.0, 10.0])
        R, idx, w = make_system()
        R = R.copy(); R[idx] += np.array([9.0, 0.0, 0.0])  # would be far without wrap
        origin = np.array([0.0, 0.0, 0.0])
        got = E_jax(R, idx, w, axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(origin),
                    radius=RADIUS, k_r=KR, box=jnp.asarray(box), periodic=True)
        exp = ref_energy(R, idx, np.asarray(w), AXIS, origin, RADIUS, KR,
                         box=box, periodic=True)
        npt.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)


class TestGeneralAxis:
    def test_projection_matches_reference(self):
        axis = _unit([1.0, 1.0, 0.0])
        R, idx, w = make_system()
        R = R.copy(); R[idx] += np.array([0.0, 0.0, 5.0])
        got = E_jax(R, idx, w, axis_hat=jnp.asarray(axis), origin=jnp.asarray(ORIGIN),
                    radius=1.0, k_r=KR, axial_min=-2.0, axial_max=2.0, k_z=1.5)
        exp = ref_energy(R, idx, np.asarray(w), axis, ORIGIN, 1.0, KR,
                         axial_min=-2.0, axial_max=2.0, k_z=1.5)
        npt.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


class TestSelectivity:
    def test_non_guest_atom_does_not_matter(self):
        R, idx, w = make_system()
        R = R.copy(); R[idx] += np.array([5.0, 0.0, 0.0])
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                  radius=RADIUS, k_r=KR)
        e0 = E_jax(R, idx, w, **kw)
        others = [i for i in range(R.shape[0]) if i not in idx]
        R2 = R.copy(); R2[others[0]] += np.array([1.0, -2.0, 3.0])
        e1 = E_jax(R2, idx, w, **kw)
        npt.assert_allclose(e0, e1, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Tier 2 - settings parsing / validation / regression
# --------------------------------------------------------------------------- #
class TestParseSettings:
    def test_disabled_returns_none(self):
        assert parse_cylinder_settings({}) is None
        assert parse_cylinder_settings({'cylinder_restraint_enabled': False}) is None

    def test_valid_settings(self):
        s = parse_cylinder_settings({
            'cylinder_restraint_enabled': True,
            'cylinder_restraint_atom_indices': [3, 4, 5],
            'cylinder_restraint_radius': 6.0,
            'cylinder_restraint_axis': [0, 0, 1],
            'cylinder_restraint_force_constant': 2.5,
            'cylinder_restraint_axial_min': -1.0,
            'cylinder_restraint_axial_max': 10.0,
        })
        assert s['atom_indices'] == [3, 4, 5]
        assert s['radius'] == 6.0
        assert s['k_r'] == 2.5
        assert s['k_z'] == 2.5           # falls back to k_r
        assert s['axial_min'] == -1.0 and s['axial_max'] == 10.0
        npt.assert_array_equal(s['axis'], np.array([0.0, 0.0, 1.0]))
        assert s['origin'] is None

    @pytest.mark.parametrize("settings", [
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_radius': 5.0},  # no indices
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_atom_indices': [1]},  # no radius
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_atom_indices': [1],
         'cylinder_restraint_radius': -1.0},  # bad radius
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_atom_indices': [1],
         'cylinder_restraint_radius': 5.0, 'cylinder_restraint_force_constant': -1.0},  # bad k_r
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_atom_indices': [1],
         'cylinder_restraint_radius': 5.0, 'cylinder_restraint_axis': [0, 0, 0]},  # zero axis
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_atom_indices': [1],
         'cylinder_restraint_radius': 5.0, 'cylinder_restraint_axis': [1, 0]},  # wrong length
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_atom_indices': [1],
         'cylinder_restraint_radius': 5.0, 'cylinder_restraint_axial_min': 5.0,
         'cylinder_restraint_axial_max': 1.0},  # min > max
        {'cylinder_restraint_enabled': True, 'cylinder_restraint_atom_indices': [1],
         'cylinder_restraint_radius': 5.0, 'cylinder_restraint_axis_origin': [0, 0]},  # bad origin
    ])
    def test_invalid_settings_raise(self, settings):
        with pytest.raises(ValueError):
            parse_cylinder_settings(settings)


class TestRegression:
    def test_fixed_config_energy_and_force(self):
        R, idx, w = make_system(seed=42)
        R = R.copy(); R[idx] += np.array([4.0, 1.0, 7.0])
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                  radius=RADIUS, k_r=KR, axial_min=-4.0, axial_max=4.0, k_z=2.0)

        def scalar(Rj):
            return cylindrical_restraint_energy(Rj, jnp.asarray(idx), w, **kw)

        e = float(scalar(jnp.asarray(R)))
        # self-checking against the independent numpy reference (catches any drift)
        e_ref = ref_energy(R, idx, np.asarray(w), AXIS, ORIGIN, RADIUS, KR,
                           axial_min=-4.0, axial_max=4.0, k_z=2.0)
        npt.assert_allclose(e, e_ref, rtol=1e-12, atol=1e-12)
        # and against a stored golden value so numeric changes are caught explicitly
        npt.assert_allclose(e, 30.731965467359114, rtol=1e-9)


# --------------------------------------------------------------------------- #
# Tier 3 - thermodynamic correctness (Boltzmann distribution)   [slow]
# --------------------------------------------------------------------------- #
def _brownian_dynamics(force_fn, x0, kT, steps, dt, seed):
    """Overdamped Langevin (unit mobility): dx = F dt + sqrt(2 kT dt) N(0,1).

    Samples the Boltzmann distribution exp(-U/kT) for any conservative ``force_fn``,
    independent of the (unit) mobility. The force is taken from the *implementation*
    (``jax.grad`` of the restraint energy) so this exercises the exact force MD uses,
    including the on-axis safe-norm path. Walkers should be initialized near the
    stationary distribution (the interior is force-free, hence uniform) so a modest
    number of steps suffices to equilibrate the thin tails. Returns pooled samples.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float)
    noise_scale = np.sqrt(2.0 * kT * dt)
    collect = []
    burn = steps // 2
    for t in range(steps):
        F = np.asarray(force_fn(jnp.asarray(x)))
        x = x + F * dt + noise_scale * rng.normal(size=x.shape)
        if t >= burn:
            collect.append(x.copy())
    return np.concatenate(collect, axis=0)


@pytest.mark.slow
def test_radial_boltzmann_distribution():
    """A free particle under the radial restraint must sample P(r) ∝ r·exp(-U/kT).

    The gradient-driven sampler also exercises the on-axis safe-norm path (r_perp→0),
    which previously produced NaN forces.
    """
    radius, k_r, kT = 3.0, 2.0, 0.05
    idx = jnp.array([0])
    w = jnp.ones((1, 1))
    kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
              radius=radius, k_r=k_r)

    @jax.jit
    def force_fn(x):  # x: (M, 3)
        def U(xi):
            return cylindrical_restraint_energy(xi[None, :], idx, w, **kw)
        return -jax.vmap(jax.grad(U))(x)

    # initialize uniformly in the disk of radius `radius` (the force-free interior is
    # uniform at equilibrium); BD then fills the thin outer tail.
    M = 6000
    rng = np.random.default_rng(1)
    r0 = radius * np.sqrt(rng.random(M))
    th = 2 * np.pi * rng.random(M)
    x0 = np.stack([r0 * np.cos(th), r0 * np.sin(th), np.zeros(M)], axis=1)

    samples = _brownian_dynamics(force_fn, x0, kT=kT, steps=4000, dt=4e-3, seed=1)
    r = np.linalg.norm(samples[:, :2], axis=1)   # perpendicular distance (axis = z)

    # analytic pdf p(r) ∝ r * exp(-0.5*k_r*relu(r-R)^2 / kT)
    grid = np.linspace(0, radius + 8 * np.sqrt(kT / k_r), 6000)
    U = 0.5 * k_r * np.clip(grid - radius, 0, None) ** 2
    pdf = grid * np.exp(-U / kT)
    pdf /= np.trapezoid(pdf, grid)
    mean_analytic = np.trapezoid(grid * pdf, grid)
    frac_inside_analytic = np.trapezoid(pdf[grid <= radius], grid[grid <= radius])

    # stochastic sampler: tolerances sized to catch gross errors (wrong exponent,
    # missing tail, NaN forces) while tolerating BD sampling noise.
    npt.assert_allclose(r.mean(), mean_analytic, rtol=0.04)
    npt.assert_allclose((r <= radius).mean(), frac_inside_analytic, atol=0.03)


@pytest.mark.slow
def test_axial_boltzmann_distribution():
    """A free particle under the axial slab must sample a uniform band with Gaussian tails."""
    zmin, zmax, k_z, kT = -4.0, 4.0, 2.0, 0.05
    idx = jnp.array([0])
    w = jnp.ones((1, 1))
    # huge radius so the radial term never activates; confine only along z
    kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
              radius=1e6, k_r=1.0, axial_min=zmin, axial_max=zmax, k_z=k_z)

    @jax.jit
    def force_fn(x):
        def U(xi):
            return cylindrical_restraint_energy(xi[None, :], idx, w, **kw)
        return -jax.vmap(jax.grad(U))(x)

    # initialize uniformly across the flat band (its stationary shape); BD fills tails.
    M = 6000
    rng = np.random.default_rng(2)
    x0 = np.zeros((M, 3))
    x0[:, 2] = rng.uniform(zmin, zmax, M)

    samples = _brownian_dynamics(force_fn, x0, kT=kT, steps=1500, dt=5e-3, seed=2)
    s = samples[:, 2]

    grid = np.linspace(zmin - 8 * np.sqrt(kT / k_z), zmax + 8 * np.sqrt(kT / k_z), 8000)
    U = 0.5 * k_z * (np.clip(zmin - grid, 0, None) ** 2 + np.clip(grid - zmax, 0, None) ** 2)
    pdf = np.exp(-U / kT); pdf /= np.trapezoid(pdf, grid)
    frac_inside_analytic = np.trapezoid(pdf[(grid >= zmin) & (grid <= zmax)],
                                        grid[(grid >= zmin) & (grid <= zmax)])

    npt.assert_allclose(s.mean(), 0.0, atol=0.1)                        # symmetric band
    npt.assert_allclose((np.abs(s) <= zmax).mean(), frac_inside_analytic, atol=0.02)
