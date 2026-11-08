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

from so3lr.restraints import cylindrical_restraint_energy, parse_cylinder_settings


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
    X = R[idx]
    if periodic and box is not None:
        box = np.asarray(box, dtype=float)
        diag = np.diag(box) if box.ndim == 2 else box
        X = X[0] + (X - X[0]) - diag * np.round((X - X[0]) / diag)   # the guest made whole
    com = np.sum(w * X, axis=0)
    p = com - origin
    if periodic and box is not None:
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


class TestOnAxisSafeNorm:
    """Pin the double-``where`` safe norm at r_perp == 0 exactly.

    ``jnp.linalg.norm`` has a NaN gradient at the origin, and the guest sits on the
    axis for most of a host-guest run, so the implementation guards it. Nothing else
    in the fast suite pins that guard: ``TestFlatBottom`` looks like it covers the
    on-axis case, but subtracting the COM in floating point leaves a residual
    r_perp ~ 2.2e-16, so the ``jnp.where`` branch is never taken there and a naive
    ``jnp.linalg.norm`` passes it. Only the two ``slow`` Boltzmann/ABF tests caught
    such a regression before this class existed.

    Each case below places the COM *exactly* on the axis (a single guest atom whose
    perpendicular components are exact zeros), so the guarded branch is the one that
    runs. NaN would propagate to every component via the chain rule -- including the
    axial one -- so asserting finiteness of the whole gradient is the sharp check.
    """

    IDX = np.array([0])

    def _grad(self, R, **kw):
        w = jnp.ones((1, 1))
        g = jax.grad(lambda Rj: cylindrical_restraint_energy(
            Rj, jnp.asarray(self.IDX), w, **kw))(jnp.asarray(R))
        return np.asarray(g)

    def test_gradient_finite_and_zero_inside_on_axis(self):
        # On the axis and inside the flat bottom: force is identically zero.
        g = self._grad([[0.0, 0.0, 0.0]], axis_hat=jnp.asarray(AXIS),
                       origin=jnp.asarray(ORIGIN), radius=RADIUS, k_r=KR)
        assert np.all(np.isfinite(g)), f"non-finite gradient on axis: {g}"
        npt.assert_array_equal(g, np.zeros_like(g))

    def test_gradient_finite_on_axis_with_zero_radius(self):
        # radius=0 puts the wall *at* the axis, so r_perp == 0 is the wall itself:
        # the radial term is active yet the force must still be finite (zero by
        # symmetry). This is the regime the PySAGES ABF test runs in.
        g = self._grad([[0.0, 0.0, 0.0]], axis_hat=jnp.asarray(AXIS),
                       origin=jnp.asarray(ORIGIN), radius=0.0, k_r=KR)
        assert np.all(np.isfinite(g)), f"non-finite gradient at r_perp=0 wall: {g}"
        npt.assert_array_equal(g, np.zeros_like(g))

    def test_gradient_finite_on_axis_with_axial_term_active(self):
        # Exactly on the axis but outside the axial band: E > 0 and the axial force
        # is non-zero, while the radial term is inactive. A NaN from the norm would
        # contaminate the axial component too, so this catches leakage through the
        # chain rule rather than just a masked-out radial term.
        s = 9.0
        zmax, k_z = 4.0, 2.0
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                  radius=RADIUS, k_r=KR, axial_min=-4.0, axial_max=zmax, k_z=k_z)
        R = [[0.0, 0.0, s]]
        e = float(cylindrical_restraint_energy(
            jnp.asarray(R), jnp.asarray(self.IDX), jnp.ones((1, 1)), **kw))
        npt.assert_allclose(e, 0.5 * k_z * (s - zmax) ** 2, rtol=1e-12)

        g = self._grad(R, **kw)
        assert np.all(np.isfinite(g)), f"non-finite gradient on axis: {g}"
        # transverse components exactly zero, axial component the analytic force
        npt.assert_array_equal(g[:, :2], np.zeros((1, 2)))
        npt.assert_allclose(g[0, 2], k_z * (s - zmax), rtol=1e-12)

    def test_multi_atom_com_exactly_on_axis(self):
        # Same guard, reached through a genuine mass-weighted COM: two atoms placed
        # symmetrically about the axis cancel exactly in x and y.
        idx = jnp.array([0, 1])
        w = jnp.full((2, 1), 0.5)
        R = jnp.asarray([[1.5, -2.5, 0.0], [-1.5, 2.5, 0.0]])
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(ORIGIN),
                  radius=RADIUS, k_r=KR)
        g = np.asarray(jax.grad(
            lambda Rj: cylindrical_restraint_energy(Rj, idx, w, **kw))(R))
        assert np.all(np.isfinite(g)), f"non-finite gradient for on-axis COM: {g}"
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


    def test_guest_across_a_box_face(self):
        """Atoms wrapped one at a time (as the integrator wraps them) leave the guest
        in two pieces at a box face: energy and forces are those of the whole guest."""
        box = np.array([10.0, 10.0, 10.0])
        R, idx, w = make_system()
        R = R.copy(); R[idx] = np.array([[-0.8, 0.3, 0.1], [0.4, -0.2, 0.5], [1.1, 0.9, -0.3]])
        R[idx] += np.array([0.0, 4.0, 2.0])         # the guest whole, around x = 0
        wrapped = R.copy(); wrapped[idx] %= box     # its first atom jumps to x = 9.2
        assert np.ptp(wrapped[idx, 0]) > box[0] / 2
        kw = dict(axis_hat=jnp.asarray(AXIS), origin=jnp.asarray(np.array([1.0, 1.0, 0.0])),
                  radius=RADIUS, k_r=KR, axial_min=0.0, axial_max=1.0, k_z=2.0,
                  box=jnp.asarray(box), periodic=True)
        e_whole, e_wrapped = E_jax(R, idx, w, **kw), E_jax(wrapped, idx, w, **kw)
        assert e_whole > 0.1                        # both walls push
        npt.assert_allclose(e_wrapped, e_whole, rtol=1e-12)
        grad = jax.grad(lambda X: cylindrical_restraint_energy(X, jnp.asarray(idx), w, **kw))
        npt.assert_allclose(grad(jnp.asarray(wrapped)), grad(jnp.asarray(R)), rtol=1e-10, atol=1e-12)


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

    @pytest.mark.parametrize("spec,expected", [
        (0, [0]),                                  # a single index, atom 0 included
        ([0], [0]),
        ("3-5", [3, 4, 5]),                        # the range strings PySAGES groups take
        ([0, "3-5", 8], [0, 3, 4, 5, 8]),
        (np.array([2, 7]), [2, 7]),
    ])
    def test_index_forms(self, spec, expected):
        s = parse_cylinder_settings({'cylinder_restraint_enabled': True,
                                     'cylinder_restraint_atom_indices': spec,
                                     'cylinder_restraint_radius': 5.0})
        assert s['atom_indices'] == expected

    @pytest.mark.parametrize("spec,match", [
        ([3, 3, 4], "repeats atoms \\[3\\]"),    # would double atom 3's weight
        ("4-2", "end < start"),
        ([], "requires cylinder_restraint_atom_indices"),
        ("x", "cylinder_restraint_atom_indices"),
    ])
    def test_bad_indices_raise(self, spec, match):
        with pytest.raises(ValueError, match=match):
            parse_cylinder_settings({'cylinder_restraint_enabled': True,
                                     'cylinder_restraint_atom_indices': spec,
                                     'cylinder_restraint_radius': 5.0})

    def test_null_means_default(self):
        """A settings file's null takes the documented default."""
        s = parse_cylinder_settings({'cylinder_restraint_enabled': True,
                                     'cylinder_restraint_atom_indices': [1],
                                     'cylinder_restraint_radius': 5.0,
                                     'cylinder_restraint_force_constant': None,
                                     'cylinder_restraint_axis': None})
        assert s['k_r'] == 5.0 and s['k_z'] == 5.0
        npt.assert_array_equal(s['axis'], [0.0, 0.0, 1.0])


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


# --------------------------------------------------------------------------- #
# Tier 4 - the restraint inside an MD run (perform_md)
# --------------------------------------------------------------------------- #
# Everything above tests the energy term on its own. These runs check what perform_md
# builds around it: the guest's mass weights from the input structure, the default origin
# (box centre), the normalised axis, the minimum image in the current box, and the
# conversion of the integrator's positions to Cartesian ones, whose force must stay in real
# space (jax-md's convention for the model's force too) with fractional positions (periodic
# MD without PySAGES, and NPT) as with Cartesian ones (PySAGES in NVT).
#
# Benzoic acid in a 25 A box: its centre of mass starts 0.435 A from the z axis through the
# box centre and 0.005 A below the centre along it, so with R_cyl = 0.2 A and an upper axial
# bound at -0.3 A both walls push from the first step on (about 0.15 eV/A on an atom). The
# force perform_md stores with its checkpoint must equal the model's force (the
# neighbour-list potential in Cartesian coordinates, built here on its own) plus the
# restraint's, written out below by hand.

MD_XYZ = __import__("pathlib").Path(__file__).parent / "test_data" / "benzoic_acid_aq_1.xyz"
MD_RESTRAINT = dict(radius=0.2, k_r=5.0, axial_max=-0.3, k_z=3.0)


def _restraint_force_by_hand(atoms, R, L, axis, origin, radius, k_r, axial_max=None, k_z=None):
    """-dE/dR of the flat-bottom cylinder (radial term, upper axial wall) on all atoms, the
    centre of mass taken over all of them, minimum image in the orthorhombic box L."""
    w = atoms.get_masses() / atoms.get_masses().sum()
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    p = (w[:, None] * R).sum(0) - np.asarray(origin, float)
    p -= L * np.round(p / L)
    s = p @ a
    perp = p - s * a
    r = np.linalg.norm(perp)
    dE_dcom = np.zeros(3)
    if r > radius:
        dE_dcom += k_r * (r - radius) * perp / r
    if axial_max is not None and s > axial_max:
        dE_dcom += k_z * (s - axial_max) * a
    return -w[:, None] * dE_dcom[None, :]


def _model_force(atoms, R, L):
    """The model's force at Cartesian positions R in the box L, float64, without the restraint."""
    from functools import partial
    from so3lr.cli import so3lr_md as M
    geo = M.atoms_to_jnp(atoms, precision=jnp.float64)
    R, box, disp, _, _ = M.handle_box("periodic_cart", jnp.asarray(R), np.diag(L), precision=jnp.float64)
    pot = M.load_model("so3lr-tauri-m-2606", precision=jnp.float64, lr_cutoff=12.0)
    nfn, nfn_lr, efn = M.process_model(pot, geo["species"], disp, box, precision=jnp.float64)
    nb, nb_lr = nfn.allocate(R, box=box), nfn_lr.allocate(R, box=box)
    energy = partial(efn, k_smearing=None, k_smearing_disp=None, has_aux=False)
    return -np.asarray(jax.grad(energy)(R, neighbor=nb.idx, neighbor_lr=nb_lr.idx, box=box,
                                        k_grid=None, k_grid_disp=None))


@pytest.mark.slow
@pytest.mark.parametrize("ensemble,cartesian,tilted", [
    ("nvt", False, False),   # fractional positions: periodic MD without PySAGES
    ("nvt", True, False),    # Cartesian positions: as PySAGES runs NVT
    ("npt", False, False),   # fractional positions in a box the barostat moves
    ("nvt", False, True),    # a general axis through an origin one lattice vector away
])
def test_restraint_force_in_md(tmp_path, ensemble, cartesian, tilted):
    """``so3lr --settings`` (perform_md) adds exactly the restraint's force to the model's
    (float64, 3 steps). Each run is its own process, as a CLI run is (an NVT run after an
    NPT run in one Python process fails in jax-md's pytree checks, with or without the
    restraint)."""
    import subprocess
    import yaml
    from ase.io import read
    from subprocess_env import so3lr_subprocess_env

    atoms = read(MD_XYZ)
    L0 = np.diag(atoms.cell[:])
    if tilted:
        # an axis through a point 1 A from the centre of mass (perpendicular to the axis),
        # given one lattice vector away along x and z
        w = atoms.get_masses() / atoms.get_masses().sum()
        com = (w[:, None] * atoms.positions).sum(0)
        axis = np.array([1.0, 2.0, 0.5])
        a = axis / np.linalg.norm(axis)
        side = np.cross(a, [0.0, 0.0, 1.0])
        origin = com + side / np.linalg.norm(side) + np.array([L0[0], 0.0, -L0[2]])
        restraint = dict(radius=0.2, k_r=5.0)              # radial wall only
        settings = dict(cylinder_restraint_axis=axis.tolist(), cylinder_restraint_axis_origin=origin.tolist())
    else:
        axis, origin = [0.0, 0.0, 1.0], L0 / 2               # the defaults: z through the box centre
        restraint = MD_RESTRAINT
        settings = dict(cylinder_restraint_axial_max=restraint["axial_max"],
                        cylinder_restraint_axial_force_constant=restraint["k_z"])
    settings.update(
        input_file=str(MD_XYZ), output_file="out.hdf5", log_file="out.log", ensemble=ensemble,
        precision="float64", model_path="so3lr-tauri-m-2606", lr_cutoff=12.0, engine="standard",
        relax_before_run=False, md_cycles=1, md_steps=3, save_buffer=1, seed=7, restart_save_path="r.npz",
        enforce_cartesian_coordinates=cartesian, cylinder_restraint_enabled=True,
        cylinder_restraint_atom_indices=list(range(len(atoms))),
        cylinder_restraint_radius=restraint["radius"], cylinder_restraint_force_constant=restraint["k_r"])
    if ensemble == "npt":
        settings["md_P"] = 1.0
    (tmp_path / "md.yaml").write_text(yaml.safe_dump(settings))
    result = subprocess.run(["so3lr", "--settings", "md.yaml"], cwd=tmp_path, capture_output=True, text=True,
                            env=so3lr_subprocess_env(), timeout=1500)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Cylindrical / funnel restraint enabled" in (tmp_path / "out.log").read_text()

    with np.load(tmp_path / "r.npz") as saved:
        assert bool(saved["fractional_coordinates"]) == (not cartesian)
        L = np.asarray(saved["box"], float).reshape(-1)
        L = np.diag(np.asarray(saved["box"], float)) if L.size == 9 else L
        R = np.asarray(saved["position"]) * (1.0 if cartesian else L)
        stored = np.asarray(saved["force"])
    if ensemble == "npt":
        assert np.abs(L - L0).max() > 1e-6               # the box moved: the restraint sees the new one
    by_hand = _restraint_force_by_hand(atoms, R, L, axis, origin, **restraint)
    # the walls push (so a missing restraint, or one off by a factor, cannot pass) ...
    assert np.abs(by_hand).max() > 0.05
    # ... and the stored force is the model's plus exactly the restraint's
    npt.assert_allclose(stored, _model_force(atoms, R, L) + by_hand, rtol=0, atol=1e-8)
