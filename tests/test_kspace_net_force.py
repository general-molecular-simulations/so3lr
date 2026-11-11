"""The k-space net force (kspace_remove_net_force).

Smooth PME spreads the charges on a mesh fixed in space, so its forces are not
translation invariant and sum to a small net force. In MD nothing cancels it, and
it accumulates into a drift of the whole box (seen as superdiffusive MSDs in slow
bulk liquids). With k-space on, the MD energy function hands the integrator the
forces minus their mean; the energy and its box derivative (the pressure) stay as
they were.
"""
import functools
import pathlib

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import numpy.testing as npt
import pytest
from ase.io import read

from so3lr.jaxmd_utils import without_net_force

jax.config.update('jax_enable_x64', True)

WATER = pathlib.Path(__file__).parent / 'test_data' / 'water_64.xyz'
LR_CUTOFF = 6.0      # minimum image holds in the 12.4 A cell


def test_without_net_force_is_identity_with_mean_free_gradient():
    R = jax.random.uniform(jax.random.PRNGKey(0), (7, 3), dtype=jnp.float64)
    a = jax.random.normal(jax.random.PRNGKey(1), (7, 3), dtype=jnp.float64)

    def energy(R):
        return jnp.sum(a * jnp.sin(3.0 * R))

    npt.assert_array_equal(without_net_force(R), R)
    g = jax.grad(energy)(R)
    g_free = jax.grad(lambda R: energy(without_net_force(R)))(R)
    npt.assert_allclose(g_free, g - g.mean(0), atol=1e-14)
    npt.assert_allclose(g_free.sum(0), 0.0, atol=1e-13)
    # forward mode agrees (custom_jvp, so jacfwd works too)
    npt.assert_allclose(jax.jacfwd(lambda R: energy(without_net_force(R)))(R), g_free, atol=1e-14)


@pytest.mark.parametrize('settings, expected', [
    (dict(kspace_electrostatics='pme'), True),
    (dict(kspace_dispersion='pme'), True),
    (dict(kspace_electrostatics='ewald', kspace_remove_net_force=True), True),
    (dict(kspace_electrostatics='pme', kspace_remove_net_force=False), False),
    (dict(), False),
    (dict(kspace_electrostatics=None, kspace_dispersion=None), False),
])
def test_remove_kspace_net_force_settings(settings, expected):
    from so3lr.cli.so3lr_md import remove_kspace_net_force
    assert remove_kspace_net_force(settings) is expected


@pytest.mark.parametrize('settings, expected', [
    (dict(ensemble='nvt'), True),
    (dict(ensemble='npt', kspace_electrostatics='pme'), True),
    (dict(ensemble='nvt', kspace_remove_net_force=False), False),                 # the old behaviour
    (dict(ensemble='nvt-langevin', langevin_removecmmotion=True), True),
    (dict(ensemble='nvt-langevin', langevin_removecmmotion=False), False),        # thermal COM motion
])
def test_restart_removes_com_motion_settings(settings, expected):
    from so3lr.cli.so3lr_md import restart_removes_com_motion
    assert restart_removes_com_motion(settings) is expected


def test_restart_keeps_pysages_and_held_atoms_momenta():
    """A restarted PySAGES state keeps its snapshot's momenta, and held atoms their zero
    momenta, in both engines."""
    from so3lr.cli.so3lr_md import restart_removes_com_motion
    assert restart_removes_com_motion(dict(ensemble='nvt'), restart_pysages=True) is False
    assert restart_removes_com_motion(dict(ensemble='nvt'), held_atoms=True) is False


def test_flash_single_points_keep_the_model_forces(monkeypatch):
    """so3lr eval and the flash ASE calculator return the forces So3lrCalculator does; only
    MD and opt take them without the net force."""
    import so3lr.flash.md as flash_md
    from so3lr.cli.so3lr_md import remove_kspace_net_force
    seen = {}

    class Engine:
        device_mesh, builds = None, 0

        def energy_forces(self, R):
            return 0.0, jnp.zeros_like(R)

    def build_engine(atoms, settings, dtype, box_margin=None, positions=None):
        seen['remove_net_force'] = remove_kspace_net_force(settings)
        return Engine(), False

    monkeypatch.setattr(flash_md, 'build_engine', build_engine)
    for check in ('flash_problem', 'structure_problem'):          # no GPU, a small box
        monkeypatch.setattr(flash_md, check, lambda *args, **kwargs: None)
    points = flash_md.SinglePoints(dict(kspace_electrostatics='pme', precision='float64'), task='eval')
    points(read(WATER))
    assert seen['remove_net_force'] is False


@functools.cache
def _md_energy_fn(remove_net_force):
    """The float64 energy function `so3lr npt` integrates for water_64 with PME
    electrostatics and dispersion (fractional positions, as in NPT), and its
    arguments."""
    from so3lr import kspace_defaults as KS
    from so3lr.cli import so3lr_md
    from so3lr.model_registry import DEFAULT_MODEL

    atoms = read(WATER)
    cell = so3lr_md.cell_from_atoms_boundary_conditions(atoms, LR_CUTOFF, 'pme', KS.DEFAULT_KSPACE_SMEARING)
    atoms.wrap()
    geometry = so3lr_md.atoms_to_jnp(atoms, precision='float64')
    position, box, displacement, _, frac = so3lr_md.handle_box(
        'periodic', geometry['positions'], cell, precision=jnp.float64)
    potential = so3lr_md.load_model(DEFAULT_MODEL, precision=jnp.float64, lr_cutoff=LR_CUTOFF,
                                    dispersion_damping=2.0, kspace_electrostatics='pme',
                                    kspace_dispersion='pme')
    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = so3lr_md.process_model(
        potential=potential, species=geometry['species'], displacement=displacement, box=box,
        total_charge=0, num_unpaired_electrons=0, precision='float64', fractional_coordinates=frac,
        buffer_size_multiplier_lr=1.25, buffer_size_multiplier_sr=1.25,
        minimum_cell_size_multiplier=1.0, remove_net_force=remove_net_force)
    k_grid, k_smearing = so3lr_md.setup_kspace_grid(
        kspace_electrostatics='pme', kspace_spacing=KS.DEFAULT_KSPACE_SPACING,
        kspace_smearing=KS.DEFAULT_KSPACE_SMEARING, box=box)
    k_grid_disp, k_smearing_disp = so3lr_md.setup_kspace_grid(
        kspace_electrostatics='pme', kspace_spacing=KS.DEFAULT_KSPACE_DISPERSION_SPACING,
        kspace_smearing=KS.DEFAULT_KSPACE_DISPERSION_SMEARING, box=box)
    energy_fn = functools.partial(energy_or_obs_fn, has_aux=False, k_smearing=k_smearing,
                                  k_smearing_disp=k_smearing_disp)
    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box)
    kw = dict(neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box, k_grid=k_grid, k_grid_disp=k_grid_disp)
    return energy_fn, position, kw


def _energy_forces_strain(remove_net_force):
    energy_fn, R, kw = _md_energy_fn(remove_net_force)
    e = float(jax.jit(energy_fn)(R, **kw))
    f = np.asarray(jax.jit(jax_md.quantity.force(energy_fn))(R, **kw))
    # dE/d(strain), box and positions scaled together: what jax-md's NPT takes
    w = float(jax.jit(jax.grad(lambda eps: energy_fn(R, perturbation=1.0 + eps, **kw)))(0.0))
    return e, f, w


def test_md_forces_sum_to_zero_with_pme():
    e0, f0, w0 = _energy_forces_strain(False)
    e1, f1, w1 = _energy_forces_strain(True)
    net0 = np.linalg.norm(f0.sum(0))
    # the mesh leaves a net force well above float64 roundoff ...
    assert net0 > 1e-6, net0
    # ... which is gone, with every atom's force moved by the same mean
    assert np.linalg.norm(f1.sum(0)) < 1e-10 * len(f1)
    npt.assert_allclose(f1, f0 - f0.mean(0), atol=1e-10)
    # energy and pressure unchanged
    assert e1 == e0
    npt.assert_allclose(w1, w0, rtol=1e-12, atol=1e-12)


def _drifting(n=6, speed_internal=0.01, seed=0):
    """Thermal-looking momenta plus a centre-of-mass velocity along x."""
    mass = jnp.asarray(np.random.default_rng(seed).uniform(1.0, 16.0, n))
    p = jax.random.normal(jax.random.PRNGKey(seed), (n, 3), dtype=jnp.float64)
    p = p - mass[:, None] * (p.sum(0) / mass.sum())
    v_com = jnp.array([speed_internal, 0.0, 0.0])
    return p + mass[:, None] * v_com, p, mass


def test_restart_loses_com_motion(caplog):
    """Momentum states, the flexible-cell state (which stores velocity) and bare momenta;
    a warning only when the box was drifting."""
    from so3lr.cli.so3lr_md import remove_restart_com_momentum, remove_restart_com_motion
    drifting, still, mass = _drifting()
    zeros = jnp.zeros_like(drifting)
    nve = jax_md.simulate.NVEState(position=zeros, momentum=drifting, force=zeros, mass=mass[:, None])
    flex = jax_md.simulate.NPTNoseHooverFlexState(
        position=zeros, velocity=drifting / mass[:, None], force=zeros, mass=mass[:, None],
        reference_box=None, box_position=None, box_velocity=None, box_mass=None,
        barostat=None, thermostat=None, stress_tensor=None)
    with caplog.at_level('INFO', logger='SO3LR'):
        for m in (mass, mass[:, None]):                  # jax-md keeps [n] or [n, 1]
            npt.assert_allclose(remove_restart_com_momentum(drifting, m), still, atol=1e-13)
        npt.assert_allclose(remove_restart_com_motion(nve).momentum, still, atol=1e-13)
        npt.assert_allclose(remove_restart_com_motion(flex).momentum, still, atol=1e-13)
    assert caplog.text.count('drifting as a whole') == 4
    assert f"{0.01 * jax_md.units.metal_unit_system()['time']:.2e} A/ps" in caplog.text
    caplog.clear()
    with caplog.at_level('INFO', logger='SO3LR'):
        npt.assert_allclose(remove_restart_com_motion(nve.set(momentum=still)).momentum, still, atol=1e-13)
    assert 'drifting' not in caplog.text and 'removed centre-of-mass motion' in caplog.text


@pytest.mark.slow
def test_cli_restart_removes_com_drift(tmp_path):
    """A restart file whose box drifts (written before the net-force removal) continues
    without the drift, and the log says so."""
    import subprocess
    from subprocess_env import so3lr_subprocess_env
    acetone = pathlib.Path(__file__).parent / 'test_data' / 'acetone.xyz'

    def so3lr(*args):
        run = subprocess.run(['so3lr', 'nvt', '--input', str(acetone), '--engine', 'standard', '--no-relax',
                              '--md-cycles', '2', '--md-steps', '2', *args],
                             cwd=tmp_path, capture_output=True, text=True, env=so3lr_subprocess_env(), timeout=900)
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]

    so3lr('--output', 'a.xyz', '--restart-save', 'a.npz')
    a = dict(np.load(tmp_path / 'a.npz', allow_pickle=False))
    mass = a['mass'].reshape(-1, 1)
    a['momentum'] = a['momentum'] + mass * np.array([0.01, 0.0, 0.0])    # ~1 A/ps
    np.savez(tmp_path / 'drift.npz', **a)
    so3lr('--output', 'b.xyz', '--restart-load', 'drift.npz', '--restart-save', 'b.npz')
    assert 'drifting as a whole' in (tmp_path / 'b.log').read_text()
    total = np.load(tmp_path / 'b.npz')['momentum'].sum(0)
    assert np.abs(total).max() < 1e-8 * mass.sum(), total
