"""Tests for restart coordinate conversion between fractional and cartesian.

When a restart file is saved with one coordinate system (e.g., fractional)
and loaded into a simulation using a different coordinate system (e.g.,
cartesian), positions and momenta must be converted. These tests verify
that save_state/load_state handle this correctly; the end-to-end restarts
across coordinate systems are in tests/test_pysages_cli.py.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax_md

from so3lr.cli.so3lr_md import (
    save_state,
    load_state,
    RestartInNewEnsembleError,
    RestartSchemaMismatchError,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_nose_hoover_chain(n_atoms):
    """Create a minimal NoseHooverChain for testing."""
    return jax_md.simulate.NoseHooverChain(
        position=jnp.array([0.0, 0.0, 0.0]),
        momentum=jnp.array([0.1, 0.1, 0.1]),
        mass=jnp.array([1.0, 1.0, 1.0]),
        tau=jnp.array(1.0),
        kinetic_energy=jnp.array(0.5),
        degrees_of_freedom=jnp.array(3 * n_atoms),
    )


def _make_nvt_state(position, momentum, force, mass):
    """Create a minimal NVTNoseHooverState for testing."""
    return jax_md.simulate.NVTNoseHooverState(
        position=position,
        momentum=momentum,
        force=force,
        mass=mass,
        chain=_make_nose_hoover_chain(position.shape[0]),
    )


def _make_npt_state(position, momentum, force, mass, box):
    """Create a minimal NPTNoseHooverState for testing."""
    n_atoms = position.shape[0]
    return jax_md.simulate.NPTNoseHooverState(
        position=position,
        momentum=momentum,
        force=force,
        mass=mass,
        reference_box=box,
        box_position=jnp.zeros(3),
        box_momentum=jnp.zeros(3),
        box_mass=jnp.array([1.0, 1.0, 1.0]),
        thermostat=_make_nose_hoover_chain(n_atoms),
        barostat=_make_nose_hoover_chain(n_atoms),
        dUdV=jnp.zeros((3, 3)),
    )


def _make_nve_state(position, momentum, force, mass):
    """Create a minimal NVEState for testing."""
    return jax_md.simulate.NVEState(
        position=position,
        momentum=momentum,
        force=force,
        mass=mass,
    )


def _make_langevin_state(position, momentum, force, mass, rng=None):
    """Create a minimal NVTLangevinState for testing.

    The BAOAB Langevin state is fully described by the core
    position/momentum/force/mass fields plus an RNG key.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    return jax_md.simulate.NVTLangevinState(
        position=position,
        momentum=momentum,
        force=force,
        mass=mass,
        rng=rng,
    )


# ===========================================================================
# Unit tests: save_state / load_state coordinate flag and conversion
# ===========================================================================

class TestLangevinRestart:
    """Regression tests for NVT-Langevin restart (save_state/load_state).

    A langevin run records ``ensemble='nvt-langevin'``. Previously load_state
    had no branch for it and fell through to ``NotImplementedError``, so any
    attempt to resume an NVT-Langevin trajectory crashed. The Langevin state
    (jax_md ``NVTLangevinState``) is fully described by the core
    position/momentum/force/mass fields plus the RNG key, which save_state now
    persists so the stochastic trajectory continues deterministically.
    """

    def test_langevin_restart_roundtrip(self, tmp_path):
        """Saving then loading an NVT-Langevin state recovers the full state.

        Before the fix this raised NotImplementedError on load.
        """
        pos = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mom = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])
        state = _make_langevin_state(pos, mom, force, mass,
                                     rng=jax.random.PRNGKey(42))
        box = jnp.array([10.0, 10.0, 10.0])
        path = str(tmp_path / "restart_langevin.npz")

        save_state(state, box, 7, path, ensemble='nvt-langevin',
                   fractional_coordinates=True)
        loaded_state, _, cycle = load_state(
            path, ensemble='nvt-langevin', fractional_coordinates=True
        )

        assert isinstance(loaded_state, jax_md.simulate.NVTLangevinState)
        np.testing.assert_allclose(loaded_state.position, pos, atol=1e-10)
        np.testing.assert_allclose(loaded_state.momentum, mom, atol=1e-10)
        np.testing.assert_allclose(loaded_state.force, force, atol=1e-10)
        assert int(np.asarray(cycle)) == 7

    def test_langevin_restart_persists_rng(self, tmp_path):
        """The RNG key is written to the .npz and restored exactly, so the
        stochastic trajectory continues deterministically across the restart."""
        pos = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mom = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])
        rng = jax.random.PRNGKey(123)
        state = _make_langevin_state(pos, mom, force, mass, rng=rng)
        box = jnp.array([10.0, 10.0, 10.0])
        path = str(tmp_path / "restart_langevin.npz")

        save_state(state, box, 0, path, ensemble='nvt-langevin')

        data = np.load(path, allow_pickle=True)
        assert 'rng' in list(data.files)
        np.testing.assert_array_equal(np.asarray(data['rng']), np.asarray(rng))

        loaded_state, _, _ = load_state(
            path, ensemble='nvt-langevin', fractional_coordinates=True
        )
        np.testing.assert_array_equal(np.asarray(loaded_state.rng),
                                      np.asarray(rng))

    def test_langevin_restart_without_rng_raises_schema_mismatch(self, tmp_path):
        """A restart file written before Langevin restarts were supported has
        no 'rng' field; loading it must raise a clear RestartSchemaMismatchError
        rather than the historical opaque NotImplementedError."""
        path = str(tmp_path / "legacy_langevin.npz")
        # Emulate a legacy langevin .npz: core fields + ensemble, but no rng.
        np.savez(
            path,
            position=np.array([[0.1, 0.2, 0.3]]),
            momentum=np.array([[1.0, 2.0, 3.0]]),
            force=np.array([[0.0, 0.0, 0.0]]),
            mass=np.array([[1.0]]),
            box=np.array([10.0, 10.0, 10.0]),
            step=0,
            fractional_coordinates=True,
            ensemble='nvt-langevin',
        )
        with pytest.raises(RestartSchemaMismatchError):
            load_state(path, ensemble='nvt-langevin',
                       fractional_coordinates=True)


# Restart state builders, per ensemble.
_STATE_BUILDERS = {
    'nve': lambda pos, mom, force, mass, box: _make_nve_state(pos, mom, force, mass),
    'nvt': lambda pos, mom, force, mass, box: _make_nvt_state(pos, mom, force, mass),
    'npt': _make_npt_state,
    'nvt-langevin': lambda pos, mom, force, mass, box: _make_langevin_state(pos, mom, force, mass),
}


class TestCoordinateConversion:
    """Verify positions are converted when coordinate systems differ."""

    @pytest.mark.parametrize('ensemble', list(_STATE_BUILDERS))
    @pytest.mark.parametrize('saved_fractional,load_fractional', [
        (True, False),   # fractional restart into a cartesian run
        (False, True),   # cartesian restart into a fractional run
        (True, True),    # same coordinate system: no conversion
    ], ids=['frac_to_cart', 'cart_to_frac', 'same'])
    def test_load_state_converts_positions(self, tmp_path, ensemble,
                                           saved_fractional, load_fractional):
        """Loading across coordinate systems multiplies (frac -> cart) or
        divides (cart -> frac) positions by the box. Momenta and forces are
        left unchanged: jax_md stores them identically in both coordinate
        systems. The conversion runs after the per-ensemble state rebuild,
        so every ensemble branch must hand it a complete state."""
        box = jnp.array([25.0, 25.0, 25.0])
        frac_pos = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        cart_pos = jnp.array([[2.5, 5.0, 7.5], [10.0, 12.5, 15.0]])
        mom = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])
        # Expected values use the same operation as the conversion, so the
        # comparison stays exact in float32.
        if saved_fractional and not load_fractional:
            saved_pos, expected_pos = frac_pos, frac_pos * box
        elif load_fractional and not saved_fractional:
            saved_pos, expected_pos = cart_pos, cart_pos / box
        else:
            saved_pos, expected_pos = frac_pos, frac_pos
        state = _STATE_BUILDERS[ensemble](saved_pos, mom, force, mass, box)
        path = str(tmp_path / f"{ensemble}_restart.npz")

        save_state(state, box, 3, path, ensemble=ensemble,
                   fractional_coordinates=saved_fractional)
        loaded_state, _, cycle = load_state(
            path, ensemble=ensemble, fractional_coordinates=load_fractional
        )

        assert type(loaded_state) is type(state)
        np.testing.assert_allclose(loaded_state.position, expected_pos, atol=1e-10)
        np.testing.assert_allclose(loaded_state.momentum, mom, atol=1e-10)
        np.testing.assert_allclose(loaded_state.force, force, atol=1e-10)
        assert int(np.asarray(cycle)) == 3
        if ensemble == 'npt':
            # NPT-specific fields are preserved
            np.testing.assert_allclose(loaded_state.reference_box, box, atol=1e-10)

    def test_roundtrip_frac_cart_frac(self, tmp_path):
        """Fractional -> save -> load as cartesian -> save -> load as fractional
        should recover original state."""
        box = jnp.array([25.0, 25.0, 25.0])
        orig_pos = jnp.array([[0.12, 0.34, 0.56], [0.78, 0.91, 0.23]])
        orig_mom = jnp.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        orig_force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])
        state = _make_nvt_state(orig_pos, orig_mom, orig_force, mass)

        # Save as fractional, load as cartesian
        path1 = str(tmp_path / "restart1.npz")
        save_state(state, box, 0, path1, ensemble='nvt', fractional_coordinates=True)
        cart_state, _, _ = load_state(
            path1, ensemble='nvt', fractional_coordinates=False
        )

        # Save as cartesian, load as fractional
        path2 = str(tmp_path / "restart2.npz")
        save_state(cart_state, box, 0, path2, ensemble='nvt', fractional_coordinates=False)
        final_state, _, _ = load_state(
            path2, ensemble='nvt', fractional_coordinates=True
        )

        np.testing.assert_allclose(final_state.position, orig_pos, atol=1e-10)
        np.testing.assert_allclose(final_state.momentum, orig_mom, atol=1e-10)
        np.testing.assert_allclose(final_state.force, orig_force, atol=1e-10)

    def test_old_restart_without_flag_assumes_fractional(self, tmp_path):
        """Old restart files without fractional_coordinates flag should be
        treated as fractional (the historical default)."""
        box = jnp.array([25.0, 25.0, 25.0])
        frac_pos = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        frac_mom = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frac_force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])

        # Manually create an old-format restart file without the flag
        chain = jax_md.simulate.NoseHooverChain(
            position=jnp.array([0.0, 0.0, 0.0]),
            momentum=jnp.array([0.1, 0.1, 0.1]),
            mass=jnp.array([1.0, 1.0, 1.0]),
            tau=jnp.array(1.0),
            kinetic_energy=jnp.array(0.5),
            degrees_of_freedom=jnp.array(6),
        )
        path = str(tmp_path / "old_restart.npz")
        np.savez(
            path,
            position=np.array(frac_pos),
            momentum=np.array(frac_mom),
            force=np.array(frac_force),
            mass=np.array(mass),
            box=np.array(box),
            step=0,
            ensemble='nvt',
            thermostat_position=np.array(chain.position),
            thermostat_momentum=np.array(chain.momentum),
            thermostat_mass=np.array(chain.mass),
            thermostat_tau=np.array(chain.tau),
            thermostat_kinetic_energy=np.array(chain.kinetic_energy),
            thermostat_degrees_of_freedom=np.array(chain.degrees_of_freedom),
            # NOTE: no fractional_coordinates key
        )

        # Load as cartesian — should convert since default assumption is fractional
        loaded_state, _, _ = load_state(
            path, ensemble='nvt', fractional_coordinates=False
        )

        expected_pos = frac_pos * box
        np.testing.assert_allclose(loaded_state.position, expected_pos, atol=1e-10)


# ===========================================================================
# Unit tests: ensemble change + coordinate conversion (RestartInNewEnsembleError)
# ===========================================================================

class TestEnsembleChangeWithCoordinateConversion:
    """RestartInNewEnsembleError is raised when ensembles differ and carries
    what perform_md's handler needs to rebuild the state in the new ensemble."""

    def test_nvt_to_npt_raises_error(self, tmp_path):
        """Loading an NVT restart with ensemble='npt' raises RestartInNewEnsembleError."""
        box = jnp.array([25.0, 25.0, 25.0])
        pos = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mom = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])
        state = _make_nvt_state(pos, mom, force, mass)
        path = str(tmp_path / "nvt_restart.npz")

        save_state(state, box, 5, path, ensemble='nvt', fractional_coordinates=True)

        with pytest.raises(RestartInNewEnsembleError) as exc_info:
            load_state(path, ensemble='npt', fractional_coordinates=True)

        # Verify the exception carries the loaded data
        e = exc_info.value
        np.testing.assert_allclose(e.loaded_state['position'], pos, atol=1e-10)
        np.testing.assert_allclose(e.loaded_state['box'], box, atol=1e-10)
        np.testing.assert_allclose(e.loaded_state['momentum'] / e.loaded_state['mass'], mom / mass, atol=1e-10)
        assert e.loaded_state.get('ensemble', None) == 'nvt'
        assert bool(e.loaded_state.get('fractional_coordinates', True)) is True

    def test_npt_to_nvt_raises_error(self, tmp_path):
        """Loading an NPT restart with ensemble='nvt' raises RestartInNewEnsembleError."""
        box = jnp.array([25.0, 25.0, 25.0])
        pos = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mom = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])
        state = _make_npt_state(pos, mom, force, mass, box)
        path = str(tmp_path / "npt_restart.npz")

        save_state(state, box, 0, path, ensemble='npt', fractional_coordinates=False)

        with pytest.raises(RestartInNewEnsembleError) as exc_info:
            load_state(path, ensemble='nvt', fractional_coordinates=False)

        e = exc_info.value
        assert e.loaded_state.get('ensemble', None) == 'npt'
        assert bool(e.loaded_state.get('fractional_coordinates', True)) is False


class TestRestartResetWithCoordinateConversion:
    """restart_reset=True forces RestartInNewEnsembleError even when ensembles match."""

    def test_restart_reset_same_ensemble_raises_error(self, tmp_path):
        """restart_reset=True with matching ensemble still raises the error."""
        box = jnp.array([25.0, 25.0, 25.0])
        pos = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mom = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        force = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        mass = jnp.array([[1.0], [16.0]])
        state = _make_nvt_state(pos, mom, force, mass)
        path = str(tmp_path / "restart.npz")

        save_state(state, box, 0, path, ensemble='nvt', fractional_coordinates=True)

        with pytest.raises(RestartInNewEnsembleError):
            load_state(
                path, ensemble='nvt', restart_reset=True,
                fractional_coordinates=True
            )
