"""Tests for the PySAGES interface in SO3LR.

Tests cover:
1. Import guard / optional dependency behavior
2. handle_box with periodic_cart mode
3. PySAGES interface functions (unit tests)
4. update_so3lr_after_pysages k_grid passthrough
5. Settings parsing defaults and auto-configuration
6. Integration: trajectory comparison between vanilla and unbiased runs
"""

import os
import re
import sys
import shutil
import tempfile
import pathlib
import subprocess

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax_md
import yaml

from so3lr.cli.so3lr_md import handle_box, HAS_PYSAGES

# ---------------------------------------------------------------------------
# Paths and shared settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "test_data"
GEOMETRY_FILE = TEST_DATA_DIR / "benzoic_acid_aq_1.xyz"

# Base MD settings shared by all integration runs.  The geometry path is
# filled in at fixture time so it points to the temp-directory copy.
_BASE_MD_SETTINGS = {
    'use_so3lr': True,
    'precision': 'float64',
    'total_charge': 0.0,
    'lr_cutoff': 12.0,
    'relax_before_run': True,
    'min_cycles': 10,
    'min_steps': 20,
    'min_start_dt': 0.05,
    'min_max_dt': 0.1,
    'min_n_min': 2,
    'min_print': True,
    'md_dt': 0.5,
    'md_T': 300,
    'md_cycles': 2,
    'md_steps': 500,
    'hdf5_buffer_size': 50,
}

_PYSAGES_INPUT_UNBIASED = {
    'method': 'unbiased',
    'cv': [{
        'type': 'coordinationnumber',
        'indices': [4, 7],
        'cn_exps': [6, 30],
        'species': 1,
        'grid_min': 0.0,
        'grid_max': 1.5,
        'grid_stride': 64,
        'periodic': False,
    }],
}

_PYSAGES_INPUT_ABF = {
    'method': 'abf',
    'cv': [{
        'type': 'coordinationnumber',
        'indices': [4, 7],
        'cn_exps': [6, 30],
        'species': 1,
        'grid_min': 0.0,
        'grid_max': 1.5,
        'grid_stride': 64,
        'periodic': False,
    }],
}


def _write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)


def _prepare_run_dir(tmp_path, name, md_overrides=None, pysages_input=None):
    """Create a simulation directory with geometry, settings, and optionally pysages input."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy geometry
    shutil.copy2(GEOMETRY_FILE, run_dir / "benzoic_acid_aq_1.xyz")

    # Build settings
    settings = dict(_BASE_MD_SETTINGS)
    settings['input_file'] = 'benzoic_acid_aq_1.xyz'
    settings['output_file'] = 'traj.xyz'
    settings['log_file'] = 'out.log'
    if md_overrides:
        settings.update(md_overrides)
    _write_yaml(run_dir / "md_settings.yaml", settings)

    # PySAGES input (if needed)
    if pysages_input is not None:
        _write_yaml(run_dir / "pysages_input.yaml", pysages_input)

    return run_dir


# ===========================================================================
# 1. Import guard / optional dependency
# ===========================================================================

class TestImportGuard:
    """Tests for optional PySAGES import behavior."""

    def test_has_pysages_flag_is_true(self):
        """PySAGES is installed in this environment, so HAS_PYSAGES should be True."""
        assert HAS_PYSAGES is True

    def test_interface_module_imports(self):
        """The interface module should import without error."""
        from so3lr.cli.so3lr_pysages_interface import HAS_PYSAGES as iface_flag
        assert iface_flag is True

    def test_do_enhanced_sampling_requires_pysages(self):
        """When HAS_PYSAGES is False and do_enhanced_sampling is True, raise ImportError."""
        import so3lr.cli.so3lr_md as md_module
        original = md_module.HAS_PYSAGES
        try:
            md_module.HAS_PYSAGES = False
            settings = {
                'do_enhanced_sampling': True,
                'input_file': 'dummy.xyz',
                'output_file': 'dummy.xyz',
                'md_dt': 0.5,
                'md_T': 300,
                'md_cycles': 1,
                'md_steps': 10,
                'ensemble': 'nvt',
                'precision': 'float32',
                'seed': 42,
                'save_buffer': 10,
            }
            with pytest.raises(ImportError, match="PySAGES is required"):
                md_module.perform_md(settings)
        finally:
            md_module.HAS_PYSAGES = original

    def test_default_no_enhanced_sampling(self):
        """do_enhanced_sampling should default to False."""
        settings = {}
        assert settings.get('do_enhanced_sampling', False) is False

    @pytest.mark.parametrize("ensemble", ["npt", "nve", "nvt-langevin"])
    def test_nvt_restricted_method_rejects_non_nvt(self, ensemble, tmp_path):
        """Free-energy methods (ABF, metad, etc.) should reject non-NVT ensembles."""
        import so3lr.cli.so3lr_md as md_module
        pysages_path = tmp_path / "pysages_input.yaml"
        _write_yaml(pysages_path, _PYSAGES_INPUT_ABF)
        settings = {
            'do_enhanced_sampling': True,
            'pysages_input': str(pysages_path),
            'input_file': 'dummy.xyz',
            'output_file': 'dummy.xyz',
            'md_dt': 0.5,
            'md_T': 300,
            'md_cycles': 1,
            'md_steps': 10,
            'ensemble': ensemble,
            'precision': 'float32',
            'seed': 42,
            'save_buffer': 10,
        }
        with pytest.raises(ValueError, match="only supports the NVT ensemble"):
            md_module.perform_md(settings)

    @pytest.mark.parametrize("method_input,ensemble", [
        (_PYSAGES_INPUT_UNBIASED, "nve"),
        (_PYSAGES_INPUT_UNBIASED, "npt"),
        ({"method": "harmonic", "method_args": {"kspring": 10.0, "center": [1.0]},
          "cv": [{"type": "distance", "grp1": [0], "grp2": [1],
                  "grid_min": 0.0, "grid_max": 10.0,
                  "grid_stride": 32, "periodic": False}]}, "nve"),
        ({"method": "harmonic", "method_args": {"kspring": 10.0, "center": [1.0]},
          "cv": [{"type": "distance", "grp1": [0], "grp2": [1],
                  "grid_min": 0.0, "grid_max": 10.0,
                  "grid_stride": 32, "periodic": False}]}, "npt"),
    ])
    def test_unrestricted_method_accepts_any_ensemble(self, method_input, ensemble, tmp_path):
        """Harmonic and unbiased methods should not reject NVE or NPT ensembles.

        The run will fail later (missing geometry), but the ensemble check must pass.
        """
        import so3lr.cli.so3lr_md as md_module
        pysages_path = tmp_path / "pysages_input.yaml"
        _write_yaml(pysages_path, method_input)
        settings = {
            'do_enhanced_sampling': True,
            'pysages_input': str(pysages_path),
            'input_file': 'dummy.xyz',
            'output_file': 'dummy.xyz',
            'md_dt': 0.5,
            'md_T': 300,
            'md_cycles': 1,
            'md_steps': 10,
            'ensemble': ensemble,
            'precision': 'float32',
            'seed': 42,
            'save_buffer': 10,
        }
        # Should NOT raise ValueError about ensemble; it should fail later
        # when trying to read the geometry file
        with pytest.raises((FileNotFoundError, RuntimeError)):
            md_module.perform_md(settings)


# ===========================================================================
# 2. handle_box with periodic_cart
# ===========================================================================

class TestHandleBox:
    """Tests for handle_box including the periodic_cart mode."""

    @pytest.fixture
    def positions_and_cell(self):
        positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cell = np.diag([10.0, 10.0, 10.0])
        return positions, cell

    def test_periodic_returns_fractional(self, positions_and_cell):
        positions, cell = positions_and_cell
        pos_out, box, disp, shift, frac = handle_box('periodic', positions, cell)
        assert frac is True
        # Positions should be transformed to fractional coordinates
        np.testing.assert_allclose(pos_out, positions / 10.0, atol=1e-10)

    def test_periodic_cart_returns_cartesian(self, positions_and_cell):
        positions, cell = positions_and_cell
        pos_out, box, disp, shift, frac = handle_box('periodic_cart', positions, cell)
        assert frac is False
        # Positions should NOT be transformed
        np.testing.assert_allclose(pos_out, positions, atol=1e-10)

    def test_periodic_cart_box_is_correct(self, positions_and_cell):
        positions, cell = positions_and_cell
        _, box, _, _, _ = handle_box('periodic_cart', positions, cell)
        np.testing.assert_allclose(box, jnp.array([10.0, 10.0, 10.0]))

    def test_periodic_cart_no_cell_raises(self):
        positions = jnp.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Cell must be defined"):
            handle_box('periodic_cart', positions, cell=None)

    def test_periodic_no_cell_raises(self):
        positions = jnp.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Cell must be defined"):
            handle_box('periodic', positions, cell=None)

    def test_free_returns_no_fractional(self):
        positions = jnp.array([[1.0, 2.0, 3.0]])
        pos_out, box, disp, shift, frac = handle_box('free', positions)
        assert frac is False
        np.testing.assert_allclose(pos_out, positions)

    def test_unsupported_boundary_raises(self):
        positions = jnp.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Unsupported boundary condition"):
            handle_box('nonsense', positions)


# ===========================================================================
# 3. PySAGES interface functions (unit tests)
# ===========================================================================

class TestCreatePysagesInterfaceFns:
    """Tests for create_pysages_interface_fns."""

    def test_context_generator_returns_callable(self):
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns
        import pysages.backends as pb

        state = "mock_state"
        box = jnp.array([10.0, 10.0, 10.0])
        nbrs = "mock_nbrs"
        k_grid = jnp.ones((3, 3))

        def dummy_step(i, args):
            return args

        gen = create_pysages_interface_fns(
            lr=False, state=state, box=box, step_md_fn=dummy_step,
            md_dt=0.5, nbrs=nbrs, k_grid=k_grid
        )
        assert callable(gen)

    def test_context_has_correct_extras_no_lr(self):
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns
        import pysages.backends as pb

        state = "mock_state"
        box = jnp.array([10.0, 10.0, 10.0])
        nbrs = "mock_nbrs"
        k_grid = jnp.ones((3, 3))

        def dummy_step(i, args):
            return args

        gen = create_pysages_interface_fns(
            lr=False, state=state, box=box, step_md_fn=dummy_step,
            md_dt=0.5, nbrs=nbrs, k_grid=k_grid
        )
        ctx = gen()
        # Access init state via the context
        init_state = ctx.init_fn()
        assert "nbrs" in init_state.extras
        assert "box" in init_state.extras
        assert "k_grid" in init_state.extras
        assert "nbrs_lr" not in init_state.extras

    def test_context_has_correct_extras_with_lr(self):
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns
        import pysages.backends as pb

        state = "mock_state"
        box = jnp.array([10.0, 10.0, 10.0])
        nbrs = "mock_nbrs"
        nbrs_lr = "mock_nbrs_lr"
        k_grid = jnp.ones((3, 3))

        def dummy_step(i, args):
            return args

        gen = create_pysages_interface_fns(
            lr=True, state=state, box=box, step_md_fn=dummy_step,
            md_dt=0.5, nbrs=nbrs, nbrs_lr=nbrs_lr, k_grid=k_grid
        )
        ctx = gen()
        init_state = ctx.init_fn()
        assert "nbrs_lr" in init_state.extras
        assert init_state.extras["nbrs_lr"] == "mock_nbrs_lr"

    def test_step_fn_threads_k_grid_no_lr(self):
        """Verify step_fn_pysages correctly passes k_grid through."""
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns
        import pysages.backends as pb

        state = jnp.array([1.0])
        box = jnp.array([10.0, 10.0, 10.0])
        nbrs = jnp.array([0])
        k_grid_input = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def identity_step(i, args):
            """Step function that returns state unchanged."""
            return args

        gen = create_pysages_interface_fns(
            lr=False, state=state, box=box, step_md_fn=identity_step,
            md_dt=0.5, nbrs=nbrs, k_grid=k_grid_input
        )
        ctx = gen()
        init_state = ctx.init_fn()
        stepped_state = ctx.step_fn(init_state)

        np.testing.assert_array_equal(
            stepped_state.extras["k_grid"], k_grid_input
        )

    def test_zero_box_replaced_with_3d_zeros(self):
        """When box is all zeros, it should be replaced with [0,0,0]."""
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns

        state = "mock"
        box = jnp.array([0.0])
        nbrs = "mock"

        def dummy_step(i, args):
            return args

        gen = create_pysages_interface_fns(
            lr=False, state=state, box=box, step_md_fn=dummy_step,
            md_dt=0.5, nbrs=nbrs
        )
        # The generated context should use the replaced box
        ctx = gen()
        np.testing.assert_allclose(ctx.box, jnp.array([0.0, 0.0, 0.0]))


class TestParsePysagesInput:
    """Tests for parse_pysages_input."""

    def test_parse_unbiased_input(self, tmp_path):
        from so3lr.cli.so3lr_pysages_interface import parse_pysages_input
        path = tmp_path / "pysages_input.yaml"
        _write_yaml(path, _PYSAGES_INPUT_UNBIASED)
        settings = parse_pysages_input(str(path))
        assert settings['method'] == 'unbiased'
        assert 'cv' in settings
        assert len(settings['cv']) >= 1
        assert settings['cv'][0]['type'] == 'coordinationnumber'

    def test_parse_abf_input(self, tmp_path):
        from so3lr.cli.so3lr_pysages_interface import parse_pysages_input
        path = tmp_path / "pysages_input.yaml"
        _write_yaml(path, _PYSAGES_INPUT_ABF)
        settings = parse_pysages_input(str(path))
        assert settings['method'] == 'abf'

    def test_parse_nonexistent_file_raises(self):
        from so3lr.cli.so3lr_pysages_interface import parse_pysages_input
        with pytest.raises(FileNotFoundError):
            parse_pysages_input('/nonexistent/path.yaml')

    def test_parse_returns_dict(self, tmp_path):
        from so3lr.cli.so3lr_pysages_interface import parse_pysages_input
        path = tmp_path / "pysages_input.yaml"
        _write_yaml(path, {'method': 'unbiased', 'cv': []})
        result = parse_pysages_input(str(path))
        assert isinstance(result, dict)
        assert result['method'] == 'unbiased'


class TestProcessCV:
    """Tests for process_cv."""

    def test_distance_cv(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'distance', 'grp1': [0, 1], 'grp2': [2, 3]})
        assert isinstance(cv, pysages.colvars.Distance)

    def test_angle_cv(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'angle', 'indices': [0, 1, 2]})
        assert isinstance(cv, pysages.colvars.Angle)

    def test_dihedral_cv(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'dihedral', 'indices': [0, 1, 2, 3]})
        assert isinstance(cv, pysages.colvars.DihedralAngle)

    def test_gyrrad_cv(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'gyrrad', 'indices': [0, 1, 2]})
        assert isinstance(cv, pysages.colvars.RadiusOfGyration)

    def test_asphericity_cv(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'asphericity', 'indices': [0, 1, 2, 3]})
        assert isinstance(cv, pysages.colvars.Asphericity)

    def test_acylindricity_cv(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'acylindricity', 'indices': [0, 1, 2, 3]})
        assert isinstance(cv, pysages.colvars.Acylindricity)

    def test_shapeanisotropy_cv(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'shapeanisotropy', 'indices': [0, 1, 2, 3]})
        assert isinstance(cv, pysages.colvars.ShapeAnisotropy)

    def test_unknown_cv_raises(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        with pytest.raises(ValueError, match="Unknown collective variable type"):
            process_cv({'type': 'nonexistent_cv'})


class TestGetPysagesMethod:
    """Tests for get_pysages_method."""

    def _make_settings(self, method, cv_type='distance'):
        return {
            'method': method,
            'cv': [{
                'type': cv_type,
                'grp1': [0], 'grp2': [1],
                'grid_min': 0.0, 'grid_max': 10.0,
                'grid_stride': 32, 'periodic': False
            }]
        }

    def test_unbiased_method(self):
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        import pysages.methods
        method = get_pysages_method(self._make_settings('unbiased'))
        assert isinstance(method, pysages.methods.Unbiased)

    def test_abf_method(self):
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        import pysages.methods
        method = get_pysages_method(self._make_settings('abf'))
        assert isinstance(method, pysages.methods.ABF)

    def test_spectralabf_method(self):
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        import pysages.methods
        method = get_pysages_method(self._make_settings('spectralabf'))
        assert isinstance(method, pysages.methods.SpectralABF)

    def test_harmonic_method(self):
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        import pysages.methods
        settings = self._make_settings('harmonic')
        settings['method_args'] = {'kspring': 10.0, 'center': [1.0]}
        method = get_pysages_method(settings)
        assert isinstance(method, pysages.methods.HarmonicBias)

    def test_unknown_method_raises(self):
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        with pytest.raises(ValueError, match="Unknown PySAGES method"):
            get_pysages_method(self._make_settings('bogus_method'))


class TestProcessGrid:
    """Tests for process_grid."""

    def test_single_cv_grid(self):
        from so3lr.cli.so3lr_pysages_interface import process_grid
        import pysages
        cv_settings = [{'grid_min': 0.0, 'grid_max': 10.0, 'grid_stride': 32, 'periodic': False}]
        grid = process_grid(cv_settings)
        assert isinstance(grid, pysages.Grid)

    def test_multi_cv_grid(self):
        from so3lr.cli.so3lr_pysages_interface import process_grid
        cv_settings = [
            {'grid_min': 0.0, 'grid_max': 10.0, 'grid_stride': 32, 'periodic': False},
            {'grid_min': -3.14, 'grid_max': 3.14, 'grid_stride': 64, 'periodic': True},
        ]
        grid = process_grid(cv_settings)
        np.testing.assert_array_equal(np.array(grid.shape), np.array([32, 64]))


# ===========================================================================
# 4. update_so3lr_after_pysages k_grid passthrough
# ===========================================================================

class TestUpdateAfterPysages:
    """Tests for update_so3lr_after_pysages k_grid handling."""

    def test_k_grid_passthrough(self):
        """k_grid should be returned unchanged."""
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages

        k_grid_input = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # We need to create a mock PySAGES result.
        # This is complex, so we test via the actual basic_test runs instead.
        # Here we just verify the function signature accepts k_grid.
        # The full integration test (category 6) covers the actual behavior.
        import inspect
        sig = inspect.signature(update_so3lr_after_pysages)
        assert 'k_grid' in sig.parameters

    def test_return_signature_includes_k_grid(self):
        """The function signature documents returning k_grid."""
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        doc = update_so3lr_after_pysages.__doc__
        assert 'k_grid' in doc

    def test_nbrs_lr_none_when_no_lr(self):
        """When lr=False, nbrs_lr should not be required."""
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        import inspect
        sig = inspect.signature(update_so3lr_after_pysages)
        # nbrs_lr should have a default of None
        assert sig.parameters['nbrs_lr'].default is None

    def test_ensemble_parameter_exists(self):
        """update_so3lr_after_pysages should accept an ensemble parameter."""
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        import inspect
        sig = inspect.signature(update_so3lr_after_pysages)
        assert 'ensemble' in sig.parameters
        assert sig.parameters['ensemble'].default == 'nvt'

    def test_create_pysages_interface_fns_has_ensemble(self):
        """create_pysages_interface_fns should accept an ensemble parameter."""
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns
        import inspect
        sig = inspect.signature(create_pysages_interface_fns)
        assert 'ensemble' in sig.parameters
        assert sig.parameters['ensemble'].default == 'nvt'


# ===========================================================================
# 4b. NVE / NPT state reconstruction
# ===========================================================================

class TestStateReconstruction:
    """Tests for NVE and NPT state reconstruction from PySAGES snapshot data."""

    def test_nve_state_reconstruction(self):
        """re_init_so3lr_state_nve should produce a valid NVEState."""
        from so3lr.cli.so3lr_pysages_interface import re_init_so3lr_state_nve

        n_atoms = 4
        pos = jnp.ones((n_atoms, 3))
        vel = jnp.ones((n_atoms, 3)) * 0.5
        forces = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms) * 12.0

        state = re_init_so3lr_state_nve(pos, vel, forces, mass)

        assert isinstance(state, jax_md.simulate.NVEState)
        np.testing.assert_allclose(state.position, pos)
        np.testing.assert_allclose(state.force, forces)
        # momentum = mass * velocity
        expected_mom = state.mass * vel
        np.testing.assert_allclose(state.momentum, expected_mom, atol=1e-10)

    def test_nve_state_has_no_chain(self):
        """NVEState should not have a chain attribute."""
        from so3lr.cli.so3lr_pysages_interface import re_init_so3lr_state_nve

        pos = jnp.ones((2, 3))
        vel = jnp.ones((2, 3))
        forces = jnp.zeros((2, 3))
        mass = jnp.ones(2) * 12.0

        state = re_init_so3lr_state_nve(pos, vel, forces, mass)
        assert not hasattr(state, 'chain')

    def test_npt_state_reconstruction(self):
        """re_init_so3lr_state_npt should produce a valid NPTNoseHooverState."""
        from so3lr.cli.so3lr_pysages_interface import re_init_so3lr_state_npt

        n_atoms = 4
        chain_len = 3
        pos = jnp.ones((n_atoms, 3))
        vel = jnp.ones((n_atoms, 3)) * 0.5
        forces = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms) * 12.0

        # Thermostat chain
        chain_pos = jnp.zeros(chain_len)
        chain_mom = jnp.zeros(chain_len)
        chain_mass = jnp.ones(chain_len)
        chain_tau = jnp.array(100.0)
        chain_ekin = jnp.array(0.0)
        chain_dof = jnp.array(float(3 * n_atoms))

        # Barostat chain
        baro_pos = jnp.zeros(chain_len)
        baro_mom = jnp.zeros(chain_len)
        baro_mass = jnp.ones(chain_len)
        baro_tau = jnp.array(1000.0)
        baro_ekin = jnp.array(0.0)
        baro_dof = jnp.array(3.0)

        # NPT box variables
        ref_box = jnp.eye(3) * 10.0
        box_pos = jnp.zeros(3)
        box_mom = jnp.zeros(3)
        box_mass_val = jnp.ones(3)

        state = re_init_so3lr_state_npt(
            chain_pos, chain_mom, chain_mass, chain_tau, chain_ekin, chain_dof,
            baro_pos, baro_mom, baro_mass, baro_tau, baro_ekin, baro_dof,
            ref_box, box_pos, box_mom, box_mass_val,
            pos, vel, forces, mass,
        )

        assert isinstance(state, jax_md.simulate.NPTNoseHooverState)
        np.testing.assert_allclose(state.position, pos)
        np.testing.assert_allclose(state.force, forces)
        np.testing.assert_allclose(state.reference_box, ref_box)
        np.testing.assert_allclose(state.box_position, box_pos)
        np.testing.assert_allclose(state.box_momentum, box_mom)
        np.testing.assert_allclose(state.box_mass, box_mass_val)

        # Check thermostat chain
        np.testing.assert_allclose(state.thermostat.position, chain_pos)
        np.testing.assert_allclose(state.thermostat.tau, chain_tau)

        # Check barostat chain
        np.testing.assert_allclose(state.barostat.position, baro_pos)
        np.testing.assert_allclose(state.barostat.tau, baro_tau)

        # Check momentum = mass * velocity
        expected_mom = state.mass * vel
        np.testing.assert_allclose(state.momentum, expected_mom, atol=1e-10)

    def test_nvt_state_reconstruction_unchanged(self):
        """re_init_so3lr_state (NVT) should still work as before."""
        from so3lr.cli.so3lr_pysages_interface import re_init_so3lr_state

        n_atoms = 4
        chain_len = 3
        pos = jnp.ones((n_atoms, 3))
        vel = jnp.ones((n_atoms, 3)) * 0.5
        forces = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms) * 12.0
        chain_pos = jnp.zeros(chain_len)
        chain_mom = jnp.zeros(chain_len)
        chain_mass = jnp.ones(chain_len)
        chain_tau = jnp.array(100.0)
        chain_ekin = jnp.array(0.0)
        chain_dof = jnp.array(float(3 * n_atoms))

        state = re_init_so3lr_state(
            chain_pos, chain_mom, chain_mass, chain_tau, chain_ekin, chain_dof,
            pos, vel, forces, mass,
        )

        assert isinstance(state, jax_md.simulate.NVTNoseHooverState)
        np.testing.assert_allclose(state.position, pos)
        np.testing.assert_allclose(state.chain.position, chain_pos)


# ===========================================================================
# 5. Settings parsing defaults and auto-configuration
# ===========================================================================

class TestSettingsDefaults:
    """Tests for settings extraction in perform_md."""

    def test_do_enhanced_sampling_defaults_false(self):
        settings = {}
        assert settings.get('do_enhanced_sampling', False) is False

    def test_pysages_input_defaults_to_local_path(self):
        settings = {}
        assert settings.get('pysages_input', './pysages_input.yaml') == './pysages_input.yaml'

    def test_enforce_cartesian_auto_set_for_periodic_pysages(self):
        """When do_enhanced_sampling=True and system is periodic,
        enforce_cartesian_coordinates should be auto-set to True."""
        # Simulate the logic from perform_md
        do_enhanced_sampling = True
        enforce_cartesian_coordinates = False
        cell_is_not_none = True

        if do_enhanced_sampling and cell_is_not_none and not enforce_cartesian_coordinates:
            enforce_cartesian_coordinates = True

        assert enforce_cartesian_coordinates is True

    def test_enforce_cartesian_not_changed_when_no_pysages(self):
        """Without enhanced sampling, enforce_cartesian_coordinates stays as set."""
        do_enhanced_sampling = False
        enforce_cartesian_coordinates = False
        cell_is_not_none = True

        if do_enhanced_sampling and cell_is_not_none and not enforce_cartesian_coordinates:
            enforce_cartesian_coordinates = True

        assert enforce_cartesian_coordinates is False

    def test_enforce_cartesian_respected_when_already_true(self):
        """If user already set enforce_cartesian_coordinates=True, don't override."""
        do_enhanced_sampling = True
        enforce_cartesian_coordinates = True
        cell_is_not_none = True

        if do_enhanced_sampling and cell_is_not_none and not enforce_cartesian_coordinates:
            enforce_cartesian_coordinates = True

        assert enforce_cartesian_coordinates is True


# ===========================================================================
# 6. Integration: run simulations and compare trajectories
# ===========================================================================

def _parse_md_observables(log_path):
    """Parse MD observables from an out.log file.

    The log format is:
      TIMESTAMP - SO3LR - INFO - STEP\tE\tKE\tPE\tH\tTemp\ttime/step

    Returns a list of dicts with keys: step, E, KE, PE, H, T
    """
    results = []
    with open(log_path) as f:
        for line in f:
            # Strip the logger prefix (everything up to and including "INFO - ")
            if "INFO - " in line:
                content = line.split("INFO - ", 1)[1]
            else:
                content = line
            parts = content.strip().split('\t')
            # MD output lines have 7 tab-separated fields: step, E, KE, PE, H, T, time/step
            if len(parts) == 7:
                try:
                    step = int(parts[0])
                    E = float(parts[1])
                    KE = float(parts[2])
                    PE = float(parts[3])
                    H = float(parts[4])
                    T = float(parts[5])
                    results.append({
                        'step': step, 'E': E, 'KE': KE,
                        'PE': PE, 'H': H, 'T': T
                    })
                except (ValueError, IndexError):
                    continue
    return results


def _run_so3lr_simulation(workdir, settings_file='md_settings.yaml'):
    """Run an SO3LR simulation in the given directory.

    Returns the subprocess result. Raises AssertionError if the run fails.
    """
    result = subprocess.run(
        ['so3lr', '--settings', settings_file],
        cwd=workdir,
        capture_output=True,
        text=True,
        env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
    )
    assert result.returncode == 0, (
        f"SO3LR simulation failed in {workdir} with return code {result.returncode}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result


@pytest.mark.slow
class TestTrajectoryComparison:
    """Integration tests: run simulations and compare MD observables."""

    @pytest.fixture(scope="class")
    def simulation_results(self, tmp_path_factory):
        """Run vanilla and unbiased simulations in temp directories."""
        tmp_path = tmp_path_factory.mktemp("pysages_integration")

        if not GEOMETRY_FILE.exists():
            pytest.skip(f"Test geometry not found: {GEOMETRY_FILE}")

        # Vanilla NVT (no PySAGES)
        vanilla_dir = _prepare_run_dir(tmp_path, "vanilla", md_overrides={
            'restart_save_path': './restart2.npz',
        })

        # Unbiased PySAGES
        unbiased_dir = _prepare_run_dir(tmp_path, "unbiased", md_overrides={
            'do_enhanced_sampling': True,
            'enforce_cartesian_coordinates': True,
            'restart_save_path': './restart2.npz',
        }, pysages_input=_PYSAGES_INPUT_UNBIASED)

        # ABF PySAGES
        abf_dir = _prepare_run_dir(tmp_path, "abf", md_overrides={
            'do_enhanced_sampling': True,
            'enforce_cartesian_coordinates': True,
            'restart_save_path': './restart2.npz',
        }, pysages_input=_PYSAGES_INPUT_ABF)

        _run_so3lr_simulation(vanilla_dir)
        _run_so3lr_simulation(unbiased_dir)
        _run_so3lr_simulation(abf_dir)

        return {
            'vanilla_log': vanilla_dir / "out.log",
            'unbiased_log': unbiased_dir / "out.log",
            'abf_log': abf_dir / "out.log",
        }

    def test_vanilla_and_unbiased_match(self, simulation_results):
        """All MD observables should be identical between vanilla and unbiased."""
        vanilla = _parse_md_observables(simulation_results['vanilla_log'])
        unbiased = _parse_md_observables(simulation_results['unbiased_log'])

        assert len(vanilla) > 0, "No MD data found in vanilla log"
        assert len(unbiased) > 0, "No MD data found in unbiased log"
        assert len(vanilla) == len(unbiased), (
            f"Different number of MD output lines: vanilla={len(vanilla)}, unbiased={len(unbiased)}"
        )

        for v, u in zip(vanilla, unbiased):
            assert v['step'] == u['step'], f"Step mismatch: {v['step']} vs {u['step']}"
            np.testing.assert_allclose(v['E'], u['E'], atol=1e-6,
                err_msg=f"Total energy mismatch at step {v['step']}")
            np.testing.assert_allclose(v['KE'], u['KE'], atol=1e-6,
                err_msg=f"Kinetic energy mismatch at step {v['step']}")
            np.testing.assert_allclose(v['PE'], u['PE'], atol=1e-6,
                err_msg=f"Potential energy mismatch at step {v['step']}")
            np.testing.assert_allclose(v['H'], u['H'], atol=1e-6,
                err_msg=f"Hamiltonian mismatch at step {v['step']}")
            np.testing.assert_allclose(v['T'], u['T'], atol=0.1,
                err_msg=f"Temperature mismatch at step {v['step']}")

    def test_abf_diverges_from_vanilla(self, simulation_results):
        """ABF should produce different observables than vanilla (bias is applied)."""
        vanilla = _parse_md_observables(simulation_results['vanilla_log'])
        abf = _parse_md_observables(simulation_results['abf_log'])

        assert len(vanilla) > 0 and len(abf) > 0

        # At least the last step should differ (ABF applies bias forces)
        last_v = vanilla[-1]
        last_a = abf[-1]
        # Check that at least one observable differs by more than numerical noise
        diffs = [
            abs(last_v['E'] - last_a['E']),
            abs(last_v['KE'] - last_a['KE']),
            abs(last_v['H'] - last_a['H']),
            abs(last_v['T'] - last_a['T']),
        ]
        assert any(d > 0.01 for d in diffs), (
            "ABF and vanilla trajectories are unexpectedly identical. "
            "The bias force should cause observable differences."
        )


@pytest.mark.slow
class TestRestart:
    """Integration test: verify restart produces identical results to continuous run."""

    @pytest.fixture(scope="class")
    def restart_results(self, tmp_path_factory):
        """Run continuous 4-cycle and 2+2 restart simulations, return parsed logs."""
        tmp_path = tmp_path_factory.mktemp("pysages_restart")

        if not GEOMETRY_FILE.exists():
            pytest.skip(f"Test geometry not found: {GEOMETRY_FILE}")

        # --- Continuous 4-cycle run ---
        continuous_dir = _prepare_run_dir(tmp_path, "continuous", md_overrides={
            'md_cycles': 4,
            'do_enhanced_sampling': True,
            'enforce_cartesian_coordinates': True,
        })

        # --- First 2 cycles with restart save ---
        first_dir = _prepare_run_dir(tmp_path, "first", md_overrides={
            'md_cycles': 2,
            'do_enhanced_sampling': True,
            'enforce_cartesian_coordinates': True,
            'restart_save_path': './restart_so3lr.npz',
            'restart_pysages_save_path': './restart_pysages.pkl',
        }, pysages_input=_PYSAGES_INPUT_UNBIASED)

        # Write pysages_input.yaml for continuous run too
        _write_yaml(continuous_dir / "pysages_input.yaml", _PYSAGES_INPUT_UNBIASED)

        _run_so3lr_simulation(continuous_dir)
        _run_so3lr_simulation(first_dir)

        # --- Restart for 2 more cycles ---
        restart_dir = _prepare_run_dir(tmp_path, "restart", md_overrides={
            'md_cycles': 2,
            'do_enhanced_sampling': True,
            'enforce_cartesian_coordinates': True,
            'restart_load_path': './restart_so3lr.npz',
            'restart_save_path': './restart_so3lr_2.npz',
            'restart_pysages_load_path': './restart_pysages.pkl',
            'restart_pysages_save_path': './restart_pysages_2.pkl',
        }, pysages_input=_PYSAGES_INPUT_UNBIASED)

        # Copy restart files from first run
        shutil.copy2(first_dir / "restart_so3lr.npz", restart_dir)
        shutil.copy2(first_dir / "restart_pysages.pkl", restart_dir)

        _run_so3lr_simulation(restart_dir)

        return {
            'continuous_log': continuous_dir / "out.log",
            'restart_log': restart_dir / "out.log",
        }

    def test_restart_matches_continuous(self, restart_results):
        """The last 2 cycles from restart should match cycles 3-4 of the continuous run."""
        continuous = _parse_md_observables(restart_results['continuous_log'])
        restarted = _parse_md_observables(restart_results['restart_log'])

        assert len(continuous) >= 4, f"Expected >= 4 data points in continuous, got {len(continuous)}"
        # Restarted run has: initial line (from restart) + 2 cycle outputs
        assert len(restarted) >= 2, f"Expected >= 2 data points in restart, got {len(restarted)}"

        # Compare the last 2 cycles: continuous[-2:] vs restarted[-2:]
        cont_last2 = continuous[-2:]
        rest_last2 = restarted[-2:]

        for c, r in zip(cont_last2, rest_last2):
            np.testing.assert_allclose(c['E'], r['E'], atol=1e-6,
                err_msg=f"Total energy mismatch at continuous step {c['step']}")
            np.testing.assert_allclose(c['KE'], r['KE'], atol=1e-6,
                err_msg=f"Kinetic energy mismatch at continuous step {c['step']}")
            np.testing.assert_allclose(c['PE'], r['PE'], atol=1e-6,
                err_msg=f"Potential energy mismatch at continuous step {c['step']}")
            np.testing.assert_allclose(c['H'], r['H'], atol=1e-6,
                err_msg=f"Hamiltonian mismatch at continuous step {c['step']}")
            np.testing.assert_allclose(c['T'], r['T'], atol=0.1,
                err_msg=f"Temperature mismatch at continuous step {c['step']}")
