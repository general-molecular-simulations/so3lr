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
import shutil
import pathlib
import subprocess

import pytest

pytest.importorskip("pysages", reason="PySAGES is an optional dependency")

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
    'precision': 'float64',
    'total_charge': 0.0,
    'lr_cutoff': 12.0,
    'relax_before_run': True,
    'min_cycles': 10,
    'min_steps': 20,
    'min_start_dt': 0.05,
    'min_max_dt': 0.1,
    'min_n_min': 2,
    'md_dt': 0.5,
    'md_T': 300,
    'md_cycles': 2,
    'md_steps': 500,
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

    def test_has_pysages_flags_are_true(self):
        """PySAGES is installed here, so both import guards must report it.

        Each module wraps its PySAGES imports in ``try/except ImportError``, so
        a broken import (e.g. a renamed symbol after a PySAGES pin bump) would
        silently turn enhanced sampling into "PySAGES is not installed". The
        two flags are independent: the interface module can lose a pysages
        import while so3lr_md still imports the interface successfully.
        """
        from so3lr.cli.so3lr_pysages_interface import HAS_PYSAGES as iface_flag
        assert HAS_PYSAGES is True
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
            # Valid for the nvt-langevin parametrization so the langevin-thermo
            # input check passes and we reach the method/ensemble gate (the
            # behaviour under test). Ignored for the npt/nve cases.
            'langevin_thermo': 1.0,
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

    @pytest.mark.parametrize("method_name", ["spectralabf", "metad"])
    def test_nvt_restricted_extra_methods_reject_non_nvt(self, method_name, tmp_path):
        """The original parametrized rejection test only covers ABF. The same
        perform_md gate at lines 1710-1717 must also reject spectralabf and
        metad for non-NVT ensembles. A regression that adds a method to
        PYSAGES_ENSEMBLE_UNRESTRICTED_METHODS by mistake (or forgets to
        include a new method in the gate) would not be caught by the ABF-only
        parametrization.
        """
        import so3lr.cli.so3lr_md as md_module
        pysages_path = tmp_path / "pysages_input.yaml"
        cv_block = [{
            'type': 'distance', 'grp1': [0], 'grp2': [1],
            'grid_min': 0.0, 'grid_max': 10.0,
            'grid_stride': 32, 'periodic': False,
        }]
        method_block = {'method': method_name, 'cv': cv_block}
        if method_name == 'metad':
            method_block['method_args'] = {
                'height': 0.1, 'sigma': [0.2], 'stride': 100,
                'ngauss': 500, 'deltaT': 1500.0,
            }
        _write_yaml(pysages_path, method_block)
        settings = {
            'do_enhanced_sampling': True,
            'pysages_input': str(pysages_path),
            'input_file': 'dummy.xyz',
            'output_file': 'dummy.xyz',
            'md_dt': 0.5, 'md_T': 300, 'md_cycles': 1, 'md_steps': 10,
            'ensemble': 'npt',
            'precision': 'float32', 'seed': 42, 'save_buffer': 10,
        }
        with pytest.raises(ValueError, match="only supports the NVT ensemble"):
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

    def test_zero_box_replaced_with_3d_zeros(self):
        """When box is all zeros, it should be replaced with [0,0,0]."""
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns

        state = "mock"
        box = jnp.array([0.0])
        nbrs = "mock"

        def dummy_step(i, args):
            return args

        gen = create_pysages_interface_fns(
            state=state, box=box, step_md_fn=dummy_step,
            md_dt=0.5, nbrs=nbrs
        )
        # The generated context should use the replaced box
        ctx = gen()
        np.testing.assert_allclose(ctx.box, jnp.array([0.0, 0.0, 0.0]))

    # ------------------------------------------------------------------
    # NPT coverage: Cartesian<->fractional conversions in init_fn and
    # step_fn, and threading of nbrs_lr through the step_fn.
    # ------------------------------------------------------------------

    class _SettableState:
        """Minimal stand-in for a jax_md state. Supports `.set(position=...)`
        and exposes the result via `.position`. Used to exercise the position-
        transformation code paths without constructing a real jax_md state."""
        def __init__(self, position):
            self.position = position

        def set(self, *, position):
            return TestCreatePysagesInterfaceFns._SettableState(position)

    def test_init_fn_npt_transforms_position_to_cartesian(self):
        """For ensemble='npt', init_fn_pysages must convert the incoming
        fractional-coords state (jax-md NPT internal representation) into
        Cartesian (what PySAGES sees). The conversion is
        ``transform(box, frac)`` which is element-wise multiplication for
        orthorhombic boxes. If the NPT branch at lines 58-59/67-68 is
        skipped, the PySAGES backend would see fractional coords as if they
        were Cartesian and every CV evaluation would be broken.
        """
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns

        box = jnp.array([10.0, 10.0, 10.0])
        frac = jnp.array([[0.1, 0.2, 0.3]])
        state = self._SettableState(frac)

        def dummy_step(i, args):
            return args

        gen = create_pysages_interface_fns(
            state=state, box=box, step_md_fn=dummy_step,
            md_dt=0.5, nbrs="mock", ensemble='npt',
        )
        ctx = gen()
        ctx_state = ctx.init_fn()
        # Expected: box * frac element-wise = [[1.0, 2.0, 3.0]]
        np.testing.assert_allclose(
            ctx_state.state.position, jnp.array([[1.0, 2.0, 3.0]]), atol=1e-10
        )

    def test_init_fn_nvt_does_not_transform_position(self):
        """The mirror-image test of the above: for non-NPT ensembles, init_fn
        must NOT touch the state's position. Catches a regression where the
        NPT branch is run for every ensemble, which would silently scale
        positions in NVT by the box on every cycle boundary.
        """
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns

        box = jnp.array([10.0, 10.0, 10.0])
        positions = jnp.array([[1.0, 2.0, 3.0]])
        state = self._SettableState(positions)

        def dummy_step(i, args):
            return args

        gen = create_pysages_interface_fns(
            state=state, box=box, step_md_fn=dummy_step,
            md_dt=0.5, nbrs="mock", ensemble='nvt',
        )
        ctx = gen()
        ctx_state = ctx.init_fn()
        np.testing.assert_allclose(
            ctx_state.state.position, positions, atol=1e-10
        )

    def test_step_fn_npt_roundtrips_position(self):
        """For ensemble='npt', step_fn_pysages converts state.position from
        Cartesian to fractional before step_md_fn and back after it. With an
        identity step_md_fn the round trip is the identity; the test checks
        both halves (fractional inside, Cartesian out).
        """
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns

        box = jnp.array([10.0, 10.0, 10.0])
        cart_initial = jnp.array([[1.0, 2.0, 3.0]])
        state = self._SettableState(cart_initial / 10.0)

        seen_inside_step = []

        def capture_step(i, args):
            inner_state, _nbrs, _nbrs_lr, _box, _k, _k_disp = args
            seen_inside_step.append(np.array(inner_state.position))
            return args

        gen = create_pysages_interface_fns(
            state=state, box=box, step_md_fn=capture_step,
            md_dt=0.5, nbrs="mock", nbrs_lr="mock_lr", ensemble='npt',
        )
        ctx = gen()
        ctx_state_after_init = ctx.init_fn()
        ctx_state_after_step = ctx.step_fn(ctx_state_after_init)

        assert seen_inside_step
        np.testing.assert_allclose(
            seen_inside_step[0], np.array([[0.1, 0.2, 0.3]]), atol=1e-10
        )
        np.testing.assert_allclose(
            ctx_state_after_step.state.position, cart_initial, atol=1e-10
        )

    def test_step_fn_threads_nbrs_lr_and_k_grid(self):
        """init_fn seeds the extras and step_fn packs/unpacks the 6-tuple
        ``(state, nbrs, nbrs_lr, box, k_grid, k_grid_disp)`` through
        step_md_fn, propagating ``nbrs``, ``nbrs_lr`` and ``k_grid`` in the
        returned extras dict.
        """
        from so3lr.cli.so3lr_pysages_interface import create_pysages_interface_fns

        box = jnp.array([10.0, 10.0, 10.0])
        nbrs = "mock_nbrs"
        nbrs_lr_token = "mock_nbrs_lr"
        k_grid = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        observed_arity = []

        def capture_step(i, args):
            observed_arity.append(len(args))
            return args

        gen = create_pysages_interface_fns(
            state="mock", box=box, step_md_fn=capture_step,
            md_dt=0.5, nbrs=nbrs, nbrs_lr=nbrs_lr_token, k_grid=k_grid,
        )
        ctx = gen()
        ctx_state_after_init = ctx.init_fn()
        assert ctx_state_after_init.extras["nbrs_lr"] == nbrs_lr_token
        ctx_state_after_step = ctx.step_fn(ctx_state_after_init)

        assert observed_arity == [6], (
            f"step_md_fn should receive a 6-tuple "
            f"(state, nbrs, nbrs_lr, box, k_grid, k_grid_disp); "
            f"got arity {observed_arity}."
        )
        assert ctx_state_after_step.extras["nbrs_lr"] == nbrs_lr_token
        assert ctx_state_after_step.extras["nbrs"] == nbrs
        np.testing.assert_array_equal(ctx_state_after_step.extras["k_grid"], k_grid)


class TestParsePysagesInput:
    """Tests for parse_pysages_input."""

    def test_parse_nonexistent_file_raises(self, tmp_path):
        """When do_enhanced_sampling=True but pysages_input points to a
        nonexistent file, perform_md must propagate FileNotFoundError to
        the caller rather than catching it or substituting a less
        informative error. Drives perform_md (rather than calling
        parse_pysages_input directly) so the test verifies the error
        surfaces through perform_md's setup path — a typo in the YAML
        path should fail loudly at the CLI boundary, not get swallowed.
        """
        import so3lr.cli.so3lr_md as md_module
        settings = {
            'do_enhanced_sampling': True,
            'pysages_input': '/nonexistent/path/that/cannot/exist.yaml',
            'input_file': str(GEOMETRY_FILE),
            'output_file': str(tmp_path / 'dummy.xyz'),
            'md_dt': 0.5, 'md_T': 300, 'md_cycles': 1, 'md_steps': 1,
            'ensemble': 'nvt', 'precision': 'float32',
            'seed': 42, 'save_buffer': 10,
            'lr_cutoff': 12.0, 'total_charge': 0.0,
        }
        with pytest.raises(FileNotFoundError):
            md_module.perform_md(settings)


# CV dispatch table: settings 'type' -> the pysages.colvars class it must build.
# The *_pbc entries need the simulation box.
_CV_DISPATCH_CASES = [
    ({'type': 'distance', 'grp1': [0, 1], 'grp2': [2, 3]}, False, 'Distance'),
    ({'type': 'angle', 'indices': [0, 1, 2]}, False, 'Angle'),
    ({'type': 'dihedral', 'indices': [0, 1, 2, 3]}, False, 'DihedralAngle'),
    ({'type': 'gyrrad', 'indices': [0, 1, 2]}, False, 'RadiusOfGyration'),
    ({'type': 'asphericity', 'indices': [0, 1, 2, 3]}, False, 'Asphericity'),
    ({'type': 'acylindricity', 'indices': [0, 1, 2, 3]}, False, 'Acylindricity'),
    ({'type': 'shapeanisotropy', 'indices': [0, 1, 2, 3]}, False, 'ShapeAnisotropy'),
    ({'type': 'differenceofdistances', 'indices': [0, 1, 2, 3]}, False,
     'DifferenceOfDistances'),
    ({'type': 'distance_pbc', 'grp1': [0, 1], 'grp2': [2, 3]}, True, 'DistancePBC'),
    ({'type': 'differenceofdistances_pbc', 'indices': [0, 1, 2, 3]}, True,
     'DifferenceOfDistancesPBC'),
]


class TestProcessCV:
    """Tests for process_cv."""

    @pytest.mark.parametrize(
        "cv_dict,needs_box,cls_name", _CV_DISPATCH_CASES,
        ids=[case[0]['type'] for case in _CV_DISPATCH_CASES],
    )
    def test_cv_dispatch(self, cv_dict, needs_box, cls_name):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        box = jnp.array([10.0, 10.0, 10.0]) if needs_box else None
        cv = process_cv(cv_dict, box=box)
        assert isinstance(cv, getattr(pysages.colvars, cls_name))

    def test_unknown_cv_raises(self):
        from so3lr.cli.so3lr_pysages_interface import process_cv
        with pytest.raises(ValueError, match="Unknown collective variable type"):
            process_cv({'type': 'nonexistent_cv'})

    @pytest.mark.parametrize("cv_dict", [
        {'type': 'distance_pbc', 'grp1': [0, 1], 'grp2': [2, 3]},
        {'type': 'differenceofdistances_pbc', 'indices': [0, 1, 2, 3]},
    ], ids=['distance_pbc', 'differenceofdistances_pbc'])
    def test_pbc_cv_without_box_raises(self, cv_dict):
        """The *_pbc dispatchers raise if `box` is not given. Without this
        guard the underlying colvar would fail with an opaque error far from
        the user-facing settings file."""
        from so3lr.cli.so3lr_pysages_interface import process_cv
        with pytest.raises(ValueError, match=f"{cv_dict['type']} requires"):
            process_cv(cv_dict)

    # ------------------------------------------------------------------
    # Argument plumbing, asserted where the resulting CV exposes the
    # plumbed values (PrincipalMoment.axis, CoordinationNumber.{num_exp,
    # denom_exp, species_nn}, Distance.groups).
    # ------------------------------------------------------------------

    def test_princmom_cv_plumbs_axis(self):
        """PrincipalMoment is the only CV with an `axis` argument; this is the
        cheapest test that catches the dispatcher silently swapping the axis
        argument for something else (or dropping it entirely)."""
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars
        cv = process_cv({'type': 'princmom', 'indices': [0, 1, 2, 3], 'axis': 2})
        assert isinstance(cv, pysages.colvars.PrincipalMoment)
        assert cv.axis == 2

    def test_coordinationnumber_cv(self):
        """CoordinationNumber is the most complex dispatcher: it pulls
        ``n_atoms`` from ``nbrs.reference_position.shape[0]`` to build the
        neighbor-pool indices, threads ``species``/``box``, and forwards
        ``species`` (renamed to ``species_nn`` on the colvar) plus
        ``cn_exps`` (mapped to ``cn_exponents``, exposed as
        ``num_exp``/``denom_exp``). A regression in any of these
        translations would not be caught by an isinstance-only check.

        We use a minimal stand-in ``nbrs`` with the two attributes
        CoordinationNumber.__init__ touches: ``reference_position``
        (for n_atoms) and ``idx`` (for max-neighbors estimation)."""
        from so3lr.cli.so3lr_pysages_interface import process_cv
        import pysages.colvars.coordinates
        import types
        n_atoms = 5
        fake_nbrs = types.SimpleNamespace(
            reference_position=jnp.zeros((n_atoms, 3)),
            idx=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
        )
        species = jnp.array([1, 1, 1, 1, 6])
        box = jnp.array([10.0, 10.0, 10.0])
        cv = process_cv(
            {'type': 'coordinationnumber', 'indices': [4],
             'cn_exps': [6, 12], 'species': 1},
            nbrs=fake_nbrs,
            species=species,
            box=box,
        )
        assert isinstance(
            cv, pysages.colvars.coordinates.CoordinationNumber
        )
        # cn_exps (n, m) -> num_exp=n, denom_exp=m on the colvar
        assert cv.num_exp == 6
        assert cv.denom_exp == 12
        # 'species' from the YAML maps to species_nn on the colvar
        assert int(cv.species_nn) == 1

    def test_distance_cv_indices_plumbed(self):
        """Beyond isinstance, verify that grp1/grp2 from the settings dict
        actually reach the resulting Distance colvar. Distance stores its two groups in `cv.groups`,
        each as a sorted index array. Verify both groups arrive unchanged.
        """
        from so3lr.cli.so3lr_pysages_interface import process_cv
        cv = process_cv({'type': 'distance',
                         'grp1': [0, 1], 'grp2': [2, 3]})
        # cv.groups is [group_a_indices, group_b_indices]
        np.testing.assert_array_equal(np.array(cv.groups[0]), [0, 1])
        np.testing.assert_array_equal(np.array(cv.groups[1]), [2, 3])


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

    @pytest.mark.parametrize("method_name,method_args,cls_name", [
        ('unbiased', None, 'Unbiased'),
        ('abf', None, 'ABF'),
        ('spectralabf', None, 'SpectralABF'),
        ('harmonic', {'kspring': 10.0, 'center': [1.0]}, 'HarmonicBias'),
    ], ids=['unbiased', 'abf', 'spectralabf', 'harmonic'])
    def test_method_dispatch(self, method_name, method_args, cls_name):
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        import pysages.methods
        settings = self._make_settings(method_name)
        if method_args is not None:
            settings['method_args'] = method_args
        method = get_pysages_method(settings)
        assert isinstance(method, getattr(pysages.methods, cls_name))

    def test_unknown_method_raises(self):
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        with pytest.raises(ValueError, match="Unknown PySAGES method"):
            get_pysages_method(self._make_settings('bogus_method'))

    # ------------------------------------------------------------------
    # Additional coverage for branches that were untested previously:
    #   * the metad dispatcher (lines 477-489) — verifies every method_arg
    #     reaches the constructed Metadynamics object;
    #   * the ABF N parameter (default 500 vs override from method_args);
    #   * the restraints branch (lines 458-466) and its absence path.
    # ------------------------------------------------------------------

    def test_metad_method_plumbs_method_args(self):
        """The metad branch passes five positional method_args
        (height, sigma, stride, ngauss [-> ngaussians], deltaT) and uses
        the metal_unit_system temperature for kB. A regression that swaps
        any two of these would not be caught by an isinstance-only check.
        """
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        import pysages.methods
        settings = self._make_settings('metad')
        settings['method_args'] = {
            'height': 0.05,
            'sigma': [0.3],
            'stride': 200,
            'ngauss': 750,
            'deltaT': 1500.0,
        }
        method = get_pysages_method(settings)
        assert isinstance(method, pysages.methods.Metadynamics)
        assert method.height == 0.05
        assert list(method.sigma) == [0.3]
        assert method.stride == 200
        assert method.ngaussians == 750
        assert method.deltaT == 1500.0

    def test_abf_N_defaults_to_500_when_method_args_absent(self):
        """With no 'method_args' block, the ABF branch must default N to 500."""
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        method = get_pysages_method(self._make_settings('abf'))
        assert method.N == 500

    def test_abf_N_read_from_method_args(self):
        """With a 'method_args' block, the ABF branch must read N from it
        rather than the hardcoded default. A regression that always uses
        500 (e.g. dropping the .get) would not be caught by the default-
        path test alone."""
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        settings = self._make_settings('abf')
        settings['method_args'] = {'N': 1234}
        method = get_pysages_method(settings)
        assert method.N == 1234

    def test_restraints_propagate_to_method(self):
        """When a 'restraints' block is present, it must surface on the
        method as a CVRestraints with matching lower/upper/kl/ku. Catches a
        regression that builds CVRestraints but never passes it into the
        method constructor."""
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        import pysages
        settings = self._make_settings('abf')
        settings['restraints'] = {
            'lower': (0.5,), 'upper': (9.5,),
            'kl': (100.0,), 'ku': (200.0,),
        }
        method = get_pysages_method(settings)
        assert method.restraints is not None
        # CVRestraints stores its fields as jnp arrays — coerce for comparison.
        np.testing.assert_allclose(np.array(method.restraints.lower), [0.5])
        np.testing.assert_allclose(np.array(method.restraints.upper), [9.5])
        np.testing.assert_allclose(np.array(method.restraints.kl), [100.0])
        np.testing.assert_allclose(np.array(method.restraints.ku), [200.0])

    def test_no_restraints_when_block_absent(self):
        """Absence of a 'restraints' block must yield method.restraints=None
        (the `else: restraints = None` branch at line 468). Without this
        explicit test, dropping the else branch could leave a stale or
        wrong-default value on the method."""
        from so3lr.cli.so3lr_pysages_interface import get_pysages_method
        method = get_pysages_method(self._make_settings('abf'))
        assert method.restraints is None


class TestProcessGrid:
    """Tests for process_grid."""

    def test_multi_cv_grid(self):
        from so3lr.cli.so3lr_pysages_interface import process_grid
        cv_settings = [
            {'grid_min': 0.0, 'grid_max': 10.0, 'grid_stride': 32, 'periodic': False},
            {'grid_min': -3.14, 'grid_max': 3.14, 'grid_stride': 64, 'periodic': True},
        ]
        grid = process_grid(cv_settings)
        np.testing.assert_array_equal(np.array(grid.shape), np.array([32, 64]))

    # ------------------------------------------------------------------
    # Beyond the shape, the three other fields the function constructs from
    # the settings must arrive on the Grid object: lower (from grid_min),
    # upper (from grid_max), and the periodic flag (computed as
    # `all(periodic)`). A regression that uses `any(periodic)` instead of
    # `all(periodic)` is invisible to a shape-only check.
    # ------------------------------------------------------------------

    def test_grid_lower_and_upper_propagate(self):
        from so3lr.cli.so3lr_pysages_interface import process_grid
        cv_settings = [
            {'grid_min': -2.5, 'grid_max': 7.5, 'grid_stride': 32, 'periodic': False},
            {'grid_min': 0.25, 'grid_max': 6.28, 'grid_stride': 64, 'periodic': False},
        ]
        grid = process_grid(cv_settings)
        np.testing.assert_allclose(np.array(grid.lower), [-2.5, 0.25], atol=1e-10)
        np.testing.assert_allclose(np.array(grid.upper), [7.5, 6.28], atol=1e-10)

    def test_grid_periodic_uses_all_not_any(self):
        """The function builds `periodic=all(periodic)`. With mixed flags the
        result must be False (any-flag-False -> not-fully-periodic). Catches a
        mutation from `all` to `any`, which would produce a periodic Grid in
        a setting where one or more CVs is non-periodic — a silent
        correctness bug."""
        from so3lr.cli.so3lr_pysages_interface import process_grid
        cv_settings_mixed = [
            {'grid_min': 0.0, 'grid_max': 10.0, 'grid_stride': 32, 'periodic': True},
            {'grid_min': 0.0, 'grid_max': 10.0, 'grid_stride': 32, 'periodic': False},
        ]
        cv_settings_all = [
            {'grid_min': 0.0, 'grid_max': 10.0, 'grid_stride': 32, 'periodic': True},
            {'grid_min': 0.0, 'grid_max': 10.0, 'grid_stride': 32, 'periodic': True},
        ]
        assert process_grid(cv_settings_mixed).is_periodic is False
        assert process_grid(cv_settings_all).is_periodic is True


# ===========================================================================
# 4. update_so3lr_after_pysages k_grid passthrough
# ===========================================================================

class _RecordingNeighbors:
    """Stand-in for a jax_md NeighborList.

    Records the arguments passed to ``update`` so the test can assert that
    update_so3lr_after_pysages refreshes the neighbor list with the correct
    box and positions, and returns self so the resulting object is reusable.
    """

    def __init__(self, idx=None):
        self.idx = idx if idx is not None else jnp.zeros((1,), dtype=jnp.int32)
        self.last_update = None

    def update(self, positions, neighbor=None, box=None):
        self.last_update = {
            'positions': positions, 'neighbor': neighbor, 'box': box,
        }
        return self


def _make_snapshot(positions, velocities, masses, forces, box_H,
                   chain_data=None, barostat_data=None, npt_box_data=None):
    """Build a minimal object matching what update_so3lr_after_pysages reads
    out of ``raw_result.snapshots[-1]``."""
    import types
    snap = types.SimpleNamespace()
    snap.positions = positions
    snap.vel_mass = (velocities, masses)
    snap.forces = forces
    snap.box = types.SimpleNamespace(H=box_H)
    snap.chain_data = chain_data
    snap.barostat_data = barostat_data
    snap.npt_box_data = npt_box_data
    return snap


def _make_result(snapshot, bias):
    """Build a minimal stand-in for pysages.Result.

    Only the attributes that update_so3lr_after_pysages touches are populated:
    ``snapshots[-1]`` and ``states[0].bias``.
    """
    import types
    state0 = types.SimpleNamespace(bias=bias)
    return types.SimpleNamespace(snapshots=[snapshot], states=[state0])


def _nvt_chain_data(n_atoms, chain_len=3):
    return {
        'position':  jnp.zeros(chain_len),
        'momentum':  jnp.zeros(chain_len),
        'mass':      jnp.ones(chain_len),
        'tau':       jnp.array(100.0),
        'kinetic_energy': jnp.array(0.0),
        'degrees_of_freedom': jnp.array(float(3 * n_atoms)),
    }


def _barostat_data(chain_len=3):
    return {
        'position':  jnp.zeros(chain_len),
        'momentum':  jnp.zeros(chain_len),
        'mass':      jnp.ones(chain_len),
        'tau':       jnp.array(1000.0),
        'kinetic_energy': jnp.array(0.0),
        'degrees_of_freedom': jnp.array(3.0),
    }


def _npt_box_data(ref_box, box_pos=None):
    return {
        'reference_box': ref_box,
        'box_position':  jnp.zeros(3) if box_pos is None else box_pos,
        'box_momentum':  jnp.zeros(3),
        'box_mass':      jnp.ones(3),
    }


class TestUpdateAfterPysages:
    """Behavioural tests for update_so3lr_after_pysages.

    These tests build a fake PySAGES Result and call the function directly so
    the assertions check actual behaviour (state reconstruction, bias-force
    restoration, neighbor-list refresh, NPT box recomputation) rather than
    inspecting the function signature.
    """

    DT = 0.5
    MD_T = 300.0

    def _make_nvt_inputs(self, n_atoms=4, bias=None, box_side=10.0):
        positions = jnp.ones((n_atoms, 3))
        velocities = jnp.ones((n_atoms, 3)) * 0.5
        forces = jnp.zeros((n_atoms, 3))
        masses = jnp.ones(n_atoms) * 12.0
        box_H = jnp.eye(3) * box_side
        snap = _make_snapshot(
            positions, velocities, masses, forces, box_H,
            chain_data=_nvt_chain_data(n_atoms),
        )
        return _make_result(snap, bias), positions, velocities, forces, masses

    # ---- k_grid passthrough (real, not signature-only) -------------------

    @pytest.mark.parametrize("with_k_grid", [True, False], ids=['array', 'none'])
    def test_k_grid_passed_through(self, with_k_grid):
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        result, *_ = self._make_nvt_inputs()
        k_grid_in = jnp.array([[1.0, 2.0], [3.0, 4.0]]) if with_k_grid else None

        _, _, _, _, k_grid_out, _ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=_RecordingNeighbors(), nbrs_lr=_RecordingNeighbors(),
            ensemble='nvt', k_grid=k_grid_in,
        )
        # Same object identity (so also the same values, or None).
        assert k_grid_out is k_grid_in

    # ---- NVT reconstruction ---------------------------------------------

    def test_nvt_reconstructs_state_without_bias(self):
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        result, positions, velocities, forces, masses = self._make_nvt_inputs(bias=None)
        nbrs = _RecordingNeighbors()

        state, nbrs_out, nbrs_lr_out, new_box, _, _ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=nbrs, nbrs_lr=_RecordingNeighbors(), ensemble='nvt', k_grid=None,
        )

        assert isinstance(state, jax_md.simulate.NVTNoseHooverState)
        np.testing.assert_allclose(state.position, positions)
        # No bias => force should be exactly the unbiased force from the snapshot
        np.testing.assert_allclose(state.force, forces)
        # momentum = canonicalized_mass * velocity
        np.testing.assert_allclose(state.momentum, state.mass * velocities, atol=1e-10)
        # The thermostat chain is rebuilt from the snapshot's chain_data.
        chain_data = result.snapshots[-1].chain_data
        np.testing.assert_allclose(state.chain.position, chain_data['position'])
        np.testing.assert_allclose(state.chain.tau, chain_data['tau'])
        # Box came from snapshot.box.H
        np.testing.assert_allclose(new_box, jnp.eye(3) * 10.0)
        assert nbrs_lr_out.last_update is not None
        # Neighbor list was refreshed with the snapshot positions and new box
        assert nbrs.last_update is not None
        np.testing.assert_allclose(nbrs.last_update['positions'], positions)
        np.testing.assert_allclose(nbrs.last_update['box'], new_box)

    def test_nvt_bias_restores_force_momentum_and_chain_ke(self):
        """When raw_result.states[0].bias != 0, update_so3lr_after_pysages must
        re-apply the post-step bias kick that the PySAGES jax-md backend
        already applied to the integrator but did *not* capture in the snapshot:

          force   <- snapshot.forces + bias
          momentum <- mass*velocity + 0.5 * dt * bias
          chain.kinetic_energy <- KE(new_momentum)         (NVT invariant)

        These three steps are the block at so3lr_pysages_interface.py lines
        336-361 and are easy to break (the historical half-bias bug missed
        the 2nd-half kick).
        """
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        from jax_md.quantity import kinetic_energy

        n_atoms = 4
        bias = jnp.full((n_atoms, 3), 0.3)  # non-zero, asymmetric vs. force
        result, _, velocities, forces, _ = self._make_nvt_inputs(n_atoms=n_atoms, bias=bias)

        state, *_ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=_RecordingNeighbors(), nbrs_lr=_RecordingNeighbors(),
            ensemble='nvt', k_grid=None,
        )

        np.testing.assert_allclose(state.force, forces + bias, atol=1e-10)
        expected_mom = state.mass * velocities + 0.5 * self.DT * bias
        np.testing.assert_allclose(state.momentum, expected_mom, atol=1e-10)
        # The chain KE must be consistent with the (post-kick) momentum,
        # otherwise the next thermostat half-step rescales by the wrong factor.
        expected_KE = kinetic_energy(momentum=expected_mom, mass=state.mass)
        np.testing.assert_allclose(state.chain.kinetic_energy, expected_KE, atol=1e-10)

    # ---- NVE reconstruction ---------------------------------------------

    def test_nve_reconstructs_state(self):
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages

        n_atoms = 3
        positions = jnp.ones((n_atoms, 3)) * 2.0
        velocities = jnp.ones((n_atoms, 3)) * 0.25
        forces = jnp.zeros((n_atoms, 3))
        masses = jnp.ones(n_atoms) * 16.0
        snap = _make_snapshot(positions, velocities, masses, forces,
                              jnp.eye(3) * 20.0)
        result = _make_result(snap, bias=None)

        state, _, nbrs_lr_out, new_box, _, _ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=_RecordingNeighbors(), nbrs_lr=_RecordingNeighbors(),
            ensemble='nve', k_grid=None,
        )

        assert isinstance(state, jax_md.simulate.NVEState)
        assert not hasattr(state, 'chain')
        np.testing.assert_allclose(state.position, positions)
        np.testing.assert_allclose(state.force, forces)
        np.testing.assert_allclose(state.momentum, state.mass * velocities, atol=1e-10)
        assert nbrs_lr_out.last_update is not None
        np.testing.assert_allclose(new_box, jnp.eye(3) * 20.0)

    def test_nve_bias_skips_chain_ke_resync(self):
        """NVE has no thermostat chain — the chain.kinetic_energy resync at
        lines 354-357 must not fire (NVEState has no ``chain`` field). The
        force/momentum kick is still applied."""
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages

        n_atoms = 3
        positions = jnp.ones((n_atoms, 3))
        velocities = jnp.ones((n_atoms, 3)) * 0.25
        forces = jnp.zeros((n_atoms, 3))
        masses = jnp.ones(n_atoms) * 16.0
        bias = jnp.full((n_atoms, 3), -0.2)

        snap = _make_snapshot(positions, velocities, masses, forces, jnp.eye(3) * 20.0)
        result = _make_result(snap, bias=bias)

        state, *_ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=_RecordingNeighbors(), nbrs_lr=_RecordingNeighbors(),
            ensemble='nve', k_grid=None,
        )

        np.testing.assert_allclose(state.force, forces + bias, atol=1e-10)
        expected_mom = state.mass * velocities + 0.5 * self.DT * bias
        np.testing.assert_allclose(state.momentum, expected_mom, atol=1e-10)
        assert not hasattr(state, 'chain')

    # ---- NPT reconstruction ---------------------------------------------

    def test_npt_recomputes_box_from_state_not_snapshot(self):
        """For NPT, the returned new_box must come from
        ``jax_md.simulate.npt_box(reconstructed_state)`` — NOT from
        ``snapshot.box.H``. The snapshot lags by one step in NPT because the
        barostat updates the box after the snapshot is taken; trusting it
        feeds a stale box back into the neighbor list and the next step's
        coordinate conversion.

        We seed ``box_position`` to a non-zero value so npt_box(state) and
        snapshot.box.H differ, then verify the returned box matches the former.
        """
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages

        n_atoms = 2
        ref_box = jnp.array([10.0, 10.0, 10.0])
        # box_position=ln(1.2) ~ 0.1823 -> npt_box = ref_box * 1.2 = [12,12,12]
        # snapshot.box.H is intentionally wrong (different value).
        box_pos = jnp.log(jnp.array([1.2, 1.2, 1.2]))
        cart_pos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # cartesian
        velocities = jnp.zeros((n_atoms, 3))
        forces = jnp.zeros((n_atoms, 3))
        masses = jnp.ones(n_atoms) * 12.0
        wrong_box_H = jnp.array([7.0, 7.0, 7.0])  # different from npt_box(state)

        snap = _make_snapshot(
            cart_pos, velocities, masses, forces, wrong_box_H,
            chain_data=_nvt_chain_data(n_atoms),
            barostat_data=_barostat_data(),
            npt_box_data=_npt_box_data(ref_box, box_pos=box_pos),
        )
        result = _make_result(snap, bias=None)
        nbrs = _RecordingNeighbors()
        nbrs_lr = _RecordingNeighbors()

        state, _, _, new_box, _, _ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=nbrs, ensemble='npt',
            nbrs_lr=nbrs_lr, k_grid=None,
        )

        # Box was recomputed from the state, so it must equal npt_box(state),
        # not the (deliberately wrong) snapshot.box.H value.
        np.testing.assert_allclose(
            new_box, jax_md.simulate.npt_box(state), atol=1e-10
        )
        # And specifically not the stale snapshot value.
        assert not jnp.allclose(new_box, wrong_box_H)
        # Neighbor list was refreshed against the recomputed box.
        np.testing.assert_allclose(nbrs.last_update['box'], new_box)
        np.testing.assert_allclose(nbrs_lr.last_update['box'], new_box)

    def test_npt_converts_positions_to_fractional(self):
        """JAX-MD NPT runs in fractional coordinates internally; the function
        must transform the snapshot's cartesian positions back to fractional
        using the *recomputed* box before storing them on the state."""
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages

        n_atoms = 2
        # Orthorhombic NPT boxes in jax-md are 1-D vectors (the diagonal),
        # not 3x3 matrices — npt_box returns the same shape, and the
        # interface inverts the box element-wise via 1.0/box.
        ref_box = jnp.array([10.0, 10.0, 10.0])
        cart_pos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = jnp.zeros((n_atoms, 3))
        forces = jnp.zeros((n_atoms, 3))
        masses = jnp.ones(n_atoms) * 12.0

        snap = _make_snapshot(
            cart_pos, velocities, masses, forces, jnp.array([99.0, 99.0, 99.0]),
            chain_data=_nvt_chain_data(n_atoms),
            barostat_data=_barostat_data(),
            npt_box_data=_npt_box_data(ref_box),  # box_position=0 => npt_box == ref_box
        )
        result = _make_result(snap, bias=None)

        state, _, _, new_box, _, _ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=_RecordingNeighbors(), nbrs_lr=_RecordingNeighbors(),
            ensemble='npt', k_grid=None,
        )

        # box_position=0 => new_box == reference_box
        np.testing.assert_allclose(new_box, ref_box, atol=1e-10)
        # positions on state must be fractional = cart / box_diag
        np.testing.assert_allclose(state.position, cart_pos / 10.0, atol=1e-10)

    # ---- Error handling --------------------------------------------------

    def test_unsupported_ensemble_raises(self):
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        result, *_ = self._make_nvt_inputs()
        with pytest.raises(ValueError, match="Unsupported ensemble"):
            update_so3lr_after_pysages(
                result, init_fn=None, rng_key=None,
                md_T=self.MD_T, dt=self.DT, nbrs=_RecordingNeighbors(), nbrs_lr=_RecordingNeighbors(),
                ensemble='nonsense', k_grid=None,
            )

    # ---- NPT bias-kick thermostat KE resync -----------------------------

    def test_npt_bias_restores_thermostat_kinetic_energy(self):
        """The NVT analogue of this test is
        ``test_nvt_bias_restores_force_momentum_and_chain_ke``; here is the
        NPT counterpart for the symmetric code at lines 358-361.

        For NPT with non-zero bias, the post-step bias kick
        (``momentum += 0.5*dt*bias``) invalidates the snapshot's
        chain.kinetic_energy field (which was KE of the *pre-kick* momentum).
        ``update_so3lr_after_pysages`` must recompute KE from the post-kick
        momentum and write it back as ``state.thermostat.kinetic_energy``
        (NPT stores its NHC-thermostat under ``thermostat`` rather than
        ``chain``). If this resync is skipped — e.g. because the NPT branch
        falls through to the NVT branch and writes to ``state.chain`` (which
        doesn't exist on NPTNoseHooverState), or just isn't there at all —
        the next thermostat half-step rescales by an inconsistent factor.
        """
        from so3lr.cli.so3lr_pysages_interface import update_so3lr_after_pysages
        from jax_md.quantity import kinetic_energy

        n_atoms = 2
        ref_box = jnp.array([10.0, 10.0, 10.0])
        cart_pos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = jnp.full((n_atoms, 3), 0.3)
        forces = jnp.zeros((n_atoms, 3))
        masses = jnp.ones(n_atoms) * 12.0
        bias = jnp.full((n_atoms, 3), 0.4)

        snap = _make_snapshot(
            cart_pos, velocities, masses, forces, jnp.array([99.0, 99.0, 99.0]),
            chain_data=_nvt_chain_data(n_atoms),
            barostat_data=_barostat_data(),
            npt_box_data=_npt_box_data(ref_box),
        )
        result = _make_result(snap, bias=bias)

        state, *_ = update_so3lr_after_pysages(
            result, init_fn=None, rng_key=None,
            md_T=self.MD_T, dt=self.DT, nbrs=_RecordingNeighbors(), nbrs_lr=_RecordingNeighbors(),
            ensemble='npt', k_grid=None,
        )

        # Sanity: the bias kick was applied to momentum (already tested
        # elsewhere for NVT/NVE; included here as the precondition for the
        # KE assertion below).
        expected_mom = state.mass * velocities + 0.5 * self.DT * bias
        np.testing.assert_allclose(state.momentum, expected_mom, atol=1e-10)

        # The NPT thermostat KE must reflect KE(post-kick momentum).
        expected_KE = kinetic_energy(momentum=expected_mom, mass=state.mass)
        np.testing.assert_allclose(
            state.thermostat.kinetic_energy, expected_KE, atol=1e-10
        )


# ===========================================================================
# 4b. NVE / NPT state reconstruction
# ===========================================================================

class TestStateReconstruction:
    """Tests for NPT state reconstruction from PySAGES snapshot data.

    The NVE and NVT rebuilds are covered through update_so3lr_after_pysages
    in TestUpdateAfterPysages (it calls the same re_init_* helpers)."""

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

        # The isotropic npt_nose_hoover integrator caches dUdV (dE/deps w.r.t. a
        # scalar perturbation) and consumes it on the first half-step, so the seed
        # MUST be a scalar. A (3, 3) seed broadcasts box_momentum to (3, 3) and
        # crashes the first NPT step after a PySAGES re-init. Pin the shape here as
        # a cheap guard; see test_npt_reinit_state_drives_real_integrator for the
        # behavioral check.
        assert jnp.shape(state.dUdV) == ()

    def test_npt_reinit_state_drives_real_integrator(self):
        """A state rebuilt by re_init_so3lr_state_npt must integrate without error.

        Regression guard for the dUdV seed shape. The npt-anisotropic-1 jax-md
        branch caches dUdV in NPTNoseHooverState and consumes it on the first
        integrator half-step (box_force), before recomputing it. For the isotropic
        npt_nose_hoover, dUdV is a SCALAR; seeding (3, 3) broadcasts box_momentum
        to (3, 3) and raises a shape error on the first step after a PySAGES cycle.

        Unlike test_step_fn_npt_*, which mock step_md_fn, this drives a REAL
        jax_md.simulate.npt_nose_hoover apply_fn on a re-init'd state, so it
        actually exercises the path that crashed. Robust to future jax-md changes
        because it only asserts the integrator runs and the carry stays stable.
        """
        from so3lr.cli.so3lr_pysages_interface import re_init_so3lr_state_npt
        from jax_md import space, simulate, quantity

        n_atoms = 12
        box = jnp.eye(3) * 12.0
        key = jax.random.PRNGKey(1)
        R = jax.random.uniform(key, (n_atoms, 3)) * 12.0
        disp, shift = space.periodic_general(box, fractional_coordinates=False)

        # Toy soft-repulsion energy; must accept box + perturbation like jax-md's NPT.
        def energy_fn(R, box=box, perturbation=None, **kw):
            d = space.map_product(disp)
            dr = space.distance(d(R, R))
            r = dr + jnp.eye(n_atoms) * 1e3
            return jnp.sum(jnp.where(r < 6.0, (6.0 - r) ** 2, 0.0)) * 0.5

        init_fn, apply_fn = simulate.npt_nose_hoover(
            energy_fn, shift, dt=1e-3, pressure=1.0, kT=1.0
        )

        # A real state (correct scalar dUdV), advanced so chains/box are populated.
        st = init_fn(key, R, box=box)
        for _ in range(5):
            st = apply_fn(st)

        # Round-trip through the interface exactly as update_so3lr_after_pysages does.
        th, ba = st.thermostat, st.barostat
        reinit = re_init_so3lr_state_npt(
            th.position, th.momentum, th.mass, th.tau, th.kinetic_energy,
            int(quantity.count_dof(st.position)),
            ba.position, ba.momentum, ba.mass, ba.tau, ba.kinetic_energy, 1,
            st.reference_box, st.box_position, st.box_momentum, st.box_mass,
            st.position, st.velocity, st.force, st.mass,
        )
        assert jnp.shape(reinit.dUdV) == ()

        # Drive the re-init'd state through real integrator steps (this used to crash).
        s = reinit
        for _ in range(10):
            s = apply_fn(s)

        assert jnp.shape(s.box_momentum) == ()
        assert jnp.shape(s.dUdV) == ()
        assert bool(jnp.all(jnp.isfinite(s.position)))
        assert bool(jnp.isfinite(jnp.asarray(s.box_momentum)))
        assert bool(jnp.isfinite(jnp.asarray(s.box_position)))


# ===========================================================================
# 5. Settings parsing defaults and auto-configuration
# ===========================================================================

_WARN_SUBSTR = "cartesian coordinates for periodic systems"
_NPT_INFO_SUBSTR = "NPT+PySAGES: overriding enforce_cartesian_coordinates=False"


class TestPerformMdEnforceCartesianGating:
    """Tests for the enforce_cartesian_coordinates gating in perform_md.

    For periodic systems with do_enhanced_sampling=True, perform_md must:

      * flip ``enforce_cartesian_coordinates`` False -> True for non-NPT
        ensembles (warning logged), because PySAGES requires Cartesian
        internal coords;
      * flip True -> False for NPT (info logged), because JAX-MD NPT
        requires fractional internal coords (the PySAGES interface
        converts per step);
      * leave the flag untouched in every other case, with no log line.

    With do_enhanced_sampling False *or absent*, no PySAGES code path may run:
    no "Parsed settings for PySAGES" line even though a valid
    pysages_input.yaml is passed. The absent row catches the default for
    ``do_enhanced_sampling`` flipping from False to True.

    These tests drive perform_md directly and inspect stderr for the
    warning/info lines. The settings name a model path that does not exist,
    so perform_md stops with FileNotFoundError at the model load, the first
    step after the gating block -- no model is loaded and no MD runs.
    """

    @pytest.fixture
    def base_settings(self, tmp_path):
        """Minimal settings dict that lets perform_md reach the gating block."""
        pysages_path = tmp_path / "pysages_input.yaml"
        _write_yaml(pysages_path, _PYSAGES_INPUT_UNBIASED)
        return {
            'pysages_input': str(pysages_path),
            'input_file': str(GEOMETRY_FILE),
            'output_file': str(tmp_path / 'dummy.xyz'),
            # Not a registered name and not a path: load_model raises right
            # after the gate, before anything is loaded.
            'model_path': str(tmp_path / 'no_such_model'),
            'md_dt': 0.5, 'md_T': 300, 'md_cycles': 1, 'md_steps': 1,
            'precision': 'float32',
            'seed': 42, 'save_buffer': 10,
            'lr_cutoff': 12.0, 'total_charge': 0.0,
        }

    # The matrix of inputs that decides the gating outcome.
    #   do_es (None = key absent), ensemble, init_flag, expect_warn, expect_info
    # All cases assume a periodic cell (which the test geometry provides).
    @pytest.mark.parametrize("do_es,ensemble,init_flag,expect_warn,expect_info", [
        # PySAGES on, non-NPT: must force cartesian True (warning on F->T)
        (True,  'nvt', False, True,  False),
        (True,  'nvt', True,  False, False),  # already True, no transition
        # PySAGES on, NPT: must force cartesian False (info on T->F)
        (True,  'npt', True,  False, True),
        (True,  'npt', False, False, False),  # already False, no transition
        # PySAGES off: flag passes through untouched, no log either way
        (False, 'nvt', False, False, False),
        (False, 'nvt', True,  False, False),
        (False, 'npt', False, False, False),
        # Key absent: exactly the inputs that would trigger the override
        # (periodic, enforce_cartesian_coordinates=False) if the default flipped.
        (None,  'nvt', False, False, False),
    ])
    def test_enforce_cartesian_logging(self, base_settings, capfd,
                                       do_es, ensemble, init_flag,
                                       expect_warn, expect_info):
        import so3lr.cli.so3lr_md as md_module
        settings = dict(base_settings)
        settings.update({
            'ensemble': ensemble,
            'enforce_cartesian_coordinates': init_flag,
        })
        if do_es is not None:
            settings['do_enhanced_sampling'] = do_es

        # Reaching the model load proves the geometry was read and the gating
        # block ran; any earlier failure raises something else.
        with pytest.raises(FileNotFoundError, match="not a known name"):
            md_module.perform_md(settings)
        captured = capfd.readouterr()
        output = captured.out + captured.err

        parsed_seen = "Parsed settings for PySAGES" in output
        warn_seen = _WARN_SUBSTR in output
        info_seen = _NPT_INFO_SUBSTR in output
        assert parsed_seen == bool(do_es), (
            f"parse_pysages_input ran={parsed_seen} for do_es={do_es}; "
            f"PySAGES must be entered only when do_enhanced_sampling is True.\n"
            f"--- output ---\n{output}"
        )
        assert warn_seen == expect_warn, (
            f"warning expectation mismatch for "
            f"(do_es={do_es}, ensemble={ensemble!r}, init_flag={init_flag}): "
            f"expected_warn={expect_warn}, saw warn={warn_seen}.\n"
            f"--- output ---\n{output}"
        )
        assert info_seen == expect_info, (
            f"info expectation mismatch for "
            f"(do_es={do_es}, ensemble={ensemble!r}, init_flag={init_flag}): "
            f"expected_info={expect_info}, saw info={info_seen}.\n"
            f"--- output ---\n{output}"
        )


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
