"""
Pytest configuration for so3lr tests.

Adds a --full option to run comprehensive versions of tests.
By default, slow tests run a lighter version (e.g., fewer atoms).

Adds a --shard i/n option that splits a run into duration-balanced slices, used
by CI to keep any single job well under the runner's practical wall-clock and
memory ceiling.
"""
import jax
import pytest

# Measured wall-clock seconds, used to (a) order slow tests first and (b) balance
# --shard slices. They only need to be roughly right, since both consumers care
# about relative size.
#
# This table needs to be manually maintained by developers as new tests are added
# and timings are collected. A test that is missing here is assumed to cost
# DEFAULT_TEST_DURATION, which is deliberately non-zero so unknown tests still
# get spread across shards instead of piling into shard 1.
KNOWN_SLOW_TEST_DURATIONS = {
    # Measured 2026-10-21 on MeluXina (pytest --durations, 8 xdist workers).
    # tests/test_integration.py
    'tests/test_integration.py::test_integration_case[restart_checkpoint]': 60.0,
    'tests/test_integration.py::test_integration_case[restart_ensemble_change]': 56.0,
    'tests/test_integration.py::test_integration_case[observables]': 41.0,
    'tests/test_integration.py::test_integration_case[kspace_electrostatics_p3m]': 37.0,
    'tests/test_integration.py::test_integration_case[npt_ensemble]': 35.0,
    'tests/test_integration.py::test_integration_case[nvt_langevin]': 31.0,
    'tests/test_integration.py::test_integration_case[initial_velocities]': 30.0,
    'tests/test_integration.py::test_integration_case[nve_ensemble]': 28.0,
    'tests/test_integration.py::TestSO3LRCLIOptions::test_nvt_with_relax[trajectory.hdf5]': 39.0,
    'tests/test_integration.py::TestSO3LRCLIOptions::test_nvt_with_relax[trajectory.xyz]': 40.0,
    'tests/test_integration.py::TestSO3LRCLISubcommands::test_opt_subcommand': 30.0,
    'tests/test_integration.py::TestSO3LRCLISubcommands::test_nvt_subcommand': 29.0,
    'tests/test_integration.py::TestSO3LRCLISubcommands::test_nve_subcommand': 28.0,
    # other suites
    # CLI subprocess runs: the shared `so3lr nvt` runs land on whichever test
    # asks for them first.
    'tests/test_charge_embed_override.py::TestChargeEmbedOverrideCLI::test_settings_yaml_route': 112.0,
    'tests/test_charge_embed_override.py::TestChargeEmbedOverrideCLI::test_command_line_flag_changes_the_run': 111.0,
    'tests/test_charge_embed_override.py::TestChargeEmbedOverrideCLI::test_command_line_flag_matches_settings_yaml': 1.0,
    'tests/test_cli_train.py::test_cli_train_forces_and_energy_with_lse_shifts': 79.0,
    'tests/test_cli_train.py::test_cli_train_water_pbc_with_long_range': 60.0,
    'tests/test_cli_finetune.py::test_cli_finetune[last_layer]': 35.0,
    'tests/test_cli_finetune.py::test_cli_finetune[forces_only]': 31.0,
    'tests/mlff_tests/test_data_loader_ase.py::test_data_load_with_pbc[True]': 13.0,
}

# Cost assumed for a test with no entry above.
DEFAULT_TEST_DURATION = 30.0


def _test_duration(item):
    return KNOWN_SLOW_TEST_DURATIONS.get(item.nodeid, DEFAULT_TEST_DURATION)


def _parse_shard(value):
    """Parse an ``i/n`` shard spec into a validated ``(index, count)`` pair."""
    try:
        index_str, count_str = value.split('/')
        index, count = int(index_str), int(count_str)
    except ValueError:
        raise pytest.UsageError(
            f"--shard expects 'i/n' (e.g. '2/3'), got {value!r}"
        )
    if count < 1 or not 1 <= index <= count:
        raise pytest.UsageError(
            f"--shard index must be within 1..n, got {value!r}"
        )
    return index, count


def _shard_items(items, index, count):
    """Split items into ``count`` duration-balanced bins; return bin ``index``.

    Greedy longest-processing-time bin packing: repeatedly place the most
    expensive remaining test into the lightest bin. For a workload dominated by
    a few long tests -- which this suite is -- that lands within a few percent
    of optimal, and unlike splitting by name it needs no maintenance when a
    test case is added.

    Ties break on nodeid so every shard sees the identical partition; sharding
    on inconsistent orderings would silently drop or duplicate tests.
    """
    ordered = sorted(items, key=lambda i: (-_test_duration(i), i.nodeid))
    bins = [[] for _ in range(count)]
    loads = [0.0] * count
    for item in ordered:
        lightest = loads.index(min(loads))
        bins[lightest].append(item)
        loads[lightest] += _test_duration(item)
    return bins[index - 1]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run full test suite (comprehensive numerical checks)"
    )
    parser.addoption(
        "--shard",
        action="store",
        default=None,
        metavar="i/n",
        help="Run only duration-balanced slice i of n (e.g. --shard 2/3)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers and full-precision matmul.

    The numerical tests (symmetry, batching, PME-vs-direct, gradients) assert
    tight tolerances that hold in exact arithmetic. On GPU, JAX's default f32
    matmul uses reduced-precision tensor-core passes (bf16/TF32-style) that break
    those tolerances, so they pass on CPU (true f32) but fail on GPU. Force full
    f32 matmul for the test session so results are backend-independent. Production
    keeps the fast `default` precision via the matmul_precision setting.
    """
    jax.config.update("jax_default_matmul_precision", "highest")
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselected by default, use --full to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection:
    - Unless --full: skip @pytest.mark.slow and non-'dha' molecule params
    - Order known long-running tests first so xdist workers pick them up earlier
    - With --shard i/n: keep only this shard's duration-balanced slice
    """
    if not config.getoption("--full"):
        skip_slow = pytest.mark.skip(reason="Slow test (use --full to run)")
        # Molecule names to skip in short mode (keep only 'dha')
        skip_molecules = {"atat", "bb"}

        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

            # Skip non-default molecule parametrizations
            if hasattr(item, "callspec") and "name" in item.callspec.params:
                if item.callspec.params["name"] in skip_molecules:
                    item.add_marker(pytest.mark.skip(
                        reason=f"Molecule '{item.callspec.params['name']}' only runs with --full"
                    ))

        items.sort(key=lambda item: _test_duration(item), reverse=True)

    shard = config.getoption("--shard")
    if shard is not None:
        index, count = _parse_shard(shard)
        selected = _shard_items(items, index, count)
        keep = {id(item) for item in selected}
        deselected = [item for item in items if id(item) not in keep]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture
def full_test(request):
    """Fixture that returns True if --full flag is set."""
    return request.config.getoption("--full")


@pytest.fixture(scope="session")
def pysages_cli_runs(tmp_path_factory):
    """Shared `so3lr` CLI runs for the PySAGES and restart-coordinate tests.

    tests/test_pysages_interface.py and tests/test_restart_coordinates.py
    compare the MD log rows of these runs (vanilla vs unbiased PySAGES,
    restart vs continuous). They used to launch 12 subprocesses, four of them
    duplicates; here each distinct run happens at most once per session, on
    first use. A restart run first launches the run whose files it loads.

    Every run: one benzoic acid molecule in a periodic 25 A box, float64,
    100 steps per cycle and no relaxation (a restart skips relaxation anyway,
    so the continuous runs it is compared with must not relax either).

    Returns ``run(name)`` -> list of ``{step, E, KE, PE, H, T}`` rows parsed
    from that run's log.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    import yaml

    geometry = Path(__file__).resolve().parent / 'test_data' / 'benzoic_acid_aq_1.xyz'
    base_settings = {
        'precision': 'float64',
        'total_charge': 0.0,
        'lr_cutoff': 12.0,
        'relax_before_run': False,
        'md_dt': 0.5,  # fs (the CLI/YAML convention; converted fs->ps internally)
        'md_T': 300,
        'md_steps': 100,
        'input_file': geometry.name,
        'output_file': 'traj.xyz',
        'log_file': 'out.log',
    }
    cv = [{
        'type': 'coordinationnumber',
        'indices': [4, 7],
        'cn_exps': [6, 30],
        'species': 1,
        'grid_min': 0.0,
        'grid_max': 1.5,
        'grid_stride': 64,
        'periodic': False,
    }]
    pysages = {'do_enhanced_sampling': True, 'enforce_cartesian_coordinates': True}
    # name -> (settings overrides, PySAGES method or None, run whose restart files it loads)
    specs = {
        'vanilla_4': ({'md_cycles': 4}, None, None),
        'unbiased_4': ({'md_cycles': 4, **pysages}, 'unbiased', None),
        'abf_4': ({'md_cycles': 4, **pysages}, 'abf', None),
        'vanilla_2_saved': (
            {'md_cycles': 2, 'restart_save_path': './restart.npz'}, None, None),
        'unbiased_2_saved': (
            {'md_cycles': 2, **pysages, 'restart_save_path': './restart.npz',
             'restart_pysages_save_path': './restart_pysages.pkl'}, 'unbiased', None),
        # Fractional-coordinate (vanilla) restart loaded into a Cartesian PySAGES run.
        'restart_vanilla_to_unbiased': (
            {'md_cycles': 2, **pysages, 'restart_load_path': './restart.npz',
             'restart_save_path': './restart2.npz'}, 'unbiased', 'vanilla_2_saved'),
        # Cartesian (PySAGES) restart loaded into a fractional-coordinate vanilla run.
        'restart_unbiased_to_vanilla': (
            {'md_cycles': 2, 'restart_load_path': './restart.npz',
             'restart_save_path': './restart2.npz'}, None, 'unbiased_2_saved'),
        # PySAGES restart that also resumes the saved PySAGES state.
        'restart_unbiased_with_state': (
            {'md_cycles': 2, **pysages, 'restart_load_path': './restart.npz',
             'restart_save_path': './restart2.npz',
             'restart_pysages_load_path': './restart_pysages.pkl',
             'restart_pysages_save_path': './restart_pysages_2.pkl'},
            'unbiased', 'unbiased_2_saved'),
    }
    run_dirs = {}
    rows = {}

    def parse_md_rows(log_path):
        # Log rows: "<prefix> INFO - step\tE\tKE\tPE\tH\tT\ttime/step"
        parsed = []
        for line in Path(log_path).read_text().splitlines():
            content = line.split("INFO - ", 1)[1] if "INFO - " in line else line
            parts = content.strip().split('\t')
            if len(parts) != 7:
                continue
            try:
                parsed.append({
                    'step': int(parts[0]), 'E': float(parts[1]),
                    'KE': float(parts[2]), 'PE': float(parts[3]),
                    'H': float(parts[4]), 'T': float(parts[5]),
                })
            except ValueError:
                continue
        return parsed

    def run(name):
        if name in rows:
            return rows[name]
        overrides, method, restart_from = specs[name]
        run_dir = tmp_path_factory.mktemp(f"pysages_{name}")
        shutil.copy2(geometry, run_dir / geometry.name)
        if restart_from is not None:
            run(restart_from)
            for key in ('restart_load_path', 'restart_pysages_load_path'):
                if key in overrides:
                    restart_file = Path(overrides[key]).name
                    shutil.copy2(run_dirs[restart_from] / restart_file, run_dir / restart_file)
        with open(run_dir / 'md_settings.yaml', 'w') as f:
            yaml.dump({**base_settings, **overrides}, f)
        if method is not None:
            with open(run_dir / 'pysages_input.yaml', 'w') as f:
                yaml.dump({'method': method, 'cv': cv}, f)
        result = subprocess.run(
            ['so3lr', '--settings', 'md_settings.yaml'],
            cwd=run_dir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
            timeout=1500,
        )
        assert result.returncode == 0, (
            f"SO3LR run '{name}' failed in {run_dir} with return code "
            f"{result.returncode}\nSTDERR:\n{result.stderr}"
        )
        run_dirs[name] = run_dir
        rows[name] = parse_md_rows(run_dir / 'out.log')
        return rows[name]

    return run
