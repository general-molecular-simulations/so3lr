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
# --shard slices. Values come from a passing ubuntu-latest CI run; they only need
# to be roughly right, since both consumers care about relative size.
#
# This table needs to be manually maintained by developers as new tests are added
# and timings are collected. A test that is missing here is assumed to cost
# DEFAULT_TEST_DURATION, which is deliberately non-zero so unknown tests still
# get spread across shards instead of piling into shard 1.
KNOWN_SLOW_TEST_DURATIONS = {
    # tests/test_integration.py -- measured 2026-08 (previously ~5x stale)
    # restart_checkpoint: 621 s before its input shrank from 3072 to 1536 atoms.
    # 340 s is that 621 s scaled by the 1.81x speedup measured locally through
    # pytest (95.8 s -> 52.9 s); refresh from a real CI run when convenient.
    'tests/test_integration.py::test_integration_case[restart_checkpoint]': 340.0,
    'tests/test_integration.py::test_integration_case[kspace_electrostatics_p3m]': 411.0,
    'tests/test_integration.py::test_integration_case[npt_ensemble]': 318.0,
    'tests/test_integration.py::test_integration_case[restart_ensemble_change]': 59.0,
    'tests/test_integration.py::test_integration_case[observables]': 55.0,
    'tests/test_integration.py::test_integration_case[initial_velocities]': 31.0,
    'tests/test_integration.py::test_integration_case[nve_ensemble]': 29.0,
    'tests/test_integration.py::test_integration_case[nvt_langevin]': 28.0,
    'tests/test_integration.py::TestSO3LRCLIOptions::test_nvt_with_relax[trajectory.hdf5]': 49.0,
    'tests/test_integration.py::TestSO3LRCLIOptions::test_nvt_with_relax[trajectory.xyz]': 49.0,
    'tests/test_integration.py::TestSO3LRCLISubcommands::test_opt_subcommand': 30.0,
    'tests/test_integration.py::TestSO3LRCLISubcommands::test_nvt_subcommand': 27.0,
    'tests/test_integration.py::TestSO3LRCLISubcommands::test_nve_subcommand': 26.0,
    # other suites
    # CLI subprocess runs: each spawns a real `so3lr nvt`, so they dominate the
    # charge-embed file. Measured locally 2026-08-19.
    'tests/test_charge_embed_override.py::TestChargeEmbedOverrideCLI::test_settings_yaml_route': 20.0,
    'tests/test_charge_embed_override.py::TestChargeEmbedOverrideCLI::test_command_line_flag_matches_settings_yaml': 10.0,
    'tests/test_charge_embed_override.py::TestChargeEmbedOverrideCLI::test_command_line_flag_changes_the_run': 10.0,
    'tests/mlff_tests/test_data_loader_ase.py::test_data_load_with_pbc[True]': 198.23,
    'tests/test_cli_train.py::test_cli_train_water_pbc_with_long_range': 112.14,
    'tests/test_cli_train.py::test_cli_train_forces_and_energy_with_lse_shifts': 91.94,
    'tests/test_cli_finetune.py::test_cli_finetune[full]': 80.94,
    'tests/test_cli_finetune.py::test_cli_finetune[last_layer]': 80.94,
    'tests/test_cli_finetune.py::test_cli_finetune[final_mlp]': 80.94,
    'tests/test_cli_finetune.py::test_cli_finetune[forces_only]': 80.94,
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
