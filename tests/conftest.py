"""
Pytest configuration for so3lr tests.

Adds a --full option that also runs the tests marked @pytest.mark.slow, which
are skipped by default.

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
    'tests/test_charge_embed_override.py::TestChargeEmbedOverrideCLI::test_command_line_flag_changes_the_run': 1.0,
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
    - Unless --full: skip @pytest.mark.slow
    - Order the modules holding known long-running tests first, each module kept
      together in file order, so module- and class-scoped fixtures are built once
    - With --shard i/n: keep only this shard's duration-balanced slice
    """
    if not config.getoption("--full"):
        skip_slow = pytest.mark.skip(reason="Slow test (use --full to run)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

        # Sorting single tests by duration would split a class or module, and pytest
        # then tears down its class/module fixtures and builds them again.
        def module(item):
            return item.nodeid.split('::', 1)[0]
        weight = {}
        for item in items:
            weight[module(item)] = max(weight.get(module(item), 0.0), _test_duration(item))
        position = {id(item): i for i, item in enumerate(items)}
        items.sort(key=lambda item: (-weight[module(item)], position[id(item)]))

    shard = config.getoption("--shard")
    if shard is not None:
        index, count = _parse_shard(shard)
        selected = _shard_items(items, index, count)
        keep = {id(item) for item in selected}
        deselected = [item for item in items if id(item) not in keep]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        # _shard_items packs by duration; run the shard in the order above so a
        # module's tests stay together.
        order = {id(item): i for i, item in enumerate(items)}
        items[:] = sorted(selected, key=lambda item: order[id(item)])


@pytest.fixture(scope="module")
def x64():
    """float64 for the requesting module. Without x64, a calculator asked for
    dtype=np.float64 silently computes in float32. The flag is global to the
    process, so it is restored afterwards."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.fixture
def full_test(request):
    """Fixture that returns True if --full flag is set."""
    return request.config.getoption("--full")
