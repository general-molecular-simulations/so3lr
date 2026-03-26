"""
Pytest configuration for so3lr tests.

Adds a --full option to run comprehensive versions of tests.
By default, slow tests run a lighter version (e.g., fewer atoms).
"""
import pytest

# This table needs to be manually maintained by developers as new tests are added and timings are collected.
KNOWN_SLOW_TEST_DURATIONS = {
    'tests/mlff_tests/test_data_loader_ase.py::test_data_load_with_pbc[True]': 198.23,
    'tests/test_integration.py::test_integration_case[opt]': 116.55,
    'tests/test_integration.py::test_integration_case[restart_checkpoint]': 115.70,
    'tests/test_cli_train.py::test_cli_train_water_pbc_with_long_range': 112.14,
    'tests/test_integration.py::test_integration_case[restart_ensemble_change]': 111.27,
    'tests/test_cli_train.py::test_cli_train_forces_and_energy_with_lse_shifts': 91.94,
    'tests/test_cli_train.py::test_cli_train_water_pbc_forces_only': 91.76,
    'tests/test_integration.py::test_integration_case[kspace_electrostatics_p3m]': 90.02,
    'tests/test_cli_train.py::test_cli_train_forces_only': 84.79,
    'tests/test_cli_finetune.py::test_cli_finetune': 80.94,
}


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run full test suite (comprehensive numerical checks)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselected by default, use --full to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection unless --full is specified:
    - Skip tests marked with @pytest.mark.slow
    - For parametrized molecule tests ['atat', 'dha', 'bb'], only run 'dha'
    - Order known long-running tests first so xdist workers pick them up earlier
    """
    if config.getoption("--full"):
        return

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

    items.sort(key=lambda item: KNOWN_SLOW_TEST_DURATIONS.get(item.nodeid, 0.0), reverse=True)


@pytest.fixture
def full_test(request):
    """Fixture that returns True if --full flag is set."""
    return request.config.getoption("--full")
