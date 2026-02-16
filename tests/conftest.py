"""
Pytest configuration for so3lr tests.

Adds a --full option to run comprehensive versions of tests.
By default, slow tests run a lighter version (e.g., fewer atoms).
"""
import pytest


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


@pytest.fixture
def full_test(request):
    """Fixture that returns True if --full flag is set."""
    return request.config.getoption("--full")
