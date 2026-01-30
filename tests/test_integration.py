"""
SO3LR CLI Integration Tests.

These tests run actual simulations with the CLI to verify end-to-end functionality.
"""
import os
import subprocess
import shutil
import pathlib
import pytest
import yaml

# ============================================================================
# Test case discovery
# ============================================================================

def get_integration_test_cases():
    """
    Discover all integration test cases.
    
    Test cases are subdirectories under tests/integration/ that contain
    a settings.yaml file.
    """
    integration_dir = pathlib.Path(__file__).parent / 'integration'
    if not integration_dir.exists():
        return []
    test_cases = []
    for folder in integration_dir.iterdir():
        if folder.is_dir() and (folder / 'settings.yaml').exists():
            test_cases.append(folder.name)
    return sorted(test_cases)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return pathlib.Path(__file__).parent / 'test_data'

@pytest.fixture
def integration_workdir(tmp_path, request):
    """
    Set up a working directory for an integration test case.
    Copies all files from the test case directory to a temporary directory.
    """
    test_case = request.param
    integration_dir = pathlib.Path(__file__).parent / 'integration' / test_case
    if not integration_dir.exists():
        pytest.skip(f"Test case directory not found: {integration_dir}")
    for item in integration_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, tmp_path)
        elif item.is_dir() and item.name not in ['__pycache__', 'hdf5', 'expected']:
            shutil.copytree(item, tmp_path / item.name)
    expected_dir = integration_dir / 'expected'
    if expected_dir.exists():
        shutil.copytree(expected_dir, tmp_path / 'expected')
    return tmp_path, test_case, integration_dir

# ============================================================================
# SO3LR CLI Integration Tests
# ============================================================================

@pytest.mark.parametrize('integration_workdir', get_integration_test_cases(), indirect=True)
def test_integration_case(integration_workdir):
    """
    Run CLI integration test for a test case.
    """
    workdir, test_case, source_dir = integration_workdir
    settings_file = workdir / 'settings.yaml'
    print(f"\n{'='*60}")
    print(f"Running test case: {test_case}")
    print(f"Working directory: {workdir}")
    print(f"{'='*60}")
    setup_settings = workdir / 'setup_settings.yaml'
    if setup_settings.exists():
        print("Running setup phase...")
        setup_result = subprocess.run(
            ['so3lr', '--settings', 'setup_settings.yaml'],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        (workdir / 'setup_stdout.log').write_text(setup_result.stdout)
        (workdir / 'setup_stderr.log').write_text(setup_result.stderr)
        assert setup_result.returncode == 0, (
            f"Setup phase failed with return code {setup_result.returncode}\n"
            f"STDERR:\n{setup_result.stderr}"
        )
        print("Setup phase completed")
    result = subprocess.run(
        ['so3lr', '--settings', 'settings.yaml'],
        cwd=workdir,
        capture_output=True,
        text=True,
        env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
    )
    (workdir / 'stdout.log').write_text(result.stdout)
    (workdir / 'stderr.log').write_text(result.stderr)
    assert result.returncode == 0, (
        f"CLI failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert 'completed successfully' in result.stderr.lower() or \
           'completed successfully' in result.stdout.lower(), (
        f"Success message not found in output\n"
        f"STDERR:\n{result.stderr}"
    )
    validate_script = workdir / 'validate.py'
    if validate_script.exists():
        validate_result = subprocess.run(
            ['python', 'validate.py'],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        assert validate_result.returncode == 0, (
            f"Validation failed:\n{validate_result.stderr}"
        )

class TestSO3LRCLISubcommands:
    """Test individual SO3LR CLI subcommands with minimal configurations."""
    @pytest.fixture
    def sample_molecule(self, tmp_path, test_data_dir):
        src = test_data_dir / 'acetone.xyz'
        dst = tmp_path / 'input.xyz'
        shutil.copy2(src, dst)
        return tmp_path
    def test_opt_subcommand(self, sample_molecule):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'opt',
                '--input', 'input.xyz',
                '--output', 'optimized.xyz',
                '--min-cycles', '2',
                '--min-steps', '2',
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        assert result.returncode == 0, f"opt failed: {result.stderr}"
        assert (workdir / 'optimized.xyz').exists()
    def test_nvt_subcommand(self, sample_molecule):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'nvt',
                '--input', 'input.xyz',
                '--output', 'trajectory.hdf5',
                '--temperature', '300',
                '--dt', '0.5',
                '--md-cycles', '2',
                '--md-steps', '2',
                '--no-relax',
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        assert result.returncode == 0, f"nvt failed: {result.stderr}"
    def test_nve_subcommand(self, sample_molecule):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'nve',
                '--input', 'input.xyz',
                '--output', 'trajectory_nve.hdf5',
                '--dt', '0.5',
                '--md-cycles', '2',
                '--md-steps', '2',
                '--no-relax',
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        assert result.returncode == 0, f"nve failed: {result.stderr}"
    def test_eval_subcommand(self, sample_molecule):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'eval',
                '--datafile', 'input.xyz',
                '--save-to', 'evaluated.extxyz',
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        assert result.returncode == 0, f"eval failed: {result.stderr}"
        assert (workdir / 'evaluated.extxyz').exists(), "Output file not created"

class TestSO3LRCLIOptions:
    """Test various SO3LR CLI subcommand options and configurations."""
    @pytest.fixture
    def sample_molecule(self, tmp_path, test_data_dir):
        src = test_data_dir / 'acetone.xyz'
        dst = tmp_path / 'input.xyz'
        shutil.copy2(src, dst)
        return tmp_path
    def test_nvt_with_relax(self, sample_molecule):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'nvt',
                '--input', 'input.xyz',
                '--output', 'trajectory.hdf5',
                '--temperature', '300',
                '--dt', '0.5',
                '--md-cycles', '2',
                '--md-steps', '2',
                '--relax',  # Enable relaxation
                '--force-conv', '0.5',  # Fast convergence
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        assert result.returncode == 0, f"nvt with relax failed: {result.stderr}"
    def test_eval_with_targets(self, sample_molecule):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'eval',
                '--datafile', 'input.xyz',
                '--save-to', 'evaluated.extxyz',
                '--targets', 'forces,dipole_vec',
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        assert result.returncode == 0, f"eval with targets failed: {result.stderr}"
    def test_nvt_xyz_output(self, sample_molecule):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'nvt',
                '--input', 'input.xyz',
                '--output', 'trajectory.xyz',  # XYZ format instead of HDF5
                '--temperature', '300',
                '--dt', '0.5',
                '--md-cycles', '2',
                '--md-steps', '2',
                '--no-relax',
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            env={**os.environ, 'JAX_TRACEBACK_FILTERING': 'off'},
        )
        assert result.returncode == 0, f"nvt xyz output failed: {result.stderr}"
        assert (workdir / 'trajectory.xyz').exists()

# ============================================================================
# Utility functions for test validation
# ============================================================================

def validate_hdf5_output(filepath):
    import h5py
    with h5py.File(filepath, 'r') as f:
        assert 'positions' in f or 'R' in f, "Missing positions dataset"
        return True

def validate_xyz_output(filepath):
    from ase.io import read
    atoms = read(filepath)
    assert len(atoms) > 0, "Empty structure"
    return True
