"""
SO3LR CLI Integration Tests.

These tests run actual simulations with the CLI to verify end-to-end functionality.
"""
import subprocess
import shutil
import pathlib
import pytest

# A CLI run that stalls must fail on its own, with its so3lr.log on disk, rather
# than take the whole shard to the 60-minute cancel with nothing to read.
# ~4x the slowest case ever measured (kspace_electrostatics_p3m at 411 s, when it
# still ran a 3072-atom water box).
SUBPROCESS_TIMEOUT = 1500

from subprocess_env import so3lr_subprocess_env

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
def integration_workdir(tmp_path, request, test_data_dir):
    """
    A working directory for an integration test case: the shared structures in
    tests/test_data, then the case's own files (which win), and the validation
    utilities its validate.py imports.
    """
    test_case = request.param
    integration_dir = pathlib.Path(__file__).parent / 'integration' / test_case
    shutil.copytree(test_data_dir, tmp_path, dirs_exist_ok=True)
    for item in integration_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, tmp_path)
    shutil.copy2(integration_dir.parent / 'validation_utils.py', tmp_path)
    return tmp_path, test_case

# ============================================================================
# SO3LR CLI Integration Tests
# ============================================================================

@pytest.mark.parametrize('integration_workdir', get_integration_test_cases(), indirect=True)
def test_integration_case(integration_workdir):
    """
    Run CLI integration test for a test case.
    """
    workdir, test_case = integration_workdir
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
            env=so3lr_subprocess_env(),
            timeout=SUBPROCESS_TIMEOUT,
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
        env=so3lr_subprocess_env(),
        timeout=SUBPROCESS_TIMEOUT,
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
    # Every case ships a validate.py.
    validate_result = subprocess.run(
        ['python', 'validate.py'],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    # Validator.report() prints its errors to stdout.
    assert validate_result.returncode == 0, (
        f"Validation failed:\n{validate_result.stdout}\n{validate_result.stderr}"
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
            env=so3lr_subprocess_env(),
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0, f"opt failed: {result.stderr}"
        assert (workdir / 'optimized.xyz').exists()
        from ase.io import read
        assert len(read(workdir / 'optimized.xyz')) == len(read(workdir / 'input.xyz'))
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
            env=so3lr_subprocess_env(),
            timeout=SUBPROCESS_TIMEOUT,
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
            env=so3lr_subprocess_env(),
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0, f"nve failed: {result.stderr}"

class TestSO3LRCLIOptions:
    """Test various SO3LR CLI subcommand options and configurations."""
    @pytest.fixture
    def sample_molecule(self, tmp_path, test_data_dir):
        src = test_data_dir / 'acetone.xyz'
        dst = tmp_path / 'input.xyz'
        shutil.copy2(src, dst)
        return tmp_path
    @pytest.mark.parametrize('output_file', ['trajectory.hdf5', 'trajectory.xyz'])
    def test_nvt_with_relax(self, sample_molecule, output_file):
        workdir = sample_molecule
        result = subprocess.run(
            [
                'so3lr', 'nvt',
                '--input', 'input.xyz',
                '--output', output_file,
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
            env=so3lr_subprocess_env(),
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0, f"nvt with relax failed: {result.stderr}"
        assert (workdir / 'trajectory_opt.xyz').exists()
        if output_file.endswith('.hdf5'):
            import h5py
            with h5py.File(workdir / output_file) as trajectory:
                assert trajectory['positions'].shape[0] == 2
        else:
            from ase.io import read
            assert len(read(workdir / output_file, index=':')) == 2
