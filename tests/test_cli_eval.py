import os
import subprocess
import pathlib

import pytest
from ase.io import read


EXPECTED_RMSE = {
    'dipole_vec_rmse': '5.079e-02',
    'forces_rmse': '5.246e-02',
    'hirshfeld_ratios_rmse': '1.168e-02',
}

EVAL_XYZ = 'ac_eval.xyz'
EVAL_LOG = 'ac_eval.log'
EVAL_INTERMEDIATES_XYZ = 'ac_eval_intermediates.xyz'
EVAL_INTERMEDIATES_LOG = 'ac_eval_intermediates.log'


@pytest.fixture(autouse=True)
def cleanup():
    """Remove output files before and after the test."""
    for f in [EVAL_XYZ, EVAL_LOG, EVAL_INTERMEDIATES_XYZ, EVAL_INTERMEDIATES_LOG]:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in [EVAL_XYZ, EVAL_LOG, EVAL_INTERMEDIATES_XYZ, EVAL_INTERMEDIATES_LOG]:
        if os.path.exists(f):
            os.remove(f)


def test_cli_eval():
    """Run `so3lr eval` on ac.xyz and verify output files and RMSE metrics."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'ac.xyz'

    result = subprocess.run(
        [
            'so3lr', 'eval',
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', EVAL_XYZ,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"so3lr eval failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Check that output files were created
    assert os.path.exists(EVAL_XYZ), f"{EVAL_XYZ} was not created"
    assert os.path.exists(EVAL_LOG), f"{EVAL_LOG} was not created"

    # Read log file and verify RMSE values
    log_content = pathlib.Path(EVAL_LOG).read_text()

    for key, expected_value in EXPECTED_RMSE.items():
        assert f"'{key}': '{expected_value}'" in log_content, (
            f"Expected '{key}': '{expected_value}' not found in {EVAL_LOG}.\n"
            f"Log content:\n{log_content}"
        )


def test_cli_eval_with_intermediates():
    """Run `so3lr eval` with intermediate quantities (observables) as targets."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'ac.xyz'

    # Include observable targets that require output_intermediate_quantities
    intermediate_targets = 'forces,nn_energy,partial_charges,dispersion_energy'
    
    result = subprocess.run(
        [
            'so3lr', 'eval',
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', EVAL_INTERMEDIATES_XYZ,
            '--log-file', EVAL_INTERMEDIATES_LOG,
            '--targets', intermediate_targets,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"so3lr eval with intermediates failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Check that output files were created
    assert os.path.exists(EVAL_INTERMEDIATES_XYZ), f"{EVAL_INTERMEDIATES_XYZ} was not created"
    assert os.path.exists(EVAL_INTERMEDIATES_LOG), f"{EVAL_INTERMEDIATES_LOG} was not created"

    # Read the output xyz file and verify intermediate quantities are present
    atoms_list = read(EVAL_INTERMEDIATES_XYZ, index=':')
    assert len(atoms_list) > 0, "No structures found in output file"

    # Check that intermediate quantities are in the output
    first_atoms = atoms_list[0]
    
    # Check for observable quantities in arrays (per-atom quantities)
    expected_arrays = ['nn_energy_so3lr', 'partial_charges_so3lr', 'dispersion_energy_so3lr']
    for key in expected_arrays:
        assert key in first_atoms.arrays, (
            f"Expected intermediate quantity '{key}' not found in atoms.arrays. "
            f"Available keys: {list(first_atoms.arrays.keys())}"
        )

    # Read log file and check that observable targets were logged
    log_content = pathlib.Path(EVAL_INTERMEDIATES_LOG).read_text()
    assert 'Observable targets' in log_content, (
        f"Expected 'Observable targets' message not found in log.\n"
        f"Log content:\n{log_content}"
    )
    
    # Check that forces RMSE is still calculated (since it has ground truth)
    assert 'forces_rmse' in log_content, (
        f"Expected 'forces_rmse' not found in log.\n"
        f"Log content:\n{log_content}"
    )
