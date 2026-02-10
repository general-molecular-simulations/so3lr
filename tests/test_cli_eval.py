import os
import subprocess
import pathlib

import pytest


EXPECTED_RMSE = {
    'dipole_vec_rmse': '5.079e-02',
    'forces_rmse': '5.246e-02',
    'hirshfeld_ratios_rmse': '1.168e-02',
}

EVAL_XYZ = 'ac_eval.xyz'
EVAL_LOG = 'ac_eval.log'


@pytest.fixture(autouse=True)
def cleanup():
    """Remove output files before and after the test."""
    for f in [EVAL_XYZ, EVAL_LOG]:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in [EVAL_XYZ, EVAL_LOG]:
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
