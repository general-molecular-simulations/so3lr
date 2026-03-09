import subprocess
import pathlib

import pytest
from ase.io import read


EXPECTED_RMSE = {
    'dipole_vec_rmse': '5.079e-02',
    'forces_rmse': '5.246e-02',
    'hirshfeld_ratios_rmse': '1.168e-02',
}


@pytest.fixture
def eval_dir(tmp_path):
    """Provide an isolated temporary directory for eval output files."""
    return tmp_path


def test_cli_eval(eval_dir):
    """Run `so3lr eval` on ac.xyz and verify output files and RMSE metrics."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'ac.xyz'
    eval_xyz = eval_dir / 'ac_eval.xyz'
    eval_log = eval_dir / 'ac_eval.log'

    result = subprocess.run(
        [
            'so3lr', 'eval',
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', str(eval_xyz),
            '--log-file', str(eval_log),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=eval_dir,
    )

    assert result.returncode == 0, (
        f"so3lr eval failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    assert eval_xyz.exists(), f"{eval_xyz} was not created"
    assert eval_log.exists(), f"{eval_log} was not created"

    log_content = eval_log.read_text()

    for key, expected_value in EXPECTED_RMSE.items():
        assert f"'{key}': '{expected_value}'" in log_content, (
            f"Expected '{key}': '{expected_value}' not found in {eval_log}.\n"
            f"Log content:\n{log_content}"
        )


def test_cli_eval_with_intermediates(eval_dir):
    """Run `so3lr eval` with intermediate quantities (observables) as targets."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'ac.xyz'
    eval_xyz = eval_dir / 'ac_eval_intermediates.xyz'
    eval_log = eval_dir / 'ac_eval_intermediates.log'

    intermediate_targets = 'forces,nn_energy,partial_charges,dispersion_energy'

    result = subprocess.run(
        [
            'so3lr', 'eval',
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', str(eval_xyz),
            '--log-file', str(eval_log),
            '--targets', intermediate_targets,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=eval_dir,
    )

    assert result.returncode == 0, (
        f"so3lr eval with intermediates failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    assert eval_xyz.exists(), f"{eval_xyz} was not created"
    assert eval_log.exists(), f"{eval_log} was not created"

    atoms_list = read(str(eval_xyz), index=':')
    assert len(atoms_list) > 0, "No structures found in output file"

    first_atoms = atoms_list[0]
    expected_arrays = ['nn_energy_so3lr', 'partial_charges_so3lr', 'dispersion_energy_so3lr']
    for key in expected_arrays:
        assert key in first_atoms.arrays, (
            f"Expected intermediate quantity '{key}' not found in atoms.arrays. "
            f"Available keys: {list(first_atoms.arrays.keys())}"
        )

    log_content = eval_log.read_text()
    assert 'Observable targets' in log_content, (
        f"Expected 'Observable targets' message not found in log.\n"
        f"Log content:\n{log_content}"
    )

    assert 'forces_rmse' in log_content, (
        f"Expected 'forces_rmse' not found in log.\n"
        f"Log content:\n{log_content}"
    )


def test_cli_eval_with_embeddings(eval_dir):
    """Run `so3lr eval --targets forces,x` and verify _x.npz is saved correctly."""
    import numpy as np
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'ac.xyz'
    eval_xyz = eval_dir / 'ac_eval_embeddings.xyz'
    eval_log = eval_dir / 'ac_eval_embeddings.log'
    eval_npz = eval_dir / 'ac_eval_embeddings_x.npz'

    result = subprocess.run(
        [
            'so3lr', 'eval',
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', str(eval_xyz),
            '--log-file', str(eval_log),
            '--targets', 'forces,x',
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=eval_dir,
    )

    assert result.returncode == 0, (
        f"so3lr eval with embeddings failed (return code {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert eval_xyz.exists(), f"{eval_xyz} was not created"
    assert eval_npz.exists(), f"{eval_npz} was not created"

    data = np.load(eval_npz)
    assert 'embeddings' in data, f"'embeddings' key missing from NPZ. Keys: {list(data.keys())}"
    assert 'n_atoms' in data, f"'n_atoms' key missing from NPZ. Keys: {list(data.keys())}"

    embeddings = data['embeddings']
    n_atoms = data['n_atoms']

    assert embeddings.ndim == 2, f"Expected 2D embeddings array, got shape {embeddings.shape}"
    assert embeddings.shape[1] > 0, "Feature dimension must be non-zero"

    n_structures = len(read(str(datafile), index=':'))
    assert n_atoms.shape == (n_structures,), (
        f"Expected n_atoms shape ({n_structures},), got {n_atoms.shape}"
    )
    assert n_atoms.sum() == embeddings.shape[0], (
        f"n_atoms.sum()={n_atoms.sum()} != embeddings.shape[0]={embeddings.shape[0]}"
    )

    atoms_list = read(str(eval_xyz), index=':')
    assert len(atoms_list) == n_structures
    for atoms in atoms_list:
        assert 'x_so3lr' not in atoms.arrays, (
            "'x_so3lr' should not appear in extxyz arrays (embeddings go to NPZ only)"
        )

    log_content = eval_log.read_text()
    assert 'Embeddings saved to' in log_content, (
        f"Expected 'Embeddings saved to' in log.\nLog content:\n{log_content}"
    )
