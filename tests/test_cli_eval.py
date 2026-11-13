import math
import re
import subprocess
import pathlib

import pytest
from ase.io import read

from subprocess_env import so3lr_subprocess_env


def _parse_metric_table_rmse(log_content, target):
    """Return the RMSE for ``target`` from the eval metrics table.

    The eval CLI logs a per-target table whose rows are
    ``<target>  <MAE>  <RMSE>  <N>  <unit>`` (embedded in a timestamped log
    line). We key on the target name and read the second numeric column.
    """
    m = re.search(
        rf"\b{re.escape(target)}\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(\d+)\b",
        log_content,
    )
    assert m is not None, (
        f"No metrics-table row for target '{target}'.\nLog content:\n{log_content}"
    )
    return float(m.group(2))  # group(1)=MAE, group(2)=RMSE


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
            # The legacy (v1) model, whose ratio head is 'hirshfeld_ratios'; the
            # no-arg default is so3lr-tauri-m-2606 (see test_cli_eval_default_model).
            '--model', 'so3lr',
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', str(eval_xyz),
            '--log-file', str(eval_log),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=eval_dir,
        env=so3lr_subprocess_env(),
    )

    assert result.returncode == 0, (
        f"so3lr eval failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    assert eval_xyz.exists(), f"{eval_xyz} was not created"
    assert eval_log.exists(), f"{eval_log} was not created"

    log_content = eval_log.read_text()

    # Legacy 'so3lr' model: the ratio head is reported as 'hirshfeld_ratios'.
    for target in ('forces', 'dipole_vec', 'hirshfeld_ratios'):
        rmse = _parse_metric_table_rmse(log_content, target)
        assert math.isfinite(rmse) and rmse > 0.0, (
            f"{target} RMSE = {rmse} is not a finite positive float.\n"
            f"Log content:\n{log_content}"
        )


def test_cli_eval_default_model(eval_dir):
    """Run `so3lr eval` with no --model (default so3lr-tauri-m-2606)."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'ac.xyz'
    eval_xyz = eval_dir / 'ac_eval_default.xyz'
    eval_log = eval_dir / 'ac_eval_default.log'

    result = subprocess.run(
        [
            'so3lr', 'eval',  # no --model -> default so3lr-tauri-m-2606
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', str(eval_xyz),
            '--log-file', str(eval_log),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=eval_dir,
        env=so3lr_subprocess_env(),
    )

    assert result.returncode == 0, (
        f"so3lr eval failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert eval_xyz.exists(), f"{eval_xyz} was not created"
    assert eval_log.exists(), f"{eval_log} was not created"

    log_content = eval_log.read_text()

    # The default is a v2 model: metrics differ from the legacy
    # ones and the ratio head is reported as 'a0_ratios'. Assert each RMSE key is
    # present and parses to a finite, positive float (version-robust).
    for target in ('dipole_vec', 'forces', 'a0_ratios'):
        rmse = _parse_metric_table_rmse(log_content, target)
        assert math.isfinite(rmse) and rmse > 0.0, (
            f"{target} RMSE = {rmse} is not a finite positive float.\n"
            f"Log content:\n{log_content}"
        )


def test_cli_eval_without_labels(eval_dir):
    """Prediction-only eval: a structure with no labels (acetone.xyz has positions
    only) must be predicted and saved without metrics, and the NaN labels the
    dataloader fills in must not be written next to the predictions."""
    import numpy as np

    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'acetone.xyz'
    eval_xyz = eval_dir / 'acetone_eval.xyz'

    result = subprocess.run(
        ['so3lr', 'eval', '--datafile', str(datafile), '--save-to', str(eval_xyz)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=eval_dir,
        env=so3lr_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"so3lr eval failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    atoms = read(eval_xyz)
    assert np.isfinite(atoms.info['energy_so3lr'])
    assert atoms.arrays['forces_so3lr'].shape == (len(atoms), 3)
    assert np.isfinite(atoms.arrays['forces_so3lr']).all()
    # Checked in the file: ASE's reader moves an 'energy' key off atoms.info.
    assert 'energy=nan' not in eval_xyz.read_text().lower(), (
        "a NaN energy label was written next to the predictions"
    )


def test_cli_eval_with_intermediates_and_embeddings(eval_dir):
    """Run `so3lr eval` with observables and the atom embeddings `x` as targets: the
    observables land in the extxyz arrays, the embeddings only in <save-to stem>_x.npz."""
    import numpy as np
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'tests' / 'test_data' / 'ac.xyz'
    eval_xyz = eval_dir / 'ac_eval_intermediates.xyz'
    eval_log = eval_dir / 'ac_eval_intermediates.log'
    eval_npz = eval_dir / 'ac_eval_intermediates_x.npz'

    targets = 'forces,nn_energy,partial_charges,dispersion_energy,x'

    result = subprocess.run(
        [
            'so3lr', 'eval',
            '--datafile', str(datafile),
            '--lr-cutoff', '100.0',
            '--save-to', str(eval_xyz),
            '--log-file', str(eval_log),
            '--targets', targets,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=eval_dir,
        env=so3lr_subprocess_env(),
    )

    assert result.returncode == 0, (
        f"so3lr eval with intermediates and embeddings failed (return code {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert eval_xyz.exists(), f"{eval_xyz} was not created"
    assert eval_log.exists(), f"{eval_log} was not created"
    assert eval_npz.exists(), f"{eval_npz} was not created"

    n_structures = len(read(str(datafile), index=':'))
    atoms_list = read(str(eval_xyz), index=':')
    assert len(atoms_list) == n_structures

    expected_arrays = ['nn_energy_so3lr', 'partial_charges_so3lr', 'dispersion_energy_so3lr']
    for key in expected_arrays:
        assert key in atoms_list[0].arrays, (
            f"Expected intermediate quantity '{key}' not found in atoms.arrays. "
            f"Available keys: {list(atoms_list[0].arrays.keys())}"
        )
    for atoms in atoms_list:
        assert 'x_so3lr' not in atoms.arrays, (
            "'x_so3lr' should not appear in extxyz arrays (embeddings go to NPZ only)"
        )

    data = np.load(eval_npz)
    assert 'embeddings' in data, f"'embeddings' key missing from NPZ. Keys: {list(data.keys())}"
    assert 'n_atoms' in data, f"'n_atoms' key missing from NPZ. Keys: {list(data.keys())}"

    embeddings = data['embeddings']
    n_atoms = data['n_atoms']

    assert embeddings.ndim == 2, f"Expected 2D embeddings array, got shape {embeddings.shape}"
    assert embeddings.shape[1] > 0, "Feature dimension must be non-zero"
    assert n_atoms.shape == (n_structures,), (
        f"Expected n_atoms shape ({n_structures},), got {n_atoms.shape}"
    )
    assert n_atoms.sum() == embeddings.shape[0], (
        f"n_atoms.sum()={n_atoms.sum()} != embeddings.shape[0]={embeddings.shape[0]}"
    )

    log_content = eval_log.read_text()
    assert 'Observable targets' in log_content, (
        f"Expected 'Observable targets' message not found in log.\n"
        f"Log content:\n{log_content}"
    )
    assert 'Embeddings saved to' in log_content, (
        f"Expected 'Embeddings saved to' in log.\nLog content:\n{log_content}"
    )

    forces_rmse = _parse_metric_table_rmse(log_content, 'forces')
    assert math.isfinite(forces_rmse) and forces_rmse > 0.0, (
        f"forces RMSE = {forces_rmse} is not a finite positive float.\n"
        f"Log content:\n{log_content}"
    )


def test_cli_eval_reports_binding_energies_without_saving(eval_dir):
    """A prepare_dimer_xyz file reports binding energies even without --save-to:
    the predictions are kept in memory for them, and nothing is written."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    datafile = project_root / 'so3lr' / 'cli' / 'examples' / 'sapt10k_20_dimers.xyz'
    result = subprocess.run(
        ['so3lr', 'eval', '--datafile', str(datafile), '--lr-cutoff', '100', '--targets', 'energy'],
        capture_output=True, text=True, timeout=600, cwd=eval_dir, env=so3lr_subprocess_env(),
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'Binding energies (dimer - separated) for 20 dimers' in output
    mae = float(re.search(r'MAE ([0-9.]+) kcal/mol', output).group(1))
    assert 0 < mae < 3, output
    assert list(eval_dir.iterdir()) == []
