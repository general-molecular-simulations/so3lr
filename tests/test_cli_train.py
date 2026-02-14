import json
import re
import subprocess
import pathlib

import yaml
import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATAFILE = PROJECT_ROOT / 'tests' / 'test_data' / 'ac.xyz'
DEFAULT_CONFIG = PROJECT_ROOT / 'so3lr' / 'config' / 'config.yaml'


@pytest.fixture
def workdir(tmp_path):
    """Provide a temporary working directory for training."""
    wd = tmp_path / 'train_test'
    yield wd
    # Cleanup happens automatically via tmp_path


def _base_config(workdir):
    """Load default config and apply common test overrides."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg['workdir'] = str(workdir)
    cfg['data']['filepath'] = str(DATAFILE)
    cfg['training']['num_epochs'] = 2
    cfg['training']['num_train'] = 90
    cfg['training']['num_valid'] = 10
    cfg['training']['use_wandb'] = False
    cfg['training']['eval_every_num_steps'] = 5
    cfg['training']['batch_max_num_graphs'] = 4
    return cfg


def _write_config(cfg, tmp_path, name='test_train.yaml'):
    config_path = tmp_path / name
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    return config_path


def _run_train(config_path):
    """Run so3lr train and return the result."""
    return subprocess.run(
        ['so3lr', 'train', '--config', str(config_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _assert_training_succeeded(result, workdir):
    """Assert that training completed and produced expected outputs."""
    assert result.returncode == 0, (
        f"so3lr train failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    assert workdir.exists(), "Workdir was not created"
    assert (workdir / 'hyperparameters.json').exists(), "hyperparameters.json not created"
    assert (workdir / 'hyperparameters.yaml').exists(), "hyperparameters.yaml not created"
    assert (workdir / 'checkpoints').exists(), "checkpoints directory not created"

    combined_output = result.stdout + result.stderr
    val_lines = [line for line in combined_output.splitlines() if line.startswith('val_')]
    assert len(val_lines) > 0, (
        f"No validation lines found in output.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    last_val_line = val_lines[-1]
    match = re.search(r'eval_loss=([\d.]+)', last_val_line)
    assert match is not None, f"Could not parse eval_loss from: {last_val_line}"

    final_loss = float(match.group(1))
    assert 0 < final_loss < 10.0, (
        f"Final eval_loss={final_loss} is outside reasonable range (0, 10.0).\n"
        f"Last validation line: {last_val_line}"
    )


def test_cli_train_forces_only(workdir, tmp_path):
    """Train with forces-only loss (no energy shifts)."""
    cfg = _base_config(workdir)
    cfg['training']['loss_weights'] = {'forces': 1.0}
    config_path = _write_config(cfg, tmp_path)

    result = _run_train(config_path)
    _assert_training_succeeded(result, workdir)


def test_cli_train_forces_and_energy_with_lse_shifts(workdir, tmp_path):
    """Train with forces + energy loss and shift_mode=lse, verify carbon shift."""
    cfg = _base_config(workdir)
    cfg['data']['shift_mode'] = 'lse'
    cfg['training']['loss_weights'] = {'forces': 1.0, 'energy': 0.01}
    config_path = _write_config(cfg, tmp_path)

    result = _run_train(config_path)
    _assert_training_succeeded(result, workdir)

    # Verify per-element energy shifts were computed and saved
    with open(workdir / 'hyperparameters.json') as f:
        hyp = json.load(f)

    energy_shifts = hyp['data']['energy_shifts']
    carbon_shift = float(energy_shifts['6'])
    assert carbon_shift == pytest.approx(-3.7451, abs=1e-3), (
        f"Carbon (Z=6) energy shift {carbon_shift} does not match expected value"
    )
