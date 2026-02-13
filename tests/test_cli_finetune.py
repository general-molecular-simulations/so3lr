import os
import re
import subprocess
import pathlib

import yaml
import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATAFILE = PROJECT_ROOT / 'tests' / 'test_data' / 'ac.xyz'
DEFAULT_CONFIG = PROJECT_ROOT / 'so3lr' / 'config' / 'finetune.yaml'


@pytest.fixture
def workdir(tmp_path):
    """Provide a temporary working directory for fine-tuning."""
    wd = tmp_path / 'finetune_test'
    yield wd
    # Cleanup happens automatically via tmp_path


@pytest.fixture
def test_config(tmp_path):
    """Create a test config with few epochs and wandb disabled."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg['training']['num_epochs'] = 2
    cfg['training']['use_wandb'] = False
    cfg['training']['eval_every_num_steps'] = 5
    cfg['training']['batch_max_num_graphs'] = 4

    config_path = tmp_path / 'test_finetune.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)

    return config_path


def test_cli_finetune(workdir, test_config):
    """Run `so3lr finetune` on ac.xyz with 90 train / 10 valid and verify it completes."""
    result = subprocess.run(
        [
            'so3lr', 'finetune',
            '--datafile', str(DATAFILE),
            '--workdir', str(workdir),
            '--num-train', '90',
            '--num-valid', '10',
            '--config', str(test_config),
            '--strategy', 'full',
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"so3lr finetune failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Check that the workdir was created with expected outputs
    assert workdir.exists(), "Workdir was not created"
    assert (workdir / 'hyperparameters.json').exists(), "hyperparameters.json not created"
    assert (workdir / 'fine_tuning.json').exists(), "fine_tuning.json not created"
    assert (workdir / 'checkpoints').exists(), "checkpoints directory not created"

    # Parse the final validation loss from output.
    # Lines look like: val_1_60:: eval_forces_mae=0.0339, ..., eval_loss=0.0024, ...
    # Validation metrics are printed via print() (stdout).
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
    assert 0 < final_loss < 0.01, (
        f"Final eval_loss={final_loss} is outside reasonable range (0, 0.01).\n"
        f"Last validation line: {last_val_line}"
    )
