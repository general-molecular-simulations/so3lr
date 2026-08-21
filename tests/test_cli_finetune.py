import os
import re
import subprocess
import pathlib

import yaml
import pytest

from subprocess_env import so3lr_subprocess_env, _parse_final_table_loss


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATAFILE = PROJECT_ROOT / 'tests' / 'test_data' / 'ac.xyz'
DATAFILE_WATER = PROJECT_ROOT / 'tests' / 'test_data' / 'water_pbc_dummy.xyz'
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


@pytest.fixture
def test_config_water(tmp_path):
    """Create a test config for water with long-range cutoff."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg['training']['num_epochs'] = 2
    cfg['training']['use_wandb'] = False
    cfg['training']['eval_every_num_steps'] = 5
    cfg['training']['batch_max_num_graphs'] = 4
    cfg['data']['neighbors_lr_cutoff'] = 12.0

    config_path = tmp_path / 'test_finetune_water.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)

    return config_path


def _parse_param_counts(result):
    """Pull (total, trainable) out of the parameter summary the trainer prints.

    Exit code and final loss alone cannot tell a real fine-tune from a run in
    which every parameter was frozen: a no-op run completes normally and reports
    the pretrained loss. These counts can.
    """
    text = result.stdout + result.stderr
    total = re.search(r'Total number of parameters:\s*([\d,]+)', text)
    trainable = re.search(r'Trainable parameters:\s*([\d,]+)', text)
    if total is None or trainable is None:
        return None, None
    return int(total.group(1).replace(',', '')), int(trainable.group(1).replace(',', ''))


def _assert_finetuning_succeeded(result, workdir, max_loss=0.1):
    """Assert that fine-tuning completed and produced expected outputs."""
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

    # Parse the final eval loss from the printed metrics table. The trainer logs
    # validation checkpoints as rows of a table (print_metrics_header/row):
    #   "        epoch |          step |    forces_mae | ... |          loss | ..."
    final_loss = _parse_final_table_loss(result)
    assert final_loss is not None, (
        f"No metrics table row found in output.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert 0 < final_loss < max_loss, (
        f"Final loss={final_loss} is outside reasonable range (0, {max_loss}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize('strategy', ['full', 'last_layer'])
def test_cli_finetune(workdir, test_config, strategy):
    """Run `so3lr finetune` on ac.xyz with 20 train / 5 valid and verify it completes."""
    result = subprocess.run(
        [
            'so3lr', 'finetune',
            '--datafile', str(DATAFILE),
            '--workdir', str(workdir),
            '--num-train', '20',
            '--num-valid', '5',
            '--config', str(test_config),
            '--strategy', strategy,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=so3lr_subprocess_env(),
    )

    _assert_finetuning_succeeded(result, workdir, max_loss=0.1)

    # Pin what the strategy actually selected. Asserting only on exit code and
    # loss would pass even if `last_layer` resolved to a `layers_N` that does
    # not exist: every parameter would be frozen, two epochs of zero updates
    # would complete, and the pretrained loss would sit well under the bound.
    total, trainable = _parse_param_counts(result)
    assert total is not None, (
        f"No parameter summary in output.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    if strategy == 'full':
        assert trainable == total
    else:
        assert 0 < trainable < total, (
            f"strategy={strategy} trained {trainable}/{total} parameters"
        )

@pytest.mark.slow
def test_cli_finetune_water_pbc(workdir, test_config_water):
    """Run `so3lr finetune` on periodic water box with 13 train / 2 valid and verify it completes."""
    result = subprocess.run(
        [
            'so3lr', 'finetune',
            '--datafile', str(DATAFILE_WATER),
            '--workdir', str(workdir),
            '--num-train', '13',
            '--num-valid', '2',
            '--config', str(test_config_water),
            '--strategy', 'full',
        ],
        capture_output=True,
        text=True,
        timeout=240,
        env=so3lr_subprocess_env(),
    )

    _assert_finetuning_succeeded(result, workdir, max_loss=0.2)
