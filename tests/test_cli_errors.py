"""Fast CLI error-path tests (CliRunner only).

These cover error contracts that the rest of the suite leaves unverified
(every other CLI test only checks ``exit_code == 0``). The first group fails
at Click parse/validation time, so it is sub-second and never builds a model.

The eval overwrite-refusal path is marked ``@pytest.mark.slow`` because the
``eval`` body calls ``setup_logger`` and ``get_hardware_info`` (both import the
heavy stack) before reaching its refusal check, so it is not sub-second.
"""
import pytest
import yaml
from click.testing import CliRunner

from so3lr.cli.so3lr_cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_xyz(tmp_path):
    """A minimal but real (exists=True) input geometry file."""
    p = tmp_path / "mol.xyz"
    p.write_text("1\n\nH 0.0 0.0 0.0\n")
    return str(p)


def test_yaml_missing_input_file_raises(runner, tmp_path):
    """A --settings YAML without 'input_file' must fail with a clear error."""
    y = tmp_path / "run.yaml"
    y.write_text(yaml.safe_dump({"md_T": 300}))
    result = runner.invoke(cli, ["--settings", str(y)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "input_file" in str(result.exception)


def test_opt_rejects_hdf5_output(runner, tmp_xyz):
    """`so3lr opt` must reject an .hdf5 output extension."""
    result = runner.invoke(cli, ["opt", "--input", tmp_xyz, "--output", "out.hdf5"])
    assert result.exit_code != 0
    assert "HDF5 output is not supported" in result.output


def test_opt_rejects_h5_output(runner, tmp_xyz):
    """`so3lr opt` must reject an .h5 output extension too."""
    result = runner.invoke(cli, ["opt", "--input", tmp_xyz, "--output", "out.h5"])
    assert result.exit_code != 0
    assert "HDF5 output is not supported" in result.output


def test_eval_bad_model_name_suggests(runner):
    """A bad --model name fails in the Click callback with a 'did you mean'."""
    result = runner.invoke(cli, ["eval", "--model", "so3lr-x", "--datafile", "nope.xyz"])
    assert result.exit_code != 0
    assert "is not a bundled name" in result.output
    assert "Did you mean" in result.output
    assert "so3lr-m" in result.output  # a bundled name appears as a suggestion


def test_opt_bad_model_name_suggests(runner, tmp_xyz):
    """The bad-model callback also fires for `so3lr opt`."""
    result = runner.invoke(cli, ["opt", "--model", "so3lr-mm", "--input", tmp_xyz])
    assert result.exit_code != 0
    assert "Did you mean" in result.output


# NOTE: eval reaches setup_logger() and get_hardware_info() before the
# overwrite guard, both of which import jax/flax, so this path is not
# sub-second and is intentionally marked slow (deselected by default).
@pytest.mark.slow
def test_eval_refuses_overwrite(runner, tmp_xyz, tmp_path):
    """`eval --save-to <existing>` without --overwrite must refuse and exit."""
    save_to = tmp_path / "preds.extxyz"
    save_to.write_text("already here")
    result = runner.invoke(
        cli, ["eval", "--datafile", tmp_xyz, "--save-to", str(save_to)]
    )
    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output
