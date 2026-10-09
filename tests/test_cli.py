"""General tests for the SO3LR command-line interface."""
import os
import yaml
import tempfile
import pathlib

import pytest
from click.testing import CliRunner

from so3lr.cli.so3lr_cli import cli, BASIC_HELP_STRING


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MD_SETTINGS_PATH = PROJECT_ROOT / "so3lr" / "cli" / "md_settings.yaml"

SUBCOMMANDS = ["opt", "nvt", "npt", "nve", "eval", "finetune", "train", "tune-ewald"]
# Subcommands listed in the brief help text (tune-ewald is accessible but not shown)
HELP_LISTED_COMMANDS = ["opt", "nvt", "npt", "nve", "eval", "finetune", "train"]


# ---------------------------------------------------------------------------
# Top-level CLI
# ---------------------------------------------------------------------------

class TestTopLevelCLI:
    """Tests for the top-level `so3lr` command."""

    def test_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_main_subcommands(self):
        """Main subcommands should appear in the basic help string."""
        for cmd in HELP_LISTED_COMMANDS:
            assert f"so3lr {cmd}" in BASIC_HELP_STRING, (
                f"Subcommand '{cmd}' missing from BASIC_HELP_STRING"
            )

    def test_no_args_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "so3lr" in result.output.lower()


# ---------------------------------------------------------------------------
# Subcommand help
# ---------------------------------------------------------------------------

class TestSubcommandHelp:
    """Each subcommand should render --help without errors."""

    @pytest.mark.parametrize("cmd", SUBCOMMANDS)
    def test_subcommand_help(self, cmd):
        runner = CliRunner()
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, (
            f"`so3lr {cmd} --help` failed (exit {result.exit_code}):\n{result.output}"
        )

    def test_opt_help_mentions_input(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["opt", "--help"])
        assert "--input" in result.output

    def test_nvt_help_mentions_temperature(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["nvt", "--help"])
        assert "--temperature" in result.output

    def test_npt_help_mentions_pressure(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["npt", "--help"])
        assert "--pressure" in result.output

    def test_nve_help_mentions_temperature(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["nve", "--help"])
        assert "--temperature" in result.output

    def test_eval_help_mentions_datafile(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--help"])
        assert "--datafile" in result.output

    def test_finetune_help_mentions_workdir(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["finetune", "--help"])
        assert "--workdir" in result.output

    def test_finetune_workdir_help_text(self):
        """--workdir description should be about the working directory, not predictions."""
        runner = CliRunner()
        result = runner.invoke(cli, ["finetune", "--help"])
        assert "predictions" not in result.output.lower()


# ---------------------------------------------------------------------------
# Help text consistency
# ---------------------------------------------------------------------------

class TestHelpTextConsistency:
    """Verify help text is accurate."""

    def test_nve_described_as_nve(self):
        """The NVE line in BASIC_HELP_STRING should say NVE, not NVT."""
        for line in BASIC_HELP_STRING.splitlines():
            if line.strip().startswith("so3lr nve"):
                assert "NVE" in line
                assert "NVT" not in line
                break
        else:
            pytest.fail("'so3lr nve' line not found in BASIC_HELP_STRING")

    def test_nvt_described_as_nvt(self):
        for line in BASIC_HELP_STRING.splitlines():
            if line.strip().startswith("so3lr nvt"):
                assert "NVT" in line
                break
        else:
            pytest.fail("'so3lr nvt' line not found in BASIC_HELP_STRING")

    def test_npt_described_as_npt(self):
        for line in BASIC_HELP_STRING.splitlines():
            if line.strip().startswith("so3lr npt"):
                assert "NPT" in line
                break
        else:
            pytest.fail("'so3lr npt' line not found in BASIC_HELP_STRING")


# ---------------------------------------------------------------------------
# md_settings.yaml
# ---------------------------------------------------------------------------

class TestMdSettingsYaml:
    """Validate the default md_settings.yaml template."""

    @pytest.fixture(scope="class")
    def settings(self):
        with open(MD_SETTINGS_PATH) as f:
            return yaml.safe_load(f)

    def test_file_loads(self, settings):
        assert isinstance(settings, dict)

    def test_required_keys_present(self, settings):
        required = [
            "md_dt", "md_T", "md_cycles", "md_steps",
            "seed", "precision", "lr_cutoff",
        ]
        for key in required:
            assert key in settings, f"md_settings.yaml missing key '{key}'"

    def test_buffer_keys_are_internal_names(self, settings):
        """Buffer keys in the YAML should use the internal (long-form) names."""
        assert "buffer_size_multiplier_sr" in settings
        assert "buffer_size_multiplier_lr" in settings
        assert "buffer-sr" not in settings
        assert "buffer-lr" not in settings

    def test_md_dt_is_in_femtoseconds(self, settings):
        """md_dt in the YAML template should be in femtoseconds (0.5 fs)."""
        assert settings["md_dt"] == 0.5

    def test_temperature_default(self, settings):
        assert settings["md_T"] == 300.0

    def test_precision_valid(self, settings):
        assert settings["precision"] in ("float32", "float64")


# ---------------------------------------------------------------------------
# Timestep unit conversion
# ---------------------------------------------------------------------------

class TestTimestepConversion:
    """md_dt from a settings YAML (in fs) must be converted to ps internally."""

    def test_yaml_md_dt_converted_to_ps(self):
        """Simulate the conversion that cli() applies after loading YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"input_file": "dummy.xyz", "md_dt": 1.0}, f)
            tmp = f.name

        try:
            with open(tmp) as f:
                settings_dict = yaml.safe_load(f)

            assert settings_dict["md_dt"] == 1.0  # fs, as written

            # Apply the same conversion the CLI does
            if "md_dt" in settings_dict:
                settings_dict["md_dt"] = settings_dict["md_dt"] / 1000

            assert settings_dict["md_dt"] == pytest.approx(0.001)  # ps
        finally:
            os.unlink(tmp)


def test_cli_import_does_not_load_jax():
    """`so3lr --help` must not pay for JAX/Flax start-up (see AGENTS.md)."""
    import subprocess
    import sys
    code = ("import sys, so3lr.cli.so3lr_cli; "
            "heavy = sorted(m for m in ('jax', 'flax', 'jax_md', 'glp') if m in sys.modules); "
            "print(heavy); sys.exit(1 if heavy else 0)")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f"importing the CLI loaded {result.stdout.strip()}"


def test_usage_error_is_one_line_not_a_traceback(tmp_path):
    """The console entry point reports click usage errors cleanly and exits non-zero."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, '-m', 'so3lr.cli.so3lr_cli', 'nvt', '--input', str(tmp_path / 'missing.xyz')],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert 'does not exist' in result.stderr
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('given,expected', [
    ({}, 'nvt'), ({'md_P': 1.0}, 'npt'), ({'ensemble': 'NVT'}, 'nvt'),
    ({'ensemble': 'nvt_langevin'}, 'nvt-langevin'), ({'ensemble': 'nvt-langevin'}, 'nvt-langevin'),
])
def test_canonicalize_ensemble_names(given, expected):
    from so3lr.cli.so3lr_cli import canonicalize_ensemble
    assert canonicalize_ensemble(dict(given))['ensemble'] == expected


def test_canonicalize_ensemble_rejects_unknown_names_before_the_run():
    """A typo used to fall into perform_md's NPT branch and crash after relaxation."""
    from so3lr.cli.so3lr_cli import canonicalize_ensemble
    with pytest.raises(ValueError, match="did you mean 'nvt-langevin'"):
        canonicalize_ensemble({'ensemble': 'nvt-langevn'})


def test_canonicalize_ensemble_pressure():
    """`ensemble: npt` alone gets the CLI default pressure (it used to reach jax-md as None);
    another ensemble drops a stray md_P (perform_md reads a pressure as NPT)."""
    from so3lr.cli.so3lr_cli import canonicalize_ensemble, DEFAULT_PRESSURE
    assert canonicalize_ensemble({'ensemble': 'npt'})['md_P'] == DEFAULT_PRESSURE
    assert canonicalize_ensemble({'ensemble': 'npt', 'md_P': 5.0})['md_P'] == 5.0
    assert canonicalize_ensemble({'ensemble': 'nvt', 'md_P': 5.0})['md_P'] is None
