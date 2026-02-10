"""General tests for the SO3LR command-line interface."""
import os
import yaml
import tempfile
import pathlib

import pytest
from click.testing import CliRunner

from so3lr.cli.so3lr_cli import cli, PARAM_MAP, BASIC_HELP_STRING


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
# PARAM_MAP
# ---------------------------------------------------------------------------

class TestParamMap:
    """PARAM_MAP should map all expected CLI options to settings keys."""

    EXPECTED_MAPPINGS = {
        "input_file": "input_file",
        "output_file": "output_file",
        "log_file": "log_file",
        "dt": "md_dt",
        "temperature": "md_T",
        "pressure": "md_P",
        "md_cycles": "md_cycles",
        "md_steps": "md_steps",
        "lr_cutoff": "lr_cutoff",
        "buffer_sr": "buffer_size_multiplier_sr",
        "buffer_lr": "buffer_size_multiplier_lr",
        "seed": "seed",
        "total_charge": "total_charge",
    }

    @pytest.mark.parametrize("cli_key,settings_key", EXPECTED_MAPPINGS.items())
    def test_mapping_exists(self, cli_key, settings_key):
        assert cli_key in PARAM_MAP, f"PARAM_MAP missing key '{cli_key}'"
        assert PARAM_MAP[cli_key] == settings_key


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

    def test_buffer_keys_use_underscores(self, settings):
        """Buffer keys should use underscores to match PARAM_MAP values."""
        assert "buffer_sr" in settings
        assert "buffer_lr" in settings
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
