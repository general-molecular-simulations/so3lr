"""General tests for the SO3LR command-line interface."""
import subprocess
import sys
import yaml
import pathlib

import pytest
from click.testing import CliRunner

from so3lr.cli.so3lr_cli import cli, main, BASIC_HELP_STRING
from so3lr.md_settings_validation import DEFAULT_PRESSURE, load_settings_yaml, resolve_ensemble


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



def test_cli_import_does_not_load_jax():
    """`so3lr --help` must not pay for JAX/Flax start-up (see AGENTS.md); a fresh
    interpreter, since this one has imported JAX already."""
    code = ("import sys, so3lr.cli.so3lr_cli; "
            "heavy = sorted(m for m in ('jax', 'flax', 'jax_md', 'glp') if m in sys.modules); "
            "print(heavy); sys.exit(1 if heavy else 0)")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f"importing the CLI loaded {result.stdout.strip()}"


def test_usage_error_is_one_line_not_a_traceback(tmp_path, monkeypatch, capsys):
    """main() reports a click usage error as one line and exits 2."""
    monkeypatch.setattr(sys, 'argv', ['so3lr', 'nvt', '--input', str(tmp_path / 'missing.xyz')])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    assert 'does not exist' in capsys.readouterr().err


@pytest.mark.parametrize('given,ensemble,md_P', [
    ({}, 'nvt', None),
    ({'md_P': 1.0}, 'npt', 1.0),
    ({'ensemble': 'NVT'}, 'nvt', None),
    ({'ensemble': 'nvt_langevin'}, 'nvt-langevin', None),
    ({'ensemble': 'npt'}, 'npt', DEFAULT_PRESSURE),     # jax-md used to get pressure=None
    ({'ensemble': 'npt', 'md_P': 5.0}, 'npt', 5.0),
    ({'ensemble': 'nvt', 'md_P': 5.0}, 'nvt', None),    # perform_md reads a pressure as NPT
])
def test_resolve_ensemble(given, ensemble, md_P):
    before = dict(given)
    settings = resolve_ensemble(given)
    assert (settings['ensemble'], settings['md_P']) == (ensemble, md_P)
    assert given == before  # it returns a copy


def test_resolve_ensemble_rejects_unknown_names_before_the_run():
    """A typo used to fall into perform_md's NPT branch and crash after relaxation."""
    with pytest.raises(ValueError, match="did you mean 'nvt-langevin'"):
        resolve_ensemble({'ensemble': 'nvt-langevn'})


def test_resolve_ensemble_warns_when_a_pressure_is_dropped(caplog):
    with caplog.at_level('WARNING', logger='SO3LR'):
        resolve_ensemble({'ensemble': 'nvt', 'md_P': 1.0})
    assert 'md_P=1.0 is ignored' in caplog.text


def test_settings_yaml_reads_exponent_floats(tmp_path):
    """PyYAML's YAML 1.1 rules load 1e-3 as the string '1e-3'."""
    path = tmp_path / 'settings.yaml'
    path.write_text("accuracy: 1e-3\nforce_convergence: 5E-2\nmd_steps: 10\nname: '1e-3'\n")
    assert load_settings_yaml(path) == {'accuracy': 1e-3, 'force_convergence': 5e-2,
                                        'md_steps': 10, 'name': '1e-3'}
