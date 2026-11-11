"""General tests for the SO3LR command-line interface."""
import re
import subprocess
import sys
import pathlib
from unittest import mock

import pytest
from click.testing import CliRunner

from so3lr.cli.so3lr_cli import cli, main, known_yaml_keys, BASIC_HELP_STRING, YAML_DEFAULTS
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


# ---------------------------------------------------------------------------
# md_settings.yaml
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ensemble", ["nve", "nvt", "npt"])
def test_help_names_each_ensemble_correctly(ensemble):
    """Each `so3lr <ensemble>` line of the help names its own ensemble and no other
    (`so3lr nve` was once described as NVT)."""
    line = next((ln for ln in BASIC_HELP_STRING.splitlines()
                 if ln.strip().startswith(f"so3lr {ensemble}")), None)
    assert line is not None, f"'so3lr {ensemble}' line not found in BASIC_HELP_STRING"
    for other in ("NVE", "NVT", "NPT"):
        assert (other in line) == (other == ensemble.upper()), line


def test_md_settings_template_matches_the_yaml_defaults():
    """md_settings.yaml is the documented template users copy: every key in it must be
    one the settings validator knows (else every copy warns of a typo), it must list
    every key that has a YAML default, and the values it shows must be those defaults.
    md_dt is in fs in the file and in ps in YAML_DEFAULTS; convert as `so3lr --settings` does."""
    template = load_settings_yaml(MD_SETTINGS_PATH)
    assert sorted(set(template) - known_yaml_keys()) == []
    assert sorted(set(YAML_DEFAULTS) - set(template)) == []
    template['md_dt'] = template['md_dt'] / 1000
    assert {k: (template[k], v) for k, v in YAML_DEFAULTS.items() if template[k] != v} == {}


@pytest.fixture
def stubbed_run(monkeypatch):
    """The settings perform_md receives from an in-process CLI run, with perform_md stubbed
    (no model is built) and no JAX, GPU or log-file setup."""
    from so3lr.cli import so3lr_cli, so3lr_md

    captured = {}
    monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: None)
    monkeypatch.setattr(so3lr_md, 'perform_md', lambda settings, *args, **kwargs: captured.update(settings))
    monkeypatch.setattr(so3lr_cli, 'reserve_gpu', lambda *args, **kwargs: None)
    # setup_logger would swap the console handlers and could open a log file.
    monkeypatch.setattr(so3lr_md, 'setup_logger', lambda *args, **kwargs: None)
    return captured


def test_unknown_settings_key_warns_and_the_run_goes_on(tmp_path, stubbed_run):
    """A misspelt settings key (md_t for md_T) is a warning naming the likely key,
    not an error: one typo must not abort a queued multi-hour job (AGENTS.md)."""
    from so3lr.cli import so3lr_cli

    captured, warnings = stubbed_run, []

    yaml_path = tmp_path / 'typo.yaml'
    yaml_path.write_text(
        f"input_file: {PROJECT_ROOT / 'tests' / 'test_data' / 'acetone.xyz'}\n"
        f"output_file: {tmp_path / 'out.hdf5'}\n"
        # standard engine: no flash or device checks before perform_md
        "engine: standard\nrelax_before_run: false\n"
        "md_t: 350.0\n"
    )
    # mock.patch.object, not monkeypatch: on exit it deletes the instance attribute
    # instead of leaving a bound method shadowing Logger.warning on the shared logger.
    with mock.patch.object(so3lr_cli.logger, 'warning',
                           side_effect=lambda msg, *args, **kwargs: warnings.append(msg)):
        result = CliRunner().invoke(cli, ['--settings', str(yaml_path)])

    assert result.exit_code == 0, result.output
    assert captured, "perform_md was never called: the typo aborted the run"
    unknown = [w for w in warnings if w.startswith('Unknown setting')]
    assert len(unknown) == 1, warnings
    assert re.search(r"'md_t'.*Did you mean: .*\bmd_T\b", unknown[0]), unknown[0]
    # ignored, not applied: md_T keeps its default
    assert captured['md_T'] == YAML_DEFAULTS['md_T']


@pytest.mark.parametrize("args, precision", [
    ([], 'mixed'), (['--precision', 'float32'], 'float32'), (['--precision', 'float64'], 'float64'),
    (['--no-flash-mixed'], 'float32'), (['--precision', 'float32', '--flash-mixed'], 'mixed')])
def test_the_run_gets_the_precision(tmp_path, stubbed_run, args, precision):
    """--precision is the run's (default mixed); the deprecated --flash-mixed/--no-flash-mixed
    still makes a float32 model's run mixed/float32."""
    result = CliRunner().invoke(cli, [
        'nvt', '--input', str(PROJECT_ROOT / 'tests' / 'test_data' / 'acetone.xyz'),
        '--output', str(tmp_path / 'out.hdf5'), '--no-relax', '--engine', 'standard', *args])
    assert result.exit_code == 0, result.output
    assert stubbed_run['precision'] == precision and stubbed_run['flash_mixed'] is None


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
