"""The `so3lr start` task menu and the release notice."""
import json
from pathlib import Path

import click
from click.testing import CliRunner
import pytest
import yaml

from so3lr.cli import setup, setup_tasks, updates


@click.command()
def wizard():
    result = setup.launch_setup()
    click.echo(f'RETURN:{result!r}')


@pytest.fixture
def structure_file(tmp_path):
    path = tmp_path / 'water.extxyz'
    path.write_text('3\nProperties=species:S:1:pos:R:3 charge=0 pbc="F F F"\n'
                    'O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n')
    return path


@pytest.fixture(autouse=True)
def interactive(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, '_interactive', lambda: True)


def task(name):
    return str([kind for _, kind in setup_tasks.TASKS].index(name) + 1)


def test_menu_offers_md_first_then_the_other_tasks():
    assert [kind for _, kind in setup_tasks.TASKS] == ['md', 'opt', 'eval', 'show', 'other']
    assert setup_tasks.TASK_CHOICES[0] == 'Run molecular dynamics'


def test_optimization_bundle_replays_through_the_opt_command(tmp_path, structure_file, monkeypatch):
    from so3lr.cli.optimization_settings import optimization_argv
    from so3lr.cli.so3lr_cli import fire_optimization
    destination = tmp_path / 'opt'
    text = '\n'.join([task('opt'), str(len(setup.available_presets()) + 2), str(structure_file),
                      '1', '', '', '0', '0', str(destination), '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'RETURN:None' in result.output
    settings = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert settings['operation'] == 'optimization'
    assert (settings['force_convergence'], settings['min_steps'], settings['min_cycles']) == (0.05, 10, 100)
    assert settings['lr_cutoff'] == 100.0  # gas phase
    # Every exported key is accepted by the real opt options (run.sh runs from the bundle).
    monkeypatch.chdir(destination)
    optimization_argv({k: v for k, v in settings.items() if k != 'operation'}, fire_optimization)
    assert set(p.name for p in destination.iterdir()) == {'structure.extxyz', 'settings.yaml', 'run.sh'}


def test_optimization_start_now_returns_the_settings_path(tmp_path, structure_file):
    destination = tmp_path / 'opt-now'
    text = '\n'.join([task('opt'), str(len(setup.available_presets()) + 2), str(structure_file),
                      '1', '0.01', '200', '0', '0', str(destination), '1']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert f"RETURN:{(destination / 'settings.yaml')!r}" in result.output
    settings = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert (settings['force_convergence'], settings['min_cycles']) == (0.01, 20)


def test_eval_saves_a_launcher_and_can_start_now(tmp_path, structure_file):
    destination = tmp_path / 'eval'
    text = '\n'.join([task('eval'), str(structure_file), '1', '', str(destination), '1']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    launcher = (destination / 'run.sh').read_text()
    assert 'exec so3lr eval --datafile' in launcher and str(structure_file) in launcher
    assert '--save-to predictions.extxyz' in launcher and '--lr-cutoff 12' in launcher
    assert "RETURN:Action(kind='eval'" in result.output
    assert list(destination.iterdir()) == [destination / 'run.sh']


def test_watch_and_other_tasks(tmp_path):
    result = CliRunner().invoke(wizard, input='\n'.join([task('show'), '']) + '\n')
    assert result.exit_code == 0, result.output
    assert "RETURN:Action(kind='show', argv=('.',)" in result.output
    result = CliRunner().invoke(wizard, input=task('other') + '\n')
    assert result.exit_code == 0, result.output
    for command in ('so3lr finetune', 'so3lr train', 'so3lr tune-ewald', setup_tasks.DOCS_URL):
        assert command in ' '.join(result.output.split())
    assert 'RETURN:None' in result.output


def test_cli_runs_the_chosen_action_from_its_folder(tmp_path, monkeypatch):
    from so3lr.cli import monitor, so3lr_cli
    seen = []
    monkeypatch.setattr(monitor, 'show', lambda path, **kwargs: seen.append(('show', path)))
    monkeypatch.setattr(setup, 'launch_setup', lambda: setup_tasks.Action('show', ('runs',)))
    assert CliRunner().invoke(so3lr_cli.cli, ['start']).exit_code == 0
    assert seen == [('show', Path('runs'))]
    folder = tmp_path / 'bundle'
    folder.mkdir()
    monkeypatch.setattr(setup, 'launch_setup', lambda: setup_tasks.Action('eval', ('eval', '--help'), folder))
    result = CliRunner().invoke(so3lr_cli.cli, ['start'])
    assert result.exit_code == 0 and '--datafile' in result.output
    assert Path.cwd() == tmp_path  # restored


# -- release notice --------------------------------------------------------------------

@pytest.mark.parametrize('latest, current, expected', [
    ('v0.3.0', '0.2.0', True), ('0.2.1', '0.2.0', True), ('v0.10.0', '0.9.9', True),
    ('v0.2.0', '0.2.0', False), ('v0.1.0', '0.2.0', False), ('nightly', '0.2.0', False), ('', '0.2.0', False),
])
def test_version_comparison(latest, current, expected):
    assert updates.newer(latest, current) is expected


class _TTY:
    def isatty(self):
        return True


def test_check_runs_only_in_interactive_non_batch_sessions():
    tty = _TTY()
    assert updates.allowed({}, tty, tty)
    assert not updates.allowed({'SO3LR_NO_UPDATE_CHECK': '1'}, tty, tty)
    assert not updates.allowed({'SLURM_JOB_ID': '7'}, tty, tty)
    assert not updates.allowed({'CI': 'true'}, tty, tty)
    assert not updates.allowed({}, tty, tty, quiet=True)
    assert not updates.allowed({}, object(), tty)


def test_refresh_is_daily_cached_and_never_raises(tmp_path):
    cache = tmp_path / 'latest-release.json'
    calls = []

    def fetch():
        calls.append(1)
        return 'v9.9.9'
    updates.refresh(cache, now=1000.0, fetch=fetch)
    updates.refresh(cache, now=1000.0 + 3600, fetch=fetch)  # within a day: no request
    assert len(calls) == 1
    assert updates.notice('0.2.0', cache).startswith('SO3LR 9.9.9 is available (you have 0.2.0).')

    def offline():
        raise OSError('no network')
    updates.refresh(cache, now=1000.0 + 2 * updates.CHECK_EVERY, fetch=offline)
    assert json.loads(cache.read_text())['latest'] == 'v9.9.9'  # keeps the last answer
    assert updates.notice('9.9.9', cache) == ''


def test_cache_respects_xdg_cache_home(tmp_path):
    assert updates.cache_path({'XDG_CACHE_HOME': str(tmp_path)}) == tmp_path / 'so3lr' / 'latest-release.json'


# -- going back ---------------------------------------------------------------------

def scripted(monkeypatch, answers):
    """Replace menu choices with a script: an index, or 'back' for Esc/Backspace."""
    from so3lr.cli.prompts import Back
    seen = []

    def choose(console, title, choices, default=1):
        seen.append(title)
        answer = answers.pop(0)
        if answer == 'back':
            raise Back()
        return answer
    monkeypatch.setattr(setup, '_choose', choose)
    monkeypatch.setattr(setup_tasks, '_choose', choose)
    return seen


def test_esc_steps_back_through_the_md_wizard_and_review(tmp_path, structure_file, monkeypatch):
    path_choice = len(setup.available_presets()) + 1
    seen = scripted(monkeypatch, [
        0, 'back',                      # MD, then back from the structure menu to the task menu
        0, path_choice, 'back',         # MD again; path; back from Dynamics to the structure menu
        path_choice, 0, 'back',         # path again; NVT; back from the model to Dynamics
        0, 0,                           # NVT, model M
        'back', 'back',                 # back from the review opens Edit settings; back again returns
        1,                              # Save for later
    ])
    destination = tmp_path / 'md'
    text = '\n'.join([str(structure_file), str(structure_file), '300', '5', '0.05', '0', '0',
                      str(destination)]) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert seen == ['What would you like to do?', 'Choose a structure', 'What would you like to do?',
                    'Choose a structure', 'Dynamics', 'Choose a structure', 'Dynamics',
                    'Model (canonical release)', 'Dynamics', 'Model (canonical release)',
                    'Next action · Enter starts MD now', 'Edit settings', 'Next action · Enter starts MD now']
    assert (destination / 'settings.yaml').is_file()


def test_esc_at_the_task_menu_closes_setup(monkeypatch):
    scripted(monkeypatch, ['back'])
    result = CliRunner().invoke(wizard)
    assert result.exit_code == 0 and 'Setup cancelled. No files written.' in result.output


def test_esc_steps_back_in_optimization_and_evaluation(tmp_path, structure_file, monkeypatch):
    path_choice = len(setup.available_presets()) + 1
    seen = scripted(monkeypatch, [1, path_choice, 'back', path_choice, 0, 'back', 1])
    destination = tmp_path / 'opt'
    text = '\n'.join([str(structure_file), str(structure_file), '', '', '0', '0', str(destination),
                      '', '', '0', '0', str(tmp_path / 'opt2')]) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    # Back from the model returns to the structure; back from the review re-asks the details.
    assert seen == ['What would you like to do?', 'Choose a structure', 'Model (canonical release)',
                    'Choose a structure', 'Model (canonical release)', 'Next action · Enter starts now',
                    'Next action · Enter starts now']
    assert (tmp_path / 'opt2' / 'settings.yaml').is_file() and not destination.exists()

    seen = scripted(monkeypatch, [2, 'back', 0, 1])
    text = '\n'.join([str(structure_file), '', '', str(tmp_path / 'ev')]) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    # Back from the model re-asks the dataset, with the previous answer as default.
    assert seen == ['What would you like to do?', 'Model (canonical release)', 'Model (canonical release)',
                    'Next action · Enter starts now']
    assert (tmp_path / 'ev' / 'run.sh').is_file()
