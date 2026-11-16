"""The `so3lr start` task menu and the release notice."""
import json
from pathlib import Path

import click
from click.testing import CliRunner
import pytest

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


def test_optimization_bundle_holds_a_valid_opt_command(tmp_path, structure_file, monkeypatch):
    import shlex
    from so3lr.cli.so3lr_cli import cli
    destination = tmp_path / 'opt'
    text = '\n'.join([task('opt'), str(len(setup.available_presets()) + 2), str(structure_file),
                      '1', '', '', '0', '0', str(destination), '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'RETURN:None' in result.output
    assert set(p.name for p in destination.iterdir()) == {'structure.extxyz', 'run.sh'}
    command = (destination / 'run.sh').read_text().splitlines()[-1]
    assert command.startswith('exec so3lr opt --input structure.extxyz --output optimization.extxyz')
    argv = shlex.split(command)[2:]
    assert argv[argv.index('--force-conv') + 1] == '0.05' and argv[argv.index('--min-cycles') + 1] == '100'
    assert argv[argv.index('--lr-cutoff') + 1] == '100'  # gas phase
    # run.sh runs from the bundle: the real opt command accepts every option.
    monkeypatch.chdir(destination)
    with cli.commands['opt'].make_context('opt', argv[1:]) as context:
        assert context.params['force_conv'] == 0.05 and context.params['min_steps'] == 10


def test_optimization_start_now_returns_the_command(tmp_path, structure_file):
    destination = tmp_path / 'opt-now'
    text = '\n'.join([task('opt'), str(len(setup.available_presets()) + 2), str(structure_file),
                      '1', '0.01', '200', '0', '0', str(destination), '1']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert "RETURN:Action(kind='opt', argv=('opt', '--input', 'structure.extxyz'" in result.output
    assert "'--force-conv', '0.01', '--min-steps', '10', '--min-cycles', '20'" in result.output


PASTE_DATASET = str(len(setup_tasks.EVAL_EXAMPLES) + 2)  # 'Enter or paste a path'


def test_eval_saves_a_launcher_and_can_start_now(tmp_path, structure_file):
    destination = tmp_path / 'eval'
    text = '\n'.join([task('eval'), PASTE_DATASET, str(structure_file), '1', '', str(destination), '1']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    launcher = (destination / 'run.sh').read_text()
    assert 'exec so3lr eval --datafile' in launcher and str(structure_file) in launcher
    # A gas-phase dataset defaults to the long cutoff; ordinary data evaluates every target.
    assert '--save-to predictions.extxyz' in launcher and '--lr-cutoff 100' in launcher
    assert '--targets' not in launcher
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
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'runs').mkdir()
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


def test_watching_a_file_that_is_not_a_journal_is_a_one_line_error(tmp_path, monkeypatch):
    from so3lr.cli import so3lr_cli
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'settings.yaml').write_text('input_file: x.xyz\n')
    monkeypatch.setattr(setup, 'launch_setup', lambda: setup_tasks.Action('show', ('settings.yaml',)))
    result = CliRunner().invoke(so3lr_cli.cli, ['start'])
    assert result.exit_code == 2 and 'Traceback' not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


# -- release notice --------------------------------------------------------------------

@pytest.mark.parametrize('latest, current, expected', [
    ('v0.3.0', '0.2.0', True), ('v0.10.0', '0.9.9', True),
    ('v0.2.0', '0.2.0', False), ('v0.1.0', '0.2.0', False), ('nightly', '0.2.0', False),
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
    assert (tmp_path / 'opt2' / 'run.sh').is_file() and not destination.exists()

    seen = scripted(monkeypatch, [2, 0, 'back', 0, 0, 1])
    text = '\n'.join(['', str(tmp_path / 'ev')]) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    # Back from the model returns to the dataset menu, which offers the previous choice.
    assert seen == ['What would you like to do?', 'Choose a dataset', 'Model (canonical release)',
                    'Choose a dataset', 'Model (canonical release)', 'Next action · Enter starts now']
    launcher = (tmp_path / 'ev' / 'run.sh').read_text()
    assert 'sapt10k_10_dimers.xyz' in launcher and '--targets energy' in launcher


# -- binding energies -------------------------------------------------------------------

def test_dimer_pairs_and_binding_energy_errors():
    from so3lr.binding_energies import KCAL_PER_EV, binding_energy_summary, dimer_references
    infos = [{'structure_type': 'dimer_original', 'benchmark_Eint': -5.0},
             {'structure_type': 'dimer_translated', 'benchmark_Eint': -5.0},
             {'structure_type': 'dimer', 'benchmark_Eint': -1.0},
             {'structure_type': 'dimer_translated'}]
    assert dimer_references(infos) == [-5.0, -1.0]
    assert dimer_references(infos[:3]) is None
    assert dimer_references([infos[1], infos[0]]) is None
    energies = [-10.0, -10.0 + 4.0 / KCAL_PER_EV, -3.0, -3.0 + 1.5 / KCAL_PER_EV]  # -4 and -1.5 kcal/mol
    summary = binding_energy_summary(energies, [-5.0, -1.0])
    assert summary['binding_kcal'] == pytest.approx([-4.0, -1.5])
    assert (summary['n'], summary['mae'], summary['max_abs']) == (2, pytest.approx(0.75), pytest.approx(1.0))


def test_bundled_dimer_example_is_paired_and_bound():
    import numpy as np
    from ase.io import iread
    from so3lr.binding_energies import dimer_references
    frames = list(iread(str(Path(setup_tasks.__file__).parent / 'examples' / 'sapt10k_10_dimers.xyz'), index=':'))
    assert len(dimer_references(atoms.info for atoms in frames)) == 10
    for bound in frames[0::2]:
        cut = bound.info['num_a']
        gap = np.min(np.linalg.norm(bound.positions[:cut, None] - bound.positions[None, cut:], axis=-1))
        assert gap < 5.0  # really in contact, not already separated


def test_eval_reports_binding_energies_for_dimer_files(tmp_path, caplog):
    import logging
    from ase import Atoms
    from ase.io import write
    from so3lr.binding_energies import KCAL_PER_EV
    from so3lr.cli.so3lr_eval import _dimer_pairs, _report_binding_energies
    pairs = []
    for reference in (-2.0, -6.0):
        for kind in ('dimer_original', 'dimer_translated'):
            pairs.append(Atoms('He2', positions=[[0, 0, 0], [0, 0, 3]],
                               info={'structure_type': kind, 'benchmark_Eint': reference}))
    data = tmp_path / 'pairs.xyz'
    write(data, pairs, format='extxyz')
    energies = []
    for binding in (-2.5, -6.0):
        energies += [-1.0 + binding / KCAL_PER_EV, -1.0]
    with caplog.at_level(logging.INFO, logger='SO3LR'):
        summary = _report_binding_energies(_dimer_pairs(data), energies)
    assert summary['mae'] == pytest.approx(0.25)
    assert 'Binding energies (dimer - separated) for 2 dimers' in caplog.text
    assert _report_binding_energies(_dimer_pairs(data), energies[:2]) is None


def test_training_progress_lines_print_when_a_script_set_up_no_logging(capsys, monkeypatch):
    import logging
    from so3lr.mlff.utils.console import say
    logger = logging.getLogger('SO3LR')
    monkeypatch.setattr(logger, 'handlers', [])
    monkeypatch.setattr(logger, 'propagate', False)
    say('Epoch 1 of 2')
    assert capsys.readouterr().out == 'Epoch 1 of 2\n'
