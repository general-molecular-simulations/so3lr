"""The CLI without a terminal: scripts, batch jobs and LLM agents.

No model is loaded: every command here stops before JAX starts.
"""
import json
import pathlib
import shutil

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from so3lr.cli.so3lr_cli import _hoist_display_options, cli
from so3lr.structure_info import spin_key_note

WATER_64 = pathlib.Path(__file__).resolve().parent.parent / 'so3lr' / 'cli' / 'examples' / 'water_64.xyz'


@pytest.fixture
def water(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return pathlib.Path(shutil.copy(WATER_64, tmp_path / 'water.xyz'))


def test_display_options_may_follow_the_subcommand(tmp_path):
    assert _hoist_display_options(cli, ['nvt', '--input', 'x.xyz', '-q']) == ['-q', 'nvt', '--input', 'x.xyz']
    assert (_hoist_display_options(cli, ['--settings', 'show', 'nvt', '--display', 'plain', '--no-color'])
            == ['--settings', 'show', '--display', 'plain', '--no-color', 'nvt'])
    assert (_hoist_display_options(cli, ['nvt', '--display=rich', '--input', 'x.xyz'])
            == ['--display=rich', 'nvt', '--input', 'x.xyz'])
    assert _hoist_display_options(cli, ['nvt', '--', '-q']) == ['nvt', '--', '-q']
    result = CliRunner().invoke(cli, ['show', str(tmp_path), '--once', '-q'])
    assert result.exit_code == 0 and 'No such option' not in result.output


def test_start_with_input_prints_the_guided_proposal_for_a_small_box(water):
    result = CliRunner().invoke(cli, ['start', '--input', str(water), '--ensemble', 'npt', '--print'])
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load(result.output)
    assert settings['ensemble'] == 'npt' and settings['md_P'] == 1.0
    assert (settings['lr_cutoff'], settings['kspace_electrostatics'], settings['kspace_dispersion']) == (5.5, 'pme', 'pme')
    assert not (water.parent / 'so3lr-run').exists()


def test_start_with_input_writes_a_bundle_whose_run_script_works_under_sbatch(water):
    result = CliRunner().invoke(cli, ['start', '--input', str(water), '--duration', '1', '--output-dir', 'run1'])
    assert result.exit_code == 0, result.output
    bundle = water.parent / 'run1'
    script = (bundle / 'run.sh').read_text()
    # sbatch runs a copy from its spool directory: the bundle's own path is written in
    assert f'run_dir={bundle.resolve()}' in script
    assert 'exec so3lr resume --or-start .' in script
    assert yaml.safe_load((bundle / 'settings.yaml').read_text())['md_cycles'] == 20


def test_start_options_need_input():
    result = CliRunner().invoke(cli, ['start', '--ensemble', 'npt'])
    assert result.exit_code != 0 and 'need --input' in result.output


def test_a_refused_start_leaves_no_journal(water):
    (water.parent / 'water_nvt.xyz').write_text('previous run')
    result = CliRunner().invoke(cli, ['nvt', '--input', str(water)])
    assert result.exit_code == 1 and 'already exists' in result.output
    assert not list(water.parent.glob('*.events.jsonl'))


def test_nonpositive_timesteps_are_refused(water):
    result = CliRunner().invoke(cli, ['nvt', '--input', str(water), '--dt', '-1'])
    assert result.exit_code == 2 and '--dt' in result.output
    (water.parent / 'run.yaml').write_text(f'input_file: {water}\nmd_dt: 0\n')
    result = CliRunner().invoke(cli, ['--settings', 'run.yaml'])
    assert result.exit_code == 2 and 'md_dt' in result.output


def test_unknown_settings_are_dropped_after_the_warning(water, caplog):
    from so3lr.cli.so3lr_cli import load_md_settings
    path = water.parent / 'run.yaml'
    path.write_text(f'input_file: {water}\nmd_temp: 300\n')
    settings = load_md_settings(path)
    assert 'md_temp' not in settings  # kept, it would make flash decline the run
    assert "Unknown setting 'md_temp'" in caplog.text


def _bundle(tmp_path, md_cycles=100, cycle=40):
    bundle = tmp_path / 'run'
    bundle.mkdir()
    (bundle / 'settings.yaml').write_text(yaml.safe_dump({
        'input_file': 'structure.extxyz', 'output_file': 'trajectory.extxyz',
        'restart_save_path': 'restart.npz', 'md_cycles': md_cycles}))
    np.savez(bundle / 'restart.npz', step=cycle)
    return bundle


def test_resume_runs_the_cycles_left_from_the_checkpoint(tmp_path, monkeypatch):
    from so3lr.cli import so3lr_cli
    seen = {}
    monkeypatch.setattr(so3lr_cli, '_execute_md', lambda settings, **kwargs: seen.update(settings))
    monkeypatch.setattr(so3lr_cli, '_load_md_settings', lambda path, loaded: (loaded, set(loaded)))
    bundle = _bundle(tmp_path)
    result = CliRunner().invoke(cli, ['resume', str(bundle)])
    assert result.exit_code == 0, result.output
    assert seen['restart_load_path'] == 'restart.npz' and seen['md_cycles'] == 60
    assert 'Resuming from cycle 40 of 100' in result.output


def test_resume_or_start_runs_a_run_without_a_checkpoint_from_the_beginning(tmp_path, monkeypatch):
    from so3lr.cli import so3lr_cli
    seen = {}
    monkeypatch.setattr(so3lr_cli, '_execute_md', lambda settings, **kwargs: seen.update(settings))
    monkeypatch.setattr(so3lr_cli, '_load_md_settings', lambda path, loaded: (loaded, set(loaded)))
    bundle = _bundle(tmp_path)
    (bundle / 'restart.npz').unlink()
    assert CliRunner().invoke(cli, ['resume', str(bundle)]).exit_code == 2  # no checkpoint yet
    result = CliRunner().invoke(cli, ['resume', '--or-start', str(bundle)])
    assert result.exit_code == 0, result.output
    assert 'restart_load_path' not in seen and seen['md_cycles'] == 100


def test_resume_of_a_finished_run_says_so(tmp_path):
    result = CliRunner().invoke(cli, ['resume', str(_bundle(tmp_path, md_cycles=40, cycle=40))])
    assert result.exit_code == 0 and 'Already complete' in result.output


def _journal(path, events):
    with path.open('w') as stream:
        for seq, (kind, data) in enumerate(events, 1):
            stream.write(json.dumps({'schema': 1, 'run_id': 'r1', 'seq': seq, 'type': kind, 'data': data,
                                     'timestamp': '2026-11-15T12:00:00Z'}) + '\n')


def test_show_json_reports_state_progress_and_checkpoint(tmp_path):
    _journal(tmp_path / 'run.r1.events.jsonl', [
        ('run_started', {'settings': {'ensemble': 'npt', 'output_file': 'trajectory.extxyz'}}),
        ('info', {'operation': 'md', 'atoms': 192}),
        ('md_started', {'start_step': 0, 'total_steps': 1000, 'dt_fs': 0.5}),
        ('sample', {'step': 400, 'segment_step': 400, 'time_ps': 0.2, 'temperature_K': 301.0}),
        ('checkpoint', {'step': 400, 'path': 'restart.npz'}),
        ('run_failed', {'message': 'Terminated by SIGTERM', 'reason': 'signal'}),
    ])
    result = CliRunner().invoke(cli, ['show', str(tmp_path), '--json'])
    assert result.exit_code == 0, result.output
    run, = json.loads(result.output)['runs']
    assert (run['state'], run['atoms'], run['done'], run['planned']) == ('Interrupted', 192, 400, 1000)
    assert run['fraction_done'] == 0.4
    assert run['checkpoint'] == {'step': 400, 'path': 'restart.npz'}
    assert run['last_sample']['temperature_K'] == 301.0 and run['failure_reason'] == 'signal'


def test_spin_read_as_multiplicity_is_named():
    assert 'read as 2S+1' in spin_key_note({'spin': 1})
    assert spin_key_note({'spin': 1, 'multiplicity': 2}) == '' and spin_key_note({}) == ''
