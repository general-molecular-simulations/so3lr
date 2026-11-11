"""Durable run summaries and replay units do not require a numerical run."""
import json
import logging
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from so3lr.cli import presentation as ui
from so3lr.cli.run_log import resolved_settings
from so3lr.cli.so3lr_cli import load_md_settings
from so3lr.model_registry import DEFAULT_MODEL


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = logging.getLogger('SO3LR')
    monkeypatch.setattr(logger, 'handlers', [])
    previous_level = logger.level
    # Use the public API to invalidate isEnabledFor's cached decisions from
    # earlier tests; assigning .level directly leaves INFO silently disabled.
    logger.setLevel(logging.INFO)
    monkeypatch.setattr(logger, 'propagate', False)
    ui.configure(display='plain')
    yield
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(previous_level)
    ui.reset()


def ready(reporter, path):
    """Same ordering as setup_logger, without importing a numerical backend."""
    logger = logging.getLogger('SO3LR')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.addHandler(ui.make_console_handler())
    file_handler = logging.FileHandler(path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    reporter.logger_ready(path)
    return logger


def settings(tmp_path):
    (tmp_path / 'input.xyz').write_text('1\n\nH 0 0 0\n')
    return dict(input_file='input.xyz', output_file='trajectory.xyz', log_file='run.log',
                ensemble='nvt-langevin', md_dt=0.0005, md_cycles=2, md_steps=10,
                seed=52, precision='mixed', engine='auto', model_path='so3lr-tauri-m',
                relax_before_run=False, overwrite=True)


def runtime(reporter):
    reporter.update_info(model=DEFAULT_MODEL, precision='mixed', engine='standard',
                         devices=['cpu'], atoms=9)


def test_md_summary_restart_segment_units_and_paths(tmp_path, capsys):
    original = settings(tmp_path)
    original.update(restart_load_path='restart.npz', restart_save_path='restart.npz')
    with ui.run_display(original) as reporter:
        logger = ready(reporter, 'run.log')
        runtime(reporter)
        reporter.start_md(400, 20, .5)
        reporter.progress(400, 300., -10., 0., kinetic_energy=1., energy=-9.)
        logger.info('400\t-9.000\t1.000\t-10.000\t-9.000\t300.0\t0.00e+00')
        reporter.progress(420, 301., -10.1, .002, kinetic_energy=1.2, energy=-8.9,
                          volume=1200.)
        reporter.checkpoint(420, 'restart.npz')
        Path('trajectory.xyz').write_text('new trajectory')
        logger.warning('A bounded warning')
        snapshot = reporter.run_log.settings_path
        run_id = reporter.journal.run_id
    text = Path('run.log').read_text()
    assert text.count(f'=== BEGIN RUN {run_id} ===') == 1
    assert text.count(f'=== END RUN {run_id} ===') == 1
    for expected in (f'Model: {DEFAULT_MODEL}; backend: standard',
                     'Execution: completed', 'additional steps: 20 / 20',
                     'segment time: 0.01 ps', 'counter: 400 -> 420',
                     'Final total energy (eV): -8.9', 'Final volume (A^3): 1200',
                     'Warnings: 1; errors: 0', 'Last saved checkpoint: step 420',
                     'MD stage:', 'Artifact Trajectory:', 'P (when present) = bar',
                     'journal pressure = eV/A^3'):
        assert expected in text
    assert 'Run identity [' not in text
    assert 'Final conserved quantity' not in text
    assert '\x1b' not in text
    stderr = capsys.readouterr().err
    assert '400\t-9.000' in stderr and 'A bounded warning' in stderr
    assert 'RUN SUMMARY' not in stderr and f'Model: {DEFAULT_MODEL}' not in stderr
    saved = yaml.safe_load(snapshot.read_text())
    assert snapshot.name == f'run.{run_id}.settings.resolved.yaml'
    assert saved['md_dt'] == .5 and saved['engine'] == 'standard'
    assert saved['model_path'] == DEFAULT_MODEL and saved['precision'] == 'mixed'
    assert saved['overwrite'] is False and 'operation' not in saved
    assert saved['restart_load_path'] == str(tmp_path / 'restart.npz')
    assert saved['input_file'] == str(tmp_path / 'input.xyz')
    assert 'None' not in snapshot.read_text() and '!!python' not in snapshot.read_text()
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    # Absolute paths make the documented replay invocation cwd-independent.
    import os
    os.chdir(elsewhere)
    loaded = load_md_settings(snapshot)
    assert loaded['md_dt'] == original['md_dt']
    assert loaded['input_file'] == str(tmp_path / 'input.xyz')
    assert loaded['restart_save_path'] == str(tmp_path / 'restart.npz')
    assert original['overwrite'] is True and original['engine'] == 'auto'


def test_append_runs_have_unique_artifacts_and_one_header_per_invocation(tmp_path):
    source = settings(tmp_path)
    snapshots, ids = [], []
    for unused in range(2):
        with ui.run_display(source) as reporter:
            ready(reporter, 'run.log')
            runtime(reporter)
            ready(reporter, 'run.log')  # nested driver reconfigures the same log
            reporter.start_md(0, 20, .5)
            reporter.progress(20)
            snapshots.append(reporter.run_log.settings_path)
            ids.append(reporter.journal.run_id)
    text = Path('run.log').read_text()
    assert len(set(snapshots)) == 2 and all(path.exists() for path in snapshots)
    for run_id in ids:
        assert text.count(f'=== BEGIN RUN {run_id} ===') == 1
        assert text.count(f'=== END RUN {run_id} ===') == 1


def test_pre_md_relaxation_keeps_md_replay_and_summary(tmp_path):
    source = settings(tmp_path)
    source['relax_before_run'] = True
    with ui.run_display(source) as reporter:
        ready(reporter, 'run.log')
        runtime(reporter)
        reporter.update_info(operation='optimization', opt_step=30, opt_total_steps=30,
                             opt_force_max_eVA=.01, opt_force_target_eVA=.02)
        Path('trajectory_opt.xyz').write_text('relaxed geometry')
        reporter.update_info(optimization_output_file='trajectory_opt.xyz')
        snapshot = reporter.run_log.settings_path
        intermediate = yaml.safe_load(snapshot.read_text())
        assert 'operation' not in intermediate
        assert intermediate['output_file'] == str(tmp_path / 'trajectory.xyz')
        reporter.start_md(0, 20, .5)
        reporter.progress(20, energy=-1.)
    text = Path('run.log').read_text()
    assert 'Optimization: Force criterion met' not in text
    assert 'additional steps: 20 / 20' in text
    assert 'Artifact Optimization trajectory:' in text


@pytest.mark.parametrize('force,target,outcome', [
    (.01, .02, 'Force criterion met'),
    (.5, 1e-10, 'Step budget reached; force criterion not met'),
    (.5, None, 'Step budget finished (no force target)'),
])
def test_opt_execution_and_convergence_are_distinct(tmp_path, force, target, outcome):
    source = settings(tmp_path)
    source.update(force_convergence=target, min_cycles=3, min_steps=10)
    with ui.run_display(source) as reporter:
        reporter.update_info(operation='optimization')
        ready(reporter, 'run.log')
        runtime(reporter)
        reporter.update_info(opt_step=30, opt_total_steps=30, opt_energy_eV=-12.,
                             opt_force_max_eVA=force, opt_force_target_eVA=target)
    text = Path('run.log').read_text()
    assert 'Execution: completed' in text and outcome in text
    assert 'MD completed additional steps' not in text
    if target is not None:
        assert f'Force target (eV/A): {target:.12g}' in text


def test_failure_preserves_exit_and_does_not_claim_old_output(tmp_path):
    source = settings(tmp_path)
    Path('trajectory.xyz').write_text('existing user artifact')
    with pytest.raises(SystemExit) as failure:
        with ui.run_display(source) as reporter:
            logger = ready(reporter, 'run.log')
            logger.error('Output already exists', extra={'file_only': True})
            journal = reporter.journal.path
            raise SystemExit(1)
    assert failure.value.code == 1
    text = Path('run.log').read_text()
    assert 'Execution: failed/interrupted' in text
    assert 'MD integration did not start' in text
    assert 'errors: 1' in text and 'Artifact Trajectory:' not in text
    assert json.loads(journal.read_text().splitlines()[-1])['type'] == 'run_failed'
    assert Path('trajectory.xyz').read_text() == 'existing user artifact'


def test_export_io_failure_warns_once_without_losing_summary(tmp_path, monkeypatch, capsys):
    source = settings(tmp_path)
    original_open = Path.open
    def fail_yaml(path, *args, **kwargs):
        if '.settings.resolved.yaml' in str(path):
            raise PermissionError('synthetic disk failure')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_yaml)
    with ui.run_display(source) as reporter:
        ready(reporter, 'run.log')
        runtime(reporter)
        reporter.update_info(engine='standard')
        reporter.start_md(0, 20, .5)
        reporter.progress(20)
    text = Path('run.log').read_text()
    assert text.count('Resolved settings export disabled') == 1
    assert 'Execution: completed' in text and 'Warnings: 1;' in text
    assert 'Artifact Resolved settings:' not in text
    assert capsys.readouterr().err.count('Resolved settings export disabled') == 1


def test_export_never_deletes_a_preexisting_atomic_temporary(tmp_path):
    with ui.run_display(settings(tmp_path)) as reporter:
        ready(reporter, 'run.log')
        snapshot = reporter.run_log.settings_path
        original = snapshot.read_text()
        temporary = snapshot.with_suffix('.yaml.tmp')
        temporary.write_text('owned by another writer')
        runtime(reporter)
    assert temporary.read_text() == 'owned by another writer'
    assert snapshot.read_text() == original
    assert 'Resolved settings export disabled' in Path('run.log').read_text()


def test_unknown_and_nonfinite_settings_are_not_exported(tmp_path):
    source = settings(tmp_path)
    source.update(secret='private', datafile='evaluation.xyz', flash_mixed=True,
                  md_precision=None, made_up=42, kspace_spacing=float('nan'), output_format='hdf5')
    exported = resolved_settings(source, {'precision': 'mixed'}, 'md', tmp_path)
    assert all(key not in exported for key in ('secret', 'datafile', 'flash_mixed',
                                              'md_precision', 'made_up', 'kspace_spacing', 'output_format'))


def test_explicit_null_does_not_silently_restore_a_nonnull_md_default(tmp_path):
    source = settings(tmp_path)
    source.update(total_charge=None, md_T=None, restart_load_path=None)
    exported = resolved_settings(source, {}, 'md', tmp_path)
    assert exported['total_charge'] is None and exported['md_T'] is None
    assert 'restart_load_path' not in exported
    snapshot = tmp_path / 'null.yaml'
    snapshot.write_text(yaml.safe_dump(exported))
    loaded = load_md_settings(snapshot)
    assert loaded['total_charge'] is None and loaded['md_T'] is None
    assert 'None' not in snapshot.read_text()


def test_enhanced_sampling_default_input_replays_from_another_directory(tmp_path, monkeypatch):
    source = settings(tmp_path)
    source['do_enhanced_sampling'] = True
    original_cv = tmp_path / 'pysages_input.yaml'
    original_cv.write_text('method: umbrella\n')
    snapshot = tmp_path / 'resolved.yaml'
    snapshot.write_text(yaml.safe_dump(resolved_settings(source, {}, 'md', tmp_path)))
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'pysages_input.yaml').write_text('method: different\n')
    monkeypatch.chdir(other)
    replay = load_md_settings(snapshot)
    assert replay['pysages_input'] == str(original_cv)
    assert Path(replay['pysages_input']).read_text() == 'method: umbrella\n'


@pytest.mark.parametrize('enabled', [False, None])
def test_ordinary_md_does_not_gain_a_pysages_input_default(tmp_path, enabled):
    source = settings(tmp_path)
    if enabled is not None:
        source['do_enhanced_sampling'] = enabled
    assert 'pysages_input' not in resolved_settings(source, {}, 'md', tmp_path)


def test_import_is_numerical_backend_free():
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c',
                             'import so3lr.cli.run_log; import sys; '
                             'assert not any(name in sys.modules for name in '
                             '("jax", "numpy", "flax", "jax_md"))'],
                            cwd=project, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
