"""Offline events survive noninteractive use without becoming part of the physics."""
import json
import logging
from pathlib import Path

import pytest

from so3lr.cli import presentation as ui
from so3lr.cli import telemetry


@pytest.fixture(autouse=True)
def plain_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('SO3LR_JOURNAL', raising=False)
    # Numerical CLI tests may previously have installed non-propagating file
    # handlers. Keep this module's failure notifications isolated and captured.
    monkeypatch.setattr(telemetry._LOGGER, 'handlers', [])
    monkeypatch.setattr(telemetry._LOGGER, 'propagate', True)
    previous_level = telemetry._LOGGER.level
    telemetry._LOGGER.setLevel(logging.INFO)
    ui.configure(display='plain')
    yield
    telemetry._LOGGER.setLevel(previous_level)
    ui.reset()


def events(path):
    # Reject the non-standard JSON constants that json.loads otherwise accepts.
    def nonfinite(value):
        pytest.fail(f'non-finite JSON constant: {value}')
    return [json.loads(line, parse_constant=nonfinite) for line in Path(path).read_text().splitlines()]


@pytest.mark.parametrize('value', [None, '1', '', 'false', 'off', '00'])
def test_only_explicit_zero_disables_journal(tmp_path, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv('SO3LR_JOURNAL', value)
    journal = telemetry.EventJournal({'log_file': tmp_path / 'run.log'})
    assert journal.enabled and journal.active
    journal.close()
    assert events(journal.path)[0]['type'] == 'run_started'


def test_explicit_opt_out_does_not_attempt_sidecar_io(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('SO3LR_JOURNAL', '0')
    def unexpected_io(*args, **kwargs):
        pytest.fail('an opted-out journal must not create directories or files')
    monkeypatch.setattr(Path, 'mkdir', unexpected_io)
    monkeypatch.setattr(Path, 'open', unexpected_io)
    journal = telemetry.EventJournal({'log_file': tmp_path / 'missing' / 'run.log'})
    assert not journal.enabled and not journal.active
    assert journal.path is None and journal.run_id
    journal.emit('phase', {'name': 'Compiling'})
    journal.flush_initial_warning()
    journal.close()
    assert not caplog.records


def test_journal_policy_is_fixed_for_each_invocation(tmp_path, monkeypatch):
    journal = telemetry.EventJournal({'log_file': tmp_path / 'run.log'})
    try:
        monkeypatch.setenv('SO3LR_JOURNAL', '0')
        journal.emit('phase', {'name': 'Running'})
        assert journal.enabled and events(journal.path)[-1]['type'] == 'phase'
        disabled = telemetry.EventJournal({'log_file': tmp_path / 'next.log'})
        assert not disabled.enabled and disabled.path is None
        disabled.close()
    finally:
        journal.close()


@pytest.mark.parametrize('display,quiet', [('auto', False), ('plain', False), ('auto', True)])
def test_batch_and_quiet_runs_write_complete_live_journal(tmp_path, monkeypatch, capsys, display, quiet):
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '7')
    monkeypatch.setenv('SO3LR_TEST_SECRET', 'must not be copied')
    ui.configure(display=display, quiet=quiet)
    settings = {'log_file': tmp_path / 'water.log', 'output_file': 'water.xyz', 'md_dt': 0.0005}
    with ui.run_display(settings) as reporter:
        assert not reporter.enabled
        path = reporter.journal.path
        # The first line is flushed before the simulation returns.
        initial = events(path)
        assert initial[0]['type'] == 'run_started'
        assert initial[0]['data']['settings']['log_file'] == str(settings['log_file'])
        assert initial[0]['data']['scheduler']['SLURM_JOB_ID'] == '12345'
        assert initial[0]['data']['scheduler']['SLURM_ARRAY_TASK_ID'] == '7'
        assert 'SO3LR_TEST_SECRET' not in path.read_text()
        assert initial[0]['data']['host']
        assert initial[0]['data']['pid'] > 0
        reporter.phase('Loading', 'Structure')
        reporter.phase('Loading', 'Structure')
        reporter.update_info(engine='standard', atoms=3)
        reporter.start_md(10000, 4000, 0.5)
        reporter.progress(12000, temperature=300.1, potential_energy=-12., time_per_step=0.02,
                          kinetic_energy=1., energy=-11., conserved_energy=-10.5,
                          volume=1000., box=[10., 10., 10.], pressure_diagonal=[0.1, 0.2, 0.3])
        reporter.checkpoint(12000, 'restart.npz')
        assert events(path)[-1]['type'] == 'checkpoint'
    records = events(path)
    assert [record['seq'] for record in records] == list(range(1, len(records) + 1))
    assert len({record['run_id'] for record in records}) == 1
    assert all(set(record) == {'schema', 'run_id', 'seq', 'timestamp', 'elapsed_s', 'type', 'data'}
               for record in records)
    assert all(record['schema'] == 1 and record['timestamp'].endswith('Z') for record in records)
    assert all(isinstance(record['elapsed_s'], float) and record['elapsed_s'] >= 0 for record in records)
    assert [record['type'] for record in records] == [
        'run_started', 'phase', 'info', 'info', 'md_started', 'sample', 'checkpoint', 'run_completed']
    assert records[3]['data'] == {'operation': 'md'}
    sample = records[5]['data']
    assert sample == {'step': 12000, 'segment_step': 2000, 'time_ps': 1., 'temperature_K': 300.1,
                      'potential_energy_eV': -12., 'kinetic_energy_eV': 1., 'energy_eV': -11.,
                      'conserved_energy_eV': -10.5, 'seconds_per_step': 0.02, 'volume_A3': 1000.,
                      'box_A': [10., 10., 10.], 'pressure_diagonal_eVA3': [0.1, 0.2, 0.3]}
    assert not reporter.journal.active
    assert capsys.readouterr().err == ''


def test_nonfinite_values_are_null_and_device_objects_are_not_inspected(tmp_path):
    class DeviceObject:
        def __repr__(self):
            raise AssertionError('must not read an arbitrary device value')

        def tolist(self):
            raise AssertionError('must not transfer a device array')

    with ui.run_display({'output_file': tmp_path / 'trajectory.xyz', 'unused': DeviceObject(),
                         'bad_setting': float('inf')}) as reporter:
        reporter.start_md(0, 20, 0.5)
        reporter.progress(2, temperature=float('nan'), potential_energy=float('-inf'),
                          time_per_step=None, volume=float('inf'), box=[10., float('nan'), 10.])
    records = events(reporter.journal.path)
    assert records[0]['data']['settings']['unused'] is None
    assert records[0]['data']['settings']['bad_setting'] is None
    sample = next(record['data'] for record in records if record['type'] == 'sample')
    assert sample['temperature_K'] is sample['potential_energy_eV'] is sample['volume_A3'] is None
    assert sample['box_A'] == [10., None, 10.]
    assert sample['seconds_per_step'] is None


@pytest.mark.parametrize('error', [ValueError('bad input'), KeyboardInterrupt(), SystemExit(2)])
def test_failed_and_interrupted_runs_end_with_failure_and_preserve_exit(error):
    with pytest.raises(type(error)) as caught:
        with ui.run_display({}) as reporter:
            raise error
    assert caught.value is error
    assert not getattr(error, '_so3lr_reported', False)
    records = events(reporter.journal.path)
    assert records[-1]['type'] == 'run_failed'
    assert records[-1]['data']['message']
    assert not any(record['type'] == 'run_completed' for record in records)
    assert not reporter.journal.active


def test_nested_context_does_not_mix_or_duplicate_invocations(tmp_path):
    settings = {'log_file': tmp_path / 'same.log'}
    with ui.run_display(settings) as first:
        with ui.run_display(settings) as nested:
            assert nested is first
    first_bytes = first.journal.path.read_bytes()
    with ui.run_display(settings) as second:
        pass
    assert first.journal.path != second.journal.path
    assert first.journal.path.read_bytes() == first_bytes
    assert first.journal.run_id != second.journal.run_id
    assert len(list(tmp_path.glob('same.*.events.jsonl'))) == 2


def test_id_collision_never_overwrites_existing_file(tmp_path, monkeypatch, capsys):
    class FixedID:
        hex = 'fixed'
    monkeypatch.setattr(telemetry.uuid, 'uuid4', FixedID)
    existing = tmp_path / 'same.fixed.events.jsonl'
    existing.write_text('keep this\n')
    with ui.run_display({'log_file': tmp_path / 'same.log'}) as reporter:
        reporter.phase('Running')
    assert not reporter.journal.active
    assert existing.read_text() == 'keep this\n'
    assert capsys.readouterr().err.count('Run telemetry disabled') == 1


def test_failed_initialization_warns_once_and_calculation_continues(tmp_path, capsys):
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('existing file')
    completed = []
    with ui.run_display({'log_file': blocked / 'run.log'}) as reporter:
        reporter.start_md(0, 2, 0.5)
        reporter.progress(2, temperature=300.)
        completed.append(True)
    assert completed == [True]
    assert not reporter.journal.active
    assert capsys.readouterr().err.count('Run telemetry disabled') == 1


def test_startup_failure_warns_in_new_run_log_once_and_is_counted(tmp_path, monkeypatch, capsys):
    logger = telemetry._LOGGER
    previous_log = tmp_path / 'completed.log'
    previous_log.write_text('=== END RUN previous ===\n')
    previous_bytes = previous_log.read_bytes()
    old_handler = logging.FileHandler(previous_log)
    new_log = tmp_path / 'current.log'
    monkeypatch.setattr(logger, 'handlers', [old_handler])
    monkeypatch.setattr(logger, 'propagate', False)
    original_open = Path.open
    def fail_journal(path, *args, **kwargs):
        if str(path).endswith('.events.jsonl'):
            raise PermissionError('synthetic journal permission failure')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_journal)
    new_handler = None
    try:
        with ui.run_display({'log_file': new_log}) as reporter:
            assert previous_log.read_bytes() == previous_bytes
            assert capsys.readouterr().err == ''
            logger.removeHandler(old_handler)
            old_handler.close()
            new_handler = logging.FileHandler(new_log)
            logger.addHandler(ui.make_console_handler())
            logger.addHandler(new_handler)
            reporter.logger_ready(new_log)
            reporter.logger_ready(new_log)  # nested MD driver initializes again
            reporter.start_md(0, 2, .5)
            reporter.progress(2)
        text = new_log.read_text()
        assert previous_log.read_bytes() == previous_bytes
        assert text.count('Run telemetry disabled') == 1
        assert 'Warnings: 1; errors: 0 (log records)' in text
        assert 'Execution: completed' in text
        assert capsys.readouterr().err.count('Run telemetry disabled') == 1
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        old_handler.close()


def test_unconfigured_startup_failure_goes_to_stderr_without_touching_old_log(tmp_path, monkeypatch, capsys):
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('existing file')
    previous_log = tmp_path / 'completed.log'
    previous_log.write_text('=== END RUN previous ===\n')
    before = previous_log.read_bytes()
    handler = logging.FileHandler(previous_log)
    monkeypatch.setattr(telemetry._LOGGER, 'handlers', [handler])
    try:
        journal = telemetry.EventJournal({'log_file': blocked / 'new.log'})
        assert previous_log.read_bytes() == before
        assert capsys.readouterr().err == ''
        journal.close()
        journal.close()
        assert capsys.readouterr().err.count('Run telemetry disabled') == 1
        assert previous_log.read_bytes() == before
    finally:
        handler.close()


@pytest.mark.parametrize('interrupted_operation', ['write', 'flush'])
def test_interrupted_constructor_closes_its_own_stream(tmp_path, monkeypatch, interrupted_operation):
    interruption = KeyboardInterrupt()
    class InterruptedStream:
        closed = False
        def write(self, value):
            if interrupted_operation == 'write':
                raise interruption
        def flush(self):
            if interrupted_operation == 'flush':
                raise interruption
        def close(self):
            self.closed = True
    stream = InterruptedStream()
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: stream)
    with pytest.raises(KeyboardInterrupt) as caught:
        telemetry.EventJournal({'log_file': tmp_path / 'run.log'})
    assert caught.value is interruption
    assert stream.closed


@pytest.mark.parametrize('interrupted_operation', ['write', 'flush', 'partial_write'])
def test_interrupted_event_keeps_failure_readable(tmp_path, interrupted_operation):
    from so3lr.cli.monitor import JournalReader

    interruption = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with ui.run_display({'log_file': tmp_path / 'run.log'}) as reporter:
            reporter.start_md(0, 10, .5)
            reader = JournalReader(reporter.journal.path)
            reader.poll()
            assert reader.state.status == 'Running'
            original = reporter.journal._stream

            class InterruptedStream:
                interrupted = False

                def write(self, value):
                    if not self.interrupted and interrupted_operation in ('write', 'partial_write'):
                        self.interrupted = True
                        # Use the real text stream, including its buffering:
                        # either a full record or a visible prefix may precede
                        # a signal raised before write returns to the caller.
                        original.write(value if interrupted_operation == 'write' else value[:len(value) // 2])
                        original.flush()
                        reader.poll()
                        raise interruption
                    return original.write(value)

                def flush(self):
                    original.flush()
                    if not self.interrupted and interrupted_operation == 'flush':
                        self.interrupted = True
                        reader.poll()
                        raise interruption

                def close(self):
                    original.close()

            reporter.journal._stream = InterruptedStream()
            reporter.phase('Compiling', 'Interrupted host event')

    assert caught.value is interruption
    # This reader consumed the completed line or retained the partial prefix
    # before run_failed was appended; recovery must work without reopening it.
    reader.poll()
    assert reader.state.status == 'Failed'
    assert reader.state.message == 'KeyboardInterrupt: interrupted'
    assert reader.state.sequence == reporter.journal.seq
    assert reader.pending == b''
    assert reader.state.malformed == int(interrupted_operation == 'partial_write')
    valid = []
    for line in reporter.journal.path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            valid.append(json.loads(line))
        except json.JSONDecodeError:
            assert interrupted_operation == 'partial_write'
    sequences = [record['seq'] for record in valid]
    assert sequences == sorted(set(sequences))
    assert valid[-1]['type'] == 'run_failed'
    assert not any(record['type'] == 'run_completed' for record in valid)
    fresh_reader = JournalReader(reporter.journal.path)
    fresh_reader.snapshot()
    assert fresh_reader.state.status == 'Failed'


def test_failed_write_disables_journal_once_without_recursive_events(tmp_path, caplog):
    class FailingStream:
        def write(self, value):
            raise OSError('disk full')

        def close(self):
            pass

    with ui.run_display({'log_file': tmp_path / 'run.log'}) as reporter:
        original = reporter.journal._stream
        original.close()
        reporter.journal._stream = FailingStream()
        reporter.phase('Running')
        reporter.phase('Saving')
        reporter.progress(10)
    assert not reporter.journal.active
    assert [record['type'] for record in events(reporter.journal.path)] == ['run_started']
    assert sum('Run telemetry disabled' in record.message for record in caplog.records) == 1


def test_warnings_errors_are_recorded_in_plain_mode_and_not_duplicated(tmp_path, capsys):
    handler = ui.make_console_handler()
    with ui.run_display({'log_file': tmp_path / 'run.log'}) as reporter:
        handler.handle(logging.makeLogRecord({'name': 'SO3LR', 'msg': 'warning', 'levelno': logging.WARNING,
                                             'levelname': 'WARNING'}))
        handler.handle(logging.makeLogRecord({'name': 'SO3LR', 'msg': 'error', 'levelno': logging.ERROR,
                                             'levelname': 'ERROR'}))
        handler.handle(logging.makeLogRecord({'name': 'SO3LR', 'msg': 'internal', 'levelno': logging.WARNING,
                                             'levelname': 'WARNING', 'telemetry_internal': True}))
    records = events(reporter.journal.path)
    assert [record['type'] for record in records] == ['run_started', 'warning', 'error', 'run_completed']
    assert capsys.readouterr().err == 'warning\nerror\ninternal\n'


def test_library_reporter_does_not_create_files(tmp_path):
    reporter = ui.get_reporter()
    assert not reporter.enabled
    assert not reporter.observing
    reporter.phase('Loading')
    reporter.start_md(0, 100, 0.5)
    reporter.progress(1)
    assert not list(tmp_path.iterdir())


def test_plain_cli_observes_without_terminal_rendering(tmp_path):
    with ui.run_display({'log_file': tmp_path / 'run.log'}) as reporter:
        assert reporter.observing and not reporter.enabled


def test_concurrent_journal_writes_have_unique_ordered_sequences(tmp_path):
    import threading
    import time
    journal = telemetry.EventJournal({'log_file': tmp_path / 'run.log'})
    original = journal._stream
    class SlowStream:
        def write(self, value):
            time.sleep(.0001)  # release the GIL between sequence assignment and write
            return original.write(value)
        def flush(self):
            original.flush()
        def close(self):
            original.close()
    journal._stream = SlowStream()
    barrier = threading.Barrier(5)
    def write(worker):
        barrier.wait()
        for item in range(30):
            journal.emit('test', {'worker': worker, 'item': item})
    threads = [threading.Thread(target=write, args=(worker,), daemon=True) for worker in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive(), 'journal writer deadlocked'
    journal.close()
    records = events(journal.path)
    assert [record['seq'] for record in records] == list(range(1, 122))
    for worker in range(4):
        assert [record['data']['item'] for record in records[1:]
                if record['data']['worker'] == worker] == list(range(30))


def test_worker_logging_uses_invocation_handler_and_does_not_leak_after_close(tmp_path, capsys):
    import gc
    import threading
    import weakref
    with ui.run_display({'log_file': tmp_path / 'run.log'}) as reporter:
        handler = ui.make_console_handler()
        reporter.start_md(0, 20, .5)
        barrier = threading.Barrier(5)
        def warn(worker):
            assert not ui.get_reporter().observing  # ContextVars stay thread-local
            barrier.wait()
            for item in range(20):
                handler.handle(logging.makeLogRecord(dict(name='worker', levelno=logging.WARNING,
                    levelname='WARNING', msg=f'{worker}:{item}')))
        threads = [threading.Thread(target=warn, args=(worker,), daemon=True) for worker in range(4)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for step in range(20):
            reporter.progress(step + 1, temperature=300.)
        for thread in threads:
            thread.join(timeout=3)
            assert not thread.is_alive(), 'worker logging deadlocked'
        path = reporter.journal.path
        reference = weakref.ref(reporter)
    before = path.read_bytes()
    records = events(path)
    assert [record['seq'] for record in records] == list(range(1, len(records) + 1))
    assert records[-1]['type'] == 'run_completed'
    warnings = [record['data']['message'] for record in records if record['type'] == 'warning']
    assert len(warnings) == 80
    for worker in range(4):
        assert [message for message in warnings if message.startswith(f'{worker}:')] == [f'{worker}:{i}' for i in range(20)]
    handler.handle(logging.makeLogRecord(dict(levelno=logging.WARNING, msg='after close')))
    assert path.read_bytes() == before
    del reporter
    gc.collect()
    assert reference() is None, 'console handler must not retain a finished invocation'
    assert 'after close' in capsys.readouterr().err
    handler.close()


def test_journal_failure_during_console_handler_replacement_does_not_deadlock(tmp_path):
    import subprocess
    import sys
    # Isolate a deliberate lock-interleaving probe: a regression must time out
    # this subprocess, not leave pytest blocked in logging.shutdown.
    code = '''
import logging, threading
from so3lr.cli import presentation as ui
ui.configure(display='plain')
with ui.run_display({'log_file':'run.log'}) as reporter:
    old, new = ui.make_console_handler(), ui.make_console_handler()
    logger = logging.getLogger('SO3LR')
    logger.handlers[:] = [old]
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    entered, release, second_holds_handler = [threading.Event() for _ in range(3)]
    class BrokenStream:
        def write(self, line):
            entered.set()
            assert release.wait(2)
            raise OSError('simulated disk failure')
        def close(self):
            pass
    reporter.journal._stream.close()
    reporter.journal._stream = BrokenStream()
    original = new.emit
    def noted_emit(record):
        second_holds_handler.set()
        original(record)
    new.emit = noted_emit
    first = threading.Thread(target=lambda: old.handle(logging.makeLogRecord(dict(
        name='SO3LR', msg='old warning', levelno=logging.WARNING, levelname='WARNING'))), daemon=True)
    first.start()
    assert entered.wait(2)
    logger.handlers[:] = [new]
    second = threading.Thread(target=lambda: logger.warning('new warning'), daemon=True)
    second.start()
    assert second_holds_handler.wait(2)
    release.set()
    first.join(2); second.join(2)
    assert not first.is_alive() and not second.is_alive(), 'handler/journal lock inversion'
    assert not reporter.journal.active
logger.handlers.clear()
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count('Run telemetry disabled') == 1
