"""Presentation must leave computations, file logs and redirected output alone."""
import io
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest

from so3lr.cli import presentation as ui


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture(autouse=True)
def presentation_defaults(monkeypatch, tmp_path):
    ui.reset()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')
    yield
    ui.reset()
    assert not ui.get_reporter().enabled


def render(reporter, width=80, height=100):
    from rich.console import Console

    output = io.StringIO()
    console = Console(file=output, width=width, height=height, color_system=None)
    old_width, old_height = reporter.console.width, reporter.console.height
    reporter.console.width, reporter.console.height = width, height
    try:
        console.print(reporter._render())
    finally:
        reporter.console.width, reporter.console.height = old_width, old_height
    return output.getvalue()


@pytest.mark.parametrize('tty,term,display,quiet,enabled', [
    (True, 'xterm', 'auto', False, True),
    (False, 'xterm', 'auto', False, False),
    (True, 'dumb', 'auto', False, False),
    (True, 'xterm', 'plain', False, False),
    (True, 'dumb', 'rich', False, True),
    (True, 'xterm', 'rich', True, False),
])
def test_display_selection(monkeypatch, tty, term, display, quiet, enabled):
    output = Terminal() if tty else io.StringIO()
    monkeypatch.setattr(sys, 'stderr', output)
    monkeypatch.setenv('TERM', term)
    ui.configure(display=display, quiet=quiet)
    with ui.run_display({}) as reporter:
        assert reporter.enabled is enabled
    # Every display but --quiet ends with the closing line (plain output too, for batch logs).
    assert bool(output.getvalue()) is not quiet
    if not quiet:
        assert re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', output.getvalue()).rstrip().endswith('✓ Run finished in 00:00')


@pytest.mark.parametrize('quiet', [False, True])
def test_forced_rich_redirect_uses_plain_output_and_warns_once(monkeypatch, quiet):
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', output)
    monkeypatch.setenv('TERM', 'xterm-256color')
    ui.configure(display='rich', quiet=quiet)
    with ui.run_display({}) as reporter:
        assert not reporter.enabled and reporter.observing
        reporter.start_md(0, 100, .5)
        reporter.progress(50, temperature=300, energy=-38)
        with ui.run_display({}) as nested:
            assert nested is reporter
        handler = ui.make_console_handler(logging.INFO)
        handler.handle(logging.LogRecord('SO3LR', logging.WARNING, '', 0, 'plain diagnostic', (), None))
        handler.close()
    text = output.getvalue()
    assert '\x1b' not in text and 'SO3LR TAURI' not in text
    assert text.count('--display rich requires a terminal') == (0 if quiet else 1)
    assert text.count('plain diagnostic') == 1


@pytest.mark.parametrize('preserved,path,expected', [
    (False, '/tmp/run.log', ''),
    (True, '/tmp/[red]run\n\x1b.log', 'Full traceback: /tmp/[red]run\\x0a\\x1b.log'),
])
def test_failure_details_require_verified_path_and_escape_controls(preserved, path, expected):
    error = RuntimeError('failure')
    error._so3lr_traceback_preserved = preserved
    error._so3lr_traceback_path = path
    assert ui.failure_details(error) == expected


@pytest.mark.parametrize('preserved', [True, False])
def test_rich_failure_footer_follows_cleanup_and_keeps_full_journal_message(monkeypatch, tmp_path, preserved):
    import json

    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    error = RuntimeError('first line\nsecond XLA diagnostic')
    error._so3lr_traceback_preserved = preserved
    error._so3lr_traceback_path = '/tmp/verified-run.log'
    with pytest.raises(RuntimeError) as caught:
        with ui.run_display({'log_file': str(tmp_path / 'run.log')}) as reporter:
            raise error
    assert caught.value is error
    text = output.getvalue()
    if preserved:
        assert reporter.renderer.detail == 'RuntimeError: first line'
        assert text.count('Full traceback: /tmp/verified-run.log') == 1
        assert text.rfind('Full traceback:') > text.rfind('\x1b[?25h')
        assert 'second XLA diagnostic' not in text and '\\x0a' not in text
    else:
        # An unverified traceback file is never offered, so nothing is hidden.
        assert reporter.renderer.detail == 'RuntimeError: first line\nsecond XLA diagnostic'
        assert 'Full traceback:' not in text
    events = [json.loads(line) for line in reporter.journal.path.read_text().splitlines()]
    failure = next(event for event in events if event['type'] == 'run_failed')
    assert failure['data']['message'] == 'RuntimeError: first line\nsecond XLA diagnostic'


def test_module_and_plain_runs_do_not_import_rich_or_jax():
    result = subprocess.run([
        sys.executable, '-c',
        "import sys; from so3lr.cli.presentation import configure, run_display; "
        "configure(display='plain'); "
        "ctx = run_display({}); ctx.__enter__(); ctx.__exit__(None, None, None); "
        "assert not {'jax', 'jax_md', 'flax', 'numpy', 'rich'} & {m.split('.')[0] for m in sys.modules}",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('environment,explicit', [(True, False), (False, True)])
def test_no_color_keeps_layout_without_color_codes(monkeypatch, environment, explicit):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    if environment:
        monkeypatch.setenv('NO_COLOR', '')
    ui.configure(display='rich', no_color=explicit)
    with ui.run_display({'model': 'so3lr-tauri-m'}) as reporter:
        assert reporter.enabled
        reporter.phase('Compiling')
    text = output.getvalue()
    assert 'SO3LR TAURI' in text
    # Cursor movement remains necessary for layout; SGR colors must not.
    assert not re.search(r'\x1b\[(?:3[0-7]|9[0-7]|38;)[0-9;]*m', text)


def test_resumed_progress_measures_only_new_steps(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='SO3LR')
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'model': 'so3lr-tauri-m'}) as reporter:
        assert 'ETA' not in render(reporter)
        reporter.start_md(10_000, 4_000, 0.5)
        initial = render(reporter)
        assert '0/4,000 steps' in initial
        assert 'ETA estimating' in initial
        assert 'ns/day' not in initial
        assert 'Checkpoint step' not in initial
        reporter.progress(12_000, temperature=300.2, potential_energy=-12.3, energy=-11.8,
                          time_per_step=0.065)
        text = render(reporter)
        assert '50%' in text
        assert '2,000/4,000 steps' in text
        assert 'step counter 12,000' in text
        assert '1.000 / 2.000 ps this run' in text
        assert 'ETA 02:10' in text
        assert '0.665 ns/day' in text
        assert 'Temperature [K]  300.2' in text
        assert 'Total energy [eV]  -11.800' in text
        reporter.checkpoint(12_000, 'restart.npz')
    # Greppable progress lines are for plain output only.
    assert not any(record.getMessage().startswith('Progress ') for record in caplog.records)


def test_paths_are_literal_and_saved_only_after_notification(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    path = '/tmp/[red]water[/red].npz'
    with ui.run_display({'input_file': '[bold]input.xyz'}) as reporter:
        assert '[bold]input.xyz' in render(reporter)
        assert 'Checkpoint step' not in render(reporter)
        reporter.checkpoint(1_000, path)
        assert path in render(reporter)


def test_startup_shows_configured_run_then_resolved_metadata(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'model_path': 'so3lr-tauri-m', 'engine': 'auto', 'ensemble': 'nvt',
                         'md_T': 300, 'md_cycles': 40, 'md_steps': 100,
                         'md_dt': 0.0005}) as reporter:
        text = render(reporter)
        assert 'so3lr-tauri-m (requested)' in text and 'auto (requested)' in text
        assert 'NVT · 300 K target' in text
        assert '4,000 steps · 0.5 fs · 2 ps' in text
        assert 'Loading  Initializing runtime' in text
        reporter.update_info(model='so3lr-tauri-m-2606', engine='Flash', atoms=300, devices=['GPU 0'])
        text = render(reporter)
        assert 'Flash' in text and '300' in text and 'GPU 0' in text
        assert '(requested)' not in text


def test_success_lists_only_existing_output_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    output, log = tmp_path / 'water.xyz', tmp_path / 'water.log'
    log.write_text('Run started\n')
    with ui.run_display({'output_file': output, 'log_file': log}) as reporter:
        assert not reporter.artifacts
    assert reporter.artifacts == [('Metrics', str(reporter.journal.path)), ('Log', str(log))]
    assert reporter.journal.path.is_file()
    output.write_text('1\n\nH 0 0 0\n')
    with ui.run_display({'output_file': output, 'log_file': log}) as reporter:
        pass
    assert reporter.artifacts == [('Metrics', str(reporter.journal.path)),
                                  ('Trajectory', str(output)), ('Log', str(log))]
    with pytest.raises(ValueError):
        with ui.run_display({'output_file': output, 'log_file': log}) as reporter:
            raise ValueError('Failure')
    assert not reporter.artifacts


@pytest.mark.parametrize('verbose', [False, True])
def test_console_and_file_logging_are_separate(monkeypatch, tmp_path, verbose):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich', verbose=verbose)
    logger = logging.getLogger('presentation-test')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_path = tmp_path / 'run.log'
    file_handler = logging.FileHandler(file_path)
    logger.handlers = [ui.make_console_handler(), file_handler]
    try:
        with ui.run_display({}):
            logger.info('2000\t300.2\t-12.3')
            logger.info('Detailed settings [literal]')
            logger.warning('Fallback: [standard]')
            logger.error('Debug traceback', extra={'file_only': True})
    finally:
        file_handler.close()
        logger.handlers.clear()
    assert '2000\t300.2' not in output.getvalue()
    assert ('Detailed settings [literal]' in output.getvalue()) is verbose
    assert 'WARNING: Fallback: [standard]' in output.getvalue()
    assert 'Debug traceback' not in output.getvalue()
    assert file_path.read_text() == ('2000\t300.2\t-12.3\nDetailed settings [literal]\n'
                                     'Fallback: [standard]\nDebug traceback\n')


@pytest.mark.parametrize('quiet', [False, True])
def test_plain_console_preserves_rows_and_quiet_keeps_warnings(monkeypatch, quiet):
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='plain', quiet=quiet)
    handler = ui.make_console_handler()
    with ui.run_display({}) as reporter:
        reporter.phase('Hidden presentation event')
        handler.handle(logging.makeLogRecord({'msg': '2000\t300.2', 'levelno': logging.INFO}))
        handler.handle(logging.makeLogRecord({'msg': 'Warning remains', 'levelno': logging.WARNING}))
    assert output.getvalue() == (('' if quiet else '2000\t300.2\n') + 'Warning remains\n'
                                 + ('' if quiet else '✓ Run finished in 00:00\n'))


@pytest.mark.parametrize('error', [ValueError('Invalid input'), KeyboardInterrupt()])
def test_error_cleanup_restores_cursor_and_propagates(monkeypatch, error):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    with pytest.raises(type(error)):
        with ui.run_display({}) as reporter:
            raise error
    assert reporter._closed
    assert reporter.status == 'failed'
    assert getattr(error, '_so3lr_reported', False)
    assert not ui.get_reporter().enabled
    assert '\x1b[?25h' in output.getvalue()
    assert 'Complete' not in output.getvalue()


def test_reporter_restores_cursor_after_interruption_before_live_hook(monkeypatch):
    from rich.console import Console
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    original_cursor = Console.show_cursor
    interruption = KeyboardInterrupt()
    def cursor(console, show=True):
        original_cursor(console, show)
        if not show:
            raise interruption
    monkeypatch.setattr(Console, 'show_cursor', cursor)
    with pytest.raises(KeyboardInterrupt) as caught:
        with ui.run_display({}):
            pytest.fail('interrupted initialization must not enter the body')
    assert caught.value is interruption
    assert '\x1b[?25l' in output.getvalue()
    assert '\x1b[?25h' in output.getvalue()


def test_failure_uses_actual_exception_chain_instead_of_previous_error(monkeypatch):
    import json
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with pytest.raises(RuntimeError):
        with ui.run_display({}) as reporter:
            ui.make_console_handler().handle(logging.makeLogRecord({
                'msg': 'Earlier recoverable diagnostic', 'levelno': logging.ERROR,
                'levelname': 'ERROR'}))
            try:
                raise ValueError('bad geometry')
            except ValueError as cause:
                raise RuntimeError('initialization failed') from cause
    expected = 'RuntimeError: initialization failed (caused by ValueError: bad geometry)'
    assert reporter.renderer._state.message == expected
    records = [json.loads(line) for line in reporter.journal.path.read_text().splitlines()]
    assert records[-1]['data']['message'] == expected
    assert ui.exception_message(AssertionError()) == 'AssertionError'


def test_worker_warning_burst_has_no_display_queue_loss(monkeypatch):
    import json
    import threading
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    with ui.run_display({}) as reporter:
        handler = ui.make_console_handler()
        def warn():
            for index in range(300):
                handler.handle(logging.makeLogRecord(dict(name='SO3LR', msg=f'worker-{index:03d}',
                    levelno=logging.WARNING, levelname='WARNING')))
        worker = threading.Thread(target=warn, daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), 'worker output deadlocked'
        text = output.getvalue()
        assert all(f'worker-{index:03d}' in text for index in range(300))
        records = [json.loads(line) for line in reporter.journal.path.read_text().splitlines()]
        assert sum(event['type'] == 'warning' for event in records) == 300
        # Shown at once from the worker, but the view's state changes only on the owning thread.
        assert reporter.renderer._state.diagnostic is None
        reporter.phase('Running')
        assert reporter.renderer._state.diagnostic == ('warning', 'worker-299')


def test_finalization_warning_stays_in_live_and_journal(monkeypatch):
    """A warning logged while the display closes after a completed run still shows and is journaled."""
    import json
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({}) as reporter:
        handler = ui.make_console_handler()
        shown = []
        monkeypatch.setattr(reporter.renderer, 'print_log', lambda record, message: shown.append(message))
        original = reporter.renderer.close
        def close(*args):
            handler.handle(logging.makeLogRecord(dict(name='SO3LR', msg='summary disk failure',
                levelno=logging.WARNING, levelname='WARNING')))
            original(*args)
        monkeypatch.setattr(reporter.renderer, 'close', close)
        reporter.finish()
        reporter.close()
        monkeypatch.setattr(reporter.renderer, 'close', original)
        assert shown == ['summary disk failure']
        records = [json.loads(line) for line in reporter.journal.path.read_text().splitlines()]
        assert records[-1]['type'] == 'warning' and records[-1]['data']['message'] == 'summary disk failure'
        assert records[-2]['type'] == 'run_completed'


def test_reused_handler_binds_to_current_invocation(monkeypatch):
    import json
    ui.configure(display='plain')
    with ui.run_display({}) as previous:
        handler = ui.make_console_handler()
    old_path = previous.journal.path
    with ui.run_display({}) as current:
        handler.handle(logging.makeLogRecord(dict(name='SO3LR', msg='new invocation diagnostic',
            levelno=logging.WARNING, levelname='WARNING')))
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    new = [json.loads(line) for line in current.journal.path.read_text().splitlines()]
    assert [event['type'] for event in old] == ['run_started', 'run_completed']
    assert [event['type'] for event in new] == ['run_started', 'warning', 'run_completed']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signal dispositions')
def test_signal_scope_preserves_custom_ignored_and_worker_dispositions():
    import signal
    import threading
    original_term, original_hup = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
    def custom(number, frame):
        pass
    try:
        signal.signal(signal.SIGTERM, custom)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        with ui._termination_signals():
            assert signal.getsignal(signal.SIGTERM) is custom
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is custom
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        failures = []
        def worker():
            try:
                with ui._termination_signals():
                    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
            except BaseException as exc:
                failures.append(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=3)
        assert not thread.is_alive() and not failures
        with ui._termination_signals():
            assert callable(signal.getsignal(signal.SIGTERM))
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGHUP, original_hup)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signal dispositions')
@pytest.mark.parametrize('interruption', ['keyboard', 'termination'])
def test_signal_installation_is_owned_before_it_can_be_interrupted(monkeypatch, interruption):
    import signal
    original_signal = signal.signal
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    redelivered = []
    try:
        for number in previous:
            original_signal(number, signal.SIG_DFL)
        def install(number, handler):
            result = original_signal(number, handler)
            if number == signal.SIGHUP and callable(handler):
                if interruption == 'keyboard':
                    raise KeyboardInterrupt
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return result
        monkeypatch.setattr(signal, 'signal', install)
        monkeypatch.setattr(signal, 'raise_signal', redelivered.append)
        with pytest.raises(KeyboardInterrupt if interruption == 'keyboard' else SystemExit) as caught:
            with ui._termination_signals():
                pytest.fail('interrupted installation must not enter the body')
        assert all(signal.getsignal(number) == signal.SIG_DFL for number in previous)
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == previous_mask
        if interruption == 'termination':
            assert redelivered == [signal.SIGTERM]
            assert caught.value.code == 128 + signal.SIGTERM
        else:
            assert not redelivered
    finally:
        for number, handler in previous.items():
            original_signal(number, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signal dispositions and masks')
@pytest.mark.parametrize('interruption', ['before', 'after', 'termination'])
def test_signal_restore_completes_all_dispositions_and_restores_prior_mask(monkeypatch, interruption):
    import signal
    original_signal = signal.signal
    numbers = (signal.SIGTERM, signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in numbers}
    initial_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    expected_mask = initial_mask | {signal.SIGUSR1}
    redelivered, injected = [], False
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, expected_mask)
        for number in numbers:
            original_signal(number, signal.SIG_DFL)
        with pytest.raises(SystemExit if interruption == 'termination' else KeyboardInterrupt):
            with ui._termination_signals():
                hup_handler = signal.getsignal(signal.SIGHUP)
                def restore(number, handler):
                    nonlocal injected
                    if number == signal.SIGTERM and handler == signal.SIG_DFL and not injected:
                        injected = True
                        if interruption == 'before':
                            raise KeyboardInterrupt
                        result = original_signal(number, handler)
                        if interruption == 'after':
                            raise KeyboardInterrupt
                        # HUP is still a Python handler at this point; TERM has
                        # already become native SIG_DFL (the review swapped them).
                        hup_handler(signal.SIGHUP, None)
                        return result
                    return original_signal(number, handler)
                monkeypatch.setattr(signal, 'signal', restore)
                monkeypatch.setattr(signal, 'raise_signal', redelivered.append)
        assert injected
        assert all(signal.getsignal(number) == signal.SIG_DFL for number in numbers)
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == expected_mask
        assert redelivered == ([signal.SIGHUP] if interruption == 'termination' else [])
    finally:
        for number, handler in previous.items():
            original_signal(number, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, initial_mask)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX pending signals and masks')
@pytest.mark.parametrize('when', ['before', 'after'])
def test_interrupted_mask_restoration_reapplies_saved_mask(monkeypatch, when):
    import signal
    from so3lr.cli.terminal import _blocked_signals
    original_mask = signal.pthread_sigmask
    initial_mask = original_mask(signal.SIG_BLOCK, [])
    expected_mask = initial_mask | {signal.SIGUSR1}
    interrupted = False
    def mask(how, values):
        nonlocal interrupted
        if how == signal.SIG_SETMASK and not interrupted:
            interrupted = True
            if when == 'after':
                original_mask(how, values)
            raise KeyboardInterrupt
        return original_mask(how, values)
    try:
        original_mask(signal.SIG_SETMASK, expected_mask)
        monkeypatch.setattr(signal, 'pthread_sigmask', mask)
        with pytest.raises(KeyboardInterrupt):
            with _blocked_signals({signal.SIGTERM, signal.SIGHUP}):
                pass
        assert interrupted and original_mask(signal.SIG_BLOCK, []) == expected_mask
    finally:
        original_mask(signal.SIG_SETMASK, initial_mask)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX pending signals and masks')
@pytest.mark.parametrize('signal_name', ['SIGHUP'])  # TERM and HUP share this path
def test_real_signal_during_restore_waits_until_all_dispositions_are_restored(tmp_path, signal_name):
    import json
    import signal
    code = '''
import json, os, pathlib, signal, sys
from so3lr.cli.terminal import termination_signals
original_signal, original_mask = signal.signal, signal.pthread_sigmask
original_mask(signal.SIG_BLOCK, {signal.SIGUSR1})
prior_mask = original_mask(signal.SIG_BLOCK, [])
injected = False
with termination_signals():
    def restore(number, handler):
        global injected
        result = original_signal(number, handler)
        if number == signal.SIGTERM and handler == signal.SIG_DFL and not injected:
            injected = True
            os.kill(os.getpid(), getattr(signal, sys.argv[1]))
        return result
    def mask(how, values):
        if injected and how == signal.SIG_SETMASK:
            pathlib.Path('restore.json').write_text(json.dumps({
                'defaults': all(signal.getsignal(n) == signal.SIG_DFL for n in (signal.SIGTERM, signal.SIGHUP)),
                'mask_restored': set(values) == prior_mask}))
        return original_mask(how, values)
    signal.signal, signal.pthread_sigmask = restore, mask
'''
    result = subprocess.run([sys.executable, '-c', code, signal_name], cwd=tmp_path,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == -getattr(signal, signal_name), result.stderr
    assert json.loads((tmp_path/'restore.json').read_text()) == {'defaults': True, 'mask_restored': True}
    assert 'Traceback' not in result.stderr


@pytest.fixture(scope='module')
def native_signal_blocker(tmp_path_factory):
    """A native call with no Python signal checkpoints, bounded by child kill."""
    import shutil
    compiler = shutil.which('cc')
    if os.name == 'nt' or compiler is None:
        pytest.skip('POSIX C compiler required for native-blocking signal probe')
    directory = tmp_path_factory.mktemp('native-signal-blocker')
    source = directory / 'block.c'
    source.write_text('#include <unistd.h>\nvoid block_forever(int fd) {\n'
                      '  volatile unsigned long counter = 0;\n'
                      '  write(fd, "ready", 5);\n'
                      '  for (;;) { counter++; }\n}\n')
    library = directory / ('block.dylib' if sys.platform == 'darwin' else 'block.so')
    subprocess.run([compiler, '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                    '-fPIC', str(source), '-o', str(library)], check=True, capture_output=True)
    return library


def _wait_for_native_marker(process, path):
    deadline = time.monotonic() + 5
    while not path.exists() or path.stat().st_size == 0:
        assert process.poll() is None, 'signal probe exited before entering native code'
        assert time.monotonic() < deadline, 'signal probe did not enter native code'
        time.sleep(.01)


# Every non-terminal display skips the signal guard; forced rich is the one that could get it wrong.
@pytest.mark.parametrize('display,quiet', [('rich', False)])
def test_nonterminal_run_retains_native_termination_during_blocked_c_call(tmp_path, native_signal_blocker, display, quiet):
    import signal
    code = '''
import ctypes, sys
from so3lr.cli import presentation as ui
library = ctypes.PyDLL(sys.argv[1])
library.block_forever.argtypes = [ctypes.c_int]
ui.configure(display=sys.argv[2], quiet=sys.argv[3] == 'True')
with ui.run_display({'log_file': 'run.log'}):
    with open('native-ready', 'wb') as marker:
        library.block_forever(marker.fileno())
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(native_signal_blocker), display, str(quiet)],
                               cwd=tmp_path, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        _wait_for_native_marker(process, tmp_path / 'native-ready')
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=2) == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


# A different second signal: proves every disposition, not just the received one, is native again.
@pytest.mark.parametrize('second_signal', ['SIGHUP'])
def test_second_termination_is_native_after_first_exception_is_swallowed(tmp_path, native_signal_blocker, second_signal):
    import signal
    code = '''
import ctypes, pathlib, sys, time
from so3lr.cli.terminal import TerminationSignal, termination_signals
library = ctypes.PyDLL(sys.argv[1])
library.block_forever.argtypes = [ctypes.c_int]
with termination_signals():
    try:
        pathlib.Path('first-ready').write_text('ready')
        while True:
            time.sleep(.05)
    except TerminationSignal:
        # A callback/finalizer can swallow the first exception. Later signals
        # must remain native even while the main thread is blocked in C.
        with open('native-ready', 'wb') as marker:
            library.block_forever(marker.fileno())
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(native_signal_blocker)], cwd=tmp_path,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_native_marker(process, tmp_path / 'first-ready')
        process.send_signal(signal.SIGTERM)
        _wait_for_native_marker(process, tmp_path / 'native-ready')
        number = getattr(signal, second_signal)
        process.send_signal(number)
        assert process.wait(timeout=2) == -number
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX PTY and termination signals')
@pytest.mark.parametrize('signal_name', ['SIGHUP'])  # terminal hang-up; SIGTERM shares the path
def test_real_terminal_termination_restores_modes_and_preserves_signal_exit(tmp_path, signal_name):
    import fcntl
    import json
    import pty
    import select
    import signal
    import struct
    import termios
    code = '''
import fcntl, json, pathlib, sys, termios, time
from so3lr.cli import presentation as ui
fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
before = termios.tcgetattr(sys.stdin.fileno())
original = ui._Reporter.close
def close(self):
    try:
        original(self)
    finally:
        after = termios.tcgetattr(sys.stdin.fileno())
        pending = getattr(termios, 'PENDIN', 0)
        restored = (before[:3] == after[:3] and before[4:] == after[4:]
                    and before[3] & ~pending == after[3] & ~pending)
        pathlib.Path('terminal.json').write_text(json.dumps({'restored': restored}))
ui._Reporter.close = close
ui.configure(display='rich')
with ui.run_display({'log_file': 'run.log', 'ensemble': 'nve'}) as reporter:
    reporter.start_md(0, 100, .5)
    reporter.progress(0, temperature=300, energy=-10)
    pathlib.Path('ready').write_text(str(reporter.journal.path))
    while True:
        time.sleep(.05)
'''
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 80, 0, 0))
    process = subprocess.Popen([sys.executable, '-c', code], cwd=tmp_path,
                               stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
                               env=dict(os.environ, TERM='xterm', COLUMNS='80', LINES='24'))
    capture, sent = bytearray(), False
    deadline = time.monotonic() + 10
    def read():
        try:
            chunk = os.read(master, 65536)
        except OSError:
            return False
        capture.extend(chunk)
        return bool(chunk)
    try:
        while process.poll() is None:
            assert time.monotonic() < deadline, 'signal cleanup timed out'
            if select.select([master], [], [], .05)[0] and not read():
                process.wait(timeout=2)
                break
            if not sent and (tmp_path / 'ready').exists():
                os.kill(process.pid, getattr(signal, signal_name))
                sent = True
        while select.select([master], [], [], .05)[0] and read():
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)
    assert sent and process.returncode == -getattr(signal, signal_name)
    assert json.loads((tmp_path / 'terminal.json').read_text())['restored']
    assert b'\x1b[?25l' in capture and b'\x1b[?25h' in capture
    journal = next(tmp_path.glob('*.events.jsonl'))
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert records[-1]['type'] == 'run_failed'
    assert signal_name in records[-1]['data']['message']
    assert not any(record['type'] == 'run_completed' for record in records)


def test_progress_refresh_is_throttled(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    clock = [10.0]
    monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
    ui.configure(display='rich')
    with ui.run_display({}) as reporter:
        reporter.start_md(0, 100, 0.5)
        updates = []
        monkeypatch.setattr(reporter.live, 'update', lambda *args, **kwargs: updates.append(1))
        for step in range(10):
            reporter.phase('Running')
            reporter.progress(step)
            reporter.phase('Saving', 'Writing trajectory and checkpoint')
            reporter.checkpoint(step, 'restart.npz')
            reporter.phase('Running')
        # The first sample immediately replaces startup status with live plots.
        assert len(updates) == 1
        clock[0] += 0.3
        reporter.progress(10)
        assert len(updates) == 2


def test_attached_plots_update_before_completion_without_reading_journal(monkeypatch):
    from so3lr.cli import monitor

    monkeypatch.setattr(sys, 'stderr', Terminal())
    monkeypatch.setattr(monitor.JournalReader, 'poll', lambda *args: pytest.fail('Attached display must not read its journal'))
    clock = [10.0]
    monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
    ui.configure(display='rich')
    with ui.run_display({'ensemble': 'npt', 'precision': 'mixed', 'md_T': 300}) as reporter:
        reporter.update_info(atoms=9, engine='standard', devices=['CPU 0'], kernels='Pallas')
        reporter.start_md(0, 200, 0.5)
        frames = []
        monkeypatch.setattr(reporter.live, 'update', lambda *args, **kwargs: frames.append(render(reporter)))
        reporter.progress(0, temperature=280, potential_energy=-39, volume=1000)
        clock[0] += 0.3
        reporter.progress(100, temperature=300, potential_energy=-38, time_per_step=0.002,
                          energy=-37, conserved_energy=-36, volume=1100)
        assert reporter.status == 'running'
        assert len(frames) == 2
        assert 'Temperature [K]  280.0' in frames[0]
        assert 'Temperature [K]  300.0' in frames[1]
        assert 'Total energy [eV]  -37.000' in frames[1]
        assert 'Volume [Å³]  1100.000' in frames[1]
        assert frames[1].index('Temperature [K]') < frames[1].index('Total energy [eV]') < frames[1].index('Volume [Å³]')
        assert 'mixed (float32 model; float64 dynamics)' in frames[1]
        assert 'CPU 0' in frames[1] and 'Pallas' in frames[1]
        assert 'ns/day' in frames[1] and 's/atom/step' in frames[1]
        assert '50%' in frames[1] and 'MD elapsed' in frames[1] and 'ETA' in frames[1]
        assert reporter._state.samples[-1]['energy_eV'] == -37
        assert reporter._state.samples[-1]['conserved_energy_eV'] == -36
        assert 'detach' not in frames[1]


class InputTerminal(Terminal):
    def fileno(self):
        return 42


def test_optimization_progress_and_transition_to_md(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'ensemble': 'nvt', 'md_T': 300}) as reporter:
        reporter.update_info(operation='optimization', opt_step=20, opt_total_steps=100,
                             opt_energy_eV=-39.2, opt_force_max_eVA=0.01, opt_force_target_eVA=0.02)
        text = render(reporter)
        assert '20/100 optimization steps' in text
        assert 'Force criterion met' in text and '0.010000 eV/Å' in text
        reporter.start_md(0, 100, 0.5)
        reporter.progress(0, temperature=300, potential_energy=-39.2, energy=-38)
        text = render(reporter)
        assert 'Temperature [K]' in text and 'Total energy [eV]' in text
        assert 'optimization steps' not in text and reporter._state.info['operation'] == 'md'


@pytest.mark.parametrize('unicode', [True, False])
def test_activity_updates_time_without_advancing_simulation(monkeypatch, unicode):
    from rich.console import Console
    clock = [10.0]
    monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
    activity = ui._Activity('Compiling', 10.0, 10.0, unicode=unicode)
    output = io.StringIO()
    console = Console(file=output, width=100, no_color=True)
    # Sample at the actual refresh cadence: the pulse must not alias into a
    # stationary glyph while the host is blocked in compilation.
    for tick in range(11):
        clock[0] = 10.0 + tick * 0.5
        console.print(activity)
    text = output.getvalue()
    assert '00:00 since launch' in text and '00:05 since launch' in text
    assert 'last update 00:05 ago' in text
    assert '%' not in text and 'steps' not in text
    assert len({line[0] for line in text.splitlines()}) >= 3
    assert '\x1b' not in text
    if not unicode:
        assert text.isascii()
        narrow = AsciiTerminal()  # truncation must not add an ellipsis glyph
        Console(file=narrow, width=20, color_system=None).print(
            ui._Activity('Compiling a long molecular force field', 0, 0, unicode=False))
        assert narrow.getvalue().isascii()


class FakeKeyboard:
    def __init__(self):
        self.keys = []
        self.entered = self.closed = 0

    def __enter__(self):
        self.entered += 1
        return self

    def read(self, timeout=0.1):
        assert timeout == 0
        return self.keys.pop(0) if self.keys else ''

    def __exit__(self, *args):
        self.closed += 1


def test_attached_keyboard_pages_selects_energy_and_toggles_plots_without_stopping_md(monkeypatch):
    from so3lr.cli import monitor

    monkeypatch.setattr(sys, 'stderr', Terminal())
    monkeypatch.setattr(sys, 'stdin', InputTerminal())
    keyboard = FakeKeyboard()
    monkeypatch.setattr(monitor, '_Keyboard', lambda: keyboard)
    ui.configure(display='rich')
    with ui.run_display({'ensemble': 'npt', 'md_T': 300, 'precision': 'mixed'}) as reporter:
        reporter.update_info(atoms=9)
        reporter.start_md(0, 1000, 0.5)
        reporter.progress(100, temperature=300, potential_energy=-39, energy=-38,
                          volume=1000, time_per_step=0.002)
        first_page = render(reporter, height=24)
        assert len(first_page.splitlines()) <= 24
        assert 'Temperature [K]' in first_page and 'Total energy [eV]' in first_page
        assert 'q: hide plots' in first_page and 'Ctrl+C: interrupt MD' in first_page
        keyboard.keys.append('down')
        reporter.progress(200, temperature=300, potential_energy=-39, energy=-38, volume=1010)
        assert 'Volume [Å³]' in render(reporter, height=24)
        keyboard.keys.extend(['up', 'e'])
        reporter.progress(300, temperature=300, potential_energy=-39, energy=-38, conserved_energy=-37, volume=1010)
        reporter.progress(400, temperature=300, potential_energy=-39, energy=-38, conserved_energy=-37, volume=1010)
        assert 'Conserved energy [eV]' in render(reporter, height=24)
        keyboard.keys.append('escape')
        reporter.progress(500, temperature=300, potential_energy=-39)
        assert reporter.status == 'running'
        assert keyboard.entered == 1 and keyboard.closed == 0
        assert 'Temperature [K]' not in render(reporter)
        assert 'q: show plots' in render(reporter)
        keyboard.keys.append('escape')  # Escape only hides; it never toggles the plots back
        reporter.progress(600, temperature=310, potential_energy=-38, energy=-37)
        assert reporter._state.samples[-1]['step'] == 600
        assert 'Temperature [K]' not in render(reporter)
        keyboard.keys.append('q')
        reporter.progress(700, temperature=315, potential_energy=-37, energy=-36, conserved_energy=-35)
        restored = render(reporter)
        assert 'Temperature [K]  315.0' in restored
        assert 'Conserved energy [eV]' in restored
        assert reporter._state.sample_count == 7
        keyboard.keys.append('q')  # q toggles: hide again
        reporter.progress(1000, temperature=310, potential_energy=-38)
        assert reporter._state.samples[-1]['step'] == 1000
        assert 'Temperature [K]' not in render(reporter)
    assert reporter.status == 'complete' and keyboard.closed == 1
    assert 'so3lr show' in render(reporter)
    assert 'q: show plots' not in render(reporter)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal stderr redirection')
@pytest.mark.parametrize('failure', [True])  # success runs the same close()
def test_terminal_stderr_redirect_is_scoped_and_flushes_partial_lines(tmp_path, failure):
    import json
    import pty
    import select

    code = r'''
import json, logging, pathlib, sys, warnings
from so3lr.cli import presentation as ui
original_stderr, original_stdout = sys.stderr, sys.stdout
existing = logging.StreamHandler(original_stderr)
logger = logging.getLogger('existing-handler')
logger.handlers = [existing]
logger.propagate = False
ui.configure(display='rich')
inside = {}
try:
    with ui.run_display({'log_file': 'run.log'}) as reporter:
        inside['stderr_redirected'] = sys.stderr is not original_stderr
        inside['stdout_unchanged'] = sys.stdout is original_stdout
        inside['old_handler_unchanged'] = logger.handlers == [existing] and existing.stream is original_stderr
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.warn('warning-via-live', UserWarning)
        sys.stderr.write('direct-via-live\n')
        fresh = logging.StreamHandler()
        inside['new_handler_uses_current_stderr'] = fresh.stream is sys.stderr
        new_logger = logging.getLogger('new-handler')
        new_logger.handlers = [fresh]
        new_logger.propagate = False
        new_logger.warning('new-handler-via-live')
        new_logger.removeHandler(fresh)
        fresh.close()
        sys.stderr.write('partial-via-live')
        if sys.argv[1] == 'True':
            raise RuntimeError('expected run failure')
except RuntimeError:
    pass
reporter.renderer.close()  # repeated cleanup must not replay a partial line
inside['stderr_restored'] = sys.stderr is original_stderr
inside['stdout_restored'] = sys.stdout is original_stdout
inside['old_handler_retained'] = logger.handlers == [existing] and existing.stream is original_stderr
pathlib.Path('stderr.json').write_text(json.dumps(inside))
'''
    master, slave = pty.openpty()
    process = subprocess.Popen([sys.executable, '-c', code, str(failure)], cwd=tmp_path,
                               stdin=subprocess.DEVNULL, stdout=slave, stderr=slave,
                               env=dict(os.environ, TERM='xterm', COLUMNS='100', LINES='30'))
    os.close(slave)
    capture = bytearray()
    deadline = time.monotonic() + 10
    try:
        while True:
            assert time.monotonic() < deadline, 'stderr capture timed out'
            if not select.select([master], [], [], .05)[0]:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            capture.extend(chunk)
        assert process.wait(timeout=2) == 0, capture.decode(errors='replace')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
    result = json.loads((tmp_path / 'stderr.json').read_text())
    assert result and all(result.values()), result
    for marker in (b'warning-via-live', b'direct-via-live', b'new-handler-via-live', b'partial-via-live'):
        assert capture.count(marker) == 1, capture.decode(errors='replace')
    assert b'\x1b[?25l' in capture and b'\x1b[?25h' in capture


@pytest.mark.parametrize('stdin', [io.StringIO(), Terminal()])
def test_attached_never_takes_redirected_or_unusable_input(monkeypatch, stdin):
    from so3lr.cli import monitor

    monkeypatch.setattr(sys, 'stderr', Terminal())
    monkeypatch.setattr(sys, 'stdin', stdin)
    monkeypatch.setattr(monitor, '_Keyboard', lambda: pytest.fail('Must not take stdin'))
    ui.configure(display='rich')
    with ui.run_display({}) as reporter:
        reporter.start_md(0, 100, 0.5)
        reporter.progress(0, temperature=300)
        assert reporter._keyboard is None
        assert 'q: hide' not in render(reporter)


def test_attached_keyboard_restored_after_render_failure(monkeypatch):
    # Ctrl+C is covered with a real PTY in test_real_terminal_controls_restore_input...
    from so3lr.cli import monitor

    monkeypatch.setattr(sys, 'stderr', Terminal())
    monkeypatch.setattr(sys, 'stdin', InputTerminal())
    keyboard = FakeKeyboard()
    monkeypatch.setattr(monitor, '_Keyboard', lambda: keyboard)
    ui.configure(display='rich')
    with pytest.raises(RuntimeError):
        with ui.run_display({}) as reporter:
            reporter.start_md(0, 100, 0.5)
            reporter.progress(0, temperature=300, potential_energy=-39)
            # The failure recurs while fail() refreshes; the keyboard must still be released.
            monkeypatch.setattr(reporter.renderer, '_render', lambda: (_ for _ in ()).throw(RuntimeError('Render failed')))
            reporter.renderer._refresh(force=True)
    assert keyboard.entered == keyboard.closed == 1
    assert not ui.get_reporter().enabled


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal input mode')
@pytest.mark.parametrize('key', ['q', '\x03'])
def test_real_terminal_controls_restore_input_and_keep_q_independent_of_md(tmp_path, key):
    import fcntl
    import pty
    import select
    import struct
    import termios

    marker = tmp_path / 'steps.txt'
    code = """
import fcntl, pathlib, sys, termios, time
from so3lr.cli.presentation import configure, run_display
fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
before = termios.tcgetattr(sys.stdin.fileno())
configure()
try:
    with run_display({'ensemble': 'npt', 'precision': 'mixed', 'md_T': 300}) as reporter:
        reporter.update_info(atoms=9)
        reporter.start_md(0, 60, 0.5)
        for step in range(61):
            reporter.progress(step, temperature=300+step, potential_energy=-39+step/100,
                              energy=-38+step/100, volume=1000+step, time_per_step=0.03)
            pathlib.Path(sys.argv[1]).write_text(str(step))
            time.sleep(0.03)
except KeyboardInterrupt:
    sys.exit(130)
finally:
    after = termios.tcgetattr(sys.stdin.fileno())
    pending = getattr(termios, 'PENDIN', 0)
    restored = (before[:3] == after[:3] and before[4:] == after[4:]
                and before[3] & ~pending == after[3] & ~pending)
    pathlib.Path(sys.argv[1] + '.terminal').write_text(str(restored))
"""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 80, 0, 0))
    process = subprocess.Popen([sys.executable, '-c', code, str(marker)],
                               cwd=tmp_path, stdin=slave, stdout=slave, stderr=slave,
                               start_new_session=True, env=dict(os.environ, TERM='xterm', COLUMNS='80', LINES='24'))
    capture, sent_at_step, restore_offset = bytearray(), None, None
    deadline = time.monotonic() + 10

    def read_available():
        try:
            chunk = os.read(master, 1024 * 1024)
        except OSError:
            return False
        capture.extend(chunk)
        return bool(chunk)

    try:
        while process.poll() is None:
            assert time.monotonic() < deadline, 'Terminal control smoke timed out'
            if select.select([master], [], [], 0.05)[0]:
                if not read_available():
                    process.wait(timeout=2)
                    break
            if (sent_at_step is None and marker.exists() and b'Temperature [K]' in capture
                    and b'Total energy [eV]' in capture):
                sent_at_step = int(marker.read_text() or 0)
                os.write(master, key.encode())
            elif key == 'q' and restore_offset is None and b'show plots' in capture:
                restore_offset = len(capture)
                os.write(master, b'q')
        drain_deadline = time.monotonic() + 1
        while time.monotonic() < drain_deadline and select.select([master], [], [], 0.05)[0]:
            if not read_available():
                break
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)
    assert sent_at_step is not None
    assert b'\x1b[?25h' in capture
    # A controlling session's exit invalidates its PTY on macOS; inspect the
    # state in the child after run_display cleanup and before session exit.
    assert Path(str(marker) + '.terminal').read_text() == 'True'
    if key == 'q':
        assert process.returncode == 0
        assert int(marker.read_text()) == 60 > sent_at_step
        assert b'show plots' in capture
        assert restore_offset is not None
        assert b'Temperature [K]' in capture[restore_offset:]
    else:
        assert process.returncode == 130
        assert int(marker.read_text()) < 60


def test_narrow_render_keeps_labels_and_long_paths(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'input_file': 'a-long-input-directory/water.extxyz'}) as reporter:
        reporter.start_md(0, 4000, 0.5)
        text = render(reporter, width=32)
        assert 'Input' in text and 'steps' in text
        assert all(len(line) <= 32 for line in text.splitlines())


def test_initialization_failure_restores_started_terminal(monkeypatch):
    error = KeyboardInterrupt()  # a BaseException: the stronger case than an ordinary error
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    original, created = ui._Reporter, []

    def renderer(settings):
        item = original(settings)
        created.append(item)
        return item

    def fail(settings, **_):
        raise error

    monkeypatch.setattr(ui, '_Reporter', renderer)
    monkeypatch.setattr(ui, 'EventJournal', fail)
    with pytest.raises(type(error)) as captured:
        with ui.run_display({}):
            pytest.fail('Initialization must not yield an active run')
    assert captured.value is error
    assert created[0]._closed and not created[0].live.is_started
    assert '\x1b[?25l' in output.getvalue() and '\x1b[?25h' in output.getvalue()
    assert not ui.get_reporter().enabled


class AsciiTerminal(Terminal):
    encoding = 'ascii'

    def write(self, value):
        value.encode('ascii')
        return super().write(value)


@pytest.mark.parametrize('operation', ['md', 'optimization'])
def test_ascii_terminal_handles_startup_progress_and_outcome(monkeypatch, operation):
    output = AsciiTerminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    with ui.run_display({'input_file': 'molecule-α.xyz', 'ensemble': 'nvt', 'md_T': 300,
                         'md_cycles': 5, 'md_steps': 10, 'md_dt': 0.0005}) as reporter:
        if operation == 'optimization':
            reporter.update_info(operation=operation, opt_step=10, opt_total_steps=20,
                                 opt_energy_eV=-39, opt_force_max_eVA=0.03, opt_force_target_eVA=0.01)
        else:
            reporter.start_md(0, 50, 0.5)
            reporter._refresh(force=True)
            reporter.progress(10, temperature=301, potential_energy=-39, energy=-38, time_per_step=0.1)
            reporter.renderer._graphs_hidden = True
        reporter._refresh(force=True)
    assert output.getvalue().isascii()
    # an optimization that missed its force target finishes in amber, not as Complete
    assert 'SO3LR TAURI' in output.getvalue() and ('Complete' in output.getvalue() or 'target not met' in output.getvalue())
    assert '\\u2500' not in output.getvalue() and '\\u2501' not in output.getvalue()


def test_plain_runs_log_greppable_progress_with_wall_clock_eta(tmp_path, monkeypatch, caplog):
    import logging
    ui.configure(display='plain')
    clock = [100.0]
    monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
    with caplog.at_level(logging.INFO, logger='SO3LR'):
        with ui.run_display({'log_file': str(tmp_path / 'run.log')}) as reporter:
            reporter.start_md(1000, 1000, 0.5)
            reporter.progress(1000, temperature=300)
            clock[0] = 110.0
            reporter.progress(1500, temperature=300)
            reporter.checkpoint(1500, tmp_path / 'restart.npz')
            clock[0] += ui.PLAIN_PROGRESS_INTERVAL
            reporter.progress(1900, temperature=300)
            reporter.finish()
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith('Progress ')]
    # 500 steps in 10 s of wall time: 50 steps/s, 0.5 fs -> 2.16 ns/day; 500 steps left -> 10 s.
    assert lines[0] == 'Progress 50.0% · step 500/1,000 · 2.160 ns/day · ETA 00:10'
    assert lines[1].startswith('Progress 90.0% · step 900/1,000 · ')
    assert len(lines) == 2


def test_interrupted_plain_run_prints_checkpoint_and_resume_hint(tmp_path, monkeypatch, capsys):
    ui.configure(display='plain')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with ui.run_display({'log_file': str(tmp_path / 'run.log'), 'output_file': 'trajectory.extxyz',
                             'restart_save_path': 'restart.npz'}) as reporter:
            reporter.start_md(0, 1000, 0.5)
            reporter.progress(0)
            reporter.checkpoint(500, tmp_path / 'restart.npz')
            raise KeyboardInterrupt
    err = capsys.readouterr().err
    assert 'Last checkpoint: step 500 · restart.npz' in err
    assert ('Resume: --restart-load restart.npz (YAML: restart_load_path: restart.npz); '
            'relaxation is skipped and the trajectory is appended.') in err


def test_failed_run_prints_resume_hint_and_explains_missing_checkpoints(tmp_path, capsys):
    ui.configure(display='plain')
    with pytest.raises(RuntimeError):
        with ui.run_display({'log_file': str(tmp_path / 'run.log'), 'output_file': 'traj.hdf5'}) as reporter:
            reporter.start_md(0, 1000, 0.5)
            raise RuntimeError('diverged')
    assert ('No checkpoint was saved. Set restart_save_path (or --restart-save) to make runs resumable.'
            in capsys.readouterr().err)
    with pytest.raises(RuntimeError):
        with ui.run_display({'log_file': str(tmp_path / 'run2.log'), 'output_file': 'traj.hdf5',
                             'restart_save_path': 'r.npz'}) as reporter:
            reporter.start_md(0, 1000, 0.5)
            reporter.checkpoint(100, '/elsewhere/r.npz')
            raise RuntimeError('diverged')
    err = capsys.readouterr().err
    assert 'Last checkpoint: step 100 · /elsewhere/r.npz' in err
    assert 'choose a new output path for HDF5' in err
    ui.configure(display='plain', quiet=True)
    with pytest.raises(RuntimeError):
        with ui.run_display({'log_file': str(tmp_path / 'run3.log')}) as reporter:
            reporter.start_md(0, 1000, 0.5)
            raise RuntimeError('diverged')
    assert 'checkpoint' not in capsys.readouterr().err


def test_optimization_failures_have_no_md_resume_hint(tmp_path, capsys):
    ui.configure(display='plain')
    with pytest.raises(RuntimeError):
        with ui.run_display({'log_file': str(tmp_path / 'run.log')}):
            raise RuntimeError('bad input')
    assert 'checkpoint' not in capsys.readouterr().err


def test_plain_progress_lines_do_not_wait_for_a_checkpoint(tmp_path, monkeypatch, caplog):
    import logging
    ui.configure(display='plain')
    clock = [0.0]
    monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
    with caplog.at_level(logging.INFO, logger='SO3LR'):
        with ui.run_display({'log_file': str(tmp_path / 'run.log')}) as reporter:
            reporter.start_md(0, 100_000, 0.5)
            for index in range(12):  # 22 minutes of samples, no checkpoint at all
                reporter.progress(index * 1000, temperature=300)
                clock[0] += 120
            reporter.finish()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Progress ')]
    assert len(lines) == 2


def test_resume_hint_counts_remaining_cycles_and_avoids_tilde(tmp_path, monkeypatch, capsys):
    ui.configure(display='plain')
    home = tmp_path / 'home'
    (home / 'runs').mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(tmp_path)
    checkpoint = home / 'runs' / 'restart.npz'
    with pytest.raises(KeyboardInterrupt):
        with ui.run_display({'log_file': str(tmp_path / 'run.log'), 'md_steps': 100,
                             'restart_save_path': str(checkpoint)}) as reporter:
            reporter.start_md(1000, 10_000, 0.5)
            reporter.checkpoint(6000, checkpoint)
            raise KeyboardInterrupt
    err = capsys.readouterr().err
    assert '~' not in err
    assert 'runs/restart.npz' in err
    # 5,000 of 10,000 steps remain from step 6,000: 50 cycles of 100 steps.
    assert 'also set --md-cycles 50 (YAML: md_cycles: 50); md_cycles counts from the checkpoint.' in err


def test_task_progress_and_results_render_without_a_run_journal(monkeypatch, tmp_path):
    """eval, tune-ewald and training share the display but keep no journal for so3lr show."""
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    log = tmp_path / 'eval.log'
    log.write_text('started\n')
    with ui.run_display({'input_file': 'data.xyz', 'output_file': None, 'log_file': log},
                        record=False) as reporter:
        reporter.update_info(operation='evaluation')
        reporter.phase('Evaluating')
        reporter.task_progress(5, 20, 'structures')
        text = render(reporter)
        assert '25%' in text and '5/20 structures' in text
        reporter.results('Timing', [('Evaluation', '1.000 s')])
        reporter.results('Timing', [('Evaluation', '2.000 s')])  # same title replaces
        text = render(reporter)
        assert text.count('Timing') == 1 and 'Evaluation  2.000 s' in text
        reporter.phase('Saving', 'Writing predictions')
        assert '5/20 structures' in render(reporter)  # unfinished work keeps its bar
        reporter.task_progress(20, 20, 'structures')
        reporter.phase('Done')
        assert 'structures' not in render(reporter)  # a finished bar leaves with its phase
    assert reporter.journal.path is None
    assert reporter.artifacts == [('Log', str(log))]
    assert sorted(path.name for path in tmp_path.iterdir()) == ['eval.log']


def test_task_rate_starts_after_the_first_unit_and_restarts_with_the_task(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    clock = [0.0]
    monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
    with ui.run_display({}, record=False) as reporter:
        reporter.phase('Compiling')
        reporter.task_progress(0, 10, 'optimization steps')
        clock[0] = 60.0                                      # the first step compiles
        reporter.task_progress(1, 10, 'optimization steps')
        assert reporter._task_origin == (60.0, 1) and reporter._eta() is None
        reporter.phase('Validating')                         # the bar and its clock stay
        clock[0] = 62.0
        reporter.task_progress(3, 10, 'optimization steps')
        assert reporter._eta() == (7.0, False)               # 1 step/s, not 3 steps per 62 s
        reporter.task_progress(0, 10, 'optimization steps')  # a new optimization
        assert reporter._task_origin is None and reporter._task_started == 62.0
        reporter.task_progress(1, 5, 'candidates')           # another task starts its own clock
        assert reporter._task_origin is None
        reporter.task_progress(5, 5, 'candidates')
        assert reporter._eta() is None                       # nothing left to estimate


def test_live_optimization_draws_graphs_once_three_cycles_are_known(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'input_file': 'water.xyz'}) as reporter:
        reporter.update_info(operation='optimization', opt_step=0, opt_total_steps=30,
                             opt_energy_eV=-10.0, opt_force_max_eVA=1.0, opt_force_target_eVA=0.05)
        reporter.update_info(opt_step=10, opt_energy_eV=-10.5, opt_force_max_eVA=0.3)
        assert 'Maximum force [' not in render(reporter, height=60)
        reporter.update_info(opt_step=20, opt_energy_eV=-10.6, opt_force_max_eVA=0.08)
        text = render(reporter, height=60)
        assert 'Potential energy [eV]' in text and 'Target 0.05 eV/Å' in text
        assert 'Maximum force [' not in render(reporter, height=20)  # no room: numbers only


def _plain_lines(output):
    return [line for line in output.getvalue().splitlines() if line]


def test_closing_line_summarizes_md_and_names_the_output(monkeypatch, tmp_path):
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='plain')
    trajectory = tmp_path / 'water.xyz'
    trajectory.write_text('1\n\nH 0 0 0\n')
    with ui.run_display({'output_file': str(trajectory)}, record=False) as reporter:
        reporter.start_md(1000, 200, 0.5)
        reporter.progress(1200, temperature=300.0, potential_energy=-1.0)
    level, head, details = reporter.verdict()
    assert (level, head) == ('complete', 'MD finished in 00:00')
    assert details == ['200 steps', 'trajectory water.xyz']  # relative to the working directory
    assert _plain_lines(output)[-1] == '✓ MD finished in 00:00 · 200 steps · trajectory water.xyz'


@pytest.mark.parametrize('force,level', [(0.004, 'complete'), (0.02, 'warning')])
def test_optimization_is_green_only_when_the_force_target_is_met(monkeypatch, force, level):
    monkeypatch.setattr(sys, 'stderr', io.StringIO())
    ui.configure(display='plain')
    with ui.run_display({}, record=False) as reporter:
        reporter.update_info(operation='optimization', opt_step=400, opt_total_steps=400,
                             opt_energy_eV=-1.0, opt_force_max_eVA=force, opt_force_target_eVA=0.005)
    verdict = reporter.verdict()
    assert verdict[0] == level
    assert verdict[1].endswith('force target not met') is (level == 'warning')
    assert '400 optimization steps' in verdict[2]


def test_commands_add_details_and_warnings_to_the_closing_line(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', io.StringIO())
    ui.configure(display='plain')
    with ui.run_display({}, record=False) as reporter:
        reporter.update_info(operation='training')
        reporter.task_progress(6, 6, 'epochs')
        reporter.outcome('validation loss 0.1', key='validation')
        reporter.outcome('validation loss 0.01', key='validation')  # replaced, not repeated
    assert reporter.verdict() == ('complete', 'Training finished in 00:00', ['6 epochs', 'validation loss 0.01'])
    with ui.run_display({}, record=False) as reporter:
        reporter.update_info(operation='tuning')
        reporter.outcome('smallest force error 0.02 eV/Å', warning='no candidate meets the 0.001 eV/Å target')
    level, head, _ = reporter.verdict()
    assert level == 'warning' and head.endswith('no candidate meets the 0.001 eV/Å target')


@pytest.mark.parametrize('error,level,head', [
    (ValueError('bad box'), 'failed', 'MD stopped at step 1,100 after 00:00'),
    (KeyboardInterrupt(), 'interrupted', 'MD interrupted at step 1,100 after 00:00'),
])
def test_closing_line_after_a_failure(monkeypatch, error, level, head):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    with pytest.raises(type(error)):
        with ui.run_display({}, record=False) as reporter:
            reporter.start_md(1000, 200, 0.5)
            reporter.progress(1100, temperature=300.0, potential_energy=-1.0)
            raise error
    verdict = reporter.verdict()
    assert verdict[:2] == (level, head)
    # the message is on the line for errors; Ctrl+C needs no repeating
    assert verdict[2] == (['ValueError: bad box'] if level == 'failed' else [])
    shown = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', output.getvalue()).splitlines()
    # after the frame, before the resume hint
    index = next(i for i, line in enumerate(shown) if head in line)
    assert shown[index + 1].startswith('No checkpoint was saved')


def test_plain_closing_line_falls_back_to_ascii_for_ascii_logs(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding='ascii')
    monkeypatch.setattr(sys, 'stderr', stream)
    ui.configure(display='plain')
    with ui.run_display({}, record=False):
        pass
    stream.flush()
    assert raw.getvalue().decode('ascii').strip() == '+ Run finished in 00:00'


def _activity_text(activity):
    from rich.console import Console
    output = io.StringIO()
    Console(file=output, width=120, color_system=None).print(activity)
    return output.getvalue()


def test_optimization_eta_is_a_bound_when_a_force_target_can_stop_it_early(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({}, record=False) as reporter:
        clock = [50.0]
        monkeypatch.setattr(ui.time, 'monotonic', lambda: clock[0])
        reporter.update_info(operation='optimization', opt_step=0, opt_total_steps=400,
                             opt_energy_eV=-1.0, opt_force_max_eVA=1.0, opt_force_target_eVA=0.05)
        clock[0] = 55.0
        reporter.update_info(opt_step=50, opt_energy_eV=-1.05, opt_force_max_eVA=0.7)
        clock[0] = 60.0
        reporter.update_info(opt_step=100, opt_energy_eV=-1.1, opt_force_max_eVA=0.5)
        assert reporter._eta() == (30.0, True)
        activity = ui._Activity('Relaxing', 50.0, 60.0, eta=(60.0 + 30.0, True))
        assert 'ETA ≤ 00:30' in _activity_text(activity)


def test_log_banner_records_version_and_time():
    from so3lr import __version__
    from so3lr.cli import so3lr_cli
    banner = so3lr_cli.log_banner()
    assert banner.startswith(so3lr_cli.SO3LR_ASCII)
    assert re.search(rf'◆ SO3LR TAURI · v{re.escape(__version__)} · \d{{4}}-\d\d-\d\d \d\d:\d\d', banner)
