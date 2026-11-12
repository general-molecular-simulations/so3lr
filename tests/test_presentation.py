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
    (False, 'dumb', 'rich', False, True),
    (True, 'xterm', 'rich', True, False),
])
def test_display_selection(monkeypatch, tty, term, display, quiet, enabled):
    output = Terminal() if tty else io.StringIO()
    monkeypatch.setattr(sys, 'stderr', output)
    monkeypatch.setenv('TERM', term)
    ui.configure(display=display, quiet=quiet)
    with ui.run_display({}) as reporter:
        assert reporter.enabled is enabled
    assert bool(output.getvalue()) is enabled


def test_module_and_plain_runs_do_not_import_rich_or_jax():
    result = subprocess.run([
        sys.executable, '-c',
        "import sys; from so3lr.cli.presentation import configure, run_display; "
        "configure(display='plain'); "
        "ctx = run_display({}); ctx.__enter__(); ctx.__exit__(None, None, None); "
        "assert not any(m == 'jax' or m.startswith('jax.') or m == 'rich' "
        "or m.startswith('rich.') for m in sys.modules)",
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


def test_resumed_progress_measures_only_new_steps(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'model': 'so3lr-tauri-m'}) as reporter:
        assert 'ETA' not in render(reporter)
        reporter.start_md(10_000, 4_000, 0.5)
        initial = render(reporter)
        assert '0/4,000 steps' in initial
        assert 'ETA estimating' in initial
        assert 'ns/day' not in initial
        assert 'Saved restart' not in initial
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


def test_paths_are_literal_and_saved_only_after_notification(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    path = '/tmp/[red]water[/red].npz'
    with ui.run_display({'input_file': '[bold]input.xyz'}) as reporter:
        assert '[bold]input.xyz' in render(reporter)
        assert 'Saved restart' not in render(reporter)
        reporter.checkpoint(1_000, path)
        assert path in render(reporter)


def test_requested_metadata_becomes_resolved(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'engine': 'auto'}) as reporter:
        assert 'auto (requested)' in render(reporter)
        reporter.update_info(engine='Flash', atoms=300, devices=['GPU 0'])
        text = render(reporter)
        assert 'Flash' in text
        assert '(requested)' not in text
        assert '300' in text
        assert 'GPU 0' in text


def test_startup_shows_configured_run_before_md_starts(monkeypatch):
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'model_path': 'so3lr-tauri-m', 'ensemble': 'nvt',
                         'md_T': 300, 'md_cycles': 40, 'md_steps': 100,
                         'md_dt': 0.0005}) as reporter:
        text = render(reporter)
        assert 'so3lr-tauri-m (requested)' in text
        assert 'NVT · 300 K target' in text
        assert '4,000 steps · 0.5 fs · 2 ps' in text
        assert 'Loading  Initializing runtime' in text


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


def test_console_and_file_logging_are_separate(monkeypatch, tmp_path):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    logger = logging.getLogger('presentation-test')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_path = tmp_path / 'run.log'
    file_handler = logging.FileHandler(file_path)
    logger.handlers = [ui.make_console_handler(), file_handler]
    try:
        with ui.run_display({}):
            logger.info('2000\t300.2\t-12.3')
            logger.warning('Fallback: [standard]')
            logger.error('Debug traceback', extra={'file_only': True})
    finally:
        file_handler.close()
        logger.handlers.clear()
    assert '2000\t300.2' not in output.getvalue()
    assert 'WARNING: Fallback: [standard]' in output.getvalue()
    assert 'Debug traceback' not in output.getvalue()
    assert file_path.read_text() == '2000\t300.2\t-12.3\nFallback: [standard]\nDebug traceback\n'


@pytest.mark.parametrize('quiet,verbose', [(False, False), (True, False), (False, True)])
def test_plain_console_preserves_rows_and_quiet_keeps_warnings(monkeypatch, quiet, verbose):
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='plain', quiet=quiet, verbose=verbose)
    handler = ui.make_console_handler()
    with ui.run_display({}) as reporter:
        reporter.phase('Hidden presentation event')
        handler.handle(logging.makeLogRecord({'msg': '2000\t300.2', 'levelno': logging.INFO}))
        handler.handle(logging.makeLogRecord({'msg': 'Warning remains', 'levelno': logging.WARNING}))
    assert output.getvalue() == ('' if quiet else '2000\t300.2\n') + 'Warning remains\n'


def test_verbose_rich_shows_info(monkeypatch):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich', verbose=True)
    with ui.run_display({}):
        ui.make_console_handler().handle(logging.makeLogRecord({
            'msg': 'Detailed settings [literal]', 'levelno': logging.INFO,
        }))
    assert 'Detailed settings [literal]' in output.getvalue()


@pytest.mark.parametrize('error', [ValueError('Invalid input'), KeyboardInterrupt(), SystemExit(1)])
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


def test_plain_error_is_not_marked_reported():
    ui.configure(display='plain')
    error = ValueError('Bad input')
    with pytest.raises(ValueError):
        with ui.run_display({}):
            raise error
    assert not getattr(error, '_so3lr_reported', False)


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


def test_worker_warning_is_visible_immediately_but_state_updates_on_owner_thread(monkeypatch):
    import json
    import threading
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    with ui.run_display({}) as reporter:
        handler = ui.make_console_handler()
        original = reporter.renderer.remember_log
        rendered_on = []
        def render_record(record):
            rendered_on.append(threading.get_ident())
            original(record)
        monkeypatch.setattr(reporter.renderer, 'remember_log', render_record)
        thread = threading.Thread(target=lambda: handler.handle(logging.makeLogRecord({
            'msg': 'worker diagnostic', 'levelno': logging.WARNING, 'levelname': 'WARNING'})))
        thread.start()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not rendered_on
        assert 'worker diagnostic' in output.getvalue()
        assert reporter.renderer._state.diagnostic is None
        records = [json.loads(line) for line in reporter.journal.path.read_text().splitlines()]
        assert records[-1]['type'] == 'warning'
        reporter.phase('Running')
        assert rendered_on == [threading.get_ident()]
        assert 'worker diagnostic' in output.getvalue()


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
        reporter.phase('Running')
        assert reporter.renderer._state.diagnostic == ('warning', 'worker-299')


@pytest.mark.parametrize('ending', ['complete', 'failed', 'close'])
def test_finalization_warning_stays_in_live_and_journal(monkeypatch, ending):
    import json
    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({}) as reporter:
        handler = ui.make_console_handler()
        shown = []
        monkeypatch.setattr(reporter.renderer, 'print_log', lambda record, message: shown.append(message))
        target = reporter.renderer if ending == 'close' else reporter.run_log
        method = 'close' if ending == 'close' else 'finish'
        original = getattr(target, method)
        def finish(*args):
            if ending != 'close':
                original(*args)
            handler.handle(logging.makeLogRecord(dict(name='SO3LR', msg='summary disk failure',
                levelno=logging.WARNING, levelname='WARNING')))
            if ending == 'close':
                original(*args)
        monkeypatch.setattr(target, method, finish)
        if ending == 'complete':
            reporter.finish()
        elif ending == 'failed':
            reporter.fail('calculation failure')
        else:
            reporter.finish()
            reporter.close()
            monkeypatch.setattr(target, method, original)
        assert shown == ['summary disk failure']
        records = [json.loads(line) for line in reporter.journal.path.read_text().splitlines()]
        warning, terminal = (records[-1], records[-2]) if ending == 'close' else (records[-2], records[-1])
        assert warning['type'] == 'warning'
        assert warning['data']['message'] == 'summary disk failure'
        assert terminal['type'] == ('run_failed' if ending == 'failed' else 'run_completed')


@pytest.mark.parametrize('collect_previous', [False, True])
def test_reused_handler_binds_to_current_invocation(monkeypatch, collect_previous):
    import gc
    import json
    ui.configure(display='plain')
    with ui.run_display({}) as previous:
        handler = ui.make_console_handler()
    old_path = previous.journal.path
    if collect_previous:
        del previous
        gc.collect()
        assert handler._reporter() is None
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
def test_worker_sigint_cannot_leave_main_signal_mask_blocked(tmp_path):
    import json
    code = '''
import json, os, signal, threading
from pathlib import Path
from so3lr.cli.terminal import termination_signals
original = signal.pthread_sigmask
prior = original(signal.SIG_BLOCK, [])
injected = False
def unmask(how, values):
    global injected
    if how == signal.SIG_SETMASK and not injected:
        injected = True
        def interrupt():
            original(signal.SIG_UNBLOCK, {signal.SIGINT})
            os.kill(os.getpid(), signal.SIGINT)
        thread = threading.Thread(target=interrupt)
        thread.start()
        thread.join(2)
    return original(how, values)
try:
    with termination_signals():
        signal.pthread_sigmask = unmask
except KeyboardInterrupt:
    Path('restore.json').write_text(json.dumps({
        'mask_restored': original(signal.SIG_BLOCK, []) == prior,
        'defaults': all(signal.getsignal(n) == signal.SIG_DFL for n in (signal.SIGTERM, signal.SIGHUP))}))
finally:
    signal.pthread_sigmask = original
    original(signal.SIG_SETMASK, prior)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path/'restore.json').read_text()) == {'mask_restored': True, 'defaults': True}


@pytest.mark.skipif(os.name == 'nt', reason='POSIX pending signals and masks')
@pytest.mark.parametrize('signal_name', ['SIGTERM', 'SIGHUP'])
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


@pytest.mark.parametrize('display,quiet', [('plain', False), ('auto', False), ('rich', True), ('rich', False)])
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


@pytest.mark.parametrize('second_signal', ['SIGTERM', 'SIGHUP'])
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
@pytest.mark.parametrize('signal_name', ['SIGTERM', 'SIGHUP'])
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
        assert '50%' in frames[1] and 'Elapsed' in frames[1] and 'ETA' in frames[1]
        assert reporter._state.samples[-1]['energy_eV'] == -37
        assert reporter._state.samples[-1]['conserved_energy_eV'] == -36
        assert 'detach' not in frames[1]


def test_attached_history_is_bounded_and_retains_extrema(monkeypatch):
    from so3lr.cli.monitor import MAX_HISTORY

    monkeypatch.setattr(sys, 'stderr', Terminal())
    ui.configure(display='rich')
    with ui.run_display({'ensemble': 'npt'}) as reporter:
        reporter.start_md(0, 5000, 0.5)
        monkeypatch.setattr(reporter.renderer, '_refresh', lambda **kwargs: None)
        for step in range(5000):
            reporter.progress(step, temperature=1000 if step == 200 else 300,
                              potential_energy=-100 if step == 201 else -1,
                              volume=5000 if step == 202 else 1000)
        assert reporter._state.sample_count == 5000
        assert len(reporter._state.samples) <= MAX_HISTORY
        assert reporter._state.samples[0]['step'] == 0
        assert reporter._state.samples[-1]['step'] == 4999
        assert max(row['temperature_K'] for row in reporter._state.samples) == 1000
        assert min(row['potential_energy_eV'] for row in reporter._state.samples) == -100
        assert max(row['volume_A3'] for row in reporter._state.samples) == 5000


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
    assert '00:00 elapsed' in text and '00:05 elapsed' in text
    assert 'last update 00:05 ago' in text
    assert '%' not in text and 'steps' not in text
    assert len({line[0] for line in text.splitlines()}) >= 3
    assert '\x1b' not in text
    assert all(not ('\u2800' <= char <= '\u28ff') for char in text)
    if not unicode:
        assert text.isascii()


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
        assert 'Elapsed' in first_page and 'ETA' in first_page and '10%' in first_page
        assert 'mixed (float32 model; float64 dynamics)' in first_page
        assert 'ns/day' in first_page and 's/atom/step' in first_page
        assert 'q: hide plots' in first_page and 'Ctrl+C: interrupt MD' in first_page
        keyboard.keys.append('down')
        reporter.progress(200, temperature=300, potential_energy=-39, energy=-38, volume=1010)
        assert 'Volume [Å³]' in render(reporter, height=24)
        keyboard.keys.extend(['up', 'e'])
        reporter.progress(300, temperature=300, potential_energy=-39, energy=-38, conserved_energy=-37, volume=1010)
        reporter.progress(400, temperature=300, potential_energy=-39, energy=-38, conserved_energy=-37, volume=1010)
        assert 'Conserved energy [eV]' in render(reporter, height=24)
        keyboard.keys.append('q')
        reporter.progress(500, temperature=300, potential_energy=-39)
        assert reporter.status == 'running'
        assert keyboard.entered == 1 and keyboard.closed == 0
        assert 'Temperature [K]' not in render(reporter)
        assert 'q: show plots' in render(reporter)
        reporter.progress(600, temperature=310, potential_energy=-38, energy=-37)
        assert reporter._state.samples[-1]['step'] == 600
        keyboard.keys.append('q')
        reporter.progress(700, temperature=315, potential_energy=-37, energy=-36, conserved_energy=-35)
        restored = render(reporter)
        assert 'Temperature [K]  315.0' in restored
        assert 'Conserved energy [eV]' in restored
        assert reporter._state.sample_count == 7
        keyboard.keys.append('q')
        reporter.progress(1000, temperature=310, potential_energy=-38)
        assert reporter._state.samples[-1]['step'] == 1000
    assert reporter.status == 'complete' and keyboard.closed == 1
    assert 'so3lr show' in render(reporter)
    assert 'q: show plots' not in render(reporter)


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


@pytest.mark.parametrize('failure', ['interrupt', 'render'])
def test_attached_keyboard_restored_after_interrupt_or_render_failure(monkeypatch, failure):
    from so3lr.cli import monitor

    monkeypatch.setattr(sys, 'stderr', Terminal())
    monkeypatch.setattr(sys, 'stdin', InputTerminal())
    keyboard = FakeKeyboard()
    monkeypatch.setattr(monitor, '_Keyboard', lambda: keyboard)
    ui.configure(display='rich')
    error_type = KeyboardInterrupt if failure == 'interrupt' else RuntimeError
    with pytest.raises(error_type):
        with ui.run_display({}) as reporter:
            reporter.start_md(0, 100, 0.5)
            reporter.progress(0, temperature=300, potential_energy=-39)
            if failure == 'interrupt':
                raise KeyboardInterrupt()
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
            elif key == 'q' and restore_offset is None and b'q: show plots' in capture:
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
        assert b'q: show plots' in capture
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


@pytest.mark.parametrize('error', [KeyboardInterrupt(), RuntimeError('journal initialization failed')])
def test_initialization_failure_restores_started_terminal(monkeypatch, error):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    original, created = ui._Reporter, []

    def renderer(settings):
        item = original(settings)
        created.append(item)
        return item

    def fail(settings):
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


def test_interrupted_run_log_initialization_closes_its_journal(monkeypatch):
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    original, journals = ui.EventJournal, []

    def journal(settings):
        item = original(settings)
        journals.append(item)
        return item

    def fail(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(ui, 'EventJournal', journal)
    monkeypatch.setattr(ui, 'RunLog', fail)
    with pytest.raises(KeyboardInterrupt):
        with ui.run_display({}):
            pass
    assert journals and not journals[0].active
    assert '\x1b[?25h' in output.getvalue()


def test_initial_live_start_failure_restores_cursor(monkeypatch):
    from rich.live import Live
    output = Terminal()
    monkeypatch.setattr(sys, 'stderr', output)
    ui.configure(display='rich')
    original, displays = Live.start, []

    def start(display, *args, **kwargs):
        original(display, *args, **kwargs)
        displays.append(display)
        raise RuntimeError('first refresh failed')

    monkeypatch.setattr(Live, 'start', start)
    with pytest.raises(RuntimeError, match='first refresh failed'):
        with ui.run_display({}):
            pass
    assert displays and not displays[0].is_started
    assert '\x1b[?25h' in output.getvalue()


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
    assert 'SO3LR TAURI' in output.getvalue() and 'Complete' in output.getvalue()
    assert '\\u2500' not in output.getvalue() and '\\u2501' not in output.getvalue()


def test_ascii_activity_truncates_without_adding_unicode():
    from rich.console import Console
    output = AsciiTerminal()
    console = Console(file=output, width=20, color_system=None)
    console.print(ui._Activity('Compiling a long molecular force field', 0, 0, unicode=False))
    assert output.getvalue().isascii()
