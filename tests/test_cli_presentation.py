"""Public help and CLI display options, without numerical model execution."""
from click.testing import CliRunner
import pytest

from so3lr.cli.so3lr_cli import cli
from so3lr.cli import presentation


def test_help_discovers_yaml_workflow_and_preserves_executable_name():
    result = CliRunner().invoke(cli, ['--help'], prog_name='so3lr')
    assert result.exit_code == 0
    assert 'Usage: so3lr ' in result.output
    assert 'so3lr --settings run.yaml' in result.output
    assert '--display' in result.output
    assert 'bare so3lr also opens guided setup' in result.output
    assert 'including batch jobs' in result.output
    assert 'molecule.xyz --lr-cutoff 100' in result.output
    assert 'water.xyz' not in result.output


def test_version_is_available_without_model_initialization():
    result = CliRunner().invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert 'SO3LR Tauri' in result.output


def test_full_help_advertises_and_includes_its_additional_examples():
    brief = CliRunner().invoke(cli, ['--help'])
    full = CliRunner().invoke(cli, ['--help-full'])
    assert brief.exit_code == full.exit_code == 0
    assert 'with additional examples' in ' '.join(brief.output.split())
    assert 'More examples' not in brief.output
    assert 'More examples' in full.output
    assert 'so3lr --display plain --settings run.yaml' in full.output


def test_grouped_help_exposes_common_and_advanced_options():
    result = CliRunner().invoke(cli, ['nvt', '--help'])
    assert result.exit_code == 0
    for label in ('Input and output', 'Dynamics', 'Engine and performance', 'Long-range interactions'):
        assert label in result.output
    for option in ('--input', '--temperature', '--flash-kernels', '--kspace-electrostatics'):
        assert option in result.output


def test_display_configuration_does_not_leak_between_invocations():
    result = CliRunner().invoke(cli, ['--display', 'rich', '--quiet', '--help'])
    assert result.exit_code == 0
    assert presentation._configuration.display == 'auto'
    assert not presentation._configuration.quiet


def test_conflicting_verbosity_is_a_usage_error():
    result = CliRunner().invoke(cli, ['--verbose', '--quiet'])
    assert result.exit_code == 2
    assert 'cannot be combined' in result.output


def test_new_commands_are_discoverable_and_help_never_launches(monkeypatch):
    from so3lr.cli import setup
    monkeypatch.setattr(setup, 'should_launch', lambda: True)
    monkeypatch.setattr(setup, 'launch_setup', lambda: (_ for _ in ()).throw(AssertionError('prompted')))
    for args in (['--help'], ['start', '--help'], ['init', '--help'], ['show', '--help']):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, result.output
    overview = CliRunner().invoke(cli, ['--help']).output
    assert 'Prepare and monitor' in overview
    assert 'so3lr start' in overview
    assert 'so3lr init' not in overview
    assert cli.commands['init'].hidden and not cli.commands['start'].hidden
    assert 'so3lr show runs/' in overview


def test_bare_interactive_cli_dispatches_setup(monkeypatch):
    from so3lr.cli import setup
    called = []
    monkeypatch.setattr(setup, 'should_launch', lambda: True)
    monkeypatch.setattr(setup, 'launch_setup', lambda: called.append(True))
    assert CliRunner().invoke(cli).exit_code == 0
    assert called == [True]


def test_show_accepts_existing_folder_and_prints_saved_graphs(tmp_path):
    import json
    events = [
        ('run_started', {'settings': {'ensemble': 'nvt', 'md_T': 300}}),
        ('md_started', {'start_step': 0, 'total_steps': 100, 'dt_fs': 0.5}),
        ('sample', {'step': 100, 'time_ps': 0.05, 'temperature_K': 301, 'energy_eV': -10}),
        ('run_completed', {'message': 'Completed'}),
    ]
    (tmp_path / 'run.events.jsonl').write_text(''.join(json.dumps({
        'schema': 1, 'run_id': 'test-run', 'seq': seq, 'type': kind,
        'timestamp': '2026-01-01T00:00:00Z', 'data': data,
    }) + '\n' for seq, (kind, data) in enumerate(events)))
    result = CliRunner().invoke(cli, ['show', str(tmp_path), '--once'])
    assert result.exit_code == 0, result.output
    assert 'Completed' in result.output and 'Temperature [K]' in result.output
    assert '301.0' in result.output and '\x1b' not in result.output


@pytest.mark.parametrize('fails', [False, True])
def test_opt_owns_display_and_journal_before_initialization(tmp_path, monkeypatch, fails):
    import json
    from so3lr.cli import so3lr_cli
    source = tmp_path / 'input.xyz'
    source.write_text('1\n\nH 0 0 0\n')
    output = tmp_path / 'optimized.xyz'
    calls, captured = [], {}
    def record(name):
        reporter = presentation.get_reporter()
        assert reporter.journal.active
        calls.append(name)
    def reserve(settings, task):
        assert task == 'opt'
        record('reserve')
    def perform(settings):
        record('perform')
        captured.update(settings)
        if fails:
            raise RuntimeError('deliberate optimization failure')
    monkeypatch.setattr(so3lr_cli, 'setup_logger', lambda path: record('log'))
    monkeypatch.setattr(so3lr_cli, 'reserve_gpu', reserve)
    monkeypatch.setattr(so3lr_cli, 'get_hardware_info', lambda: record('hardware'))
    monkeypatch.setattr(so3lr_cli, 'perform_min', perform)
    result = CliRunner().invoke(cli, ['--display', 'plain', 'opt', '--input', str(source),
                                    '--output', str(output), '--min-cycles', '2', '--min-steps', '3'])
    assert result.exit_code == (1 if fails else 0), result.output
    assert calls == ['log', 'reserve', 'hardware', 'perform']
    assert captured['min_cycles'] == 2 and captured['min_steps'] == 3
    assert captured['log_file'] == str(output.with_suffix('.log'))
    journal, = tmp_path.glob('optimized.*.events.jsonl')
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert events[0]['data']['settings']['log_file'] == captured['log_file']
    assert events[0]['data']['settings']['output_format'] == captured['output_format']
    assert events[1]['type'] == 'info' and events[1]['data']['operation'] == 'optimization'
    assert events[-1]['type'] == ('run_failed' if fails else 'run_completed')
    assert presentation.get_reporter() is presentation._null


def test_unmarked_md_yaml_keeps_units_defaults_and_single_read(tmp_path, monkeypatch):
    import yaml
    from so3lr.cli import so3lr_cli
    settings = tmp_path / 'md.yaml'
    settings.write_text(yaml.safe_dump(dict(input_file='input.xyz', dt=0.5)))
    original = so3lr_cli.load_settings_yaml
    reads, calls = [], []
    def load(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(so3lr_cli, 'load_settings_yaml', load)
    monkeypatch.setattr(so3lr_cli, '_execute_md', lambda values, **kwargs: calls.append(values))
    result = CliRunner().invoke(cli, ['--settings', str(settings)])
    assert result.exit_code == 0, result.output
    assert reads == [settings] and len(calls) == 1
    assert calls[0]['md_dt'] == 0.0005 and calls[0]['ensemble'] == 'nvt'


@pytest.mark.parametrize('error', [KeyError('forces'), AssertionError()])
def test_main_preserves_unexpected_exceptions_outside_display(monkeypatch, error):
    from so3lr.cli import so3lr_cli
    def fail():
        raise error
    monkeypatch.setattr(so3lr_cli, 'cli', fail)
    with pytest.raises(type(error)) as caught:
        so3lr_cli.main()
    assert caught.value is error
    assert caught.traceback[-1].name == 'fail'


def test_main_unexpected_command_failure_keeps_terminal_traceback(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        'from so3lr.cli import so3lr_cli as app; '
        'app.cli = lambda: 1 / 0; app.main()'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Traceback (most recent call last)' in result.stderr
    assert 'ZeroDivisionError: division by zero' in result.stderr


def test_main_keeps_traceback_when_display_ownership_does_not_prove_a_saved_trace(tmp_path, monkeypatch):
    from so3lr.cli import so3lr_cli
    presentation.configure(display='plain')
    def fail():
        with presentation.run_display({'log_file': tmp_path / 'run.log'}):
            raise KeyError('forces')
    monkeypatch.setattr(so3lr_cli, 'cli', fail)
    try:
        with pytest.raises(KeyError) as caught:
            so3lr_cli.main()
    finally:
        presentation.reset()
    assert caught.value.args == ('forces',)
    assert caught.traceback[-1].name == 'fail'


@pytest.fixture
def failure_cli(tmp_path, monkeypatch):
    """Exercise the real command wrappers while intercepting numerical execution."""
    import logging
    from so3lr.cli import so3lr_cli as app
    monkeypatch.setattr(app.logger, 'handlers', [])
    monkeypatch.setattr(app.logger, 'propagate', False)
    old_level = app.logger.level
    app.logger.setLevel(logging.INFO)
    log_path = tmp_path / 'run.log'
    source = tmp_path / 'input.xyz'
    source.write_text('1\n\nH 0 0 0\n')
    presentation.configure(display='plain')

    def setup(path):
        app.logger.addHandler(presentation.make_console_handler())
        app.logger.addHandler(logging.FileHandler(path))
        presentation.get_reporter().logger_ready(path)

    def runtime(settings):
        raise AssertionError('runtime failure')

    monkeypatch.setattr(app, 'setup_logger', setup)
    monkeypatch.setattr(app, 'get_hardware_info', lambda: None)
    monkeypatch.setattr(app, 'reserve_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(app, 'log_simulation_settings', lambda *args: None)
    monkeypatch.setattr(app, 'run', runtime)
    monkeypatch.setattr(app, 'perform_min', runtime)

    def invoke(operation):
        if operation == 'md':
            command = lambda: app._execute_md({'log_file': str(log_path)})
        else:
            command = lambda: app.fire_optimization.main(
                ['--input', str(source), '--output', str(tmp_path / 'opt.xyz'),
                 '--log-file', str(log_path)], standalone_mode=False)
        monkeypatch.setattr(app, 'cli', command)
        return app.main()

    yield app, log_path, invoke
    presentation.reset()
    for handler in app.logger.handlers[:]:
        app.logger.removeHandler(handler)
        handler.close()
    app.logger.setLevel(old_level)


@pytest.mark.parametrize(('operation', 'stage', 'preserved'), [
    ('md', 'setup', False), ('opt', 'setup', False),
    ('md', 'banner', True), ('opt', 'banner', True), ('opt', 'metadata', False),
])
def test_preamble_failure_retains_traceback_or_proves_it_was_saved(failure_cli, monkeypatch, capsys,
                                                                operation, stage, preserved):
    app, log_path, invoke = failure_cli
    failure = RuntimeError(f'{stage} preamble failed')
    if stage == 'setup':
        def fail_setup(*args, **kwargs):
            raise failure
        monkeypatch.setattr(app, 'setup_logger', fail_setup)
    elif stage == 'banner':
        original = app.logger.info
        def fail_banner(message, *args, **kwargs):
            if str(message).startswith(app.SO3LR_ASCII):
                raise failure
            return original(message, *args, **kwargs)
        monkeypatch.setattr(app.logger, 'info', fail_banner)
    else:
        original = presentation._RunReporter.update_info
        def fail_metadata(self, **kwargs):
            if kwargs.get('operation') == 'optimization':
                raise failure
            return original(self, **kwargs)
        monkeypatch.setattr(presentation._RunReporter, 'update_info', fail_metadata)
    with pytest.raises(SystemExit if preserved else RuntimeError) as caught:
        invoke(operation)
    if preserved:
        assert caught.value.code == 1
        durable = log_path.read_text()
        assert 'Traceback (most recent call last)' in durable
        assert f'RuntimeError: {stage} preamble failed' in durable
        err = capsys.readouterr().err
        assert f'Error: RuntimeError: {stage} preamble failed' in err
        assert 'Traceback (most recent call last)' not in err
    else:
        assert caught.value is failure
        assert not getattr(failure, '_so3lr_traceback_preserved', False)
        assert caught.traceback[-1].name == ('fail_setup' if stage == 'setup' else 'fail_metadata')


@pytest.mark.parametrize('operation', ['md', 'opt'])
def test_multiline_saved_failure_prints_one_verified_log_pointer(failure_cli, monkeypatch, capsys, operation):
    app, log_path, invoke = failure_cli
    def runtime(settings):
        raise RuntimeError('first line\nsecond line')
    monkeypatch.setattr(app, 'run', runtime)
    monkeypatch.setattr(app, 'perform_min', runtime)
    with pytest.raises(SystemExit) as caught:
        invoke(operation)
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert err.count('Error: RuntimeError: first line') == 1
    assert err.count(f'Full traceback: {log_path.resolve()}') == 1
    assert 'second line' not in err
    assert 'Traceback (most recent call last)' not in err
    assert 'RuntimeError: first line\nsecond line' in log_path.read_text()


def test_optimization_step_budget_completion_remains_a_successful_command(failure_cli, monkeypatch):
    app, log_path, invoke = failure_cli
    def finish_budget(settings):
        presentation.get_reporter().update_info(
            operation='optimization', opt_step=10, opt_total_steps=10,
            opt_force_max_eVA=0.5, opt_force_target_eVA=0.01)
    monkeypatch.setattr(app, 'perform_min', finish_budget)
    assert invoke('opt') is None  # command completion remains success, not convergence


@pytest.mark.parametrize('operation', ['md', 'opt'])
def test_failed_traceback_file_write_keeps_original_exception(failure_cli, monkeypatch, capsys, operation):
    import logging
    app, log_path, invoke = failure_cli
    monkeypatch.setattr(logging, 'raiseExceptions', False)

    class FullDiskHandler(logging.FileHandler):
        def emit(self, record):
            if record.exc_info:
                try:
                    raise OSError('No space left on device')
                except OSError:
                    self.handleError(record)  # logging normally swallows failed writes
                return
            return super().emit(record)

    def setup(path):
        app.logger.addHandler(presentation.make_console_handler())
        app.logger.addHandler(FullDiskHandler(path))
        presentation.get_reporter().logger_ready(path)
    monkeypatch.setattr(app, 'setup_logger', setup)
    with pytest.raises(AssertionError, match='runtime failure') as caught:
        invoke(operation)
    assert not getattr(caught.value, '_so3lr_traceback_preserved', False)
    assert 'Traceback (most recent call last)' not in log_path.read_text()
    assert 'Full traceback:' not in capsys.readouterr().err
    assert not getattr(caught.value, '_so3lr_traceback_path', None)


@pytest.mark.parametrize('operation', ['md', 'opt'])
def test_traceback_rollover_still_produces_one_compact_terminal_error(
        failure_cli, monkeypatch, capsys, operation):
    from logging.handlers import RotatingFileHandler
    app, log_path, invoke = failure_cli
    handler = RotatingFileHandler(log_path, maxBytes=8192, backupCount=1)

    def setup(path):
        app.logger.addHandler(presentation.make_console_handler())
        app.logger.addHandler(handler)
        presentation.get_reporter().logger_ready(path)

    def runtime(settings):
        # Force the next logging record (the traceback) to roll the file.
        handler.stream.write('previous log entry ' * 500 + '\n')
        handler.flush()
        raise AssertionError('runtime failure')

    monkeypatch.setattr(app, 'setup_logger', setup)
    monkeypatch.setattr(app, 'run', runtime)
    monkeypatch.setattr(app, 'perform_min', runtime)
    with pytest.raises(SystemExit) as caught:
        invoke(operation)
    assert caught.value.code == 1
    assert log_path.with_suffix(log_path.suffix + '.1').exists()
    assert 'AssertionError: runtime failure' in log_path.read_text()
    err = capsys.readouterr().err
    assert 'Error: AssertionError: runtime failure' in err
    assert 'Traceback (most recent call last)' not in err


def test_filtered_traceback_does_not_claim_preservation(failure_cli, monkeypatch, capsys):
    import logging
    app, log_path, invoke = failure_cli

    def setup(path):
        handler = logging.FileHandler(path)
        handler.addFilter(lambda record: not record.exc_info)
        app.logger.addHandler(presentation.make_console_handler())
        app.logger.addHandler(handler)
        presentation.get_reporter().logger_ready(path)

    monkeypatch.setattr(app, 'setup_logger', setup)
    with pytest.raises(AssertionError, match='runtime failure') as caught:
        invoke('md')
    assert not getattr(caught.value, '_so3lr_traceback_preserved', False)
    assert 'Traceback (most recent call last)' not in log_path.read_text()
    assert 'Full traceback:' not in capsys.readouterr().err
    assert not getattr(caught.value, '_so3lr_traceback_path', None)


@pytest.mark.parametrize('failure', [KeyboardInterrupt(), SystemExit(7),
                                   presentation._TerminationSignal(15)])
def test_hardware_inspection_does_not_swallow_control_flow(monkeypatch, failure):
    import sys
    from types import SimpleNamespace
    from so3lr.cli import so3lr_cli as app

    def memory_stats():
        raise failure

    device = SimpleNamespace(device_kind='test GPU', memory_stats=memory_stats)
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(
        default_backend=lambda: 'gpu', local_devices=lambda **kwargs: [device]))
    with pytest.raises(type(failure)) as caught:
        app.get_hardware_info()
    assert caught.value is failure
