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
    for args in (['--help'], ['init', '--help'], ['show', '--help']):
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, result.output
    overview = CliRunner().invoke(cli, ['--help']).output
    assert 'Prepare and monitor' in overview
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


def test_resolved_optimization_yaml_reuses_cli_validation_and_reads_once(tmp_path, monkeypatch):
    import yaml
    from so3lr.cli import so3lr_cli
    source = tmp_path / 'input.xyz'
    source.write_text('1\n\nH 0 0 0\n')
    output = tmp_path / 'optimized.xyz'
    settings = tmp_path / 'optimization.yaml'
    settings.write_text(yaml.safe_dump(dict(
        operation='optimization', input_file=str(source), output_file=str(output),
        model_path='so3lr-tauri-m-2606', precision='float64', min_cycles=2, min_steps=3,
        force_convergence=0.05, buffer_size_multiplier_sr=1.5, overwrite=False,
        lr_cutoff=100.0, engine='standard')))
    reads, captured = [], {}
    original = so3lr_cli.load_settings_yaml
    def load(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(so3lr_cli, 'load_settings_yaml', load)
    monkeypatch.setattr(so3lr_cli, 'setup_logger', lambda *args: None)
    monkeypatch.setattr(so3lr_cli, 'reserve_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(so3lr_cli, 'get_hardware_info', lambda: None)
    monkeypatch.setattr(so3lr_cli, 'perform_min', lambda values: captured.update(values))
    result = CliRunner().invoke(cli, ['--display', 'plain', '--settings', str(settings)])
    assert result.exit_code == 0, result.output
    assert reads == [settings]
    assert captured['min_cycles'] == 2 and captured['min_steps'] == 3
    assert captured['force_convergence'] == 0.05 and captured['buffer_size_multiplier_sr'] == 1.5
    assert captured['precision'] == 'float64' and captured['overwrite'] is False
    assert captured['model_path'] == 'so3lr-tauri-m-2606' and captured['engine'] == 'standard'
    assert captured['output_format'] == 'extxyz'
    assert 'md_dt' not in captured


def test_optimization_yaml_model_validation_is_the_normal_click_callback(tmp_path, monkeypatch):
    import yaml
    from so3lr.cli import so3lr_cli
    source = tmp_path / 'input.xyz'
    source.write_text('1\n\nH 0 0 0\n')
    settings = tmp_path / 'invalid-model.yaml'
    settings.write_text(yaml.safe_dump(dict(operation='optimization', input_file=str(source), model='so3lr-mm')))
    monkeypatch.setattr(so3lr_cli, 'perform_min', lambda *args: pytest.fail('Started invalid model'))
    result = CliRunner().invoke(cli, ['--settings', str(settings)])
    assert result.exit_code == 2
    assert 'Did you mean' in result.output


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


def test_unknown_yaml_operation_fails_without_falling_back_to_md(tmp_path, monkeypatch):
    from so3lr.cli import so3lr_cli
    settings = tmp_path / 'unknown.yaml'
    settings.write_text('operation: typo\ninput_file: input.xyz\n')
    monkeypatch.setattr(so3lr_cli, '_execute_md', lambda *args, **kwargs: pytest.fail('Started MD'))
    result = CliRunner().invoke(cli, ['--settings', str(settings)])
    assert result.exit_code == 2
    assert "Unsupported settings operation 'typo'" in result.output


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


def test_main_display_owned_plain_failure_names_exception(tmp_path, monkeypatch, capsys):
    from so3lr.cli import so3lr_cli
    presentation.configure(display='plain')
    def fail():
        with presentation.run_display({'log_file': tmp_path / 'run.log'}):
            raise KeyError('forces')
    monkeypatch.setattr(so3lr_cli, 'cli', fail)
    try:
        with pytest.raises(SystemExit) as caught:
            so3lr_cli.main()
    finally:
        presentation.reset()
    assert caught.value.code == 1
    assert "Error: KeyError: 'forces'" in capsys.readouterr().err


def test_md_short_terminal_error_retains_complete_traceback_in_file(tmp_path, monkeypatch, capsys):
    import logging
    from so3lr.cli import so3lr_cli as app
    logger = app.logger
    old_level = logger.level
    monkeypatch.setattr(logger, 'handlers', [])
    monkeypatch.setattr(logger, 'propagate', False)
    logger.setLevel(logging.INFO)
    log_path = tmp_path / 'run.log'
    presentation.configure(display='plain')
    def setup(path):
        logger.addHandler(presentation.make_console_handler())
        logger.addHandler(logging.FileHandler(path))
        presentation.get_reporter().logger_ready(path)
    def runtime(settings):
        raise AssertionError('runtime failure')
    monkeypatch.setattr(app, 'setup_logger', setup)
    monkeypatch.setattr(app, 'get_hardware_info', lambda: None)
    monkeypatch.setattr(app, 'reserve_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(app, 'log_simulation_settings', lambda *args: None)
    monkeypatch.setattr(app, 'run', runtime)
    monkeypatch.setattr(app, 'cli', lambda: app._execute_md({'log_file': str(log_path)}))
    try:
        with pytest.raises(SystemExit) as caught:
            app.main()
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(old_level)
        presentation.reset()
    assert caught.value.code == 1
    terminal = capsys.readouterr().err
    assert 'AssertionError: runtime failure' in terminal
    assert 'Traceback (most recent call last)' not in terminal
    durable = log_path.read_text()
    assert 'Traceback (most recent call last)' in durable
    assert 'AssertionError: runtime failure' in durable
    assert 'Execution: failed' in durable
