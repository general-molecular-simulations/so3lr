"""Optimization settings must replay through Click without loading a model."""
from contextlib import contextmanager
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import click
from click.testing import CliRunner
import pytest

from so3lr.cli import so3lr_cli as app
from so3lr.cli.optimization_settings import optimization_argv, optimization_keys
from so3lr.md_settings_validation import load_settings_yaml


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / 'input [literal] $(not-a-command).xyz'
    path.write_text('1\n\nHe 0 0 0\n')
    return str(path)


def parse(settings):
    """Use actual options and callbacks with only the final runtime stubbed."""
    received = {}
    def callback(**kwargs):
        received.update(kwargs)
    command = click.Command('opt', params=app.fire_optimization.params, callback=callback)
    argv = optimization_argv(settings, app.fire_optimization)
    result = CliRunner().invoke(command, argv)
    return result, received, argv


def test_resolved_yaml_reaches_actual_optimization_callback_unchanged(monkeypatch, tmp_path, input_file):
    settings_file = tmp_path / 'resolved.yaml'
    settings_file.write_text(f'''operation: optimization
input_file: '{input_file}'
output_file: '{tmp_path / "results [literal].extxyz"}'
log_file: '{tmp_path / "results.log"}'
model_path: so3lr-tauri-s-2606
precision: mixed
engine: standard
force_convergence: 1e-7
min_cycles: 3
min_steps: 7
min_start_dt: 0.025
min_max_dt: 0.2
min_n_min: 4
buffer_size_multiplier_sr: 1.75
buffer_size_multiplier_lr: 2.0
lr_cutoff: 100
dispersion_damping: 3
total_charge: -1
num_unpaired_electrons: 1
charge_embed_total_charge: 0
charge_equilibration_total_charge: -1
kspace_electrostatics: pme
kspace_smearing: 1.1
kspace_spacing: 2.5
kspace_interp_nodes: 4
overwrite: false
flash_mixed: false
flash_mode: auto
flash_devices: 1
flash_kernels: pallas
''')
    settings = load_settings_yaml(settings_file)
    before = copy.deepcopy(settings)
    calls = []
    @contextmanager
    def display(_settings):
        yield SimpleNamespace(update_info=lambda **kwargs: None,
                              journal=SimpleNamespace(active=False))
    monkeypatch.setattr(app.presentation, 'run_display', display)
    monkeypatch.setattr(app, 'setup_logger', lambda *args, **kwargs: None)
    monkeypatch.setattr(app, 'get_hardware_info', lambda: None)
    monkeypatch.setattr(app, 'reserve_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(app, 'perform_min', lambda resolved: calls.append(resolved.copy()))
    argv = optimization_argv(settings, app.fire_optimization)
    result = CliRunner().invoke(app.fire_optimization, argv)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert settings == before
    for key, value in settings.items():
        if key != 'operation':
            assert calls[0][key] == value
    assert calls[0]['output_format'] == 'extxyz'
    assert '--no-overwrite' in argv and '--no-flash-mixed' in argv
    assert not (tmp_path / 'results [literal].extxyz').exists()
    assert not (tmp_path / 'results.log').exists()


def test_aliases_use_real_option_names_and_model_callback(input_file):
    result, values, argv = parse({'input': input_file, 'model': 'so3lr-tauri-m-2606',
                                 'force_convergence': 0.01, 'buffer_size_multiplier_sr': 1.5,
                                 'buffer_size_multiplier_lr': 1.75, 'spin': 2,
                                 'min-start-dt': 0.03})
    assert result.exit_code == 0, result.output
    assert values['model_path'] == 'so3lr-tauri-m-2606'
    assert values['force_conv'] == 0.01
    assert values['buffer_sr'] == 1.5 and values['buffer_lr'] == 1.75
    assert values['num_unpaired_electrons'] == 2
    assert values['min_start_dt'] == 0.03
    assert any(token.startswith('--model=') for token in argv)
    result, _, _ = parse({'input_file': input_file, 'model_path': 'so3lr-model-typo'})
    assert result.exit_code != 0
    assert 'not a bundled name' in result.output


@pytest.mark.parametrize('value,expected', [(True, True), (False, False), ('true', True), ('false', False), (None, None)])
def test_boolean_flags_preserve_positive_negative_and_unspecified(input_file, value, expected):
    result, values, argv = parse({'input_file': input_file, 'flash_mixed': value})
    assert result.exit_code == 0, result.output
    assert values['flash_mixed'] is expected
    if expected is False:
        assert '--no-flash-mixed' in argv
    elif expected is True:
        assert '--flash-mixed' in argv
    else:
        assert not any('flash-mixed' in arg for arg in argv)


def test_null_is_omitted_only_when_it_matches_the_declared_default(input_file):
    result, values, _ = parse({'input_file': input_file, 'model_path': None, 'force_convergence': None})
    assert result.exit_code == 0
    assert values['model_path'] is None and values['force_conv'] is None
    with pytest.raises(click.UsageError, match='null cannot override'):
        optimization_argv({'input_file': input_file, 'precision': None}, app.fire_optimization)
    with pytest.raises(click.UsageError, match='null cannot override'):
        optimization_argv({'input_file': input_file, 'overwrite': None}, app.fire_optimization)


@pytest.mark.parametrize('key,value', [
    ('md_cycles', 100), ('ensemble', 'nve'), ('output_format', 'extxyz'),
    ('unknown_key', None), ('help', True), ('kspace_npt_cycles', 10),
])
def test_unsupported_settings_are_never_silently_dropped(input_file, key, value):
    with pytest.raises(click.UsageError, match=key):
        optimization_argv({'input_file': input_file, key: value}, app.fire_optimization)


def test_inherited_but_unused_null_setting_can_be_omitted(input_file):
    argv = optimization_argv({'input_file': input_file, 'kspace_npt_cycles': None}, app.fire_optimization)
    assert not any('kspace-npt-cycles' in value for value in argv)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 'nan', 'inf', '-inf'])
def test_nonfinite_numbers_are_rejected_before_runtime(input_file, value):
    with pytest.raises(click.UsageError, match='finite'):
        optimization_argv({'input_file': input_file, 'lr_cutoff': value}, app.fire_optimization)


@pytest.mark.parametrize('value', [{}, [], object(), Path('file'), 'bad\x00path'])
def test_nonplain_values_and_nul_are_rejected(input_file, value):
    with pytest.raises(click.UsageError):
        optimization_argv({'input_file': input_file, 'output_file': value}, app.fire_optimization)


def test_conflicting_aliases_error_and_identical_values_coalesce(input_file):
    with pytest.raises(click.UsageError, match='conflicts'):
        optimization_argv({'input_file': input_file, 'force_conv': 0.01, 'force_convergence': 0.02}, app.fire_optimization)
    result, values, argv = parse({'input_file': input_file, 'force_conv': '0.01', 'force_convergence': 0.01})
    assert result.exit_code == 0
    assert values['force_conv'] == 0.01
    assert sum(arg.startswith('--force-conv=') for arg in argv) == 1


def test_values_starting_with_options_remain_literal(input_file):
    result, values, argv = parse({'input_file': input_file, 'output_file': '--help', 'log_file': 'x;$(touch sentinel).log'})
    assert result.exit_code == 0, result.output
    assert values['output_file'] == '--help'
    assert values['log_file'] == 'x;$(touch sentinel).log'
    assert '--output=--help' in argv
    assert all(isinstance(item, str) for item in argv)


@pytest.mark.parametrize('settings', [{'operation': 'optimization'}, {'input_file': None}])
def test_missing_input_cannot_be_mistaken_for_successful_replay(settings):
    with pytest.raises(click.UsageError, match='require.*input_file'):
        optimization_argv(settings, app.fire_optimization)


def test_registry_types_choices_and_defaults_remain_authoritative(input_file):
    result, values, _ = parse({'input_file': input_file})
    assert result.exit_code == 0
    context = click.Context(app.fire_optimization)
    for option in app.fire_optimization.params:
        if option.name != 'input_file':
            default = option.get_default(context)
            if default is getattr(click.core, 'UNSET', None):
                default = None
            assert values[option.name] == default
    with pytest.raises(click.UsageError):
        optimization_argv({'input_file': input_file, 'flash_devices': 0}, app.fire_optimization)
    with pytest.raises(click.UsageError):
        optimization_argv({'input_file': input_file, 'precision': 'half'}, app.fire_optimization)


def test_supported_key_registry_is_shared_with_writer():
    keys = optimization_keys(app.fire_optimization)
    assert {'input_file', 'input', 'model', 'model_path', 'force_convergence',
            'buffer_size_multiplier_sr', 'buffer_size_multiplier_lr', 'flash_mixed'} <= keys
    assert not {'output_format', 'operation', 'help', 'md_T', 'kspace_npt_cycles'} & keys


def test_one_sided_boolean_flag_cannot_silently_change_value():
    command = click.Command('opt', params=[click.Option(['--enabled'], is_flag=True, default=True)])
    with pytest.raises(click.UsageError, match='no flag'):
        optimization_argv({'enabled': False}, command)
    assert optimization_argv({'enabled': True}, command) == ['--enabled']


@pytest.mark.parametrize('settings', [[], {'operation': 'md'}, {1: 'value'}])
def test_invalid_settings_shape_or_operation(settings):
    with pytest.raises(click.UsageError):
        optimization_argv(settings, app.fire_optimization)


def test_helper_and_real_command_metadata_import_without_jax():
    result = subprocess.run([sys.executable, '-c',
        'import sys; from so3lr.cli.optimization_settings import optimization_keys; '
        'from so3lr.cli.so3lr_cli import fire_optimization; optimization_keys(fire_optimization); '
        'assert not any(name.split(".")[0] in ("jax", "flax", "jax_md") for name in sys.modules)'],
        cwd=Path(app.__file__).parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
