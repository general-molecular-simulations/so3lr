"""Replay optimization YAML through the actual Click command, without JAX.

This module translates settings to an argument vector, never to a shell command.
Click's option declarations remain authoritative for defaults, types, choices,
and the model-validation callback (which runs when the command parses argv).
"""
import math
from collections.abc import Mapping

import click


_INTERNAL_ALIASES = {
    'force_convergence': 'force_conv',
    'buffer_size_multiplier_sr': 'buffer_sr',
    'buffer_size_multiplier_lr': 'buffer_lr',
}
# opt inherits this shared flag, but its callback does not consume it. Replaying
# a non-null value would misleadingly claim an NPT setting affected optimization.
_UNUSED_OPTIONS = {'kspace_npt_cycles'}
_CLICK_UNSET = getattr(click.core, 'UNSET', object())


def _options(command):
    by_name = {option.name: option for option in command.params
               if isinstance(option, click.Option) and option.expose_value
               and option.name != 'help' and not option.is_eager}
    aliases = dict(by_name)
    for option in by_name.values():
        for flag in option.opts:
            if flag.startswith('--'):
                aliases[flag[2:].replace('-', '_')] = option
    for key, name in _INTERNAL_ALIASES.items():
        if name in by_name:
            aliases[key] = by_name[name]
    return aliases


def optimization_keys(command):
    """Supported YAML keys, derived from opt's real Click options.

    ``operation`` is metadata and is deliberately separate. Derived values such
    as output_format are not replayable settings and must not be exported.
    """
    return frozenset(key for key, option in _options(command).items()
                     if option.name not in _UNUSED_OPTIONS)


def _error(key, message):
    raise click.UsageError(f"Invalid optimization setting {key!r}: {message}")


def _scalar(value, key):
    if type(value) not in (str, bool, int, float):
        _error(key, 'expected a plain string, boolean, or finite number.')
    if isinstance(value, float) and not math.isfinite(value):
        _error(key, 'numbers must be finite.')
    if isinstance(value, str) and '\x00' in value:
        _error(key, 'NUL characters are not allowed.')
    return str(value)


def _finite_converted(value, key):
    # A YAML string such as "nan" must not bypass the numeric check when Click
    # converts a float option; string-valued paths named "nan" remain valid.
    if isinstance(value, float) and not math.isfinite(value):
        _error(key, 'numbers must be finite.')
    if isinstance(value, (tuple, list)):
        for item in value:
            _finite_converted(item, key)


def _default(option, context):
    value = option.get_default(context, call=False)
    # Newer Click versions retain a sentinel for unspecified defaults, while
    # command callbacks still receive None for those scalar options.
    return None if value is _CLICK_UNSET else value


def _arguments(option, value, key, context):
    default = _default(option, context)
    if value is None:
        if default is None:
            return None, []
        _error(key, f'null cannot override the command default {default!r}; supply a value or omit the key.')
    flag = next((item for item in option.opts if item.startswith('--')), option.opts[0])
    if option.is_bool_flag:
        converted = click.BOOL.convert(_scalar(value, key), option, context)
        if option.secondary_opts:
            return converted, [flag if converted else option.secondary_opts[0]]
        activation = getattr(option, 'flag_activation_value', option.flag_value)
        if converted == activation:
            return converted, [flag]
        if converted == default:
            return converted, []
        _error(key, 'the command has no flag for this boolean value.')
    if option.is_flag:
        _scalar(value, key)
        if value == getattr(option, 'flag_activation_value', option.flag_value):
            return value, [flag]
        if value == default:
            return value, []
        _error(key, 'the command flag cannot represent this value.')
    if option.multiple:
        if not isinstance(value, (list, tuple)):
            _error(key, 'this repeatable option requires a list of values.')
        entries = value
    else:
        entries = [value]
    converted, arguments = [], []
    for entry in entries:
        if option.nargs == 1:
            raw = _scalar(entry, key)
            tokens = [raw]
        else:
            if not isinstance(entry, (tuple, list)) or len(entry) != option.nargs:
                _error(key, f'expected {option.nargs} values.')
            raw = tuple(_scalar(item, key) for item in entry)
            tokens = list(raw)
        typed = option.type.convert(raw, option, context)
        _finite_converted(typed, key)
        converted.append(typed)
        # The equals form keeps paths beginning with "--" as literal values;
        # no shell quoting, interpolation, or evaluation is involved.
        arguments.extend([flag + '=' + tokens[0], *tokens[1:]])
    return (tuple(converted) if option.multiple else converted[0]), arguments


def optimization_argv(settings, command):
    """Build argv for the supplied opt command; raise Click UsageError on loss.

    The caller must invoke the real command parser with the returned list so
    its callbacks and validation run normally. Settings are never mutated.
    """
    if not isinstance(settings, Mapping):
        raise click.UsageError('Optimization settings must be a mapping.')
    if 'operation' in settings and settings['operation'] != 'optimization':
        _error('operation', "expected 'optimization'.")
    options = _options(command)
    context = click.Context(command)
    selected = {}
    for key, value in settings.items():
        if key == 'operation':
            continue
        if not isinstance(key, str):
            _error(key, 'setting names must be strings.')
        option = options.get(key.replace('-', '_'))
        if option is None:
            _error(key, 'not supported by the optimization command.')
        if option.name in _UNUSED_OPTIONS:
            if value is None:
                continue
            _error(key, 'this MD-only option is not used by optimization.')
        converted, arguments = _arguments(option, value, key, context)
        if option.name in selected:
            previous_key, previous_value, _ = selected[option.name]
            if previous_value != converted:
                _error(key, f'conflicts with {previous_key!r}; use only one spelling.')
            continue
        selected[option.name] = (key, converted, arguments)
    if 'input_file' in options:
        selected_input = selected.get('input_file')
        input_value = (selected_input[1] if selected_input is not None
                       else _default(options['input_file'], context))
        if input_value is None or input_value == '':
            # The interactive opt command displays help without an input.
            # A settings-file replay must not report that as a successful run.
            raise click.UsageError("Optimization settings require 'input_file' (or 'input').")
    # Stable registry order makes saved settings reproducible regardless of
    # YAML key order, while every explicitly supplied value stays explicit.
    return [argument for option in command.params if option.name in selected
            for argument in selected[option.name][2]]
