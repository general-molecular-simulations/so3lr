"""Shared display labels tolerate incomplete or invalid run metadata."""
import subprocess
import sys

import pytest

from so3lr.cli.formatting import precision_label, throughput_label, terminal_text


def test_mixed_precision_explains_model_and_dynamics():
    assert precision_label('mixed') == 'mixed (float32 model; float64 dynamics)'


def test_optimization_completion_is_separate_from_force_convergence():
    from so3lr.cli.formatting import optimization_status
    info = {'opt_force_target_eVA': 0.02, 'opt_force_max_eVA': 0.1}
    assert optimization_status(info, 'Completed') == 'Step budget reached; force criterion not met'
    info['opt_force_max_eVA'] = 0.01
    assert optimization_status(info, 'Running') == 'Force criterion met'
    assert optimization_status(info, 'Failed') == 'Optimization stopped'
    assert optimization_status({}, 'Completed') == 'Step budget finished (no force target)'


def test_zero_force_target_matches_the_actual_fire_comparison():
    from so3lr.cli.formatting import optimization_status
    info = {'opt_force_target_eVA': 0, 'opt_force_max_eVA': .01}
    assert optimization_status(info, 'Completed') == 'Step budget reached; force criterion not met'
    info['opt_force_max_eVA'] = 0.
    assert optimization_status(info, 'Completed') == 'Force criterion met'
    info['opt_force_target_eVA'] = -1.
    assert optimization_status(info, 'Completed') == 'Step budget reached; force criterion not met'


@pytest.mark.parametrize('field', ['opt_force_target_eVA', 'opt_force_max_eVA'])
@pytest.mark.parametrize('value', [True, False])
def test_boolean_metadata_cannot_report_optimization_convergence(field, value):
    from so3lr.cli.formatting import optimization_status
    info = {'opt_force_target_eVA': .02, 'opt_force_max_eVA': .01, field: value}
    assert optimization_status(info, 'Completed') != 'Force criterion met'


@pytest.mark.parametrize('value', ['float32', 'float64', 'unknown', '', None])
def test_other_precision_labels_are_preserved(value):
    assert precision_label(value) == str(value)


def test_throughput_uses_femtoseconds_and_total_atom_count():
    assert throughput_label(0.004, 0.5, 192) == '10.800 ns/day · 2.08e-05 s/atom/step'
    assert throughput_label(0.004, 0.5, 384) == '10.800 ns/day · 1.04e-05 s/atom/step'


def test_available_rates_are_independent():
    assert throughput_label(0.004, dt_fs=0.5) == '10.800 ns/day'
    assert throughput_label(0.004, atoms=192) == '2.08e-05 s/atom/step'
    assert throughput_label(0.004) == ''


@pytest.mark.parametrize('invalid', [None, 0, -1, float('nan'), float('inf'),
                                   float('-inf'), True, False, 'unknown', {}, [], 1j])
def test_invalid_values_suppress_only_dependent_rates(invalid):
    assert throughput_label(invalid, 0.5, 192) == ''
    assert throughput_label(0.004, invalid, 192) == '2.08e-05 s/atom/step'
    assert throughput_label(0.004, 0.5, invalid) == '10.800 ns/day'


def test_overflow_and_underflow_results_are_not_displayed():
    assert throughput_label(1e-300, 1e300, 100) == '1.00e-302 s/atom/step'
    assert throughput_label(1e300, 1e300, 1e-300) == '0.086 ns/day'
    assert throughput_label(1e-300, 1e-300, 1e300) == '0.086 ns/day'
    assert throughput_label(1e300, 1e-300) == ''
    assert throughput_label(10**1000, 0.5, 192) == ''


def test_formatter_imports_no_numerical_or_terminal_dependencies():
    subprocess.run([sys.executable, '-c',
                    'from so3lr.cli.formatting import precision_label, throughput_label; '
                    'precision_label("mixed"); throughput_label(.004, .5, 192); '
                    'import sys; '
                    'assert not {"jax", "jax_md", "flax", "numpy", "rich"}.intersection(sys.modules)'],
                   check=True)


@pytest.mark.parametrize('ascii_only', [False, True])
def test_terminal_text_escapes_controls_before_literal_rich_rendering(ascii_only):
    import io
    from rich.console import Console
    from rich.text import Text

    source = '[red]molecule-α.xyz\x1b[2J\x1b]0;title\x1b\\\x9b3J\nnext\rline\t\x07'
    output = io.StringIO()
    Console(file=output, force_terminal=True, color_system=None).print(Text(terminal_text(source, ascii_only)))
    rendered = output.getvalue()
    assert '\x1b' not in rendered and '\x9b' not in rendered and '\x07' not in rendered
    assert '\\x1b[2J' in rendered and '\\x9b3J' in rendered
    assert '\\x0anext\\x0dline\\x09\\x07' in rendered
    assert '[red]' in rendered  # no markup interpretation
    assert ('\\u03b1' if ascii_only else 'α') in rendered
    if ascii_only:
        assert rendered.isascii()


def test_terminal_text_preserves_only_explicitly_allowed_layout_newlines():
    source = 'first\nsecond\x1b[2J\rthird\t'
    assert terminal_text(source) == 'first\\x0asecond\\x1b[2J\\x0dthird\\x09'
    assert terminal_text(source, allow_newlines=True) == 'first\nsecond\\x1b[2J\\x0dthird\\x09'
    assert terminal_text('Volume [Å³] · target ≤ 2', ascii_only=True) == 'Volume [A^3] | target <= 2'


def test_prompt_choices_use_the_same_control_escaping():
    import io
    from rich.console import Console
    from so3lr.cli.prompts import selection_view

    output = io.StringIO()
    Console(file=output, width=100, force_terminal=True, color_system=None).print(
        selection_view('Choose molecule', ['[bold]ethanol\x1b[2J\nα.xyz']))
    assert '\x1b' not in output.getvalue()
    assert '[bold]ethanol\\x1b[2J\\x0aα.xyz' in output.getvalue()
