"""Shared display labels tolerate incomplete or invalid run metadata."""
import subprocess
import sys

import pytest

from so3lr.cli.formatting import precision_label, throughput_label


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
