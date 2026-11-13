"""Shared display labels tolerate incomplete or invalid run metadata."""
import pytest

from so3lr.cli.formatting import duration_label, optimization_status, throughput_label, terminal_text


@pytest.mark.parametrize('info,status,label', [
    # Finishing the job is separate from meeting its recorded force criterion.
    ({'opt_force_target_eVA': 0.02, 'opt_force_max_eVA': 0.1}, 'Completed',
     'Step budget reached; force criterion not met'),
    ({'opt_force_target_eVA': 0.02, 'opt_force_max_eVA': 0.01}, 'Running', 'Force criterion met'),
    ({'opt_force_target_eVA': 0.02, 'opt_force_max_eVA': 0.01}, 'Failed', 'Optimization stopped'),
    ({}, 'Completed', 'Step budget finished (no force target)'),
    # Zero is an exact-force target, as in FIRE's own comparison; a negative one is never met.
    ({'opt_force_target_eVA': 0, 'opt_force_max_eVA': .01}, 'Completed',
     'Step budget reached; force criterion not met'),
    ({'opt_force_target_eVA': 0, 'opt_force_max_eVA': 0.}, 'Completed', 'Force criterion met'),
    ({'opt_force_target_eVA': -1., 'opt_force_max_eVA': 0.}, 'Completed',
     'Step budget reached; force criterion not met'),
    # Booleans are not numbers here, so they cannot report convergence.
    ({'opt_force_target_eVA': True, 'opt_force_max_eVA': .01}, 'Completed',
     'Step budget finished (no force target)'),
    ({'opt_force_target_eVA': .02, 'opt_force_max_eVA': False}, 'Completed',
     'Step budget reached; force criterion not met'),
    # A relaxation that used its budget is not shown as still relaxing while MD runs.
    ({'opt_step': 100, 'opt_total_steps': 100, 'opt_force_max_eVA': 0.03}, 'Running',
     'Step budget finished (no force target)'),
    ({'opt_step': 40, 'opt_total_steps': 100, 'opt_force_max_eVA': 0.03}, 'Running', 'Relaxing geometry'),
    ({'opt_step': 100, 'opt_total_steps': 100, 'opt_force_max_eVA': 0.03, 'opt_force_target_eVA': 0.01},
     'Running', 'Step budget reached; force criterion not met'),
])
def test_optimization_status(info, status, label):
    assert optimization_status(info, status) == label


def test_throughput_uses_femtoseconds_and_total_atom_count():
    assert throughput_label(0.004, 0.5, 192) == '10.800 ns/day · 2.08e-05 s/atom/step'
    assert throughput_label(0.004) == ''


@pytest.mark.parametrize('invalid', [None, 0, float('nan'), True, 'unknown', 1j])
def test_invalid_values_suppress_only_dependent_rates(invalid):
    assert throughput_label(invalid, 0.5, 192) == ''
    assert throughput_label(0.004, invalid, 192) == '2.08e-05 s/atom/step'
    assert throughput_label(0.004, 0.5, invalid) == '10.800 ns/day'


def test_overflow_and_underflow_results_are_not_displayed():
    assert throughput_label(1e-300, 1e300, 100) == '1.00e-302 s/atom/step'
    assert throughput_label(1e300, 1e-300) == ''
    assert throughput_label(10**1000, 0.5, 192) == ''


@pytest.mark.parametrize('seconds, label', [(3725, '1:02:05'),
                                            (86400 * 2 + 3 * 3600 + 4 * 60 + 5, '2d 03:04:05')])
def test_duration_label_spells_out_hours_and_days(seconds, label):
    # mm:ss is asserted by every closing line ("finished in 00:00") and ETA test.
    assert duration_label(seconds) == label


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
    # Every UI glyph has a readable ASCII fallback, not a \u escape.
    folded = terminal_text('◆ ● ○ ◌ ✓ ✕ ✦ ✧ › ┄ ╸ ×', ascii_only=True)
    assert folded.isascii() and '\\u' not in folded
