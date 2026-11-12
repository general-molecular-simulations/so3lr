"""Shared terminal labels, without importing the numerical backend."""
import math


def terminal_text(value, ascii_only=False, *, allow_newlines=False):
    """Escape terminal controls, then fold UI glyphs for ASCII-only streams.

    Newlines are data by default, including in paths and diagnostic messages.
    Only layout text constructed by the application should opt into preserving
    them. ESC, carriage returns and other C0/C1 controls are always escaped.
    """
    text = ''.join(char if allow_newlines and char == '\n' else
                   f'\\x{ord(char):02x}' if ord(char) < 32 or 127 <= ord(char) < 160 else char
                   for char in str(value))
    if not ascii_only:
        return text
    for glyph, replacement in (('←/→', 'left/right'), ('↑/↓', 'up/down'),
                               ('Å³', 'A^3'), ('Å', 'A'), ('≤', '<='),
                               ('━', '-'), ('─', '-'), ('┄', '-'), ('╸', '-'), ('·', '|'), ('≈', '~'),
                               ('◆', '*'), ('●', '*'), ('○', 'o'), ('◌', 'o'), ('✓', '+'), ('✕', 'x'), ('■', '#'),
                               ('✦', '*'), ('✧', '*'), ('›', '>'), ('×', 'x')):
        text = text.replace(glyph, replacement)
    return text.encode('ascii', errors='backslashreplace').decode('ascii')


def home_path(path):
    """Abbreviate the home directory as ``~`` for display."""
    import os
    text = str(path)
    home = os.path.expanduser('~').rstrip(os.sep)
    if home and home != '~' and (text == home or text.startswith(home + os.sep)):
        return '~' + text[len(home):]
    return text


def show_command(path):
    """A copyable ``so3lr show`` argument; a leading ``~`` stays unquoted for the shell."""
    import os
    import shlex
    text = home_path(path)
    if text == '~':
        return text
    if text.startswith('~' + os.sep):
        return '~' + os.sep + shlex.quote(text[2:])
    return shlex.quote(text)


def duration_label(seconds):
    """Wall time as mm:ss, h:mm:ss, or 2d 03:04:05 for multi-day runs."""
    value = max(0, int(seconds))
    days, value = divmod(value, 86400)
    hours, value = divmod(value, 3600)
    minutes, seconds = divmod(value, 60)
    if days:
        return f'{days}d {hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{hours:d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


def precision_label(value):
    """Explain the resolved mixed precision used by both MD engines."""
    label = str(value)
    if label == 'mixed':
        return 'mixed (float32 model; float64 dynamics)'
    return label


def finite_number(value):
    """A finite float, or None for missing, non-numeric, boolean or non-finite values."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _positive_number(value):
    number = finite_number(value)
    return number if number is not None and number > 0 else None


def throughput_label(seconds_per_step, dt_fs=None, atoms=None):
    """Available cycle rates, normalized by all simulated atoms when known.

    ``seconds_per_step`` is the existing host timing, not startup or output
    time. Missing inputs suppress only the rate that depends on them.
    """
    seconds = _positive_number(seconds_per_step)
    if seconds is None:
        return ''
    labels = []
    dt = _positive_number(dt_fs)
    if dt is not None:
        ns_per_day = _positive_number(dt * 0.0864 / seconds)
        if ns_per_day is not None:
            labels.append(f'{ns_per_day:.3f} ns/day')
    atom_count = _positive_number(atoms)
    if atom_count is not None:
        seconds_per_atom = _positive_number(seconds / atom_count)
        if seconds_per_atom is not None:
            labels.append(f'{seconds_per_atom:.2e} s/atom/step')
    return ' · '.join(labels)


def optimization_status(info, status):
    """Separate finishing a job from satisfying its recorded force criterion."""
    if str(status).lower() in ('failed', 'stopped'):
        return 'Optimization stopped'
    # FIRE compares against every explicitly supplied finite threshold. Zero
    # is an exact-force target, not the absence of a stopping criterion.
    target = finite_number(info.get('opt_force_target_eVA'))
    force = finite_number(info.get('opt_force_max_eVA'))
    if target is not None and force is not None and 0 <= force <= target:
        return 'Force criterion met'
    step = finite_number(info.get('opt_step'))
    total = finite_number(info.get('opt_total_steps'))
    # A relaxation before MD finishes while the run itself is still running.
    budget_used = step is not None and total is not None and total > 0 and step >= total
    if str(status).lower() in ('complete', 'completed') or budget_used:
        return ('Step budget reached; force criterion not met' if target is not None
                else 'Step budget finished (no force target)')
    return 'Relaxing geometry'
