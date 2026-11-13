"""One visual language for every SO3LR terminal screen.

Styles are plain Rich style strings, so any Console can render them without a
registered theme. Colors are chosen to stay legible on dark and light
backgrounds; ``NO_COLOR`` and ``--no-color`` remove them while keeping bold.
Glyphs used here have ASCII fallbacks in ``formatting.terminal_text``.
"""

ACCENT = '#f0b429'          # amber: brand mark, focus, progress, primary trace
ACCENT_BOLD = f'bold {ACCENT}'
STRONG = 'bold'
MUTED = '#8b93a1'           # labels, units, secondary facts
FAINT = '#5c6370'           # axes, rules, guide lines
SUCCESS = '#3ccf8e'
WARNING = ACCENT            # 'Running' shares the brand amber
DANGER = '#ff5c5c'

# One color per plotted observable, so stacked plots read at a glance.
SERIES = {
    'temperature_K': ACCENT,
    'energy_eV': '#22c3e6',
    'potential_energy_eV': '#22c3e6',
    'conserved_energy_eV': '#22c3e6',
    'kinetic_energy_eV': '#22c3e6',
    'volume_A3': '#9b7bff',
    'opt_energy_eV': '#22c3e6',
}

BRAND = '◆'
STATUS = {
    'running': ('●', WARNING),
    'started': ('○', MUTED),
    'complete': ('✓', SUCCESS),
    'completed': ('✓', SUCCESS),
    'failed': ('✕', DANGER),
    'interrupted': ('■', WARNING),
    'stale?': ('◌', WARNING),
    'read error': ('✕', DANGER),
}


def status_style(status):
    """Color for a lifecycle word such as Running, Completed or Failed."""
    return STATUS.get(str(status).lower(), ('', MUTED))[1]


def status_glyph(status):
    return STATUS.get(str(status).lower(), ('', MUTED))[0]


def series_style(key):
    return SERIES.get(key, ACCENT)


def controls(text, ascii_only=False, *, single_line=False):
    """Style ``key: action · key: action`` hints: keys stand out, actions recede.

    The visible characters are exactly ``text`` (after terminal escaping), so
    copy/paste and plain-text consumers see the same hint. ``single_line``
    truncates instead of wrapping.
    """
    from rich.text import Text
    from .formatting import terminal_text

    result = Text(no_wrap=single_line, overflow=('crop' if ascii_only else 'ellipsis') if single_line else None)
    separator = ' | ' if ascii_only else ' · '
    for line_index, line in enumerate(terminal_text(text, ascii_only, allow_newlines=True).split('\n')):
        if line_index:
            result.append('\n')
        for index, part in enumerate(line.split(separator)):
            if index:
                result.append(separator, style=FAINT)
            key, colon, action = part.partition(': ')
            if colon and key and len(key) <= 12:
                result.append(key, style=STRONG)
                result.append(colon + action, style=MUTED)
            else:
                result.append(part, style=MUTED)
    return result


def bar(ratio, width, ascii_only=False):
    """A progress bar with half-cell resolution: filled accent, faint track.

    Rich's ProgressBar is not a Text, so it cannot share a line with labels,
    and its fill and track look identical without color.
    """
    from rich.text import Text

    ratio = max(0.0, min(1.0, float(ratio)))
    width = max(1, int(width))
    halves = round(ratio * width * 2)
    full, half = divmod(halves, 2)
    result = Text(no_wrap=True)
    if ascii_only:
        result.append('#' * full + ('#' if half else ''), style=ACCENT)
        result.append('-' * (width - full - half), style=FAINT)
        return result
    result.append('━' * full, style=ACCENT)
    if half:
        result.append('╸', style=ACCENT)
    result.append('─' * (width - full - half), style=FAINT)
    return result
