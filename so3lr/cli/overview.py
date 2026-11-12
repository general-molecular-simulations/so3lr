"""Pure, adaptive overview of already-read local run journals."""
from collections import Counter
from datetime import datetime, timezone
import math
import os
from pathlib import Path

from .formatting import finite_number as _number, terminal_text as _literal


def _shorten(value, width, ascii_only):
    """Keep both directory context and a distinguishing journal suffix."""
    from rich.cells import cell_len
    if cell_len(value) <= width:
        return value
    marker = '~' if ascii_only else '…'
    left_budget = max(0, (width - 1) // 2)
    right_budget = max(0, width - 1 - left_budget)
    left, right = '', ''
    for char in value:
        if cell_len(left + char) > left_budget:
            break
        left += char
    for char in reversed(value):
        if cell_len(char + right) > right_budget:
            break
        right = char + right
    return left + marker + right


def _names(states):
    """Disambiguate repeated directories without resolving or reading paths."""
    paths = [Path(getattr(state, 'path', '') or '.') for state in states]
    parents = [path.parent for path in paths]
    counts = Counter(parents)
    try:
        common = Path(os.path.commonpath([str(parent) for parent in parents])) if parents else None
    except ValueError:  # Mixed relative/absolute paths are valid display inputs.
        common = None
    names = []
    for path in paths:
        try:
            directory = str(path.parent.relative_to(common)) if common is not None else str(path.parent)
        except ValueError:
            directory = str(path.parent)
        if directory == '.':
            directory = path.parent.name or str(path.parent)
        if counts[path.parent] > 1:
            stem = path.name.removesuffix('.events.jsonl')
            directory = f'{directory}/{stem}'
        names.append(directory)
    return names, common


def _sample(state):
    samples = getattr(state, 'samples', None)
    if isinstance(samples, (tuple, list)) and samples and isinstance(samples[-1], dict):
        return samples[-1]
    return {}


def _optimization(state):
    info = getattr(state, 'info', {})
    return isinstance(info, dict) and info.get('operation') == 'optimization'


def _steps(state, sample):
    if _optimization(state):
        segment = _number(state.info.get('opt_step'))
        total = _number(state.info.get('opt_total_steps'))
    else:
        segment = _number(sample.get('segment_step'))
        if segment is None:
            step, start = _number(sample.get('step')), _number(getattr(state, 'start_step', None))
            if step is not None and start is not None:
                segment = step - start
        total = _number(getattr(state, 'total_steps', None))
    if segment is not None and (segment < 0 or not math.isfinite(segment)):
        segment = None
    total = total if total is not None and total > 0 else None
    done = f'{int(segment):,}' if segment is not None else '-'
    planned = f'{int(total):,}' if total is not None else '-'
    percentage = segment / total * 100 if segment is not None and total is not None else None
    percentage = f'{percentage:.0f}%' if percentage is not None and math.isfinite(percentage) else '-'
    return f'{done}/{planned}' if total is not None else done, percentage


def _metric(value, width, decimals):
    value = _number(value)
    if value is None:
        return '-'
    label = f'{value:.{decimals}f}'
    return label if len(label) <= width else f'{value:.4g}'


def _age(timestamp, now):
    if not isinstance(timestamp, datetime):
        return '-'
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    seconds = max(0, int((now - timestamp).total_seconds()))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m'
    if seconds < 86400:
        return f'{seconds // 3600}h'
    return f'{seconds // 86400}d'


def _render(readers, width, height, now, ascii_only, page):
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    states = [reader.state for reader in readers]
    has_optimization = any(_optimization(state) for state in states)
    names, folder = _names(states)
    separator = ' | ' if ascii_only else ' · '

    def line(value, style=''):
        return Text(_literal(value, ascii_only), style=style, no_wrap=True,
                    overflow='crop' if ascii_only else 'ellipsis')

    count_label = f'{len(states)} run' + ('' if len(states) == 1 else 's')
    header = [line(f'SO3LR TAURI{separator}Run overview{separator}{count_label}', 'bold yellow')]
    counts = Counter(str(getattr(state, 'status', '') or '-') for state in states)
    waiting = counts['Started'] + counts['No lifecycle event yet']
    classified = sum(counts[status] for status in ('Running', 'Completed', 'Failed')) + waiting
    summary = [f'{counts[status]} {status.lower()}' for status in ('Running', 'Completed', 'Failed')]
    summary.append(f'{waiting} started/waiting' if has_optimization else f'{waiting} waiting for MD')
    if len(states) > classified:
        summary.append(f'{len(states) - classified} other')
    header.append(line('Reported: ' + separator.join(summary), 'dim'))
    if folder is not None:
        header.append(line('Folder: ' + str(folder), 'dim'))

    progress_note = ('Steps: MD segment / optimization iteration; restart offsets excluded.' if has_optimization
                     else 'Steps/%: current segment; restart offsets excluded.')
    footer = [line(progress_note, 'dim')]
    read_errors = sum(bool(getattr(reader, 'error', '')) for reader in readers)
    if read_errors:
        label = 'error' if read_errors == 1 else 'errors'
        footer.append(line(f'{read_errors} journal read {label}; cached values may be outdated.', 'red'))
    controls = ('Esc/q: close | Left/Right: individual runs | Up/Down: overview page' if ascii_only else
                'Esc/q: close · ←/→: individual runs · ↑/↓: overview page')
    if not states:
        return Group(*header, line('No run journals found.'), line(controls, 'dim'))

    # Every table row has a fixed one-line height. Reserve a header, column
    # labels and footer before choosing the page; never cut a row in half.
    overhead = len(header) + len(footer) + 3
    if height is not None and height <= overhead:
        compact = []
        if height >= 3:
            compact.append(header[0])
        if height >= 2:
            compact.append(line('Enlarge the terminal to view run rows.'))
        compact.append(line(controls, 'dim'))
        return Group(*compact)
    capacity = len(states) if height is None else max(1, int(height) - overhead)
    pages = max(1, math.ceil(len(states) / capacity))
    page_index = int(page) % pages
    first, last = page_index * capacity, min(len(states), (page_index + 1) * capacity)

    progress_title = 'Steps' if has_optimization else 'Segment steps'
    columns = [('#', max(2, len(str(len(states)))), 'right'), ('Run', None, 'left')]
    if has_optimization and width >= 76:
        columns.append(('Mode', 4, 'left'))
    columns.extend([('State', 10, 'left'), (progress_title, 15 if width >= 76 else 12, 'right')])
    if width >= 76:
        columns.append(('%', 5, 'right'))
    if width >= 52:
        columns.extend([('T [K]', 7, 'right'), ('Total E [eV]', 12, 'right')])
    if width >= 100:
        columns.append(('Age', 6, 'right'))
    name_width = max(1, width - sum(size or 0 for _, size, _ in columns) - 2 * (len(columns) - 1))
    table = Table(box=None, padding=(0, 1), pad_edge=False, expand=False, show_edge=False,
                  width=width, highlight=False)
    for title, size, alignment in columns:
        table.add_column(Text(title), width=name_width if size is None else size, justify=alignment,
                         no_wrap=True, overflow='crop' if ascii_only else 'ellipsis', header_style='bold')
    for index in range(first, last):
        state = states[index]
        optimization = _optimization(state)
        sample = {} if optimization else _sample(state)
        steps, percent = _steps(state, sample)
        status = str(getattr(state, 'status', None) or '-')
        style = {'Running': 'yellow', 'Completed': 'green', 'Failed': 'red'}.get(status, 'dim')
        if getattr(readers[index], 'error', ''):
            status, style = 'Read error', 'red'
        name = _literal(names[index], ascii_only)
        if has_optimization and width < 76:
            prefix = 'OPT:' if optimization else 'MD:'
            name = prefix + _shorten(name, max(1, name_width - len(prefix)), ascii_only)
        else:
            name = _shorten(name, name_width, ascii_only)
        values = {'#': str(index + 1), 'Run': name,
                  'Mode': 'OPT' if optimization else 'MD',
                  'State': _literal(status, ascii_only), progress_title: steps, '%': percent,
                  'T [K]': _metric(sample.get('temperature_K'), 7, 1),
                  'Total E [eV]': _metric(sample.get('energy_eV'), 12, 6),
                  'Age': _age(getattr(state, 'sample_timestamp', None), now)}
        table.add_row(*(Text(values[title], style=style if title == 'State' else '')
                        for title, _, _ in columns))
    footer.append(line(f'Overview page {page_index + 1}/{pages}{separator}Runs {first + 1}-{last} of {len(states)}', 'dim'))
    footer.append(line(controls, 'dim'))
    return Group(*header, table, *footer)


class _OverviewView:
    def __init__(self, readers, width, height, now, ascii_only, page):
        self.readers, self.width, self.height = readers, width, height
        self.now, self.ascii_only, self.page = now, ascii_only, page

    def __rich_console__(self, console, options):
        # Rich may constrain the width after the caller measures its terminal.
        # Build the table using that final width so data columns stay aligned.
        width = max(1, min(self.width, options.max_width))
        yield _render(self.readers, width, self.height,
                      self.now or datetime.now(timezone.utc), self.ascii_only, self.page)


def render_overview(readers, width=80, height=None, now=None, ascii_only=False, page=0):
    """Render in-memory states in caller order; never poll or read a journal.

    With no height limit, include every supplied reader in one snapshot. A
    bounded height pages rows; positive or negative page indices wrap around.
    """
    return _OverviewView(readers, width, height, now, ascii_only, page)
