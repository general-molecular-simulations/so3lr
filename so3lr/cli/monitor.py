"""Offline, read-only terminal monitor for SO3LR event journals.

No simulation modules are imported here. Closing this observer never signals,
pauses, or otherwise touches the process writing the journal.
"""
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import time

from .formatting import ascii_only_console, display_path, duration_label, finite_number as _finite, optimization_status, show_command, precision_label, throughput_label, terminal_text
from .terminal import input_mode, live_display, termination_signals
from . import theme


ENERGIES = {
    'potential_energy_eV': 'Potential energy',
    'energy_eV': 'Total energy',
    'conserved_energy_eV': 'Conserved energy',
}
PLOT_FIELDS = ('temperature_K', *ENERGIES, 'volume_A3')
MAX_HISTORY = 2048
MAX_MEAN_BINS = 2048
READ_BUDGET = 4 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024


def _stamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def _checkpoint_step(checkpoint):
    step = _finite(checkpoint.get('step'))
    return f'{int(step):,}' if step is not None and step == int(step) else '?'


def _age(timestamp, now):
    seconds = _age_seconds(timestamp, now)
    if seconds is None:
        return 'unknown'
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m {seconds % 60}s'
    if seconds < 86400:
        return f'{seconds // 3600}h {(seconds % 3600) // 60}m'
    return f'{seconds // 86400}d {(seconds % 86400) // 3600}h'


def _age_seconds(timestamp, now):
    if not isinstance(timestamp, datetime):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (now - timestamp).total_seconds())


def observation_timestamp(state):
    """Age describes the last recorded event, not a remote process heartbeat."""
    return getattr(state, 'event_timestamp', getattr(state, 'sample_timestamp', None))


def stale_after_seconds(state):
    intervals = [value for item in getattr(state, 'sample_intervals', ())
                 if (value := _finite(item)) is not None and value > 0]
    return max(600.0, 20 * median(intervals)) if intervals else 600.0


def reading_history(reader):
    return getattr(reader, 'file_size', 0) > getattr(reader, 'offset', 0)


def observation_status(reader, now):
    """Qualify cached lifecycle evidence without changing its recorded state."""
    if getattr(reader, 'error', ''):
        return 'Read error'
    if reading_history(reader):
        return 'Reading'
    state = reader.state
    status = str(getattr(state, 'status', None) or 'Unknown')
    if status == 'No lifecycle event yet':
        return 'Unknown'
    if status in ('Started', 'Running'):
        age = _age_seconds(observation_timestamp(state), now)
        if age is not None and age > stale_after_seconds(state):
            return 'Stale?'
    return lifecycle_label(state)


def lifecycle_label(state):
    """The recorded state, naming runs stopped on request rather than calling them failed."""
    status = str(getattr(state, 'status', None) or 'Unknown')
    if status == 'Failed' and getattr(state, 'failure_reason', '') in ('interrupt', 'signal'):
        return 'Interrupted'
    return status


def _observation_line(reader, now):
    age = _age(observation_timestamp(reader.state), now)
    qualifier = observation_status(reader, now)
    if qualifier == 'Reading':
        return f'Age {age} · reading history; latest state not read yet'
    if qualifier == 'Read error':
        return f'Age {age} · read error; current job status unknown'
    if qualifier == 'Stale?':
        return f'Age {age} · no recent update; stale? Job status unknown'
    if reader.state.status not in ('Completed', 'Failed'):
        return f'Age {age} · last journal event; current job status unknown'
    return f'Age {age} · last journal event'


def origin_label(state, compact=False):
    host = getattr(state, 'host', '')
    scheduler = getattr(state, 'scheduler', {})
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    job = scheduler.get('SLURM_JOB_ID') or scheduler.get('SLURM_JOBID')
    if not host and not job:
        return ''
    if compact:
        return f'{host or "?"} / {job or "-"}'
    return f'Host {host or "unknown"}' + (f' · Slurm job {job}' if job else '')


class JournalPathError(ValueError):
    """The PATH given to so3lr show is not a journal or journal directory."""


def discover_journals(path, max_depth=3, max_files=128, max_dirs=512):
    """Discover journals in a bounded subtree without following directory links."""
    path = Path(path).expanduser()
    if path.is_file():
        if not path.name.endswith('.events.jsonl'):
            neighbors = discover_journals(path.parent, max_depth=0, max_files=2)
            target = neighbors[0] if len(neighbors) == 1 else path.parent
            raise JournalPathError(f'Expected a *.events.jsonl journal, not {path.name}. '
                             f'Try: so3lr show {show_command(target)}')
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Journal or directory does not exist: {path}')
    found, visited = [], 0
    pending = deque([(path, 0)])
    excluded = {'node_modules', 'site-packages', '__pycache__', 'venv', 'envs'}
    while pending and visited < max_dirs and len(found) < max_files:
        directory, depth = pending.popleft()
        visited += 1
        try:
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= 4096:
                        break
                    if entry.name.startswith('.') or entry.name in excluded:
                        continue
                    try:
                        if entry.is_file(follow_symlinks=False) and entry.name.endswith('.events.jsonl'):
                            found.append(Path(entry.path))
                            if len(found) >= max_files:
                                break
                        elif depth < max_depth and entry.is_dir(follow_symlinks=False):
                            if len(pending) + visited < max_dirs:
                                pending.append((Path(entry.path), depth + 1))
                    except OSError:
                        continue
        except OSError:
            continue
    # Writer activity must not reshuffle row numbers or next/previous navigation.
    return sorted(found, key=str)


def _compact(samples):
    """Retain actual points and every plotted field's extrema in each block."""
    result = []
    for start in range(0, len(samples), 16):
        block = samples[start:start + 16]
        indices = {0, len(block) - 1}
        for key in PLOT_FIELDS:
            valid = [(i, _finite(sample.get(key))) for i, sample in enumerate(block)]
            valid = [(i, value) for i, value in valid if value is not None]
            if valid:
                indices.add(min(valid, key=lambda pair: pair[1])[0])
                indices.add(max(valid, key=lambda pair: pair[1])[0])
        result.extend(block[index] for index in sorted(indices))
    return result


@dataclass
class _TimeBin:
    first: float
    last: float
    total: float
    count: int

    def merge(self, other):
        self.first, self.last = min(self.first, other.first), max(self.last, other.last)
        self.total += other.total
        self.count += other.count


class _LatterHalfMean:
    """A sample mean independent of the extrema retained for graph rendering.

    Keep raw values up to the cap, then use aligned time bins containing sums
    and counts. Doubling bin width bounds memory without preferentially keeping
    extrema. Whole bins contribute exact sums; only a bin crossing the current
    time midpoint requires an estimate, which is explicitly labeled in the UI.
    """
    def __init__(self):
        self.first = self.last = None
        self.exact = []
        self.bins = {}
        self.origin, self.width = 0.0, None

    def _bin(self, timestamp, value):
        index = math.floor((timestamp - self.origin) / self.width)
        item = _TimeBin(timestamp, timestamp, value, 1)
        if index in self.bins:
            self.bins[index].merge(item)
        else:
            self.bins[index] = item
        while len(self.bins) > MAX_MEAN_BINS:
            combined = {}
            for index, item in self.bins.items():
                index //= 2
                if index in combined:
                    combined[index].merge(item)
                else:
                    combined[index] = item
            self.bins = combined
            self.width *= 2

    def add(self, timestamp, value):
        timestamp, value = _finite(timestamp), _finite(value)
        if timestamp is None:
            return
        self.first = timestamp if self.first is None else min(self.first, timestamp)
        self.last = timestamp if self.last is None else max(self.last, timestamp)
        if value is None:
            return
        if self.width is None:
            self.exact.append((timestamp, value))
            if len(self.exact) <= MAX_MEAN_BINS:
                return
            self.origin = self.first
            self.width = (self.last - self.first) / max(1, MAX_MEAN_BINS // 2) or 1.0
            for time_ps, energy in self.exact:
                self._bin(time_ps, energy)
            self.exact = []
        else:
            self._bin(timestamp, value)

    def reference(self):
        if self.first is None:
            return None, False
        midpoint = self.first / 2 + self.last / 2
        if self.width is None:
            values = [value for timestamp, value in self.exact if timestamp >= midpoint]
            return (_finite(sum(values) / len(values)) if values else None), False
        total, count, approximate = 0.0, 0.0, False
        for item in self.bins.values():
            if item.last < midpoint:
                continue
            fraction = 1.0
            if item.first < midpoint:
                # Estimate only the boundary bin, assuming samples and their
                # mean are uniform within its observed time span. Preserve an
                # endpoint contribution when the midpoint equals its last time.
                fraction = (1 + (item.count - 1) * (item.last - midpoint) /
                            (item.last - item.first)) / item.count
                approximate = True
            total += item.total * fraction
            count += item.count * fraction
        return (_finite(total / count) if count else None), approximate


@dataclass
class RunState:
    path: Path
    run_id: str = ''
    sequence: int = -1
    status: str = 'No lifecycle event yet'
    phase: str = ''
    detail: str = ''
    settings: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    host: str = ''
    scheduler: dict = field(default_factory=dict)
    event_timestamp: datetime | None = None
    sample_intervals: deque = field(default_factory=lambda: deque(maxlen=128))
    start_step: int = 0
    total_steps: int = 0
    dt_fs: float | None = None
    samples: list = field(default_factory=list)
    sample_count: int = 0
    sample_timestamp: datetime | None = None
    checkpoint: dict | None = None
    checkpoint_timestamp: datetime | None = None
    message: str = ''
    failure_reason: str = ''      # 'interrupt' or 'signal' when stopped on request
    diagnostic: tuple | None = None
    malformed: int = 0
    unsupported: int = 0
    energy_mean: _LatterHalfMean = field(default_factory=_LatterHalfMean)
    property_means: dict = field(default_factory=lambda: {
        key: _LatterHalfMean() for key in PLOT_FIELDS if key != 'energy_eV'
    })

    def reference(self, key):
        mean = self.energy_mean if key == 'energy_eV' else self.property_means.get(key)
        return mean.reference() if mean is not None else (None, False)

    def consume(self, event):
        if not isinstance(event, dict):
            self.malformed += 1
            return
        if event.get('schema') != 1:
            self.unsupported += 1
            return
        data, kind = event.get('data'), event.get('type')
        run_id, seq = event.get('run_id'), event.get('seq')
        if (not isinstance(data, dict) or not isinstance(kind, str)
                or not isinstance(run_id, str) or not run_id
                or not isinstance(seq, int) or isinstance(seq, bool)):
            self.malformed += 1
            return
        if self.run_id and self.run_id != run_id:
            self.malformed += 1
            return
        if seq <= self.sequence:
            return
        self.run_id, self.sequence = run_id, seq
        if kind == 'run_started':
            self.status = 'Started'
            self.settings = data.get('settings') if isinstance(data.get('settings'), dict) else {}
            self.host = str(data.get('host') or '')
            self.scheduler = data.get('scheduler') if isinstance(data.get('scheduler'), dict) else {}
        elif kind == 'info':
            self.info.update(data)
        elif kind == 'phase':
            self.phase, self.detail = str(data.get('name', '')), str(data.get('detail', ''))
        elif kind == 'md_started':
            start, total = _finite(data.get('start_step')), _finite(data.get('total_steps'))
            if start is None or total is None or total < 0:
                self.malformed += 1
                return
            self.start_step, self.total_steps = int(start), int(total)
            self.dt_fs = _finite(data.get('dt_fs'))
            self.samples, self.sample_count, self.sample_timestamp = [], 0, None
            self.sample_intervals.clear()
            self.energy_mean = _LatterHalfMean()
            self.property_means = {key: _LatterHalfMean() for key in PLOT_FIELDS if key != 'energy_eV'}
            if self.status not in ('Completed', 'Failed'):
                self.status = 'Running'
        elif kind == 'sample':
            if _finite(data.get('time_ps')) is None or _finite(data.get('step')) is None:
                self.malformed += 1
                return
            # Retain only scalar graph/counter fields, not arbitrary large
            # objects a malformed producer could attach to every sample.
            sample = {key: _finite(data.get(key)) for key in ('step', 'segment_step', 'time_ps', 'seconds_per_step', *PLOT_FIELDS)}
            self.samples.append(sample)
            self.energy_mean.add(sample['time_ps'], sample['energy_eV'])
            for key, mean in self.property_means.items():
                mean.add(sample['time_ps'], sample[key])
            self.sample_count += 1
            timestamp = _stamp(event.get('timestamp'))
            if timestamp is not None and self.sample_timestamp is not None:
                interval = (timestamp - self.sample_timestamp).total_seconds()
                if interval > 0:
                    self.sample_intervals.append(interval)
            self.sample_timestamp = timestamp
            if len(self.samples) > MAX_HISTORY:
                self.samples = _compact(self.samples)
        elif kind == 'checkpoint':
            if data.get('path') is not None:
                self.checkpoint = data
                self.checkpoint_timestamp = _stamp(event.get('timestamp'))
        elif kind == 'run_completed':
            self.status, self.message = 'Completed', str(data.get('message', ''))
            self.phase, self.detail = 'Complete', ''
        elif kind == 'run_failed':
            self.status, self.message = 'Failed', str(data.get('message', ''))
            self.failure_reason = str(data.get('reason', ''))
            self.phase, self.detail = 'Failed', ''
        elif kind in ('warning', 'error'):
            self.diagnostic = (kind, str(data.get('message', '')))
        else:
            self.unsupported += 1
            return
        self.event_timestamp = _stamp(event.get('timestamp'))


class JournalReader:
    """Incremental binary tail with bounded reads and partial-line buffering."""
    def __init__(self, path):
        self.path = Path(path)
        self.state = RunState(self.path)
        self.offset = 0
        self.file_size = 0
        self.pending = b''
        self.identity = None
        self.discarding = False
        self.error = ''

    def poll(self, max_bytes=READ_BUDGET):
        try:
            stat = self.path.stat()
            self.file_size = stat.st_size
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.state = RunState(self.path)
                self.offset, self.pending, self.discarding = 0, b'', False
                self.identity = identity
            if stat.st_size == self.offset:
                # Nothing new: skip the open/close (metadata calls on Lustre).
                self.error = ''
                return 0
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                content = stream.read(max(0, int(max_bytes)))
            self.offset += len(content)
            self.error = ''
        except OSError as exc:
            self.error = str(exc)
            return 0
        if not content:
            return 0
        consumed = len(content)
        if self.discarding:
            _, separator, content = content.partition(b'\n')
            if not separator:
                return consumed
            self.discarding = False
        lines = (self.pending + content).split(b'\n')
        self.pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            if len(line) > MAX_LINE_BYTES:
                self.state.malformed += 1
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError, RecursionError):
                self.state.malformed += 1
                continue
            self.state.consume(event)
        if len(self.pending) > MAX_LINE_BYTES:
            self.pending, self.discarding = b'', True
            self.state.malformed += 1
        return consumed

    def snapshot(self):
        """Read up to the current file size; a growing writer cannot trap us."""
        try:
            end = self.path.stat().st_size
        except OSError:
            self.poll()
            return
        # poll checks replacement/truncation before seeking.
        self.poll(min(READ_BUDGET, end))
        while self.offset < end and not self.error:
            if not self.poll(min(READ_BUDGET, end - self.offset)):
                break


def _time_ticks(left, right, columns):
    """Place at most five labeled ticks, always leaving a space between labels."""
    if right <= left:
        return [(0, f'{left:.4g}')]
    for count in range(5, 1, -1):
        ticks = [(round(index * (columns - 1) / (count - 1)),
                  f'{left + (right - left) * index / (count - 1):.4g}')
                 for index in range(count)]
        previous_end = -2
        for position, label in ticks:
            start = max(0, min(columns - len(label), position - len(label) // 2))
            if len(label) > columns or start <= previous_end + 1:
                break
            previous_end = start + len(label) - 1
        else:
            return ticks
    return [(0, f'{left:.4g}')]


def _plot_scale(samples, key, target=None, reference=None, label_budget=24):
    points = [(_finite(row.get('time_ps')), _finite(row.get(key))) for row in samples]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if not points:
        return None
    decimals = 1 if key == 'temperature_K' else 6 if key == 'conserved_energy_eV' else 3
    low, high = min(y for _, y in points), max(y for _, y in points)
    target, reference = _finite(target), _finite(reference)
    guide = target if target is not None else reference
    if guide is not None:
        low, high = min(low, guide), max(high, guide)
    padding = (high - low) * 0.07 if high > low else max(abs(low) * 0.001, 0.001)
    low, high = low - padding, high + padding
    ticks = (high, high / 2 + low / 2, low)
    # Numeric readouts keep their usual precision, but an axis must reveal the
    # variation it magnifies. Increase tick precision before using scientific
    # notation when fixed decimals would make distinct ticks look identical.
    tick_decimals = decimals
    ylabels = [f'{value:.{tick_decimals}f}' for value in ticks]
    while len(set(ylabels)) < len(set(ticks)) and tick_decimals < 16:
        tick_decimals += 1
        ylabels = [f'{value:.{tick_decimals}f}' for value in ticks]
    if max(map(len, ylabels)) > label_budget or len(set(ylabels)) < len(set(ticks)):
        for digits in range(17, 0, -1):
            compact = [f'{value:.{digits}g}' for value in ticks]
            if max(map(len, compact)) <= label_budget:
                ylabels = compact
                break
    return points, decimals, low, high, ylabels


def _plot(samples, key, title, unit, width, target=None, ascii_only=False, rows=7,
          reference=None, reference_approximate=False, label_width=None, time_bounds=None,
          empty_message='No samples for this property yet.'):
    from rich.console import Group
    from rich.text import Text

    from rich.cells import cell_len

    trace = theme.series_style(key)
    heading = Text()
    heading.append(f'{title} [{unit}]', style=theme.STRONG)
    scale = _plot_scale(samples, key, target, reference, label_budget=max(8, width - 10))
    if scale is None:
        return Group(heading, Text(empty_message, style=theme.MUTED))
    points, decimals, low, high, ylabels = scale
    heading.append(f'  {points[-1][1]:.{decimals}f}', style=f'bold {trace}')
    target, reference = _finite(target), _finite(reference)
    caption = None
    if target is not None:
        caption = f'Thermostat target {target:g} {unit}'
    elif reference is not None:
        separator = ' | ' if ascii_only else ' · '
        approximate = ('~ ' if ascii_only else '≈ ') if reference_approximate else ''
        suffix = ' (binned)' if reference_approximate else ''
        caption = f'Reference{separator}latter half: {approximate}{reference:.{decimals}f} {unit}{suffix}'
    if caption is not None:
        # The guide's exact value sits on the heading row when it fits.
        guide_mark = '- ' if ascii_only else '─ '
        gap = width - cell_len(heading.plain) - cell_len(guide_mark + caption)
        if gap >= 2:
            heading.append(' ' * gap)
            heading.append(guide_mark, style=theme.FAINT)
            heading.append(caption, style=theme.MUTED)
            caption = None
    guide = target if target is not None else reference
    left, right = time_bounds or (min(0, min(x for x, _ in points)), max(x for x, _ in points))
    span = right - left or 1.0
    label_width = max(label_width or 0, max(map(len, ylabels)))
    columns = max(8, width - label_width - 2)
    grid = [[0] * columns for _ in range(rows)]
    bits = ((1, 8), (2, 16), (4, 32), (64, 128))
    previous = None
    for sample in samples:
        x, y = _finite(sample.get('time_ps')), _finite(sample.get(key))
        if x is None or y is None:
            previous = None
            continue
        px = min(columns * 2 - 1, max(0, round((x - left) / span * (columns * 2 - 1))))
        py = min(rows * 4 - 1, max(0, round((high - y) / (high - low) * (rows * 4 - 1))))
        # Rasterize straight connections between recorded coordinates. This
        # changes only the visual trace; all numeric readouts remain samples.
        x0, y0 = previous if previous is not None else (px, py)
        dx, dy = abs(px - x0), -abs(py - y0)
        sx, sy = (1 if x0 < px else -1), (1 if y0 < py else -1)
        error = dx + dy
        while True:
            grid[y0 // 4][x0 // 2] |= bits[y0 % 4][x0 % 2]
            if x0 == px and y0 == py:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy
        previous = (px, py)
    target_row = -1
    if guide is not None:
        target_pixel = min(rows * 4 - 1, max(0, round((high - guide) / (high - low) * (rows * 4 - 1))))
        target_row = target_pixel // 4
    result = [heading]
    for row in range(rows):
        label = ylabels[{0: 0, rows // 2: 1, rows - 1: 2}[row]] if row in (0, rows // 2, rows - 1) else ''
        line = Text(label.rjust(label_width), style=theme.MUTED)
        line.append(' |' if ascii_only else ' │', style=theme.FAINT)
        cells = []
        for mask in grid[row]:
            if mask:
                cells.append(('*' if ascii_only else chr(0x2800 + mask), trace))
            elif row == target_row:
                # A quiet, solid line behind the recorded trace. Do not merge
                # guide pixels into orange data cells: that makes the guide
                # change color and invents horizontal pieces in the trace.
                # The exact reference value is retained in the caption;
                # this line is positioned at terminal-character resolution.
                cells.append(('-' if ascii_only else '─', theme.FAINT))
            else:
                cells.append((' ', None))
        # One span per run of equal style, not one per terminal cell.
        for style, run in groupby(cells, key=lambda cell: cell[1]):
            line.append(''.join(char for char, _ in run), style=style)
        result.append(line)
    axis = list(('-' if ascii_only else '─') * columns)
    labels = [' '] * columns
    for position, label in _time_ticks(left, right, columns):
        axis[position] = '+' if ascii_only else '┬'
        start = max(0, min(columns - len(label), position - len(label) // 2))
        labels[start:start + len(label)] = label
    result.append(Text(' ' * label_width + (' +' if ascii_only else ' └') + ''.join(axis), style=theme.FAINT))
    # The time unit sits in the label gutter, under the value axis.
    tick_row = Text('ps'.rjust(label_width) + '  ', style=theme.FAINT)
    tick_row.append(''.join(labels).rstrip(), style=theme.MUTED)
    result.append(tick_row)
    if caption is not None:
        result.append(Text(caption, style=theme.MUTED))
    return Group(*result)


def _plot_specs(state, energy, ascii_only=False):
    ensemble = str(state.settings.get('ensemble', '')).lower()
    target = (state.settings.get('md_T', state.settings.get('temperature'))
              if ensemble in ('nvt', 'nvt-langevin', 'npt') else None)
    energy = energy if energy in ENERGIES else 'energy_eV'
    specs = [('temperature_K', 'Temperature', 'K', target),
             (energy, ENERGIES[energy], 'eV', None)]
    if ensemble == 'npt':
        specs.append(('volume_A3', 'Volume', 'A^3' if ascii_only else 'Å³', None))
    return specs


def _shared_label_width(state, specs, width=80):
    widths = []
    for key, _, _, target in specs:
        reference = state.reference(key)[0] if target is None else None
        scale = _plot_scale(state.samples, key, target, reference, label_budget=max(8, width - 10))
        if scale is not None:
            widths.append(max(map(len, scale[-1])))
    return max(widths, default=0)


def _state_plot(state, key, title, unit, width, target, ascii_only, rows=7, label_width=None):
    reference, approximate = state.reference(key) if target is None else (None, False)
    times = [_finite(row.get('time_ps')) for row in state.samples]
    times = [value for value in times if value is not None]
    bounds = (min(0, min(times)), max(times)) if times else None
    message = ('Not defined for Langevin dynamics (stochastic thermostat).'
               if key == 'conserved_energy_eV' and str(state.settings.get('ensemble')).lower() == 'nvt-langevin'
               else 'No samples for this property yet.')
    return _plot(state.samples, key, title, unit, width, target, ascii_only, rows,
                 reference=reference, reference_approximate=approximate,
                 label_width=label_width, time_bounds=bounds, empty_message=message)


@dataclass(frozen=True)
class LiveProgress:
    """Producer-side progress of an attached run; display state, never journaled."""
    completed: int
    total: int
    step: int = 0
    dt_fs: float | None = None
    elapsed: float | None = None
    eta: float | None = None       # None while it cannot be estimated yet
    running: bool = True
    keyboard: bool = False         # whether view keys are being read

    @property
    def ratio(self):
        return self.completed / self.total if self.total else 0.0

    def steps_text(self):
        return f'{self.ratio:.0%} · {self.completed:,}/{self.total:,} steps'

    def span_text(self):
        if self.dt_fs is None:
            return ''
        return f'{self.completed * self.dt_fs / 1000:.3f} / {self.total * self.dt_fs / 1000:.3f} ps this run'

    def timing_text(self):
        if self.elapsed is None:
            return ''
        text = f'MD elapsed {duration_label(self.elapsed)}'
        if self.running:
            text += f' · ETA {duration_label(self.eta)}' if self.eta is not None else ' · ETA estimating'
        return text


def _attached_header(state, width, ascii_only, live=None):
    """Brand and status, a progress bar, then wall-clock facts: three rows."""
    from rich.cells import cell_len
    from rich.text import Text

    def label(value):
        return terminal_text(value, ascii_only)

    status = lifecycle_label(state)
    glyph = theme.status_glyph(status)
    title = Text(no_wrap=True, overflow='ellipsis')
    title.append(label(f'{theme.BRAND} SO3LR TAURI'), style=theme.ACCENT_BOLD)
    title.append(label(f'  {glyph} {status}' if glyph else f'  {status}'),
                 style=f'bold {theme.status_style(status)}')
    model = state.info.get('model', state.settings.get('model', state.settings.get('model_path', '')))
    details = [str(state.settings.get('ensemble', '')).upper(),
               f"{state.info['atoms']:,} atoms" if isinstance(state.info.get('atoms'), int) else '',
               str(model or '')]
    precision = state.info.get('precision', state.settings.get('precision'))
    if precision is not None:
        details.append(precision_label(precision))
    rows = [title]
    right = label(' · '.join(item for item in details if item))
    gap = width - cell_len(title.plain) - cell_len(right)
    if gap >= 3:
        title.append(' ' * gap)
        title.append(right, style=theme.MUTED)
    elif right:
        rows.append(Text(right, style=theme.MUTED, no_wrap=True, overflow='ellipsis'))
    if live is not None:
        separator = label(' · ')
        rest = separator.join(label(item) for item in (f'{live.completed:,}/{live.total:,} steps', live.span_text()) if item)
        percent = f'{live.ratio:.0%}'
        line = Text(no_wrap=True, overflow='ellipsis')
        bar_width = max(10, min(40, width - cell_len(percent + separator + rest) - 2))
        line.append_text(theme.bar(live.ratio, bar_width, ascii_only))
        line.append('  ')
        line.append(percent, style=theme.STRONG)
        line.append(separator + rest, style=theme.MUTED)
        rows.append(line)
    timing = live.timing_text() if live is not None else ''
    cycle_seconds = _finite(state.samples[-1].get('seconds_per_step')) if state.samples else None
    throughput = throughput_label(cycle_seconds, state.dt_fs, state.info.get('atoms'))
    timing = ' · '.join(item for item in (timing, throughput) if item)
    if timing:
        rows.append(Text(label(timing), style=theme.MUTED, no_wrap=True, overflow='ellipsis'))
    return rows


def _controls(state, attached=False, paged=False, multiple=False, keyboard=False):
    if attached:
        if state.status in ('Completed', 'Failed'):
            if state.info.get('_journal_enabled') is False:
                return 'Saved graphs unavailable: SO3LR_JOURNAL=0. Text log retained.'
            # Escape the path here: the paged view keeps layout newlines.
            return 'View saved graphs: so3lr show ' + terminal_text(show_command(state.path.parent))
        if not keyboard:
            if state.info.get('_journal_enabled') is False:
                return 'Ctrl+C: interrupt MD · saved graphs disabled (SO3LR_JOURNAL=0)'
            return 'Ctrl+C: interrupt MD · more plots: so3lr show <run-folder>'
        return ('↑/↓: plots · ' if paged else '') + 'e: energy · Esc/q: hide plots · Ctrl+C: interrupt MD'
    close = 'Esc/q: close viewer' if state.status in ('Completed', 'Failed') else 'Esc/q: detach'
    if state.info.get('operation') == 'optimization':
        return ('←/→: run · o: overview · ' if multiple else '') + close
    if multiple:
        return ('↑/↓: plots · ' if paged else '') + '←/→: run · o: overview · e: energy · ' + close
    if state.status not in ('Completed', 'Failed'):
        close += ' (observer only)'
    return ('↑/↓: plot page · ' if paged else '') + 'j/k or ←/→: run · e: energy · ' + close


def _render_optimization(reader, selected=0, count=1, ascii_only=False, compact=False, height=None,
                         now=None, static=False):
    """Show recorded optimization measurements without inventing MD samples."""
    from rich.console import Group
    from rich.text import Text as RichText

    def Text(value='', style=''):
        return RichText(terminal_text(value, ascii_only), style=style, no_wrap=compact,
                        overflow=('crop' if ascii_only else 'ellipsis') if compact else 'fold')

    state = reader.state
    now = now or datetime.now(timezone.utc)
    style = theme.status_style(state.status)
    rows, priorities = [], []

    def add(value, style='', priority=20):
        rows.append(Text(value, style))
        priorities.append(priority)

    add(f'{theme.BRAND} SO3LR TAURI  optimization' + (' snapshot' if static else ''), theme.ACCENT_BOLD, 3)
    add(f'Run {selected + 1}/{count} · {reader.path.parent.name}', priority=9)
    add(f'Last reported state: {state.status}', style, 4)
    add(_observation_line(reader, now), theme.WARNING if observation_status(reader, now) == 'Stale?' else theme.MUTED, 2)
    if origin := origin_label(state):
        add(origin, theme.MUTED, 9)
    add('Optimization · latest recorded cycle', theme.MUTED, 13)
    model = state.info.get('model', state.settings.get('model', state.settings.get('model_path')))
    precision = state.info.get('precision', state.settings.get('precision'))
    if model is not None:
        add(f'Model      {model}', priority=11)
    if precision is not None:
        add(f'Precision  {precision_label(precision)}', theme.MUTED, 12)
    if state.phase:
        add(f'Phase      {state.phase}' + (f' · {state.detail}' if state.detail else ''), theme.MUTED, 10)
    step, total = (_finite(state.info.get(key)) for key in ('opt_step', 'opt_total_steps'))
    if step is not None:
        progress = f'{int(step):,}/{int(total):,} steps' if total is not None else f'{int(step):,} steps'
        add('Progress   ' + progress, priority=5)
    energy, force, target = (_finite(state.info.get(key))
                             for key in ('opt_energy_eV', 'opt_force_max_eVA', 'opt_force_target_eVA'))
    if energy is not None:
        add(f'Potential energy  {energy:.6f} eV', theme.WARNING, 8)
    if force is not None:
        add(f'Maximum force     {force:.6g} eV/Å', theme.WARNING, 6)
    if target is not None:
        add(f'Force target      {target:.6g} eV/Å', theme.MUTED, 7)
    criterion = optimization_status(state.info, state.status)
    add(criterion, theme.SUCCESS if criterion == 'Force criterion met' else theme.MUTED, 1)
    if step is None:
        add('Waiting for the first recorded optimization cycle.', theme.MUTED, 2)
    if state.message:
        add(state.message, style, 2 if state.status == 'Failed' else 14)
    if state.diagnostic:
        kind, message = state.diagnostic
        add(f'{kind.upper()}: {message}', theme.DANGER if kind == 'error' else theme.WARNING, 2)
    if reader.error:
        add('Read error: ' + reader.error, theme.WARNING, 2)
    if not compact:
        for key, label in (('output_file', 'Structure path'), ('log_file', 'Log path')):
            if state.settings.get(key):
                add(f'{label}  {state.settings[key]}', theme.MUTED)
        add(f'Journal  {reader.path}', theme.MUTED)
    controls = _controls(state, multiple=count > 1)
    if compact:
        # Quit comes first, so horizontal truncation cannot hide the escape
        # route on very narrow terminals either.
        controls = 'Esc/q: close viewer' if state.status in ('Completed', 'Failed') else 'Esc/q: detach'
        if count > 1:
            controls += ' · ←/→: run · o: overview'
    if not static:
        rows.append(theme.controls(controls, ascii_only, single_line=compact))
        priorities.append(0)
    if compact and height is not None and len(rows) > height:
        # Every compact Text is exactly one row. Keep controls and the honest
        # outcome before optional metadata, while preserving their display order.
        keep = sorted(sorted(range(len(rows)), key=lambda i: (priorities[i], i))[:max(1, int(height))])
        rows = [rows[index] for index in keep]
    return Group(*rows)


def _render_monitor(readers, selected=0, energy='energy_eV', width=80, now=None, ascii_only=False, height=None, attached=False,
                    static=False, live=None):
    """Build the native view after Rich has supplied the available width."""
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text as RichText

    def Text(value='', style=''):
        return RichText(terminal_text(value, ascii_only), style=style)

    now = now or datetime.now(timezone.utc)
    rows = [Text(theme.BRAND + (' SO3LR TAURI  live MD' if attached else ' SO3LR TAURI  snapshot' if static else ' SO3LR TAURI  monitor'), style=theme.ACCENT_BOLD)]
    if not readers:
        rows.append(Text('No *.events.jsonl journals found.' if static else 'Waiting for *.events.jsonl journals.', style=theme.MUTED))
        if not static:
            rows.append(Text('Esc/q: close observer', style=theme.MUTED))
        return Group(*rows)
    reader = readers[selected % len(readers)]
    state = reader.state
    if state.info.get('operation') == 'optimization':
        return _render_optimization(reader, selected, len(readers), ascii_only, now=now, static=static)
    run_name = reader.path.parent.name or reader.path.name.removesuffix('.events.jsonl')
    if state.run_id and run_name.endswith('.' + state.run_id):
        run_name = run_name[:-(len(state.run_id) + 1)]
    short_id = f' · {state.run_id[:8]}' if state.run_id else ''
    rows.append(Text(('Run · ' if attached else f'Run {selected + 1}/{len(readers)} · ') + run_name + short_id))
    status_style = theme.status_style(state.status)
    rows.append(Text(f'{"State" if attached else "Last reported state"}: {state.status}', style=status_style))
    if not attached:
        rows.append(Text(_observation_line(reader, now), style=theme.MUTED))
        if origin := origin_label(state):
            rows.append(Text(origin, style=theme.MUTED))
    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style=theme.MUTED, no_wrap=True)
    metadata.add_column(overflow='fold')
    model = state.info.get('model', state.settings.get('model', state.settings.get('model_path')))
    if model is not None:
        metadata.add_row(Text('Model'), Text(model))
    details = [state.settings.get('ensemble'),
               f"{state.info['atoms']} atoms" if 'atoms' in state.info else None,
               state.info.get('engine')]
    details = [str(value) for value in details if value is not None]
    if details:
        metadata.add_row(Text('Run'), Text(' · '.join(details)))
    if attached:
        for key, label in (('kernels', 'Kernels'), ('devices', 'Device')):
            value = state.info.get(key)
            if isinstance(value, (list, tuple)):
                value = ', '.join(str(item) for item in value)
            if value is not None:
                metadata.add_row(Text(label), Text(value))
    precision = state.info.get('precision', state.settings.get('precision'))
    if precision is not None:
        metadata.add_row(Text('Precision'), Text(precision_label(precision)))
    if state.phase:
        metadata.add_row(Text('Phase'), Text(state.phase + (' · ' + state.detail if state.detail else '')))
    if state.message:
        metadata.add_row(Text('Message'), Text(state.message))
    rows.append(metadata)
    if attached:
        if live is not None:
            rows.extend(Text(text, style=theme.MUTED) for text in
                        (live.steps_text() + f' · step counter {live.step:,}', live.span_text(), live.timing_text())
                        if text)
    elif state.samples:
        last = state.samples[-1]
        step = _finite(last.get('step'))
        segment = _finite(last.get('segment_step'))
        segment = segment if segment is not None else step - state.start_step
        progress = f'Segment {int(segment):,}/{state.total_steps:,} steps' if state.total_steps else f'Segment step {int(segment):,}'
        rows.append(Text(f'{progress} · step counter {int(step):,} · plots show the current segment', style=theme.MUTED))
    else:
        rows.append(Text('No MD samples yet.', style=theme.MUTED))
    rows.append(Text(''))
    specs = _plot_specs(state, energy, ascii_only)
    label_width = _shared_label_width(state, specs, width)
    for index, (key, title, unit, target) in enumerate(specs):
        if index:
            rows.append(Text(''))
        rows.append(_state_plot(state, key, title, unit, max(24, width), target, ascii_only, label_width=label_width))
    rows.append(Text(f'{state.sample_count:,} samples · {len(state.samples):,} retained · lines connect recorded samples', style=theme.MUTED))
    cycle_seconds = _finite(state.samples[-1].get('seconds_per_step')) if state.samples else None
    throughput = throughput_label(cycle_seconds, state.dt_fs, state.info.get('atoms'))
    if throughput:
        rows.append(Text(f'Cycle throughput {throughput} (excludes startup/I/O)', style=theme.MUTED))
    if state.checkpoint:
        rows.append(Text(f"Saved checkpoint · step {_checkpoint_step(state.checkpoint)} · "
                         f"{state.checkpoint['path']} · age {_age(state.checkpoint_timestamp, now)}", style=theme.MUTED))
    artifacts = Table.grid(padding=(0, 2))
    artifacts.add_column(style=theme.MUTED, no_wrap=True)
    artifacts.add_column(overflow='fold')
    for key, label in (('output_file', 'Trajectory path'), ('log_file', 'Log path')):
        if state.settings.get(key):
            artifacts.add_row(Text(label), Text(state.settings[key]))
    if not attached or state.info.get('_journal_enabled') is not False:
        artifacts.add_row(Text('Journal'), Text(reader.path))
    rows.append(artifacts)
    if state.diagnostic:
        kind, message = state.diagnostic
        rows.append(Text(f'{kind.upper()}: {message}', style=theme.DANGER if kind == 'error' else theme.WARNING))
    if reader.error:
        rows.append(Text('Read error: ' + reader.error, style=theme.WARNING))
    if state.malformed or state.unsupported:
        rows.append(Text(f'Skipped {state.malformed} malformed and {state.unsupported} unsupported events.', style=theme.WARNING))
    if not static:
        rows.append(theme.controls(_controls(state, attached, multiple=len(readers) > 1,
                                             keyboard=live is not None and live.keyboard), ascii_only))
    return Group(*rows)


def _paged_monitor(console, options, readers, selected, energy, width, height, page, now, ascii_only, attached=False,
                   static=False, live=None):
    """Pack complete plot blocks into short screens; keep controls visible."""
    from rich.cells import cell_len
    from rich.console import Group
    from rich.text import Text as RichText

    def Text(value='', style='', *, allow_newlines=False):
        return RichText(terminal_text(value, ascii_only, allow_newlines=allow_newlines), style=style, no_wrap=True,
                        overflow='crop' if ascii_only else 'ellipsis')

    def line_count(renderable):
        return len(console.render_lines(renderable, options, pad=False))

    reader = readers[selected % len(readers)]
    state = reader.state
    now = now or datetime.now(timezone.utc)
    name = reader.path.parent.name or reader.path.name.removesuffix('.events.jsonl')
    if state.run_id and name.endswith('.' + state.run_id):
        name = name[:-(len(state.run_id) + 1)]
    if attached:
        header = _attached_header(state, width, ascii_only, live)
    else:
        qualifier = observation_status(reader, now)
        observation = f' · {"Reading history" if qualifier == "Reading" else qualifier}' if qualifier != state.status else ''
        header = [Text(f'{theme.BRAND} SO3LR TAURI · Run {selected + 1}/{len(readers)} · {name}', theme.ACCENT_BOLD),
                  Text(f'Last reported state: {state.status}{observation} · {state.phase}',
                       theme.status_style(state.status))]
        if origin := origin_label(state):
            header.append(Text(origin, theme.MUTED))
    if state.samples and not attached:
        last = state.samples[-1]
        segment = _finite(last.get('segment_step'))
        segment = segment if segment is not None else last['step'] - state.start_step
        progress = f'{int(segment):,}/{state.total_steps:,} steps' if state.total_steps else f'{int(segment):,} steps'
        header.append(Text(f"Age {_age(observation_timestamp(state), now)} · {str(state.settings.get('ensemble', '')).upper()} · {progress}", theme.MUTED))
    elif not attached:
        header.append(Text(f'Age {_age(observation_timestamp(state), now)} · ' + (state.detail or 'Waiting for MD samples.'), theme.MUTED))
    summary = []
    cycle_seconds = _finite(state.samples[-1].get('seconds_per_step')) if state.samples else None
    throughput = throughput_label(cycle_seconds, state.dt_fs, state.info.get('atoms'))
    if throughput and not attached:  # the attached header already shows throughput
        summary.append(f'Cycle {throughput}')
    if state.checkpoint:
        summary.append(f"Checkpoint step {_checkpoint_step(state.checkpoint)} · age {_age(state.checkpoint_timestamp, now)}")
    if not summary and not attached:
        summary.append(f'{state.sample_count:,} recorded samples · lines connect recorded values')
    footer = [Text(' · '.join(summary), theme.MUTED)] if summary else []
    if attached and state.status in ('Completed', 'Failed'):
        # The final frame stays in scrollback: say where the results are.
        saved = [str(state.settings[key]) for key in ('output_file', 'log_file') if state.settings.get(key)]
        if state.checkpoint and state.checkpoint.get('path'):
            saved.append(display_path(state.checkpoint['path']))
        if saved:
            footer.append(Text(terminal_text('Saved: ' + ' · '.join(dict.fromkeys(saved))), theme.MUTED))
    if reader.error or state.diagnostic:
        message = 'Read error: ' + reader.error if reader.error else state.diagnostic[0].upper() + ': ' + state.diagnostic[1]
        footer.append(Text(message, theme.WARNING))
    if state.status == 'Failed' and state.message:
        footer.append(Text(state.message, theme.DANGER))
    keyboard = live is not None and live.keyboard
    controls_text = _controls(state, attached, paged=True, multiple=len(readers) > 1, keyboard=keyboard)
    if (cell_len(terminal_text(controls_text, ascii_only)) > width
            and not (attached and state.status in ('Completed', 'Failed'))):
        if attached:
            controls_text = (('↑/↓: plots · e: energy · Esc/q: hide\n' if keyboard else '')
                             + 'Ctrl+C interrupts MD')
        else:
            close = 'Esc/q: close' if state.status in ('Completed', 'Failed') else 'Esc/q: detach'
            controls_text = ('←/→: run · o: overview\n' if len(readers) > 1 else '') + '↑/↓: plots · e: energy · ' + close
        if ascii_only:
            controls_text = (controls_text.replace('↑/↓', 'up/down').replace('←/→', 'left/right')
                             .replace('energy', 'E').replace('plots', 'plot'))
        if any(cell_len(terminal_text(line, ascii_only)) > width for line in controls_text.splitlines()):
            # Keep the close/hide action complete on narrow terminals, too.
            controls_text = controls_text.replace(' · Esc/q:', '\nEsc/q:')
    controls = theme.controls(controls_text, ascii_only, single_line=True)
    if attached and state.status in ('Completed', 'Failed'):
        # The saved-directory command must stay copyable even for long paths.
        controls.no_wrap, controls.overflow = False, 'fold'
    # Reserve the spacer under the header and a pager row for multi-page views.
    budget = height - line_count(Group(*header, Text(''), *footer, Text('Plot page 1/1'), controls))
    if budget < 7:
        compact = [Text('SO3LR live MD' if attached else 'SO3LR monitor', theme.ACCENT_BOLD)]
        if not attached:
            compact.append(Text(_observation_line(reader, now), theme.MUTED))
            if origin := origin_label(state):
                compact.append(Text(origin, theme.MUTED))
        compact.append(Text('Enlarge terminal for plots, or use so3lr show --once.'))
        if not attached:
            if line_count(controls) > height:
                controls = Text('Esc/q: close observer', theme.MUTED)
            available = max(0, height - line_count(controls))
            # Even a tiny terminal keeps its exit and event age before optional
            # title/origin details. Each candidate is a single cropped row.
            priorities = [3, 0, *([2] if origin_label(state) else []), 1]
            keep = sorted(sorted(range(len(compact)), key=lambda i: priorities[i])[:available])
            compact = [compact[index] for index in keep]
        return Group(*compact, controls)
    specs = _plot_specs(state, energy, ascii_only)
    label_width = _shared_label_width(state, specs, width)
    # Rows a block needs besides its plot rows: heading, axis, tick labels and,
    # when it does not fit on the heading, the guide caption. Measure it.
    overhead = max(line_count(_state_plot(state, key, title, unit, width, target, ascii_only, 2, label_width)) - 2
                   for key, title, unit, target in specs)
    plot_rows = max(2, min(12, budget - overhead))
    if budget >= 2 * (overhead + 3) + 1:  # pairs only when each plot keeps 3+ rows
        # Keep temperature and energy together on one page when possible.
        plot_rows = max(2, min(12, (budget - 2 * overhead - 1) // 2))
    blocks = []
    for key, title, unit, target in specs:
        block = _state_plot(state, key, title, unit, width, target, ascii_only, plot_rows, label_width)
        blocks.append((title, block, line_count(block)))
    pages, current, used = [], [], 0
    for title, block, count in blocks:
        separator = 1 if current else 0
        if current and used + separator + count > budget:
            pages.append(current)
            current, used, separator = [], 0, 0
        current.append((title, block))
        used += separator + count
    if current:
        pages.append(current)
    page_index = page % len(pages)
    content = []
    for _, block in pages[page_index]:
        if content:
            content.append(Text(''))
        content.append(block)
    names = ' + '.join(title for title, _ in pages[page_index])
    # A single page needs no pager; its reserved row separates plots from the footer.
    pager = [Text(f'Plot page {page_index + 1}/{len(pages)} · {names}', theme.MUTED) if len(pages) > 1 else Text('')]
    return Group(*header, Text(''), *content, *footer, *pager, controls)


class _MonitorView:
    def __init__(self, readers, **kwargs):
        self.readers, self.kwargs = readers, kwargs

    def __rich_console__(self, console, options):
        # The terminal can resize between refreshes, and Rich may constrain a
        # renderable more than the caller's requested width (notably TERM=dumb).
        # Calculate graph cell widths here, rather than letting a Table wrap an
        # already rasterized line and detach its points from the axis.
        kwargs = dict(self.kwargs)
        kwargs['width'] = min(kwargs['width'], options.max_width)
        page = kwargs.pop('page')
        full = _render_monitor(self.readers, **kwargs)
        height = kwargs.get('height')
        if height is not None and self.readers and len(console.render_lines(full, options, pad=False)) > height:
            reader = self.readers[kwargs['selected'] % len(self.readers)]
            if reader.state.info.get('operation') == 'optimization':
                yield _render_optimization(reader, kwargs['selected'], len(self.readers), kwargs['ascii_only'],
                                           compact=True, height=height, now=kwargs['now'], static=kwargs['static'])
            else:
                yield _paged_monitor(console, options, self.readers, page=page, **kwargs)
        else:
            yield full


def render_monitor(readers, selected=0, energy='energy_eV', width=80, now=None, ascii_only=False, height=None, page=0, attached=False,
                   static=False, live=None):
    """Return an adaptive Rich view without touching the simulation process."""
    return _MonitorView(readers, selected=selected, energy=energy, width=width,
                        now=now, ascii_only=ascii_only, height=None if static else height, page=page,
                        attached=attached, static=static, live=live)


class _Keyboard(AbstractContextManager):
    """Temporary cbreak input; never takes ownership of another process."""
    def __init__(self):
        self.fd = None
        self.previous = None
        self.pending = ''
        self._escape_since = None

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        if os.name != 'nt':
            import termios
            import tty
            self.previous = termios.tcgetattr(self.fd)
            try:
                tty.setcbreak(self.fd)
            except BaseException:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)
                raise
        return self

    def read(self, timeout=0.1):
        from .prompts import ESCAPE_TIMEOUT, decode_key

        def adapt(key):
            return {'left': 'previous', 'right': 'next', 'ignore': '', '\x1b': 'escape'}.get(key, key)

        if os.name == 'nt':
            import msvcrt
            if not msvcrt.kbhit():
                time.sleep(timeout)
                return ''
            char = msvcrt.getwch()
            if char in ('\x00', '\xe0'):
                char += msvcrt.getwch()
            key, _ = decode_key(char, final=True, literal=True)
            return adapt(key)
        import select
        key, remainder = decode_key(self.pending, literal=True)
        if key is None:
            if self.pending and self._escape_since is None:
                self._escape_since = time.monotonic()
            wait = timeout
            if self._escape_since is not None:
                wait = min(wait, max(0.0, ESCAPE_TIMEOUT - (time.monotonic() - self._escape_since)))
            if select.select([self.fd], [], [], wait)[0]:
                chunk = os.read(self.fd, 32).decode('utf-8', errors='ignore')
                if not chunk:
                    return 'q'
                self.pending += chunk
                key, remainder = decode_key(self.pending, literal=True)
                if key is None:
                    if self._escape_since is None:
                        self._escape_since = time.monotonic()
                    return ''
            elif self._escape_since is not None and time.monotonic() - self._escape_since >= ESCAPE_TIMEOUT:
                key, remainder = decode_key(self.pending, final=True, literal=True)
            else:
                return ''
        # Preserve queued keys. A single terminal read can contain an arrow,
        # energy selection and quit/toggle; each must get its own UI update.
        self.pending = remainder
        self._escape_since = None
        return adapt(key)

    def __exit__(self, *args):
        if self.previous is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)


def show(path, once=False, interval=1.0, no_color=False):
    """Observe a journal/directory. Non-TTY output always produces one snapshot."""
    from rich.console import Console
    from rich.live import Live
    from .overview import render_overview

    interval = _finite(interval)
    if interval is None or interval <= 0:
        raise ValueError('Monitor interval must be a positive number of seconds.')
    console = Console(file=sys.stdout, force_terminal=False if once else None,
                      no_color=no_color or 'NO_COLOR' in os.environ,
                      markup=False, highlight=False)
    paths = discover_journals(path)
    readers = [JournalReader(item) for item in paths]
    interactive = console.is_terminal and sys.stdin.isatty() and os.environ.get('TERM') != 'dumb'
    ascii_only = ascii_only_console(console)
    if once or not interactive:
        for reader in readers:
            reader.snapshot()
        render = render_overview if len(readers) > 1 else render_monitor
        console.print(render(readers, width=console.width, ascii_only=ascii_only, static=True))
        return
    # The overview is an extra navigation stop, not a synthetic run. Keep the
    # selected file stable when new journals appear or old ones are removed.
    selected, energy_index, plot_page = (-1 if len(readers) > 1 else 0), list(ENERGIES).index('energy_eV'), 0
    next_refresh = next_discovery = 0.0
    try:
        with termination_signals(enabled=interactive), input_mode(_Keyboard()) as keyboard, live_display(Live(console=console, auto_refresh=False,
                                           screen=False, transient=False,
                                           redirect_stdout=False, redirect_stderr=False)) as live:
            while True:
                now = time.monotonic()
                if now >= next_discovery:
                    previous = readers[selected].path if readers and selected >= 0 else None
                    was_overview = selected < 0
                    cached = {reader.path: reader for reader in readers}
                    paths = discover_journals(path)
                    readers = [cached.get(item) or JournalReader(item) for item in paths]
                    selected = (-1 if len(readers) > 1 and (was_overview or previous not in paths)
                                else paths.index(previous) if previous in paths else 0)
                    next_discovery = now + max(5.0, interval)
                if now >= next_refresh:
                    if selected < 0:
                        # A shared read budget prevents large experiment
                        # folders from blocking input for one budget per run.
                        budget = max(1, READ_BUDGET // max(1, len(readers)))
                        for reader in readers:
                            reader.poll(max_bytes=budget)
                        view = render_overview(readers, width=console.width,
                                               height=console.height, page=plot_page,
                                               ascii_only=ascii_only)
                    else:
                        if readers:
                            readers[selected].poll()
                        view = render_monitor(readers, selected, list(ENERGIES)[energy_index],
                                              console.width, ascii_only=ascii_only,
                                              height=console.height, page=plot_page)
                    live.update(view, refresh=True)
                    next_refresh = now + interval
                key = keyboard.read(min(0.1, interval))
                if key in ('q', 'Q', '\x04', 'escape'):
                    break
                if key in ('j', 'next', 'k', 'previous') and readers:
                    views = [-1, *range(len(readers))] if len(readers) > 1 else [0]
                    direction = 1 if key in ('j', 'next') else -1
                    selected = views[(views.index(selected) + direction) % len(views)]
                    plot_page = 0
                    next_refresh = 0
                elif key in ('o', 'O') and len(readers) > 1:
                    selected = -1
                    plot_page = 0
                    next_refresh = 0
                elif key in ('e', 'E'):
                    energy_index = (energy_index + 1) % len(ENERGIES)
                    next_refresh = 0
                elif key in ('up', 'down'):
                    plot_page += 1 if key == 'down' else -1
                    next_refresh = 0
    except KeyboardInterrupt:
        pass
