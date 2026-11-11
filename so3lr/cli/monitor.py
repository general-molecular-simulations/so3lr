"""Offline, read-only terminal monitor for SO3LR event journals.

No simulation modules are imported here. Closing this observer never signals,
pauses, or otherwise touches the process writing the journal.
"""
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

from .formatting import optimization_status, precision_label, throughput_label


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


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _stamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def _literal(value):
    """Keep arbitrary paths and messages literal, including terminal controls."""
    text = str(value)
    return ''.join(f'\\x{ord(char):02x}' if ord(char) < 32 or 127 <= ord(char) < 160 else char
                   for char in text)


def _age(timestamp, now):
    if timestamp is None:
        return 'unknown'
    seconds = max(0, int((now - timestamp).total_seconds()))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m {seconds % 60}s'
    if seconds < 86400:
        return f'{seconds // 3600}h {(seconds % 3600) // 60}m'
    return f'{seconds // 86400}d {(seconds % 86400) // 3600}h'


def discover_journals(path, max_depth=3, max_files=128, max_dirs=512):
    """Discover journals in a bounded subtree without following directory links."""
    path = Path(path).expanduser()
    if path.is_file():
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
                            found.append((entry.stat().st_mtime_ns, Path(entry.path)))
                            if len(found) >= max_files:
                                break
                        elif depth < max_depth and entry.is_dir(follow_symlinks=False):
                            if len(pending) + visited < max_dirs:
                                pending.append((Path(entry.path), depth + 1))
                    except OSError:
                        continue
        except OSError:
            continue
    return [item[1] for item in sorted(found, key=lambda item: (item[0], str(item[1])), reverse=True)]


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
    start_step: int = 0
    total_steps: int = 0
    dt_fs: float | None = None
    samples: list = field(default_factory=list)
    sample_count: int = 0
    sample_timestamp: datetime | None = None
    checkpoint: dict | None = None
    checkpoint_timestamp: datetime | None = None
    message: str = ''
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
            self.host = str(data.get('host', ''))
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
            self.sample_timestamp = _stamp(event.get('timestamp'))
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
            self.phase, self.detail = 'Failed', ''
        elif kind in ('warning', 'error'):
            self.diagnostic = (kind, str(data.get('message', '')))
        else:
            self.unsupported += 1


class JournalReader:
    """Incremental binary tail with bounded reads and partial-line buffering."""
    def __init__(self, path):
        self.path = Path(path)
        self.state = RunState(self.path)
        self.offset = 0
        self.pending = b''
        self.identity = None
        self.discarding = False
        self.error = ''

    def poll(self, max_bytes=READ_BUDGET):
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.state = RunState(self.path)
                self.offset, self.pending, self.discarding = 0, b'', False
                self.identity = identity
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
            except (ValueError, UnicodeDecodeError):
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
    ylabels = [f'{value:.{decimals}f}' for value in (high, (high + low) / 2, low)]
    if max(map(len, ylabels)) > label_budget:
        for digits in range(16, 0, -1):
            compact = [f'{value:.{digits}g}' for value in (high, (high + low) / 2, low)]
            if max(map(len, compact)) <= label_budget:
                ylabels = compact
                break
    return points, decimals, low, high, ylabels


def _plot(samples, key, title, unit, width, target=None, ascii_only=False, rows=7,
          reference=None, reference_approximate=False, label_width=None, time_bounds=None,
          empty_message='No samples for this property yet.'):
    from rich.console import Group
    from rich.text import Text

    heading = Text(f'{title} [{unit}]', style='bold')
    scale = _plot_scale(samples, key, target, reference, label_budget=max(8, width - 10))
    if scale is None:
        return Group(heading, Text(empty_message, style='dim'))
    points, decimals, low, high, ylabels = scale
    heading.append(f'  {points[-1][1]:.{decimals}f}', style='yellow')
    target, reference = _finite(target), _finite(reference)
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
        line = Text(label.rjust(label_width) + (' |' if ascii_only else ' │'), style='dim')
        for col, mask in enumerate(grid[row]):
            if mask:
                line.append('*' if ascii_only else chr(0x2800 + mask), style='yellow not dim')
            elif row == target_row:
                # A quiet, solid line behind the recorded trace. Do not merge
                # guide pixels into orange data cells: that makes the guide
                # change color and invents horizontal pieces in the trace.
                # The exact reference value is retained in the caption;
                # this line is positioned at terminal-character resolution.
                line.append('-' if ascii_only else '─', style='dim')
            else:
                line.append(' ')
        result.append(line)
    axis = list(('-' if ascii_only else '─') * columns)
    labels = [' '] * columns
    for position, label in _time_ticks(left, right, columns):
        axis[position] = '+' if ascii_only else '┬'
        start = max(0, min(columns - len(label), position - len(label) // 2))
        labels[start:start + len(label)] = label
    result.append(Text(' ' * label_width + (' +' if ascii_only else ' └') + ''.join(axis), style='dim'))
    result.append(Text(' ' * (label_width + 2) + ''.join(labels).rstrip(), style='dim'))
    result.append(Text('Time [ps] (current segment)', style='dim'))
    if target is not None:
        result.append(Text(f'Thermostat target {target:g} {unit}', style='dim'))
    elif reference is not None:
        separator = ' | ' if ascii_only else ' · '
        approximate = ('~ ' if ascii_only else '≈ ') if reference_approximate else ''
        suffix = ' (binned)' if reference_approximate else ''
        result.append(Text(f'Reference{separator}latter half: {approximate}{reference:.{decimals}f} {unit}{suffix}', style='dim'))
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


def _controls(state, attached=False, paged=False, multiple=False):
    if attached:
        if state.status in ('Completed', 'Failed'):
            return 'View saved graphs: so3lr show ' + _literal(shlex.quote(str(state.path.parent)))
        if not state.info.get('_attached_keyboard'):
            return 'Ctrl+C: interrupt MD · more plots: so3lr show <run-folder>'
        return ('↑/↓: plots · ' if paged else '') + 'e: energy · q: hide plots · Ctrl+C: interrupt MD'
    close = 'q: close viewer' if state.status in ('Completed', 'Failed') else 'q: detach'
    if state.info.get('operation') == 'optimization':
        return ('←/→: run · o: overview · ' if multiple else '') + close
    if multiple:
        return ('↑/↓: plots · ' if paged else '') + '←/→: run · o: overview · e: energy · ' + close
    if state.status not in ('Completed', 'Failed'):
        close += ' (job continues)'
    return ('↑/↓: plot page · ' if paged else '') + 'j/k or ←/→: run · e: energy · ' + close


def _render_optimization(reader, selected=0, count=1, ascii_only=False, compact=False):
    """Show recorded optimization measurements without inventing MD samples."""
    from rich.console import Group
    from rich.text import Text as RichText

    def Text(value='', style=''):
        value = _literal(value)
        if ascii_only:
            value = value.replace('←/→', 'left/right').replace('·', '|').replace('Å', 'A')
            value = value.encode('ascii', errors='backslashreplace').decode('ascii')
        return RichText(value, style=style, no_wrap=compact, overflow='ellipsis' if compact else 'fold')

    state = reader.state
    style = 'red' if state.status == 'Failed' else 'green' if state.status == 'Completed' else 'yellow'
    rows = [Text('SO3LR TAURI  optimization', 'bold yellow'),
            Text(f'Run {selected + 1}/{count} · {reader.path.parent.name}'),
            Text(f'Last reported state: {state.status}', style),
            Text('Optimization · latest recorded cycle', 'dim')]
    model = state.info.get('model', state.settings.get('model', state.settings.get('model_path')))
    precision = state.info.get('precision', state.settings.get('precision'))
    if model is not None:
        rows.append(Text(f'Model      {model}'))
    if precision is not None:
        rows.append(Text(f'Precision  {precision_label(precision)}', 'dim'))
    if state.phase:
        rows.append(Text(f'Phase      {state.phase}' + (f' · {state.detail}' if state.detail else ''), 'dim'))
    step, total = (_finite(state.info.get(key)) for key in ('opt_step', 'opt_total_steps'))
    if step is not None:
        progress = f'{int(step):,}/{int(total):,} steps' if total is not None else f'{int(step):,} steps'
        rows.append(Text('Progress   ' + progress))
    energy, force, target = (_finite(state.info.get(key))
                             for key in ('opt_energy_eV', 'opt_force_max_eVA', 'opt_force_target_eVA'))
    if energy is not None:
        rows.append(Text(f'Potential energy  {energy:.6f} eV', 'yellow'))
    if force is not None:
        rows.append(Text(f'Maximum force     {force:.6g} eV/Å', 'yellow'))
    if target is not None:
        rows.append(Text(f'Force target      {target:.6g} eV/Å', 'dim'))
    criterion = optimization_status(state.info, state.status)
    rows.append(Text(criterion, 'green' if criterion == 'Force criterion met' else 'dim'))
    if step is None:
        rows.append(Text('Waiting for the first recorded optimization cycle.', 'dim'))
    if state.message:
        rows.append(Text(state.message, style))
    if state.diagnostic:
        kind, message = state.diagnostic
        rows.append(Text(f'{kind.upper()}: {message}', 'red' if kind == 'error' else 'yellow'))
    if reader.error:
        rows.append(Text('Read error: ' + reader.error, 'yellow'))
    if not compact:
        for key, label in (('output_file', 'Structure path'), ('log_file', 'Log path')):
            if state.settings.get(key):
                rows.append(Text(f'{label}  {state.settings[key]}', 'dim'))
        rows.append(Text(f'Journal  {reader.path}', 'dim'))
    rows.append(Text(_controls(state, multiple=count > 1), 'dim'))
    return Group(*rows)


def _render_monitor(readers, selected=0, energy='energy_eV', width=80, now=None, ascii_only=False, height=None, attached=False):
    """Build the native view after Rich has supplied the available width."""
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text as RichText

    def Text(value='', style=''):
        if ascii_only:
            value = str(value).replace('←/→', 'left/right').replace('·', '|')
            value = value.encode('ascii', errors='backslashreplace').decode('ascii')
        return RichText(value, style=style)

    now = now or datetime.now(timezone.utc)
    rows = [Text('SO3LR TAURI  live MD' if attached else 'SO3LR TAURI  monitor', style='bold yellow')]
    if not readers:
        return Group(*rows, Text('Waiting for *.events.jsonl journals.', style='dim'),
                     Text('q: close observer', style='dim'))
    reader = readers[selected % len(readers)]
    state = reader.state
    if state.info.get('operation') == 'optimization':
        return _render_optimization(reader, selected, len(readers), ascii_only)
    run_name = reader.path.parent.name or reader.path.name.removesuffix('.events.jsonl')
    if state.run_id and run_name.endswith('.' + state.run_id):
        run_name = run_name[:-(len(state.run_id) + 1)]
    short_id = f' · {state.run_id[:8]}' if state.run_id else ''
    rows.append(Text(('Run · ' if attached else f'Run {selected + 1}/{len(readers)} · ') + _literal(run_name + short_id)))
    status_style = 'green' if state.status == 'Completed' else 'red' if state.status == 'Failed' else 'yellow'
    rows.append(Text(f'{"State" if attached else "Last reported state"}: {state.status}', style=status_style))
    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style='dim', no_wrap=True)
    metadata.add_column(overflow='fold')
    model = state.info.get('model', state.settings.get('model', state.settings.get('model_path')))
    if model is not None:
        metadata.add_row(Text('Model'), Text(_literal(model)))
    details = [state.settings.get('ensemble'),
               f"{state.info['atoms']} atoms" if 'atoms' in state.info else None,
               state.info.get('engine')]
    details = [str(value) for value in details if value is not None]
    if details:
        metadata.add_row(Text('Run'), Text(_literal(' · '.join(details))))
    if attached:
        for key, label in (('kernels', 'Kernels'), ('devices', 'Device')):
            value = state.info.get(key)
            if isinstance(value, (list, tuple)):
                value = ', '.join(str(item) for item in value)
            if value is not None:
                metadata.add_row(Text(label), Text(_literal(value)))
    precision = state.info.get('precision', state.settings.get('precision'))
    if precision is not None:
        metadata.add_row(Text('Precision'), Text(_literal(precision_label(precision))))
    if state.phase:
        metadata.add_row(Text('Phase'), Text(_literal(state.phase + (' · ' + state.detail if state.detail else ''))))
    if state.message:
        metadata.add_row(Text('Message'), Text(_literal(state.message)))
    rows.append(metadata)
    if attached:
        rows.extend(Text(state.info.get(key, ''), style='dim')
                    for key in ('_attached_progress', '_attached_time', '_attached_timing'))
    elif state.samples:
        last = state.samples[-1]
        step = _finite(last.get('step'))
        segment = _finite(last.get('segment_step'))
        segment = segment if segment is not None else step - state.start_step
        progress = f'Segment {int(segment):,}/{state.total_steps:,} steps' if state.total_steps else f'Segment step {int(segment):,}'
        rows.append(Text(f'{progress} · step counter {int(step):,} · sample age {_age(state.sample_timestamp, now)}', style='dim'))
    else:
        rows.append(Text('No MD samples yet.', style='dim'))
    rows.append(Text(''))
    specs = _plot_specs(state, energy, ascii_only)
    label_width = _shared_label_width(state, specs, width)
    for index, (key, title, unit, target) in enumerate(specs):
        if index:
            rows.append(Text(''))
        rows.append(_state_plot(state, key, title, unit, max(24, width), target, ascii_only, label_width=label_width))
    rows.append(Text(f'{state.sample_count:,} samples · {len(state.samples):,} retained · lines connect recorded samples', style='dim'))
    cycle_seconds = _finite(state.samples[-1].get('seconds_per_step')) if state.samples else None
    throughput = throughput_label(cycle_seconds, state.dt_fs, state.info.get('atoms'))
    if throughput:
        rows.append(Text(f'Cycle throughput {throughput} (excludes startup/I/O)', style='dim'))
    if state.checkpoint:
        rows.append(Text(f"Saved checkpoint · step {_literal(state.checkpoint.get('step', '?'))} · "
                         f"{_literal(state.checkpoint['path'])} · age {_age(state.checkpoint_timestamp, now)}", style='dim'))
    artifacts = Table.grid(padding=(0, 2))
    artifacts.add_column(style='dim', no_wrap=True)
    artifacts.add_column(overflow='fold')
    for key, label in (('output_file', 'Trajectory path'), ('log_file', 'Log path')):
        if state.settings.get(key):
            artifacts.add_row(Text(label), Text(_literal(state.settings[key])))
    artifacts.add_row(Text('Journal'), Text(_literal(reader.path)))
    rows.append(artifacts)
    if state.diagnostic:
        kind, message = state.diagnostic
        rows.append(Text(f'{kind.upper()}: {_literal(message)}', style='red' if kind == 'error' else 'yellow'))
    if reader.error:
        rows.append(Text('Read error: ' + _literal(reader.error), style='yellow'))
    if state.malformed or state.unsupported:
        rows.append(Text(f'Skipped {state.malformed} malformed and {state.unsupported} unsupported events.', style='yellow'))
    rows.append(Text(_controls(state, attached, multiple=len(readers) > 1), style='dim'))
    return Group(*rows)


def _paged_monitor(console, options, readers, selected, energy, width, height, page, now, ascii_only, attached=False):
    """Pack complete plot blocks into short screens; keep controls visible."""
    from rich.console import Group
    from rich.text import Text as RichText

    def Text(value='', style=''):
        if ascii_only:
            value = str(value).replace('←/→', 'left/right').replace('↑/↓', 'up/down').replace('·', '|')
            value = value.encode('ascii', errors='backslashreplace').decode('ascii')
        return RichText(value, style=style, no_wrap=True, overflow='ellipsis')

    def line_count(renderable):
        return len(console.render_lines(renderable, options, pad=False))

    reader = readers[selected % len(readers)]
    state = reader.state
    now = now or datetime.now(timezone.utc)
    name = reader.path.parent.name or reader.path.name.removesuffix('.events.jsonl')
    if state.run_id and name.endswith('.' + state.run_id):
        name = name[:-(len(state.run_id) + 1)]
    if attached:
        model = state.info.get('model', state.settings.get('model', state.settings.get('model_path', '')))
        header = [Text(f'SO3LR TAURI · {state.status} · {_literal(model)}', 'bold yellow')]
        precision = state.info.get('precision', state.settings.get('precision'))
        if precision is not None:
            header.append(Text('Precision ' + precision_label(precision), 'dim'))
        header.extend(Text(state.info.get(key, ''), 'dim')
                      for key in ('_attached_compact_progress', '_attached_timing'))
    else:
        header = [Text(f'SO3LR TAURI · Run {selected + 1}/{len(readers)} · {_literal(name)}', 'bold yellow'),
                  Text(f'Last reported state: {state.status} · {_literal(state.phase)}',
                       'red' if state.status == 'Failed' else 'green' if state.status == 'Completed' else 'yellow')]
    if state.samples and not attached:
        last = state.samples[-1]
        segment = _finite(last.get('segment_step'))
        segment = segment if segment is not None else last['step'] - state.start_step
        progress = f'{int(segment):,}/{state.total_steps:,} steps' if state.total_steps else f'{int(segment):,} steps'
        header.append(Text(f"{str(state.settings.get('ensemble', '')).upper()} · {progress} · sample age {_age(state.sample_timestamp, now)}", 'dim'))
    elif not attached:
        header.append(Text(state.detail or 'Waiting for MD samples.', 'dim'))
    summary = []
    cycle_seconds = _finite(state.samples[-1].get('seconds_per_step')) if state.samples else None
    throughput = throughput_label(cycle_seconds, state.dt_fs, state.info.get('atoms'))
    if throughput:
        summary.append(f'Cycle {throughput}')
    if state.checkpoint and not attached:
        summary.append(f"Checkpoint step {_literal(state.checkpoint.get('step', '?'))} · age {_age(state.checkpoint_timestamp, now)}")
    if not summary:
        summary.append(f'{state.sample_count:,} recorded samples · lines connect recorded values')
    footer = [Text(' · '.join(summary), 'dim')]
    if attached and state.checkpoint:
        footer.append(Text(f"Checkpoint step {_literal(state.checkpoint.get('step', '?'))} · age {_age(state.checkpoint_timestamp, now)}", 'dim'))
    if reader.error or state.diagnostic:
        message = 'Read error: ' + reader.error if reader.error else state.diagnostic[0].upper() + ': ' + state.diagnostic[1]
        footer.append(Text(_literal(message), 'yellow'))
    if state.status == 'Failed' and state.message:
        footer.append(Text(_literal(state.message), 'red'))
    controls_text = _controls(state, attached, paged=True, multiple=len(readers) > 1)
    if width < 60 and not (attached and state.status in ('Completed', 'Failed')):
        if attached:
            controls_text = (('↑/↓ plots · e energy · q hide\n' if state.info.get('_attached_keyboard') else '')
                             + 'Ctrl+C interrupts MD')
        else:
            controls_text = ('←/→ run · o overview\n' if len(readers) > 1 else '') + '↑/↓ plots · e energy · q close'
        if ascii_only:
            controls_text = controls_text.replace('↑/↓', 'up/down').replace('←/→', 'left/right').replace('energy', 'E')
    controls = Text(controls_text, 'dim')
    if attached and state.status in ('Completed', 'Failed'):
        # The saved-directory command must stay copyable even for long paths.
        controls.no_wrap, controls.overflow = False, 'fold'
    budget = height - line_count(Group(*header, *footer, Text('Plot page 1/1'), controls))
    if budget < 7:
        return Group(Text('SO3LR live MD' if attached else 'SO3LR monitor', 'bold yellow'),
                     Text('Enlarge terminal for plots, or use so3lr show --once.'),
                     controls)
    plot_rows = max(2, min(4, budget - 5))
    if budget >= 14:
        # Keep temperature and energy together at 80x24 when possible, while
        # retaining real axes and the progress/throughput/reference rows.
        reference_row = int(state.reference(energy)[0] is not None)
        plot_rows = max(2, min(4, (budget - 10 - reference_row) // 2))
    blocks = []
    specs = _plot_specs(state, energy, ascii_only)
    label_width = _shared_label_width(state, specs, width)
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
    return Group(*header, *content, *footer,
                 Text(f'Plot page {page_index + 1}/{len(pages)} · {names}', 'dim'), controls)


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
                yield _render_optimization(reader, kwargs['selected'], len(self.readers), kwargs['ascii_only'], compact=True)
            else:
                yield _paged_monitor(console, options, self.readers, page=page, **kwargs)
        else:
            yield full


def render_monitor(readers, selected=0, energy='energy_eV', width=80, now=None, ascii_only=False, height=None, page=0, attached=False):
    """Return an adaptive Rich view without touching the simulation process."""
    return _MonitorView(readers, selected=selected, energy=energy, width=width,
                        now=now, ascii_only=ascii_only, height=height, page=page, attached=attached)


class _Keyboard(AbstractContextManager):
    """Temporary cbreak input; never takes ownership of another process."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.previous = None
        self.pending = ''
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
        if os.name == 'nt':
            import msvcrt
            if not msvcrt.kbhit():
                time.sleep(timeout)
                return ''
            char = msvcrt.getwch()
            if char in ('\x00', '\xe0'):
                return {'K': 'previous', 'M': 'next', 'H': 'up', 'P': 'down'}.get(msvcrt.getwch(), '')
            return char
        import select
        if not self.pending or self.pending in ('\x1b', '\x1b['):
            if not select.select([self.fd], [], [], timeout)[0]:
                return ''
            chunk = os.read(self.fd, 32).decode('utf-8', errors='ignore')
            if not chunk:
                return 'q'
            self.pending += chunk
        if self.pending in ('\x1b', '\x1b['):
            return ''
        # Preserve queued keys. A single terminal read can contain an arrow,
        # energy selection and quit/toggle; each must get its own UI update.
        if self.pending.startswith('\x1b['):
            code, self.pending = self.pending[2], self.pending[3:]
            return {'D': 'previous', 'C': 'next', 'A': 'up', 'B': 'down'}.get(code, '')
        key, self.pending = self.pending[0], self.pending[1:]
        return '' if key == '\x1b' else key

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
    ascii_only = console.encoding.lower().replace('-', '') not in ('utf8', 'utf16', 'utf32')
    if once or not interactive:
        for reader in readers:
            reader.snapshot()
        render = render_overview if len(readers) > 1 else render_monitor
        console.print(render(readers, width=console.width, ascii_only=ascii_only))
        return
    # The overview is an extra navigation stop, not a synthetic run. Keep the
    # selected file stable when discovery reorders journals by modification time.
    selected, energy_index, plot_page = (-1 if len(readers) > 1 else 0), list(ENERGIES).index('energy_eV'), 0
    next_refresh = next_discovery = 0.0
    try:
        with _Keyboard() as keyboard, Live(console=console, auto_refresh=False,
                                           screen=False, transient=False,
                                           redirect_stdout=False, redirect_stderr=False) as live:
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
                if key in ('q', 'Q', '\x04'):
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
