"""Terminal presentation for CLI runs, independent of the numerical backends.

The existing text log is preserved alongside an offline event journal. This
module reports values already produced by a runner; it never reads model state
or changes a setting. Rich is loaded only for an actual terminal display.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

from .telemetry import EventJournal
from . import theme
from .formatting import ascii_only_console, display_path as _display_path, duration_label, finite_number as _finite, precision_label, throughput_label, optimization_status, terminal_text
from .terminal import TerminationSignal as _TerminationSignal, termination_signals as _termination_signals, stop_live


@dataclass(frozen=True)
class _Configuration:
    display: str = 'auto'
    verbose: bool = False
    quiet: bool = False
    no_color: bool = False


_configuration = _Configuration()
_active = ContextVar('so3lr_terminal_reporter', default=None)


def configure(display='auto', verbose=False, quiet=False, no_color=False):
    """Configure the next CLI display without importing JAX or Rich."""
    if display not in ('auto', 'rich', 'plain'):
        raise ValueError("display must be 'auto', 'rich', or 'plain'")
    global _configuration
    _configuration = _Configuration(display, verbose, quiet, no_color)


def reset():
    """Restore CLI defaults; active displays are owned by their context manager."""
    global _configuration
    _configuration = _Configuration()


def _rich_enabled():
    if _configuration.quiet or _configuration.display == 'plain':
        return False
    if not bool(getattr(sys.stderr, 'isatty', lambda: False)()):
        return False
    return _configuration.display == 'rich' or os.environ.get('TERM') != 'dumb'


class _NullReporter:
    enabled = False
    observing = False

    def phase(self, name, detail=''):
        pass

    def update_info(self, **kwargs):
        pass

    def start_md(self, start_step, total_steps, dt_fs):
        pass

    def progress(self, step, temperature=None, potential_energy=None, time_per_step=None, **optional):
        pass

    def checkpoint(self, step, path):
        pass

    def finish(self, level='complete', reason=''):
        pass

    def fail(self, message, reason='error'):
        pass

    def logger_ready(self, log_file):
        pass

    def print_hint(self, line):
        pass

    def task_progress(self, done, total, unit):
        pass

    def outcome(self, detail=None, *, key=None, warning=None):
        pass

    def print_verdict(self, level, head, details):
        pass

    def results(self, title, rows):
        pass


_null = _NullReporter()


# What a completed command's main output file is called on screen.
OPERATION_NAMES = {'md': 'MD', 'optimization': 'Optimization', 'evaluation': 'Evaluation',
                   'tuning': 'Tuning', 'training': 'Training'}
def _existing(path):
    """str(path) when it exists (a file, or a training run's model directory), else None."""
    try:
        return str(path) if path and os.path.exists(path) else None
    except (OSError, TypeError, ValueError):
        return None


def _output_artifact(settings, operation):
    """(label, path) of the run's output, path None until it exists."""
    return OUTPUT_LABELS.get(operation, 'Trajectory'), _existing(settings.get('output_file'))


OUTPUT_LABELS = {'optimization': 'Optimization trajectory', 'evaluation': 'Predictions',
                 'training': 'Model directory'}


PLAIN_PROGRESS_INTERVAL = 600  # seconds between plain progress lines without a checkpoint


class _PlainReporter(_NullReporter):
    """Plain and batch output: greppable progress lines and end-of-run hints on stderr.

    The numeric MD table and diagnostics still come from the logger; this adds
    only what a scheduler log cannot reconstruct: percent done, throughput and
    ETA from wall clock (I/O included), and how to resume.
    """

    def __init__(self):
        self._start_step = self._total_steps = self._dt_fs = None
        self._rate_origin = None      # (wall clock, step) of the first MD sample
        self._last_line = None

    def start_md(self, start_step, total_steps, dt_fs):
        self._start_step, self._total_steps, self._dt_fs = int(start_step), int(total_steps), _finite(dt_fs)
        self._rate_origin = self._last_line = None

    def progress(self, step, temperature=None, potential_energy=None, time_per_step=None, **optional):
        now = time.monotonic()
        if self._rate_origin is None:
            self._rate_origin = (now, int(step))
            self._last_line = now  # the ten-minute clock starts at the first sample
        elif now - self._last_line >= PLAIN_PROGRESS_INTERVAL:
            self._log_progress(int(step))

    def checkpoint(self, step, path):
        self._log_progress(int(step))

    def _log_progress(self, step):
        self._last_line = time.monotonic()
        if not self._total_steps or self._start_step is None:
            return
        done = max(0, min(step - self._start_step, self._total_steps))
        parts = [f'{done / self._total_steps:.1%}', f'step {done:,}/{self._total_steps:,}']
        if self._rate_origin is not None:
            started, first = self._rate_origin
            elapsed = self._last_line - started
            if elapsed > 0 and step > first:
                rate = (step - first) / elapsed  # steps per wall second, I/O included
                if self._dt_fs:
                    parts.append(throughput_label(1 / rate, self._dt_fs))
                if done < self._total_steps:
                    parts.append(f'ETA {duration_label((self._total_steps - done) / rate)}')
        logging.getLogger('SO3LR').info('Progress ' + ' · '.join(parts))

    def print_hint(self, line):
        print(terminal_text(line, _ascii_stream(sys.stderr)), file=sys.stderr)

    def print_verdict(self, level, head, details):
        line = ' · '.join([f'{theme.status_glyph(level)} {head}', *details])
        print(terminal_text(line, _ascii_stream(sys.stderr)), file=sys.stderr)


def _ascii_stream(stream):
    """True for a stream that cannot encode the display's glyphs (batch logs in C/POSIX locales)."""
    encoding = getattr(stream, 'encoding', None)
    if not encoding:  # an in-memory text stream takes any character
        return False
    try:
        '✓·Å≤'.encode(encoding)
        return False
    except (LookupError, UnicodeEncodeError):
        return True


def get_reporter():
    """Return the current display, or a harmless reporter for library callers."""
    return _active.get() or _null


def reading_progress(total):
    """A data loader's progress callback for the live display, or None (it then uses tqdm)."""
    reporter = get_reporter()
    if not reporter.enabled:
        return None
    last = [float('-inf')]

    def report(read):
        # at most 10 updates a second: reading can be far faster than the display needs
        now = time.monotonic()
        if read >= total or now - last[0] >= 0.1:
            last[0] = now
            reporter.task_progress(read, total, 'structures read')
    return report


class _TaskCounter:
    """tqdm's update/close for work counted on the live display's progress bar."""

    def __init__(self, reporter, total, unit):
        self.reporter, self.total, self.unit, self.n = reporter, total, unit, 0

    def update(self, count=1):
        self.n += count
        self.reporter.task_progress(self.n, self.total, self.unit)

    def close(self):
        pass


def progress_counter(total, unit, description):
    """Count finished work: on the live display's bar when one is open, else with tqdm."""
    reporter = get_reporter()
    if reporter.enabled:
        return _TaskCounter(reporter, total, unit)
    from tqdm import tqdm
    return tqdm(total=total, desc=description, unit=unit.removesuffix('s'), smoothing=0.1, mininterval=0.5)


def exception_message(exc):
    """Describe the actual failure, never a previously logged error record."""
    if isinstance(exc, _TerminationSignal):
        return str(exc)
    if isinstance(exc, KeyboardInterrupt):
        return 'KeyboardInterrupt: interrupted'
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen and len(parts) < 4:
        seen.add(id(exc))
        detail = str(exc)
        parts.append(type(exc).__name__ + (f': {detail}' if detail else ''))
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return ' (caused by '.join(parts) + ')' * (len(parts) - 1)


def failure_details(exc):
    """Point to a traceback only after the CLI verified this failure's write."""
    path = getattr(exc, '_so3lr_traceback_path', None)
    if not getattr(exc, '_so3lr_traceback_preserved', False) or not path:
        return ''
    return terminal_text(f'Full traceback: {path}')


@dataclass(frozen=True)
class _Activity:
    """Animate wall time from immutable host snapshots, never simulation state."""
    phase: str
    started: float
    reported: float
    unicode: bool = True
    eta: tuple | None = None  # (monotonic deadline, whether it is only an upper bound)

    def __rich_console__(self, console, options):
        from rich.text import Text
        now = time.monotonic()
        # Tauri pulse grows and recedes in place. A five-second cycle remains
        # calm during long compilations and is visible at our 2 Hz refresh.
        frames = '+×✧✦✧×' if self.unicode else '+x*x+'
        marker = frames[int(max(0, now - self.started) * len(frames) / 5) % len(frames)]
        text = Text(marker, style=theme.ACCENT, no_wrap=True,
                    overflow='ellipsis' if self.unicode else 'crop')
        text.append(terminal_text(f' {self.phase}', not self.unicode), style='default')
        timing = f' · {duration_label(now - self.started)} since launch'
        if self.eta is not None:
            deadline, bound = self.eta
            timing += f" · ETA {'≤ ' if bound else ''}{duration_label(max(0.0, deadline - now))}"
        if now - self.reported >= 2:
            timing += f' · last update {duration_label(now - self.reported)} ago'
        text.append(terminal_text(timing, not self.unicode), style=theme.MUTED)
        yield text


class _Reporter:
    enabled = True

    def __init__(self, settings):
        from rich.console import Console
        from rich.live import Live
        from .monitor import RunState

        self.config = _configuration
        self.settings = settings
        self._state = RunState(Path(settings.get('log_file') or settings.get('output_file') or 'so3lr'))
        self._state.settings = settings
        self._state.status = 'Started'
        self._reader = SimpleNamespace(state=self._state, path=self._state.path, error='')
        self._event_sequence = 0
        self._keyboard = None
        self._graphs_hidden = False
        self._plot_page, self._energy_index = 0, 1  # Total energy is the initial graph.
        self._stderr_before = sys.stderr
        self.console = Console(
            file=self._stderr_before,
            force_terminal=True if self.config.display == 'rich' else None,
            no_color=self.config.no_color or 'NO_COLOR' in os.environ,
            markup=False, highlight=False,
        )
        self.info = {}
        self._task = None
        self._task_origin = None   # (wall clock, done) the rate is measured from
        self._task_started = None  # wall clock of the task's first report
        self._task_seen = None     # (unit, done) of the last report, shown or not
        self._results = []
        self._finish_level = 'complete'
        self.requested = set()
        for key in ('model', 'precision', 'engine'):
            if settings.get(key) is not None:
                self.info[key] = settings[key]
                self.requested.add(key)
        if 'model' not in self.info and settings.get('model_path'):
            self.info['model'] = settings['model_path']
            self.requested.add('model')
        if settings.get('input_file'):
            self.info['input'] = settings['input_file']
        ensemble = str(settings.get('ensemble', '')).upper()
        temperature = _finite(settings.get('md_T'))
        if ensemble:
            self.info['dynamics'] = ensemble
            if temperature is not None and ensemble != 'NVE':
                self.info['dynamics'] += f' · {temperature:g} K target'
        cycles, steps, dt_ps = (_finite(settings.get(key))
                                for key in ('md_cycles', 'md_steps', 'md_dt'))
        if cycles is not None and steps is not None:
            count = int(cycles * steps)
            self.info['plan'] = f'{count:,} steps'
            if dt_ps is not None:
                self.info['plan'] += f' · {dt_ps * 1000:g} fs · {count * dt_ps:g} ps'
        self.phase_name, self.detail = 'Loading', 'Initializing runtime'
        self._seen_phases = {(self.phase_name, self.detail)}
        self.start_step = self.total_steps = self.step = 0
        self.dt_fs = None
        self.started_at = None
        self.temperature = self.potential_energy = self.time_per_step = None
        self.saved = None
        self.artifacts = []
        self.status = 'running'
        self._last_refresh = float('-inf')
        self._closed = False
        self._opened_at = self._reported_at = time.monotonic()
        try:
            self._heartbeat = os.isatty(sys.stderr.fileno())
        except (AttributeError, OSError, ValueError):
            self._heartbeat = False
        self._live_snapshot = self._snapshot()
        # The refresh thread sees immutable text plus a wall clock. Graph data
        # is rendered only at existing host reporting points, preserving the
        # numerical path and avoiding concurrent reads of mutable accumulators.
        self.live = Live(get_renderable=self._live_view, console=self.console,
                         auto_refresh=self._heartbeat, refresh_per_second=2,
                         transient=False, redirect_stdout=False,
                         redirect_stderr=bool(getattr(self._stderr_before, 'isatty', lambda: False)()))
        try:
            self.live.start(refresh=True)
        except BaseException:
            # Construction has not returned to run_display yet. Restore any
            # cursor/live state acquired before a failed first refresh.
            try:
                self.close()
            except BaseException:
                pass  # preserve the original interruption after cleanup attempts
            raise
        self._last_refresh = time.monotonic()

    def _text(self, text, style=''):
        from rich.text import Text
        # Text treats paths and user messages literally, including [brackets].
        return Text(self._label(text), style=style)

    @property
    def _ascii_only(self):
        return ascii_only_console(self.console)

    def _label(self, text):
        return terminal_text(text, self._ascii_only)

    def _snapshot(self):
        from rich.console import Group
        from rich.text import Text
        view = self._render()
        if not self._heartbeat:
            return view, None
        rows = []
        for segments in self.console.render_lines(view, self.console.options, pad=False):
            row = Text(no_wrap=True, overflow='crop')
            for segment in segments:
                if not segment.control:
                    row.append(segment.text, segment.style)
            rows.append(row)
        activity = None
        if self.status == 'running':
            eta = self._eta()
            activity = _Activity(self.phase_name, self._opened_at, self._reported_at, not self._ascii_only,
                                 None if eta is None else (time.monotonic() + eta[0], eta[1]))
        return Group(*rows), activity

    @property
    def _bar_width(self):
        return min(40, max(8, self.console.width - 40))

    def _eta(self):
        """(seconds left, whether only an upper bound) for the work in progress, or None.

        From the measured rate only: none while loading or compiling. With a force
        target, an optimization's step budget is a ceiling, so its ETA is a bound."""
        now = time.monotonic()
        if self.started_at is not None:
            eta = self._live_progress().eta
            return None if eta is None else (eta, False)
        if self._task is not None and self._task_origin is not None and self.status == 'running':
            done, total, _ = self._task
            started, first = self._task_origin
            if first < done < total and now > started:
                # an optimization with a force target may stop before its step budget
                bound = (self.info.get('operation') == 'optimization'
                         and _finite(self.info.get('opt_force_target_eVA')) is not None)
                return (total - done) * (now - started) / (done - first), bound
        return None

    def _live_view(self):
        from rich.console import Group
        frame, activity = self._live_snapshot
        return Group(frame, activity) if activity is not None else frame

    def _render(self):
        from rich.console import Group
        from rich.table import Table
        from .. import __version__

        if self._state.samples and not self._graphs_hidden:
            from .monitor import ENERGIES, render_monitor
            return render_monitor([self._reader], energy=list(ENERGIES)[self._energy_index],
                                  width=self.console.width, height=self.console.height - int(self._heartbeat),
                                  ascii_only=self._ascii_only,
                                  page=self._plot_page, attached=True, live=self._live_progress())

        title = self._text('')
        title.append(self._label(f'{theme.BRAND} SO3LR TAURI'), style=theme.ACCENT_BOLD)
        title.append(f'  v{__version__}', style=theme.MUTED)
        rows = [title, self._text('')]
        metadata = Table.grid(padding=(0, 2))
        metadata.add_column(style=theme.MUTED, no_wrap=True)
        metadata.add_column(overflow='fold')
        labels = (('input', 'Input'), ('model', 'Model'), ('engine', 'Engine'),
                  ('kernels', 'Kernels'), ('precision', 'Precision'),
                  ('atoms', 'Atoms'), ('devices', 'Device'),
                  ('dynamics', 'Dynamics'), ('plan', 'Run'))
        for key, label in labels:
            if key not in self.info:
                continue
            value = self.info[key]
            if isinstance(value, (list, tuple)):
                value = ', '.join(str(item) for item in value)
            value = str(value)
            if key == 'precision':
                value = precision_label(value)
            if key in self.requested:
                value += ' (requested)'
            metadata.add_row(self._text(label), self._text(value))
        rows.extend((metadata, self._text('')))
        style = theme.status_style(self.status)
        glyph = theme.status_glyph(self.status) or theme.status_glyph('running')
        if self.status == 'complete' and self._finish_level != 'complete':
            glyph, style = theme.status_glyph(self._finish_level), theme.status_style(self._finish_level)
        phase = self._text(f'{glyph} {self.phase_name}', f'bold {style}')
        if self.detail:
            phase.append(self._label(f'  {self.detail}'), style=theme.MUTED)
        rows.append(phase)
        if self._task is not None:
            done, total, unit = self._task
            ratio = done / total if total else 0.0
            bar = theme.bar(ratio, self._bar_width, self._ascii_only)
            bar.append(f'  {ratio:.0%}', style=theme.STRONG)
            bar.append(self._label(f'  {done:,}/{total:,} {unit}'), style=theme.MUTED)
            rows.append(bar)
            now = time.monotonic()
            timing = f'Elapsed {duration_label(now - self._task_started)}'
            eta = self._eta()
            if eta is not None:  # only once a rate is measured
                seconds, bound = eta
                started, first = self._task_origin
                timing += f" · ETA {'≤ ' if bound else ''}{duration_label(seconds)} · {(done - first) / (now - started):.3g} {unit}/s"
            rows.append(self._text(timing, theme.MUTED))
        if self.info.get('operation') == 'optimization':
            energy, force, target = (_finite(self.info.get(key)) for key in
                                    ('opt_energy_eV', 'opt_force_max_eVA', 'opt_force_target_eVA'))
            if energy is not None:
                rows.append(self._text(f'Potential energy  {energy:.6f} eV'))
            if force is not None:
                detail = f'Maximum force     {force:.6f} eV/Å'
                if target is not None:
                    detail += f' · target ≤ {target:g} eV/Å'
                rows.append(self._text(detail))
                outcome = optimization_status(self.info, self.status)
                if outcome != 'Relaxing geometry':  # the phase row already says so
                    rows.append(self._text(outcome, theme.SUCCESS if outcome == 'Force criterion met' else theme.MUTED))
            plots_at = len(rows)
        for title, items in self._results:
            rows.append(self._text(''))
            rows.append(self._text(title, theme.STRONG))
            table = Table.grid(padding=(0, 2))
            table.add_column(style=theme.MUTED, no_wrap=True)
            table.add_column(overflow='fold')
            for label, value in items:
                table.add_row(self._text(label), self._text(value))
            rows.append(table)
        if self.started_at is not None:
            live = self._live_progress()
            bar = theme.bar(live.ratio, self._bar_width, self._ascii_only)
            bar.append(f'  {live.ratio:.0%}', style=theme.STRONG)
            bar.append(f'  {live.completed:,}/{live.total:,} steps', style=theme.MUTED)
            rows.append(bar)
            if self.start_step:
                rows.append(self._text(f'Restart counter {self.start_step:,} · step counter {self.step:,}', theme.MUTED))
            if live.span_text():
                rows.append(self._text(live.span_text(), theme.MUTED))
            timing = live.timing_text()
            if live.completed and self.time_per_step is not None and self.time_per_step > 0:
                rate = throughput_label(self.time_per_step, self.dt_fs, self.info.get('atoms'))
                if rate:
                    timing += f' · {rate}'
            rows.append(self._text(timing, theme.MUTED))
            metrics = []
            if self.temperature is not None:
                metrics.append(f'T {self.temperature:.1f} K')
            if self.potential_energy is not None:
                metrics.append(f'Epot {self.potential_energy:,.6f} eV')
            if metrics:
                rows.append(self._text(' · '.join(metrics), theme.STRONG))
        if self.saved is not None:
            step, path = self.saved
            rows.append(self._text(f'Checkpoint step {step:,} · {_display_path(path)}', theme.MUTED))
        if self._results and self.artifacts:
            rows.append(self._text(''))
        for label, path in self.artifacts:
            rows.append(self._text(f'{label} · {_display_path(path)}', theme.MUTED))
        hint = None
        if self.info.get('operation') == 'optimization' and self.status != 'running':
            from .monitor import _controls
            hint = _controls(self._state, attached=True)
        if self._graphs_hidden and self.status == 'running' and self._keyboard is not None:
            hint = 'q: show plots · Ctrl+C: interrupt MD'
        elif self._graphs_hidden and self.status != 'running':
            from .monitor import _controls
            hint = _controls(self._state, attached=True)
        if hint:
            rows.append(theme.controls(hint, self._ascii_only))
        if self.info.get('operation') == 'optimization':
            self._insert_optimization_plots(rows, plots_at)
        return Group(*rows)

    def _insert_optimization_plots(self, rows, index):
        """Energy and force graphs in the space the terminal has left, if at least 3 rows each."""
        from rich.console import Group
        from .monitor import OPT_MIN_POINTS, optimization_plots
        if len(self._state.opt_samples) < OPT_MIN_POINTS:
            return
        width = self.console.width
        used = len(self.console.render_lines(Group(*rows), self.console.options.update_width(width), pad=False))
        free = self.console.height - int(self._heartbeat) - used
        # each graph: heading, axis and tick rows; blank rows before, between and after
        plot_rows = min(7, (free - 3) // 2 - 3)
        if plot_rows < 3:
            return
        plots = optimization_plots(self._state, width, self._ascii_only, rows=plot_rows)
        rows[index:index] = [self._text(''), plots, self._text('')]

    def print_hint(self, line):
        self.console.print(self._text(line, theme.MUTED))

    def print_verdict(self, level, head, details):
        glyph, color = theme.status_glyph(level), theme.status_style(level)
        line = self._text('')
        line.append(self._label(f'{glyph} {head}'), style=f'bold {color}')
        for detail in details:
            line.append(self._label(f' · {detail}'), style=theme.MUTED)
        self.console.print(line)

    def _live_progress(self):
        """Progress known only to this process, handed to the view instead of the journal state."""
        from .monitor import LiveProgress
        completed = max(0, min(self.step - self.start_step, self.total_steps))
        known_rate = completed and self.time_per_step is not None and self.time_per_step > 0
        # No ETA is invented during compilation or at the restored step.
        return LiveProgress(completed=completed, total=self.total_steps, step=self.step, dt_fs=self.dt_fs,
                            elapsed=time.monotonic() - self.started_at,
                            eta=(self.total_steps - completed) * self.time_per_step if known_rate else None,
                            running=self.status == 'running', keyboard=self._keyboard is not None)

    def attach_journal(self, path, run_id, *, enabled=True):
        self._state.run_id = run_id
        self._state.info['_journal_enabled'] = enabled
        if path is not None:
            self._state.path = self._reader.path = Path(path)

    def _observe(self, kind, data):
        self._event_sequence += 1
        self._state.consume({'schema': 1, 'run_id': self._state.run_id or 'attached',
                             'seq': self._event_sequence, 'type': kind, 'data': data,
                             'timestamp': datetime.now(timezone.utc).isoformat()})

    def _start_keyboard(self):
        # This is the simulation's terminal, not a separate show process.
        # Read controls only at host reporting boundaries and retain ISIG so
        # Ctrl+C keeps its ordinary interruption behavior during compiled work.
        if self._keyboard is not None:
            return
        from .monitor import _Keyboard
        try:
            if not getattr(sys.stdin, 'isatty', lambda: False)() or not getattr(sys.stderr, 'isatty', lambda: False)():
                return
            sys.stdin.fileno()
            self._keyboard = _Keyboard()
            self._keyboard.__enter__()
        except Exception:
            # Some stream wrappers claim TTY support but lack usable termios.
            # Optional input controls must not prevent the calculation.
            self._stop_keyboard()

    def _stop_keyboard(self):
        keyboard, self._keyboard = self._keyboard, None
        if keyboard is not None:
            try:
                keyboard.__exit__(None, None, None)
            except (OSError, ValueError):
                pass

    def _poll_keyboard(self):
        if self._keyboard is None:
            return False
        try:
            key = self._keyboard.read(timeout=0)
        except (OSError, ValueError):
            self._stop_keyboard()
            return False
        if key in ('e', 'E'):
            from .monitor import ENERGIES
            self._energy_index = (self._energy_index + 1) % len(ENERGIES)
        elif key in ('up', 'down'):
            self._plot_page += 1 if key == 'down' else -1
        elif key == 'escape':
            # Escape leaves the graph view without stopping MD or toggling an
            # already hidden view back on. q remains the explicit toggle.
            self._graphs_hidden = True
        elif key in ('q', 'Q'):
            # Keep input active while the compact status is visible so the
            # same key can restore the graphs, including samples gathered
            # while hidden. Terminal ownership ends with the MD invocation.
            self._graphs_hidden = not self._graphs_hidden
        else:
            return False
        return True

    def _refresh(self, force=False):
        if self._closed:
            return
        force = self._poll_keyboard() or force
        now = time.monotonic()
        if force or now - self._last_refresh >= 0.25:
            self._live_snapshot = self._snapshot()
            self.live.update(self._live_view(), refresh=True)
            self._last_refresh = now

    def phase(self, name, detail=''):
        if self.status != 'running':
            return
        phase = (str(name), str(detail))
        if phase == (self.phase_name, self.detail):
            return
        if name != self.phase_name and self._task is not None and self._task[0] >= self._task[1]:
            self._task = None  # a finished bar leaves with its phase; one in progress stays
        self.phase_name, self.detail = phase
        self._reported_at = time.monotonic()
        self._state.phase, self._state.detail = phase
        # Show the first potentially blocking preparation stage immediately;
        # repeated Running/Saving transitions share the normal refresh budget.
        first_preparation = name in ('Loading', 'Compiling', 'Relaxing') and phase not in self._seen_phases
        if first_preparation:
            self._seen_phases.add(phase)
        self._refresh(force=first_preparation)

    def task_progress(self, done, total, unit):
        """Generic progress for non-MD work: structures evaluated, training steps, candidates."""
        now = time.monotonic()
        done, total, unit = int(done), int(total), str(unit)
        seen = self._task_seen
        if seen is None or unit != seen[0] or done < seen[1] or (done == 0 and seen[1] > 0):
            self._task_started, self._task_origin = now, None  # a new task, or one restarted at 0
        elif self._task_origin is None and done > seen[1]:
            # The rate starts after the first finished unit, which often includes compilation.
            self._task_origin = (now, done)
        self._task_seen = (unit, done)
        self._task = (done, total, unit)
        self._reported_at = now
        self._refresh(force=done >= total)

    def results(self, title, rows):
        """A titled table of (label, value) rows, replaced when the same title comes again."""
        self._results = [item for item in self._results if item[0] != title] + [(str(title), list(rows))]
        self._reported_at = time.monotonic()
        self._refresh(force=True)

    def update_info(self, **kwargs):
        self._reported_at = time.monotonic()
        self.info.update({key: value for key, value in kwargs.items() if value is not None})
        self.requested.difference_update(kwargs)
        self._state.info.update({key: value for key, value in kwargs.items() if value is not None})
        self._state.record_optimization(kwargs)
        self._refresh()

    def start_md(self, start_step, total_steps, dt_fs):
        """Begin this invocation's work; total_steps excludes restored steps."""
        self.start_step = self.step = int(start_step)
        self.total_steps = max(0, int(total_steps))
        self.dt_fs = _finite(dt_fs)
        self.started_at = time.monotonic()
        self.time_per_step = self.temperature = self.potential_energy = None
        self._task = None  # a relaxation's step bar
        self._observe('md_started', {'start_step': self.start_step, 'total_steps': self.total_steps, 'dt_fs': self.dt_fs})
        self._start_keyboard()
        self.phase('Simulating')

    def progress(self, step, temperature=None, potential_energy=None, time_per_step=None, **optional):
        self._reported_at = time.monotonic()
        self.step = int(step)
        self.temperature = _finite(temperature)
        self.potential_energy = _finite(potential_energy)
        self.time_per_step = _finite(time_per_step)
        data = optional.pop('_sample', None)
        if data is None:
            segment_step = self.step - self.start_step
            data = {'step': self.step, 'segment_step': segment_step,
                    'time_ps': segment_step * self.dt_fs / 1000 if self.dt_fs is not None else None,
                    'temperature_K': self.temperature, 'potential_energy_eV': self.potential_energy,
                    'seconds_per_step': self.time_per_step}
            for argument, field in (('energy', 'energy_eV'), ('conserved_energy', 'conserved_energy_eV'), ('volume', 'volume_A3')):
                if argument in optional:
                    data[field] = _finite(optional[argument])
        self._observe('sample', data)
        self._refresh(force=self._state.sample_count == 1)

    def checkpoint(self, step, path):
        """Called only after a restart write has completed successfully."""
        self.saved = (int(step), str(path))
        self._observe('checkpoint', {'step': int(step), 'path': str(path)})
        self._refresh()

    def finish(self, level='complete', reason=''):
        if self.status != 'running':
            return
        self.status = 'complete'
        # A run that finished without reaching its goal says so in amber, as its closing line does.
        self.phase_name, self.detail = ('Complete', '') if level == 'complete' else ('Finished', reason)
        self._finish_level = level
        self._observe('run_completed', {'message': 'Run completed successfully.'})
        for label, path in (_output_artifact(self.settings, self.info.get('operation')),
                            ('Log', _existing(self.settings.get('log_file')))):
            if path:
                self.artifacts.append((label, path))
        self._refresh(force=True)

    def fail(self, message, reason='error'):
        if self.status == 'failed':
            return
        self.status = 'failed'
        self.phase_name, self.detail = 'Stopped', str(message)
        self._observe('run_failed', {'message': str(message), 'reason': reason})
        self._refresh(force=True)

    def close(self):
        try:
            if not self._closed:
                try:
                    try:
                        # Rich does not flush its partial-line buffer on stop.
                        # Only flush the proxy this display installed; existing
                        # stream handlers and the real stderr retain ownership.
                        if getattr(sys.stderr, 'rich_proxied_file', None) is self._stderr_before:
                            sys.stderr.flush()
                    finally:
                        stop_live(self.live)
                finally:
                    self._closed = True
        finally:
            self._stop_keyboard()

    def remember_log(self, record):
        """Update the latest diagnostic on the display's owning thread."""
        if record.levelno >= logging.WARNING:
            self._observe('error' if record.levelno >= logging.ERROR else 'warning', {'message': record.getMessage()})

    def print_log(self, record, message):
        """Print through Rich's synchronized console without mutating run state."""
        style = theme.DANGER if record.levelno >= logging.ERROR else theme.WARNING if record.levelno >= logging.WARNING else theme.MUTED
        label = f'{record.levelname}: ' if record.levelno >= logging.WARNING else ''
        self.console.print(self._text(label + message, style))


class _RunReporter:
    """Fan out host events to the journal and an optional terminal renderer."""

    observing = True

    def __init__(self, settings, renderer, record=True):
        self.renderer = renderer
        self._settings = settings
        self._last_checkpoint = None
        # enabled continues to mean terminal rendering, not journal availability.
        self.enabled = renderer.enabled
        # Only MD and optimization keep a run journal for so3lr show.
        self.journal = EventJournal(settings, enabled=record)
        try:
            if self.enabled:
                self.renderer.attach_journal(self.journal.path, self.journal.run_id,
                                             enabled=self.journal.enabled)
        except BaseException:
            self.journal.close()
            raise
        self.status = 'running'
        self._closed = False
        self._owner_thread = threading.get_ident()
        self._log_lock = threading.RLock()
        # Every record is printed immediately and journaled. The mutable view
        # only retains its latest diagnostic, updated on the owning thread.
        self._pending_log = None
        self._phase = None
        self._start_step = None
        self._dt_fs = None
        self._total_steps = None
        # what the closing line summarizes
        self._opened = time.monotonic()
        self._info = {}
        self._last_step = None
        self._task = None
        self._failure = None
        self._details, self._warning = {}, None
        self.final = None  # (level, head, details) once finished or failed

    def __getattr__(self, name):
        # Keep the renderer's inspection surface (live/render/artifacts) usable.
        return getattr(self.renderer, name)

    def phase(self, name, detail=''):
        self._drain_logs()
        phase = (str(name), str(detail))
        if self.status != 'running' or phase == self._phase:
            return
        self._phase = phase
        self.journal.emit('phase', {'name': phase[0], 'detail': phase[1]})
        self.renderer.phase(*phase)

    def task_progress(self, done, total, unit):
        self._drain_logs()
        self._task = (int(done), int(total), str(unit))
        self.renderer.task_progress(done, total, unit)

    def _optimization_progress(self, info):
        """An optimization's steps feed the shared progress bar, once its budget is known."""
        step, total = _finite(info.get('opt_step')), _finite(self._info.get('opt_total_steps'))
        if step is not None and total:
            self.task_progress(step, total, 'optimization steps')

    def outcome(self, detail=None, *, key=None, warning=None):
        """Add to the closing line: a detail (replaced when its key comes again), or why the
        run finished without reaching its goal, which turns the line amber."""
        if detail:
            self._details[key or detail] = str(detail)
        if warning:
            self._warning = str(warning)

    def verdict(self):
        """The closing line as (level, head, details), once the run has finished or failed."""
        info = self._info
        name = OPERATION_NAMES.get(info.get('operation'), 'Run')
        elapsed = duration_label(time.monotonic() - self._opened)
        md = info.get('operation') == 'md' and self._last_step is not None
        details = []
        if self.status == 'failed':
            message, reason = self._failure
            level = 'interrupted' if reason in ('interrupt', 'signal') else 'failed'
            head = f"{name} {'interrupted' if level == 'interrupted' else 'stopped'}"
            if md:
                head += f' at step {self._last_step:,}'
            head += f' after {elapsed}'
            # Plain output prints the error itself next; Ctrl+C needs no repeating.
            if level == 'failed' and self.enabled and message:
                details.append(str(message).splitlines()[0])
            return level, head, details
        warning = self._warning
        if info.get('operation') == 'optimization':
            force, target = (_finite(info.get(key)) for key in ('opt_force_max_eVA', 'opt_force_target_eVA'))
            met = optimization_status(info, 'complete') == 'Force criterion met'
            if target is not None and not met:
                warning = warning or 'force target not met'
            if force is not None:
                relation = '' if target is None else f" {'≤' if met else '>'} {target:g}"
                details.append(f'max force {force:.3g}{relation} eV/Å')
        level = 'warning' if warning else 'complete'
        head = f'{name} finished in {elapsed}' + (f': {warning}' if warning else '')
        if md:
            details.append(f'{self._last_step - (self._start_step or 0):,} steps')
        if self._task is not None:
            done, total, unit = self._task
            details.append(f'{done:,} {unit}' if done >= total else f'{done:,}/{total:,} {unit}')
        details.extend(self._details.values())
        label, output = _output_artifact(self._settings, info.get('operation'))
        if output:
            details.append(f'{label.lower()} {_display_path(output)}')
        return level, head, details

    def results(self, title, rows):
        self._drain_logs()
        self.renderer.results(title, rows)

    def update_info(self, **kwargs):
        self._drain_logs()
        self._info.update({key: value for key, value in kwargs.items() if value is not None})
        self.journal.emit('info', kwargs)
        self.renderer.update_info(**kwargs)
        self._optimization_progress(kwargs)

    def logger_ready(self, log_file):
        self._drain_logs()
        self.journal.flush_initial_warning()

    def start_md(self, start_step, total_steps, dt_fs):
        # Relaxation can share this reporter; MD must replace its operation
        # marker in both the journal and attached display before new samples.
        self.update_info(operation='md')
        self._task = None
        self._start_step, self._dt_fs = int(start_step), _finite(dt_fs)
        self._total_steps = int(total_steps)
        self.journal.emit('md_started', {'start_step': self._start_step,
                                       'total_steps': int(total_steps), 'dt_fs': self._dt_fs})
        self.renderer.start_md(start_step, total_steps, dt_fs)

    def progress(self, step, temperature=None, potential_energy=None, time_per_step=None, **optional):
        self._drain_logs()
        step = int(step)
        self._last_step = step
        segment_step = step - self._start_step if self._start_step is not None else None
        data = {
            'step': step,
            'segment_step': segment_step,
            'time_ps': (segment_step * self._dt_fs / 1000.0
                        if segment_step is not None and self._dt_fs is not None else None),
            'temperature_K': _finite(temperature),
            'potential_energy_eV': _finite(potential_energy),
            'seconds_per_step': _finite(time_per_step),
        }
        for argument, field in (('kinetic_energy', 'kinetic_energy_eV'), ('energy', 'energy_eV'),
                                ('conserved_energy', 'conserved_energy_eV'), ('volume', 'volume_A3')):
            if argument in optional:
                data[field] = _finite(optional[argument])
        for argument, field in (('box', 'box_A'), ('pressure_diagonal', 'pressure_diagonal_eVA3')):
            if argument in optional:
                data[field] = optional[argument]
        self.journal.emit('sample', data)
        self.renderer.progress(step, temperature=temperature, potential_energy=potential_energy,
                               time_per_step=time_per_step, _sample=data, **optional)

    def resume_hint(self):
        """How to continue an interrupted or failed MD run, or [] when unknown."""
        if self._total_steps is None:
            return []
        if self._last_checkpoint is None:
            if self._settings.get('restart_save_path'):
                return []
            return ['No checkpoint was saved. Set restart_save_path (or --restart-save) to make runs resumable.']
        step, path = self._last_checkpoint
        # No '~': YAML paths are not expanded, so the hint must work when pasted there.
        shown = _display_path(path)
        shown = str(Path(path).resolve()) if shown.startswith('~') else shown
        output = str(self._settings.get('output_file') or '')
        appended = ('choose a new output path for HDF5' if output.endswith(('.h5', '.hdf5'))
                    else 'the trajectory is appended')
        lines = [f'Last checkpoint: step {step:,} · {shown}',
                 f'Resume: --restart-load {shown} (YAML: restart_load_path: {shown}); '
                 f'relaxation is skipped and {appended}.']
        # A restart runs md_cycles more cycles from the checkpoint, not up to the old end.
        steps_per_cycle = self._settings.get('md_steps')
        remaining = self._start_step + self._total_steps - step
        if isinstance(steps_per_cycle, int) and steps_per_cycle > 0 and remaining > 0:
            cycles = -(-remaining // steps_per_cycle)
            lines.append(f'To finish the original length, also set --md-cycles {cycles} '
                         f'(YAML: md_cycles: {cycles}); md_cycles counts from the checkpoint.')
        return lines

    def checkpoint(self, step, path):
        self._drain_logs()
        self.journal.emit('checkpoint', {'step': int(step), 'path': str(path)})
        self._last_checkpoint = (int(step), str(path))
        self.renderer.checkpoint(step, path)

    def _log_data(self, record):
        if record.levelno < logging.WARNING or getattr(record, 'telemetry_internal', False):
            return
        try:
            message = record.getMessage()
        except Exception:
            # The normal handler owns malformed logging records; observation
            # must not turn its handled formatting error into a run failure.
            return
        return ('error' if record.levelno >= logging.ERROR else 'warning',
                {'message': message, 'logger': record.name, 'level': record.levelname})

    def _record_log(self, data):
        if data is None:
            return
        kind, values = data
        # An old and a replacement console handler can coexist in worker
        # threads during logger setup. Never acquire either handler's lock
        # through a failure warning while holding our lifecycle lock.
        return self.journal.emit(kind, values, defer_warning=True)

    def route_log(self, record, message):
        """Show diagnostics immediately, including during finalization."""
        data = self._log_data(record)
        error = None
        show = False
        with self._log_lock:
            if self._closed:
                return False
            error = self._record_log(data)
            show = self.enabled and (record.levelno >= logging.WARNING or self.renderer.config.verbose)
            if self.enabled and record.levelno >= logging.WARNING:
                self._pending_log = record
        if error is not None:
            self.journal._warn(error)
        # Rich serializes console output with Live. Do not wait for a host hook:
        # a worker warning may be the only clue while a compiled cycle stalls.
        # Neither console output nor diagnostic logging holds _log_lock.
        if show:
            self.renderer.print_log(record, message)
        self._drain_logs()
        return self.enabled

    def _drain_logs(self):
        if threading.get_ident() != self._owner_thread:
            return
        with self._log_lock:
            record, self._pending_log = self._pending_log, None
        if record is not None:
            self.renderer.remember_log(record)

    def finish(self):
        with self._log_lock:
            if self.status != 'running':
                return
            self.status = 'complete'
        self._drain_logs()
        self.journal.emit('run_completed', {'message': 'Run completed successfully.'})
        if self.enabled and self.journal.active:
            self.renderer.artifacts.append(('Metrics', str(self.journal.path)))
        self.final = self.verdict()
        level, head = self.final[:2]
        self.renderer.finish(level, head.partition(': ')[2])

    def fail(self, message, *, display_message=None, reason='error'):
        """Record a failure; ``reason`` is 'interrupt' (Ctrl+C), 'signal' or 'error'."""
        with self._log_lock:
            if self.status != 'running':
                return
            self.status = 'failed'
        self._failure = (message if display_message is None else display_message, reason)
        self.final = self.verdict()
        self._drain_logs()
        self.journal.emit('run_failed', {'message': str(message), 'reason': reason})
        self.renderer.fail(message if display_message is None else display_message, reason=reason)

    def close(self):
        self._drain_logs()
        try:
            if self.enabled:
                self.renderer.close()
        finally:
            # Keep routing through Rich until Live releases the terminal.
            # Stopping Live must not hold the logging/journal lifecycle lock.
            with self._log_lock:
                self._closed = True
            self._drain_logs()
            self.journal.close()


@contextmanager
def run_display(settings, *, record=True):
    """Own one CLI journal/display and restore both, including on Ctrl+C."""
    current = _active.get()
    if current is not None:
        yield current
        return
    rich_enabled = _rich_enabled()
    if _configuration.display == 'rich' and not _configuration.quiet and not rich_enabled:
        print('Warning: --display rich requires a terminal; using plain output.', file=sys.stderr)
    # _rich_enabled already requires stderr to be a terminal.
    with _termination_signals(enabled=rich_enabled):
        renderer = (_Reporter(settings) if rich_enabled
                    else _null if _configuration.quiet else _PlainReporter())
        try:
            reporter = _RunReporter(settings, renderer, record=record)
        except BaseException:
            # A journal may block on a remote filesystem and be interrupted after
            # Live has already hidden the cursor. Own that renderer immediately.
            if renderer.enabled:
                renderer.close()
            raise
        token = _active.set(reporter)
        failure_detail = ''
        resume = []
        try:
            yield reporter
        except BaseException as exc:
            if isinstance(exc, SystemExit) and exc.code in (None, 0):
                reporter.finish()
            else:
                message = exception_message(exc)
                failure_detail = failure_details(exc) if isinstance(exc, Exception) else ''
                display = message.splitlines()[0] if failure_detail else None
                reason = ('interrupt' if isinstance(exc, KeyboardInterrupt)
                          else 'signal' if isinstance(exc, _TerminationSignal) else 'error')
                if reason == 'interrupt':
                    # The journal keeps the exact record; the screen says what happened.
                    display = 'Interrupted (Ctrl+C)'
                reporter.fail(message, display_message=display, reason=reason)
                resume = reporter.resume_hint()
                try:
                    exc._so3lr_reported = reporter.enabled
                except (AttributeError, TypeError):
                    pass
            raise
        else:
            reporter.finish()
        finally:
            try:
                reporter.close()
                if reporter.final is not None:
                    renderer.print_verdict(*reporter.final)
                # One place prints the hints, whichever way the error is reported later;
                # plain runs get the traceback pointer from the CLI's error line instead.
                for line in ([failure_detail] if failure_detail and reporter.enabled else []) + resume:
                    renderer.print_hint(line)
            finally:
                _active.reset(token)


class _ConsoleHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        reporter = _active.get()
        self._reporter = weakref.ref(reporter) if reporter is not None else None

    def emit(self, record):
        if getattr(record, 'file_only', False):
            return
        try:
            # ContextVars intentionally do not cross threads. Bind a weak
            # reference to this handler's invocation rather than consulting a
            # worker's empty context, and never retain a closed display.
            reporter = self._reporter() if self._reporter is not None else None
            if reporter is None or reporter._closed:
                current = _active.get()
                if current is not None and not current._closed:
                    self._reporter = weakref.ref(current)
                    reporter = current
            if reporter is not None and reporter.route_log(record, self.format(record)):
                return
            if _configuration.quiet and record.levelno < logging.WARNING:
                return
            super().emit(record)
        except Exception:
            self.handleError(record)


def make_console_handler(level=logging.INFO):
    """Message-only stderr logging unless the active run owns a Rich display."""
    handler = _ConsoleHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(message)s'))
    return handler
