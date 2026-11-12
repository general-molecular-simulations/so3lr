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
from .run_log import RunLog
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

    def finish(self):
        pass

    def fail(self, message, reason='error'):
        pass

    def logger_ready(self, log_file):
        pass


_null = _NullReporter()


def get_reporter():
    """Return the current display, or a harmless reporter for library callers."""
    return _active.get() or _null


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
            activity = _Activity(self.phase_name, self._opened_at, self._reported_at, not self._ascii_only)
        return Group(*rows), activity

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
            completed = max(0, min(self.step - self.start_step, self.total_steps))
            ratio = completed / self.total_steps if self.total_steps else 0
            progress = f'{ratio:.0%} · {completed:,}/{self.total_steps:,} steps'
            segment_time = (f'{completed * self.dt_fs / 1000:.3f} / '
                            f'{self.total_steps * self.dt_fs / 1000:.3f} ps this run') if self.dt_fs is not None else ''
            timing = f'MD elapsed {duration_label(time.monotonic() - self.started_at)}'
            if self.status == 'running':
                timing += (f' · ETA {duration_label((self.total_steps - completed) * self.time_per_step)}'
                           if completed and self.time_per_step is not None and self.time_per_step > 0
                           else ' · ETA estimating')
            self._state.info['_attached_ratio'] = ratio
            self._state.info['_attached_progress'] = progress + f' · step counter {self.step:,}'
            self._state.info['_attached_time'] = segment_time
            self._state.info['_attached_compact_progress'] = progress + (' · ' + segment_time if segment_time else '')
            self._state.info['_attached_timing'] = timing
            self._state.info['_attached_keyboard'] = self._keyboard is not None
            return render_monitor([self._reader], energy=list(ENERGIES)[self._energy_index],
                                  width=self.console.width, height=self.console.height - int(self._heartbeat),
                                  ascii_only=self._ascii_only,
                                  page=self._plot_page, attached=True)

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
        phase = self._text(f'{glyph} {self.phase_name}', f'bold {style}')
        if self.detail:
            phase.append(self._label(f'  {self.detail}'), style=theme.MUTED)
        rows.append(phase)
        if self.info.get('operation') == 'optimization':
            step, total = (_finite(self.info.get(key)) for key in ('opt_step', 'opt_total_steps'))
            if step is not None and total is not None and total > 0:
                ratio = max(0.0, min(1.0, step / total))
                width = min(40, max(8, self.console.width - 40))
                bar = theme.bar(ratio, width, self._ascii_only)
                bar.append(f'  {int(step):,}/{int(total):,} optimization steps', style=theme.MUTED)
                rows.append(bar)
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
        if self.started_at is not None:
            completed = max(0, min(self.step - self.start_step, self.total_steps))
            ratio = completed / self.total_steps if self.total_steps else 0
            width = min(40, max(8, self.console.width - 40))
            bar = theme.bar(ratio, width, self._ascii_only)
            bar.append(f'  {ratio:.0%}', style=theme.STRONG)
            bar.append(f'  {completed:,}/{self.total_steps:,} steps', style=theme.MUTED)
            rows.append(bar)
            if self.start_step:
                rows.append(self._text(f'Restart counter {self.start_step:,} · step counter {self.step:,}', theme.MUTED))
            if self.dt_fs is not None:
                rows.append(self._text(f'{completed * self.dt_fs / 1000:.3f} / '
                                       f'{self.total_steps * self.dt_fs / 1000:.3f} ps this run', theme.MUTED))
            elapsed = time.monotonic() - self.started_at
            timing = f'MD elapsed {duration_label(elapsed)}'
            # No ETA is invented during compilation or at the restored step.
            if completed and self.time_per_step is not None and self.time_per_step > 0:
                if self.status == 'running':
                    timing += f' · ETA {duration_label((self.total_steps - completed) * self.time_per_step)}'
                rate = throughput_label(self.time_per_step, self.dt_fs, self.info.get('atoms'))
                if rate:
                    timing += f' · {rate}'
            elif self.status == 'running':
                timing += ' · ETA estimating'
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
        for label, path in self.artifacts:
            rows.append(self._text(f'{label} · {path}', theme.MUTED))
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
        return Group(*rows)

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
        self.phase_name, self.detail = phase
        self._reported_at = time.monotonic()
        self._state.phase, self._state.detail = phase
        # Show the first potentially blocking preparation stage immediately;
        # repeated Running/Saving transitions share the normal refresh budget.
        first_preparation = phase not in self._seen_phases and name in ('Loading', 'Compiling', 'Relaxing')
        self._seen_phases.add(phase)
        self._refresh(force=first_preparation)

    def update_info(self, **kwargs):
        self._reported_at = time.monotonic()
        self.info.update({key: value for key, value in kwargs.items() if value is not None})
        self.requested.difference_update(kwargs)
        self._state.info.update({key: value for key, value in kwargs.items() if value is not None})
        self._refresh()

    def start_md(self, start_step, total_steps, dt_fs):
        """Begin this invocation's work; total_steps excludes restored steps."""
        self.start_step = self.step = int(start_step)
        self.total_steps = max(0, int(total_steps))
        self.dt_fs = _finite(dt_fs)
        self.started_at = time.monotonic()
        self.time_per_step = self.temperature = self.potential_energy = None
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

    def finish(self):
        if self.status != 'running':
            return
        self.status = 'complete'
        self.phase_name, self.detail = 'Complete', ''
        self._observe('run_completed', {'message': 'Run completed successfully.'})
        output_label = 'Optimization trajectory' if self.info.get('operation') == 'optimization' else 'Trajectory'
        for key, label in (('output_file', output_label), ('log_file', 'Log')):
            path = self.settings.get(key)
            if path:
                try:
                    if os.path.isfile(path):
                        self.artifacts.append((label, str(path)))
                except (OSError, TypeError, ValueError):
                    pass
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


PLAIN_PROGRESS_INTERVAL = 600  # seconds between plain progress lines without a checkpoint


class _RunReporter:
    """Fan out host events to the journal and an optional terminal renderer."""

    observing = True

    def __init__(self, settings, renderer):
        self.renderer = renderer
        self._settings = settings
        self._last_checkpoint = None
        # enabled continues to mean terminal rendering, not journal availability.
        self.enabled = renderer.enabled
        self.journal = EventJournal(settings)
        try:
            self.run_log = RunLog(settings, self.journal)
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
        self._rate_origin = None     # (wall clock, step) of the first MD sample
        self._last_progress_line = None

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
        self.run_log.phase(phase[0])
        self.renderer.phase(*phase)

    def update_info(self, **kwargs):
        self._drain_logs()
        self.journal.emit('info', kwargs)
        self.run_log.update_info(kwargs)
        self.renderer.update_info(**kwargs)

    def logger_ready(self, log_file):
        self._drain_logs()
        self.run_log.logger_ready(log_file)
        self.journal.flush_initial_warning()

    def start_md(self, start_step, total_steps, dt_fs):
        # Relaxation can share this reporter; MD must replace its operation
        # marker in both the journal and attached display before new samples.
        self.update_info(operation='md')
        self._start_step, self._dt_fs = int(start_step), _finite(dt_fs)
        self._total_steps, self._rate_origin, self._last_progress_line = int(total_steps), None, None
        self.journal.emit('md_started', {'start_step': self._start_step,
                                       'total_steps': int(total_steps), 'dt_fs': self._dt_fs})
        self.run_log.start_md(self._start_step, int(total_steps), self._dt_fs)
        self.renderer.start_md(start_step, total_steps, dt_fs)

    def progress(self, step, temperature=None, potential_energy=None, time_per_step=None, **optional):
        self._drain_logs()
        step = int(step)
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
        self.run_log.progress(data)
        if self._rate_origin is None:
            self._rate_origin = (time.monotonic(), step)
        elif (not self.enabled and self._last_progress_line is not None
              and time.monotonic() - self._last_progress_line >= PLAIN_PROGRESS_INTERVAL):
            self._log_progress(step)
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
        shown = _display_path(path)
        output = str(self._settings.get('output_file') or '')
        appended = ('choose a new output path for HDF5' if output.endswith(('.h5', '.hdf5'))
                    else 'the trajectory is appended')
        return [f'Last checkpoint: step {step:,} · {shown}',
                f'Resume: --restart-load {shown} (YAML: restart_load_path: {shown}); '
                f'relaxation is skipped and {appended}.']

    def _log_progress(self, step):
        """One greppable progress line for plain and batch output."""
        self._last_progress_line = time.monotonic()
        if self._total_steps is None or self._start_step is None or not self._total_steps:
            return
        done = max(0, min(step - self._start_step, self._total_steps))
        parts = [f'{done / self._total_steps:.1%}', f'step {done:,}/{self._total_steps:,}']
        if self._rate_origin is not None:
            started, first = self._rate_origin
            elapsed = time.monotonic() - started
            if elapsed > 0 and step > first:
                rate = (step - first) / elapsed  # steps per wall second, I/O included
                if self._dt_fs:
                    parts.append(throughput_label(1 / rate, self._dt_fs))
                if done < self._total_steps:
                    parts.append(f'ETA {duration_label((self._total_steps - done) / rate)}')
        logging.getLogger('SO3LR').info('Progress ' + ' · '.join(parts))

    def checkpoint(self, step, path):
        self._drain_logs()
        self.journal.emit('checkpoint', {'step': int(step), 'path': str(path)})
        self.run_log.checkpoint(step, path)
        self._last_checkpoint = (int(step), str(path))
        if not self.enabled:
            self._log_progress(int(step))
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
        self.run_log.finish('complete')
        self.journal.emit('run_completed', {'message': 'Run completed successfully.'})
        if self.enabled and self.journal.active:
            self.renderer.artifacts.append(('Metrics', str(self.journal.path)))
        self.renderer.finish()

    def fail(self, message, *, display_message=None, reason='error'):
        """Record a failure; ``reason`` is 'interrupt' (Ctrl+C), 'signal' or 'error'."""
        with self._log_lock:
            if self.status != 'running':
                return
            self.status = 'failed'
        self._drain_logs()
        self.run_log.finish('failed', str(message))
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
            self.run_log.close()
            self.journal.close()


@contextmanager
def run_display(settings):
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
        renderer = _Reporter(settings) if rich_enabled else _null
        try:
            reporter = _RunReporter(settings, renderer)
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
                # One place prints the hints, whichever way the error is reported later.
                if reporter.enabled:
                    for line in ([failure_detail] if failure_detail else []) + resume:
                        renderer.console.print(renderer._text(line, theme.MUTED))
                elif not _configuration.quiet:
                    for line in resume:
                        print(terminal_text(line), file=sys.stderr)
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
