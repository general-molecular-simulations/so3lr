"""Offline, append-only CLI run events; no numerical or terminal dependencies.

By default each invocation owns one exclusively-created journal. Explicit
SO3LR_JOURNAL=0 disables its sidecars. The simulation remains authoritative:
a filesystem failure disables this optional observer, rather than aborting a
long-running calculation or changing its numerical work.
"""
from datetime import datetime, timezone
import json
import logging
import math
from numbers import Integral, Real
import os
from pathlib import Path
import socket
import sys
import threading
import time
import uuid


_LOGGER = logging.getLogger('SO3LR')
_SCHEDULER_KEYS = (
    'SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_STEP_ID',
    'SLURM_JOB_NAME', 'SLURM_JOB_NODELIST', 'SLURM_NTASKS', 'SLURM_GPUS',
    'SLURM_GPUS_ON_NODE', 'PBS_JOBID', 'PBS_JOBNAME', 'LSB_JOBID', 'LSB_JOBINDEX',
)


def json_value(value):
    """Finite JSON values, without inspecting arbitrary array/device objects.

    Runners pass host scalars and lists. Unknown objects are represented by null;
    calling their repr/item/tolist here could trigger numerical device work.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, os.PathLike):
        return os.fsdecode(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return None


class EventJournal:
    """A best-effort journal flushed at each host-side reporting boundary."""

    def __init__(self, settings):
        self.run_id = uuid.uuid4().hex
        # Requested policy is distinct from active IO: a failed journal should
        # not suppress a replay snapshot that can still be written separately.
        self.enabled = os.environ.get('SO3LR_JOURNAL') != '0'
        self.path = None
        self.seq = 0
        self._stream = None
        self._needs_separator = False
        self._warned = False
        self._lock = threading.RLock()
        self._initializing = True
        self._pending_initial_error = None
        self._started_at = time.monotonic()
        try:
            if not self.enabled:
                return
            base = Path(settings.get('log_file') or settings.get('output_file') or 'so3lr').expanduser()
            base.parent.mkdir(parents=True, exist_ok=True)
            # The exclusive create, not probability alone, guarantees that even
            # colliding IDs cannot overwrite or mix concurrent invocations.
            for attempt in range(4):
                self.path = base.with_name(f'{base.stem}.{self.run_id}.events.jsonl').absolute()
                try:
                    self._stream = self.path.open('x', encoding='utf-8')
                    break
                except FileExistsError:
                    if attempt == 3:
                        raise
                    self.run_id = uuid.uuid4().hex
            from .. import __version__
            self.emit('run_started', {
                'settings': settings,
                'host': socket.gethostname(),
                'pid': os.getpid(),
                'cwd': os.getcwd(),
                'so3lr_version': __version__,
                'scheduler': {key: os.environ[key] for key in _SCHEDULER_KEYS if key in os.environ},
                'journal_path': str(self.path),
            })
        except Exception as exc:
            self._disable(exc)
        except BaseException:
            # The caller cannot close an object whose constructor never
            # returned. Release a partially opened journal without swallowing
            # the original interrupt or exit signal.
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.close()
                except BaseException:
                    pass
            raise
        finally:
            self._initializing = False

    @property
    def active(self):
        with self._lock:
            return self._stream is not None

    def _disable_locked(self, exc):
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        if self._initializing:
            # The SO3LR logger can still own a previous invocation's handlers.
            # Wait until this invocation installs its file and warning counter.
            self._pending_initial_error = exc
            return False
        return True

    def _disable(self, exc):
        with self._lock:
            warn = self._disable_locked(exc)
        if warn:
            self._warn(exc)

    def _warn(self, exc):
        with self._lock:
            if self._warned:
                return
            self._warned = True
        # This marker prevents the reporting handler from journaling its
        # own failure notification. Logging must not fail the calculation.
        try:
            _LOGGER.warning('Run telemetry disabled: %s', exc,
                            extra={'telemetry_internal': True})
        except Exception:
            pass

    def flush_initial_warning(self):
        """Report a startup failure after this invocation's logger is ready."""
        with self._lock:
            error, self._pending_initial_error = self._pending_initial_error, None
        if error is not None:
            self._warn(error)

    def emit(self, event_type, data, *, defer_warning=False):
        """Write one ordered event; optionally return a deferred IO diagnostic.

        A caller serializing its own lifecycle must release its lock before
        logging the returned failure, since logging owns handler locks too.
        """
        error = None
        with self._lock:
            if self._stream is None:
                return
            try:
                event = {
                    'schema': 1,
                    'run_id': self.run_id,
                    'seq': self.seq + 1,
                    'timestamp': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
                    'elapsed_s': float(max(0.0, time.monotonic() - self._started_at)),
                    'type': event_type,
                    'data': json_value(data),
                }
                line = json.dumps(event, allow_nan=False, ensure_ascii=False, separators=(',', ':')) + '\n'
                # Reserve before any write: an interrupt can arrive after the
                # stream accepted the complete record but before returning.
                self.seq = event['seq']
                if self._needs_separator:
                    # A previous interrupted write may have left a JSON prefix.
                    # Keep the file append-only so a live reader's pending
                    # bytes/offset remain valid, and isolate the next event.
                    # Readers skip this separator if the prior line completed.
                    self._stream.write('\n')
                self._needs_separator = True
                self._stream.write(line)
                self._stream.flush()
                self._needs_separator = False
            except Exception as exc:
                if self._disable_locked(exc):
                    error = exc
        # Logging can acquire a console-handler lock held by another thread.
        # Never emit diagnostics while holding the journal's write lock.
        if error is not None and not defer_warning:
            self._warn(error)
        return error

    def close(self):
        error = pending = None
        with self._lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    error = exc
            if self._pending_initial_error is not None and not self._warned:
                pending, self._pending_initial_error = self._pending_initial_error, None
                self._warned = True
        if error is not None:
            self._warn(error)
        if pending is not None:
            # Startup can be interrupted before setup_logger. Preserve the
            # diagnostic without ever writing into a previous run's file.
            try:
                sys.stderr.write(f'WARNING: Run telemetry disabled: {pending}\n')
            except Exception:
                pass
