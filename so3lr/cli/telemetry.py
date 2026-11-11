"""Offline, append-only CLI run events; no numerical or terminal dependencies.

Every invocation owns one exclusively-created journal. The simulation remains
authoritative: a filesystem failure disables this optional observer, rather
than aborting a long-running calculation or changing its numerical work.
"""
from datetime import datetime, timezone
import json
import logging
import math
from numbers import Integral, Real
import os
from pathlib import Path
import socket
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
        self.path = None
        self.seq = 0
        self._stream = None
        self._warned = False
        self._started_at = time.monotonic()
        try:
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

    @property
    def active(self):
        return self._stream is not None

    def _disable(self, exc):
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        if not self._warned:
            self._warned = True
            # This marker prevents the reporting handler from journaling its
            # own failure notification. Logging must not fail the calculation.
            try:
                _LOGGER.warning('Run telemetry disabled: %s', exc,
                                extra={'telemetry_internal': True})
            except Exception:
                pass

    def emit(self, event_type, data):
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
            self._stream.write(line)
            self._stream.flush()
            self.seq += 1
        except Exception as exc:
            self._disable(exc)

    def close(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                self._disable(exc)
