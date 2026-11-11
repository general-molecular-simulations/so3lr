"""Human-readable run records and replay settings, without numerical imports.

This optional observer consumes host reports only. The scientific table and the
simulation remain authoritative; a reporting failure must not stop a run.
"""
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import socket
import time

from .formatting import optimization_status, precision_label
from .telemetry import json_value


_LOGGER = logging.getLogger('SO3LR')
_PATH_KEYS = frozenset({
    'input_file', 'output_file', 'log_file', 'restart_load_path',
    'restart_save_path', 'restart_pysages_load_path', 'restart_pysages_save_path',
    'atom_mask_path', 'pysages_input',
})
_NON_SIMULATION_KEYS = frozenset({
    'datafile', 'save_to', 'targets', 'batch_size', 'jit_compile', 'accuracy',
    'cutoff_lo', 'cutoff_it', 'cutoff_hi', 'smear_lo', 'smear_hi', 'smear_it',
    'ns_lo', 'ns_hi', 'nodes_lo', 'nodes_hi', 'mesh_lo', 'mesh_hi',
    'dispersion_damping_lo', 'dispersion_damping_hi', 'dispersion_damping_it',
    'md_precision', 'flash_mixed',
})


def _absolute(value, cwd):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else cwd / path)


def resolved_settings(settings, info, operation, cwd):
    """Serialize only replayable settings; MD's internal ps become YAML fs.

    Runtime metadata overrides requested aliases/automatic engine selection.
    Paths retain their meaning even when replay starts in another directory.
    """
    from .so3lr_cli import YAML_DEFAULTS, YAML_KEY_ALIASES, known_yaml_keys
    from ..model_registry import available_models
    source = dict(settings)
    for alias, canonical in YAML_KEY_ALIASES.items():
        if alias in source:
            source.setdefault(canonical, source.pop(alias))
    if operation == 'optimization':
        from .optimization_settings import optimization_keys
        from .so3lr_cli import fire_optimization
        allowed = optimization_keys(fire_optimization)
    else:
        allowed = known_yaml_keys() - _NON_SIMULATION_KEYS
        source = dict(YAML_DEFAULTS, **source)
        if source.get('do_enhanced_sampling', False):
            # check_enhanced_sampling reads this cwd-relative default when the
            # key is omitted. Materialize it before rebasing replay paths.
            source.setdefault('pysages_input', './pysages_input.yaml')
    result = {}
    for key, value in source.items():
        if key not in allowed:
            continue
        converted = json_value(value)
        if converted is not None:
            result[key] = converted
        elif value is None and operation != 'optimization' and YAML_DEFAULTS.get(key) is not None:
            # Explicit null can differ from an omitted key (e.g. total_charge).
            # safe_dump writes YAML null, never the invalid Python string None.
            result[key] = None
    for field, key in (('model', 'model_path'), ('precision', 'precision'), ('engine', 'engine')):
        if info.get(field) is not None:
            result[key] = str(info[field])
    if operation == 'optimization':
        result['operation'] = 'optimization'
        if info.get('optimization_output_file'):
            result['output_file'] = info['optimization_output_file']
    elif result.get('md_dt') is not None:
        result['md_dt'] = float(result['md_dt']) * 1000.0
    # Replay is protected even when the original invocation opted into overwrite.
    result['overwrite'] = False
    for key in _PATH_KEYS & result.keys():
        if result[key] is not None:
            result[key] = _absolute(result[key], cwd)
    if result.get('model_path') and result['model_path'] not in available_models():
        result['model_path'] = _absolute(result['model_path'], cwd)
    return result


class _CountRecords(logging.Handler):
    def __init__(self, owner):
        super().__init__(logging.WARNING)
        self.owner = owner

    def emit(self, record):
        # Count records, including file-only diagnostics; this does not emit or
        # format logs, so it cannot recurse through the normal console handler.
        if record.levelno >= logging.ERROR:
            self.owner.errors += 1
        else:
            self.owner.warnings += 1


class RunLog:
    """Best-effort observer attached after the invocation's logger is ready."""

    def __init__(self, settings, journal):
        self.settings = dict(settings)
        self.journal = journal
        self.cwd = Path.cwd()
        self.started = time.monotonic()
        self.started_utc = datetime.now(timezone.utc).isoformat(timespec='seconds')
        self.info = {}
        self.operation = None
        self.path = None
        self.settings_path = None
        self._settings_owned = False
        self._settings_text = None
        self._settings_disabled = False
        self._disabled = False
        self._warned = set()
        self._headers = set()
        self._identity = None
        self._counter = _CountRecords(self)
        self.warnings = self.errors = 0
        self.md = None
        self.sample = {}
        self.saved_checkpoint = None
        self._phase = None
        self._phase_started = self.started
        self._phase_seconds = {}
        self._initial_output = self._fingerprint(settings.get('output_file'))

    def _fingerprint(self, value):
        if not value:
            return None
        try:
            stat = Path(_absolute(value, self.cwd)).stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns
        except OSError:
            return None

    def _warning(self, feature, exc):
        if feature in self._warned:
            return
        self._warned.add(feature)
        try:
            _LOGGER.warning('%s disabled: %s', feature, exc)
        except Exception:
            pass

    def _write(self, message):
        _LOGGER.info(message, extra={'file_only': True})

    def _guard(self, function, *args):
        if self._disabled:
            return
        try:
            function(*args)
        except Exception as exc:
            self._disabled = True
            self._warning('Run summary', exc)

    def logger_ready(self, log_file):
        self._guard(self._logger_ready, log_file)

    def _logger_ready(self, log_file):
        if self.operation is None:
            # A later optimization report may be pre-MD relaxation. Lock the
            # invocation's operation before that nested phase can change it.
            self.operation = 'optimization' if self.info.get('operation') == 'optimization' else 'md'
        if self._counter not in _LOGGER.handlers:
            _LOGGER.addHandler(self._counter)
        self.path = Path(_absolute(log_file, self.cwd)) if log_file else None
        if self.path and str(self.path) not in self._headers:
            self._headers.add(str(self.path))
            self._write(f'=== BEGIN RUN {self.journal.run_id} ===')
            self._write(f'Run started (UTC): {self.started_utc}; operation: {self.operation}; '
                        f'host: {socket.gethostname()}; pid: {os.getpid()}')
            self._write(f'Working directory: {self.cwd}')
            self._write(f'Event journal: {self.journal.path or "unavailable"}')
            self._write('Table units: step = integration-step counter; E/PE/KE/H = eV; '
                        'T = K; cell = A; P (when present) = bar; timing = s/step. '
                        'Journal volume = A^3; journal pressure = eV/A^3. '
                        'Optimization: energy = eV; Fmax = max absolute force component in eV/A. '
                        'Physical MD time in the journal and summary is local to this invocation.')
            self._identity = None
        self._report_identity()
        self._save_settings()

    def update_info(self, values):
        self.info.update({key: json_value(value) for key, value in values.items() if value is not None})
        if self.operation is not None:
            self._guard(self._report_identity)
            if set(values) & {'model', 'precision', 'engine', 'optimization_output_file'}:
                self._save_settings()

    def _report_identity(self, force=False):
        fields = []
        for field, key in (('model', 'model_path'), ('engine', 'engine'), ('precision', 'precision')):
            actual = self.info.get(field)
            value = actual if actual is not None else self.settings.get(key, 'default')
            if field == 'precision':
                value = precision_label(value)
            label = 'backend' if field == 'engine' else field
            fields.append(f'{label}: {value}' + (' (requested; unresolved)' if actual is None else ''))
        devices = self.info.get('devices')
        if isinstance(devices, (list, tuple)):
            devices = ', '.join(map(str, devices))
        fields.extend([f'hardware: {devices or "not reported"}',
                       f'seed: {self.settings.get("seed", "not applicable")}'])
        if self.info.get('atoms') is not None:
            fields.append(f'atoms: {self.info["atoms"]}')
        identity = tuple(fields)
        if force or identity != self._identity:
            self._write(f'Model: {fields[0].removeprefix("model: ")}; {fields[1]}')
            self._write('Precision: ' + fields[2].removeprefix('precision: '))
            self._write('Hardware and system: ' + '; '.join(fields[3:]))
            self._identity = identity

    def _save_settings(self):
        if self.operation is None or self._settings_disabled:
            return
        try:
            import yaml
            values = resolved_settings(self.settings, self.info, self.operation, self.cwd)
            if self.settings_path is None:
                journal_path = self.journal.path
                if journal_path is None:
                    base = Path(self.settings.get('log_file') or self.settings.get('output_file') or 'so3lr')
                    journal_path = Path(_absolute(base, self.cwd)).with_name(
                        f'{base.stem}.{self.journal.run_id}.events.jsonl')
                self.settings_path = journal_path.with_name(
                    journal_path.name.removesuffix('.events.jsonl') + '.settings.resolved.yaml')
            unresolved = [key for key in ('model', 'precision', 'engine') if self.info.get(key) is None]
            text = (f'# SO3LR run {self.journal.run_id}\n'
                    '# Replay: so3lr --settings <this-file>\n'
                    '# MD md_dt is in fs; lengths in A; energies in eV; FIRE dt is unchanged.\n'
                    '# Paths are absolute. Choose NEW output/log/checkpoint destinations before replay.\n'
                    '# overwrite is false. Existing outputs remain protected.\n'
                    '# A restart path is preserved, not archived; its checkpoint may have advanced.\n')
            if unresolved:
                text += '# Runtime resolution incomplete for: ' + ', '.join(unresolved) + '.\n'
            text += yaml.safe_dump(values, sort_keys=True, allow_unicode=True)
            if text == self._settings_text:
                return
            if not self._settings_owned:
                with self.settings_path.open('x', encoding='utf-8') as stream:
                    stream.write(text)
                self._settings_owned = True
            else:
                # Only replace the UUID-owned file we created, never an unrelated
                # replay file. Readers observe a complete YAML document.
                temporary = self.settings_path.with_suffix('.yaml.tmp')
                temporary_owned = False
                try:
                    with temporary.open('x', encoding='utf-8') as stream:
                        temporary_owned = True
                        stream.write(text)
                    os.replace(temporary, self.settings_path)
                finally:
                    if temporary_owned and temporary.exists():
                        temporary.unlink()
            self._settings_text = text
        except Exception as exc:
            self._settings_disabled = True
            self._warning('Resolved settings export', exc)

    def phase(self, name):
        now = time.monotonic()
        if self._phase:
            self._phase_seconds[self._phase] = self._phase_seconds.get(self._phase, 0.) + now - self._phase_started
        self._phase, self._phase_started = str(name), now

    def start_md(self, start_step, total_steps, dt_fs):
        self.md = dict(start_step=start_step, total_steps=total_steps, dt_fs=dt_fs,
                       started=time.monotonic())

    def progress(self, data):
        self.sample = json_value(data)

    def checkpoint(self, step, path):
        self.saved_checkpoint = (int(step), _absolute(path, self.cwd))

    def finish(self, status, message=''):
        self._guard(self._finish, status, message)

    def _finish(self, status, message):
        if self.operation is None:
            return  # A library/mocked display that never configured CLI logging.
        now = time.monotonic()
        self.phase('Finished')
        self._save_settings()
        self._write(f'=== RUN SUMMARY {self.journal.run_id} ===')
        self._write(f'Execution: {"completed" if status == "complete" else "failed/interrupted"}'
                    + (f'; {message}' if message else ''))
        self._report_identity(force=True)
        if self.operation == 'optimization':
            self._write(f'Optimization: {optimization_status(self.info, status)}; '
                        f'completed steps: {self.info.get("opt_step", 0)} / '
                        f'{self.info.get("opt_total_steps", "not reported")}')
            for key, label in (('opt_energy_eV', 'Final potential energy (eV)'),
                               ('opt_force_max_eVA', 'Final Fmax (eV/A)'),
                               ('opt_force_target_eVA', 'Force target (eV/A)')):
                if self.info.get(key) is not None:
                    self._write(f'{label}: {self.info[key]:.12g}')
        elif self.md is not None:
            completed = self.sample.get('segment_step') or 0
            elapsed_ps = self.sample.get('time_ps') or 0.
            self._write(f'MD completed additional steps: {completed} / {self.md["total_steps"]}; '
                        f'segment time: {elapsed_ps:.12g} ps; dt: {self.md["dt_fs"]} fs; '
                        f'counter: {self.md["start_step"]} -> {self.sample.get("step", self.md["start_step"])}')
            for key, label in (('temperature_K', 'Final temperature (K)'),
                               ('energy_eV', 'Final total energy (eV)'),
                               ('potential_energy_eV', 'Final potential energy (eV)'),
                               ('kinetic_energy_eV', 'Final kinetic energy (eV)'),
                               ('conserved_energy_eV', 'Final conserved quantity (eV)'),
                               ('volume_A3', 'Final volume (A^3)')):
                if self.sample.get(key) is not None:
                    self._write(f'{label}: {self.sample[key]:.12g}')
        else:
            self._write('MD integration did not start; no completed MD steps reported.')
        self._write(f'Warnings: {self.warnings}; errors: {self.errors} (log records)')
        self._write(f'Wall time: {now - self.started:.6f} s')
        if self.md is not None:
            self._write(f'Before MD: {self.md["started"] - self.started:.6f} s; '
                        f'MD stage: {now - self.md["started"]:.6f} s '
                        '(includes first-cycle compilation and output)')
        if 'Compiling' in self._phase_seconds:
            self._write(f'Reported Compiling stage: {self._phase_seconds["Compiling"]:.6f} s '
                        '(stage wall time, not a compiler-only measurement)')
        if self.saved_checkpoint:
            step, path = self.saved_checkpoint
            self._write(f'Last saved checkpoint: step {step}; {path}')
        else:
            self._write('Last saved checkpoint: none reported for this invocation')
        for label, path in (('Log', self.path), ('Events', self.journal.path),
                            ('Resolved settings', self.settings_path if self._settings_owned else None)):
            if path:
                self._write(f'Artifact {label}: {path}')
        output = self.settings.get('output_file')
        optimization_output = self.info.get('optimization_output_file')
        same_output = (output and optimization_output
                       and _absolute(output, self.cwd) == _absolute(optimization_output, self.cwd))
        if output and not same_output and self._fingerprint(output) not in (None, self._initial_output):
            self._write(f'Artifact Trajectory: {_absolute(output, self.cwd)}')
        if optimization_output:
            self._write(f'Artifact Optimization trajectory: {_absolute(optimization_output, self.cwd)}')
        self._write(f'=== END RUN {self.journal.run_id} ===')

    def close(self):
        _LOGGER.removeHandler(self._counter)
        self._counter.close()
