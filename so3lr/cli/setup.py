"""Guided, offline MD setup. No model or JAX imports occur while drafting a run."""
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys

import click
import yaml

from . import theme

from so3lr.md_settings_validation import DEFAULT_PRECISION, DEFAULT_PRESSURE, validate_md_counts
from so3lr.model_registry import DEFAULT_MODEL, canonical_models, model_aliases


_SCHEDULER_VARIABLES = ('SLURM_JOB_ID', 'SLURM_JOBID', 'PBS_JOBID', 'LSB_JOBID',
                        'JOB_ID', 'COBALT_JOBID')
_ENSEMBLES = ('nvt', 'nvt-langevin', 'nve', 'npt')
_PRESETS = (('Example · Ala4 peptide · 42 atoms · gas phase', 'ala4.xyz'),
            ('Example · DHA · 56 atoms', 'dha.xyz'),
            ('Example · water box · 192 atoms · periodic', 'water_64.xyz'))
REVIEW_ACTIONS = ('Save setup and start MD now', 'Save setup for later', 'Edit settings…', 'Cancel')
_LAUNCHER = '''#!/bin/sh
set -eu
run_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$run_dir"
exec so3lr --settings settings.yaml
'''


def _interactive(stdin=None, stdout=None, environ=None):
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    environ = os.environ if environ is None else environ
    return (bool(getattr(stdin, 'isatty', lambda: False)())
            and bool(getattr(stdout, 'isatty', lambda: False)())
            and environ.get('TERM') != 'dumb')


def should_launch(stdin=None, stdout=None, environ=None):
    """Whether a bare ``so3lr`` may prompt; explicit start can use an allocation."""
    environ = os.environ if environ is None else environ
    return (_interactive(stdin, stdout, environ)
            and not any(environ.get(name) for name in _SCHEDULER_VARIABLES))


@dataclass(frozen=True)
class Structure:
    path: Path
    atoms: int
    formula: str
    periodic: bool
    digest: str
    frame_index: int = 0
    frame_count: int = 1
    cell_lengths: tuple = ()
    minimum_cell_height: float | None = None
    has_momenta: bool = False
    input_temperature: float | None = None
    keep_input_momenta: bool = True
    isolated: bool = False
    electrons: int = 0
    input_charge: int | None = None
    input_unpaired: int | None = None


def _info_integer(info, *keys):
    """Return the first integral value stored under one of ``keys``."""
    from .formatting import finite_number
    for key in keys:
        number = finite_number(info.get(key))
        if number is not None and number.is_integer():
            return int(number)
    return None


def structure_charge_and_unpaired(info):
    """Read total charge and unpaired electrons recorded in an XYZ comment line.

    ``spin`` is deliberately ignored: datasets disagree on whether it means S,
    2S or the multiplicity, and a wrong guess would silently change the physics.
    """
    charge = _info_integer(info, 'total_charge', 'charge')
    unpaired = _info_integer(info, 'num_unpaired_electrons', 'unpaired_electrons')
    if unpaired is None:
        multiplicity = _info_integer(info, 'multiplicity')
        if multiplicity is not None and multiplicity >= 1:
            unpaired = multiplicity - 1
    return charge, (unpaired if unpaired is None or unpaired >= 0 else None)


def electron_count_problem(electrons, charge, unpaired):
    """Explain an impossible charge/multiplicity combination, or return ''."""
    if electrons <= 0:
        return ''
    count = electrons - charge
    if count < 0:
        return f'A total charge of {charge:+d} removes more than the {electrons} electrons of the neutral structure.'
    if unpaired > count:
        return f'{unpaired} unpaired electrons exceed the {count} electrons at charge {charge:+d}.'
    if (count - unpaired) % 2:
        parity = 'an odd' if count % 2 else 'an even'
        return (f'Charge {charge:+d} leaves {count} electrons ({parity} number), which cannot have '
                f'{unpaired} unpaired electrons. Check the total charge or the multiplicity.')
    return ''


def _digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _as_isolated(structure):
    """Describe an explicitly requested transformation of the exported copy."""
    return replace(structure, periodic=False,
                   cell_lengths=(0.0, 0.0, 0.0), minimum_cell_height=None, isolated=True)


def inspect_structure(path, frame_index=None):
    """Inspect a single structure or an explicitly chosen zero-based frame."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Structure file does not exist: {path}')
    if path.suffix.lower() not in ('.xyz', '.extxyz'):
        raise ValueError('Choose an .xyz or .extxyz structure for guided setup.')
    # ASE is needed only after the user chooses a structure. Importing the
    # wizard or displaying command help must not initialize a numerical model.
    import numpy as np
    from ase.io import iread
    from so3lr.pbc import cell_heights, is_fully_periodic

    frames = iread(str(path), format='extxyz', index=':')
    first, count = None, 0
    try:
        for index, atoms in enumerate(frames):
            count += 1
            if frame_index is None and index:
                raise ValueError('The file contains multiple frames. Choose a single frame first.')
            if index == (frame_index or 0):
                first = atoms
        if first is None or len(first) == 0:
            raise ValueError('The chosen frame does not exist or contains no atoms.')
    except KeyError as exc:
        raise ValueError(f'Unrecognized element or property in the XYZ structure: {exc}') from exc
    except (IndexError, AssertionError) as exc:
        raise ValueError('The structure is not a readable single-frame XYZ file.') from exc
    finally:
        frames.close()
    if not np.isfinite(first.positions).all():
        raise ValueError('Atomic positions must contain only finite numbers.')
    cell = np.asarray(first.cell)
    if not np.isfinite(cell).all():
        raise ValueError('Cell vectors must contain only finite numbers.')
    periodic = is_fully_periodic(first.pbc, context='SO3LR MD setup')
    if periodic and (np.any(np.diag(cell) <= 0)
                     or np.any(cell != np.diag(np.diag(cell)))):
        # Both CLI engines require positive, axis-aligned orthorhombic cells
        # for every ensemble. Match check_cell's exact off-diagonal check;
        # accepting tiny tilts here would still fail after saving and launching.
        raise ValueError('Periodic MD requires a positive, axis-aligned orthorhombic cell.')
    height = None
    if periodic:
        height = float(np.min(cell_heights(cell))) if np.linalg.matrix_rank(cell) == 3 else 0.0
    has_momenta = first.has('momenta') and bool(np.any(first.get_momenta()))
    charge, unpaired = structure_charge_and_unpaired(first.info)
    return Structure(path, len(first), first.get_chemical_formula(),
                     periodic, _digest(path), frame_index or 0, count,
                     tuple(float(length) for length in first.cell.lengths()), height,
                     has_momenta, float(first.get_temperature()) if has_momenta else None,
                     electrons=int(first.numbers.sum()), input_charge=charge, input_unpaired=unpaired)


def _decimal(value, label):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f'{label} must be a finite positive number.') from None
    if not number.is_finite() or number <= 0:
        raise ValueError(f'{label} must be a finite positive number.')
    return number


def export_run(directory, structure, settings):
    """Write one new bundle after review; refuse every existing destination."""
    directory = Path(directory).expanduser()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f'The output path already exists: {directory}')
    directory = directory.resolve()
    if not directory.parent.is_dir():
        raise ValueError('The parent output directory must already exist.')
    # Re-read on save so a changed or replaced input cannot silently introduce
    # several frames after the user reviewed a single-frame structure.
    current = inspect_structure(structure.path, frame_index=structure.frame_index)
    if structure.isolated:
        current = _as_isolated(current)
    if current != replace(structure, keep_input_momenta=True):
        raise ValueError('The structure changed during setup. Go back and review it again.')
    directory.mkdir()  # exclusive: never replace a prior run or symlink
    written = []
    try:
        input_path = directory / 'structure.extxyz'
        if structure.frame_count == 1 and structure.keep_input_momenta and not structure.isolated:
            # Another process may have populated the new directory after mkdir.
            # Reserve each file exclusively and claim cleanup ownership only
            # after creation succeeds, including when ASE rewrites a frame.
            with input_path.open('xb') as handle:
                written.append(input_path)
                with structure.path.open('rb') as source:
                    shutil.copyfileobj(source, handle)
            copied_digest = _digest(input_path)
        else:
            from ase.io import read, write
            atoms = read(structure.path, index=structure.frame_index, format='extxyz')
            if not structure.keep_input_momenta:
                atoms.set_momenta(None)
            if structure.isolated:
                atoms.set_pbc(False)
                atoms.set_cell([0, 0, 0])
            with input_path.open('x', encoding='utf-8') as handle:
                written.append(input_path)
                write(handle, atoms, format='extxyz')
            copied_digest = _digest(structure.path)
        if copied_digest != structure.digest:
            raise ValueError('The structure changed while saving. Go back and review it again.')
        settings_path = directory / 'settings.yaml'
        with settings_path.open('x') as handle:
            written.append(settings_path)
            handle.write('# Generated by so3lr start. Timestep md_dt is in femtoseconds.\n')
            yaml.safe_dump(settings, handle, sort_keys=False)
        launcher = directory / 'run.sh'
        with launcher.open('x') as handle:
            written.append(launcher)
            handle.write(_LAUNCHER)
        launcher.chmod(0o755)
    except BaseException:
        # Remove only files this export owns; never recursively remove content
        # another process might have placed in the new directory.
        for path in reversed(written):
            path.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass
        raise
    return settings_path


def _prompt(label, **kwargs):
    # Indent answers under their section rule.
    return click.prompt('  ' + label, **kwargs)


def _confirm(label, **kwargs):
    return click.confirm('  ' + label, **kwargs)


def _say(console, text, style=''):
    from rich.text import Text
    from .prompts import terminal_text
    console.print(Text(terminal_text(text), style=style))


def _stage_rule(number, title):
    from rich.rule import Rule
    from rich.text import Text
    from . import theme
    from .prompts import terminal_text
    return Rule(Text(terminal_text(f'{number}/3 · {title}'), style=theme.ACCENT_BOLD),
                align='left', style=theme.FAINT)


def _stage(console, number, title):
    console.print()
    console.print(_stage_rule(number, title))


def _choose(console, title, choices, default=1):
    from .prompts import choose
    return choose(console, title, choices, default=default - 1)


def _select_model(console, previous=DEFAULT_MODEL):
    """Keep current models prominent and every canonical release reachable."""
    aliases = model_aliases()
    names = canonical_models()
    current = [(aliases[f'so3lr-tauri-{size}'], label) for size, label in
               (('m', 'M'), ('s', 'S (faster)'), ('l', 'L (recommended for reactions)'))
               if aliases.get(f'so3lr-tauri-{size}') in names]
    current_names = [name for name, _ in current]
    legacy = [name for name in names if name not in current_names]
    choices = [f'{label}{" (default)" if name == DEFAULT_MODEL else ""} · {name}'
               for name, label in current]
    if legacy:
        choices.append('Legacy models…')
    default = current_names.index(previous) + 1 if previous in current_names else len(choices)
    while True:
        selected = _choose(console, 'Model (canonical release)', choices, default)
        if selected < len(current):
            return current[selected][0]
        legacy_default = legacy.index(previous) + 1 if previous in legacy else 1
        selected = _choose(console, 'Legacy models', legacy + ['← Back to current models'], legacy_default)
        if selected < len(legacy):
            return legacy[selected]
        default = len(choices)


def available_presets():
    """Small example structures shipped with the package, never read from cwd."""
    data = Path(__file__).resolve().parent / 'examples'
    return [(label, data / filename) for label, filename in _PRESETS if (data / filename).is_file()]


def _entered_path(value):
    value = value.strip()
    if not value or '\x00' in value:
        raise ValueError('Enter a nonempty path without null characters.')
    # Terminal drag-and-drop commonly surrounds a path containing spaces with
    # quotes. Remove a matching pair only when it is not itself a real filename.
    if not Path(value).expanduser().exists() and len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return Path(value).expanduser()


def _path_prompt(default=None, *, allow_directory=False):
    from .prompts import terminal_text
    while True:
        answer = _prompt('Structure path (.xyz or .extxyz)', default=default, type=str)
        try:
            path = _entered_path(answer)
            if allow_directory and path.is_dir():
                return path
            if not path.is_file():
                raise ValueError(f'Choose an existing structure file: {path}')
            if path.suffix.lower() not in ('.xyz', '.extxyz'):
                raise ValueError('Choose an .xyz or .extxyz structure file.')
            return path
        except (OSError, ValueError, RuntimeError) as exc:
            click.echo('Error: ' + terminal_text(exc))


def browse_structure(console, start=None):
    """Navigate directories; return a file or None to revisit source choices."""
    directory = Path(start or Path.cwd()).expanduser().resolve()
    while True:
        try:
            children = list(directory.iterdir())
            folders = sorted((path for path in children if path.is_dir()), key=lambda path: path.name.casefold())
            files = sorted((path for path in children if path.is_file()
                            and path.suffix.lower() in ('.xyz', '.extxyz')), key=lambda path: path.name.casefold())
        except OSError as exc:
            _say(console, f'Cannot list this directory: {exc}', theme.DANGER)
            folders, files = [], []
        choices = ['↑ Up one directory', '← Back to structure sources', 'Enter or paste a path']
        choices += [f'{path.name}/' for path in folders] + [path.name for path in files]
        selected = _choose(console, f'Browse · {directory}', choices)
        if selected == 0:
            directory = directory.parent
        elif selected == 1:
            return None
        elif selected == 2:
            path = _path_prompt(allow_directory=True)
            if path.is_dir():
                directory = path.resolve()
            else:
                return path
        elif selected < 3 + len(folders):
            directory = folders[selected - 3].resolve()
        else:
            return files[selected - 3 - len(folders)]


def _pick_frame(console, path, old):
    from ase.io import iread
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in ('.xyz', '.extxyz'):
        return inspect_structure(path)  # standard, actionable path errors
    frames = iread(str(path), format='extxyz', index=':')
    count = 0
    try:
        for _ in frames:
            count += 1
    except KeyError as exc:
        raise ValueError(f'Unrecognized element or property in the XYZ structure: {exc}') from exc
    finally:
        frames.close()
    if count <= 1:
        return inspect_structure(path)
    default = old.get('frame_index', 0) + 1 if old.get('source') == str(path) else 1
    selected = _choose(console, f'{path.name} · {count} frames · choose the frame to copy',
                       [f'Frame {index + 1} of {count}' for index in range(count)], min(default, count))
    return inspect_structure(path, frame_index=selected)


def _select_structure(console, old):
    presets = available_presets()
    while True:
        choices = [label for label, _ in presets] + ['Browse files', 'Enter or paste a path', 'Cancel']
        if old.get('source'):
            choices.insert(0, f'Keep previous · {Path(old["source"]).name}')
        selection = _choose(console, 'Choose a structure', choices)
        if old.get('source'):
            if selection == 0:
                source = old['source']
            else:
                selection -= 1
                source = None
        else:
            source = None
        if source is None:
            if selection < len(presets):
                source = presets[selection][1]
            elif selection == len(presets):
                source = browse_structure(console, Path(old['source']).parent if old.get('source') else None)
                if source is None:
                    continue
            elif selection == len(presets) + 1:
                source = _path_prompt(old.get('source'))
            else:
                raise click.Abort()
        try:
            # A bundled example is a demo input: copy its first frame without a frame menu.
            preset = any(source == path for _, path in presets)
            structure = (inspect_structure(source, frame_index=0) if preset
                         else _pick_frame(console, source, old))
        except (ValueError, OSError, IndexError, AssertionError) as exc:
            _say(console, exc, theme.DANGER)
            continue
        dha_path = next((path for _, path in presets if path.name == 'dha.xyz'), None)
        if structure.path == dha_path and structure.periodic:
            lengths = ', '.join(f'{length:.6g}' for length in structure.cell_lengths)
            boundary = _choose(console, 'DHA boundary conditions',
                               (f'Keep input periodic cell ({lengths} Å)',
                                'Use isolated molecule (remove cell from exported copy)'),
                               2 if old.get('isolated') and old.get('source') == str(structure.path) else 1)
            if boundary == 1:
                structure = _as_isolated(structure)
                _say(console, 'The exported copy will use gas-phase boundaries; the source stays unchanged.')
        _say(console, f'{structure.formula} · {structure.atoms} atoms · '
             f'{"PBC: all three axes" if structure.periodic else "PBC: off (gas phase)"}')
        if structure.periodic:
            _say(console, 'Cell lengths: ' + ', '.join(f'{length:.6g}' for length in structure.cell_lengths) + ' Å')
        if structure.frame_count > 1:
            _say(console, f'Copying frame {structure.frame_index + 1} of {structure.frame_count}; source stays unchanged.')
        return structure


def _unused_directory(name):
    """Suggest ``name``, or ``name-2``, ``name-3``, ... when it already exists."""
    base = Path(name).expanduser()
    candidate, number = base, 2
    while candidate.exists() or candidate.is_symlink():
        candidate = base.with_name(f'{base.name}-{number}')
        number += 1
    return name if candidate == base else str(Path(name).with_name(candidate.name))


def _plain_number(value):
    """'5.00' -> '5', '0.050' -> '0.05', without scientific notation."""
    try:
        number = Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return str(value)
    return format(number, 'f')


def _positive(console, label, default):
    while True:
        answer = _prompt(label, default=_plain_number(default), type=str)
        try:
            value = _decimal(answer, label)
            if not float(value) > 0 or not math.isfinite(float(value)):
                raise ValueError(f'{label} is outside the supported numeric range.')
            return value
        except (ValueError, OverflowError) as exc:
            _say(console, exc, theme.DANGER)


def frame_plan(duration_ps, frame_ps, dt_fs):
    """Whole frames covering at least ``duration_ps``; frames are whole timesteps.

    Returns ``(frames, steps_per_frame)``. The duration is rounded up to a
    whole number of frames, and the caller shows the exact result.
    """
    duration = _decimal(duration_ps, 'Duration')
    frame = _decimal(frame_ps, 'Frame interval')
    dt = _decimal(dt_fs, 'Timestep')
    steps = frame * 1000 / dt
    if steps != steps.to_integral_value():
        raise ValueError(f'The frame interval must be a whole number of {dt:g} fs timesteps '
                         f'(for example {dt * 100 / 1000:g} ps).')
    frames = int((duration / frame).to_integral_value(rounding='ROUND_CEILING'))
    return max(1, frames), int(steps)


_EDITS = (('Structure, dynamics and model (start over)', 'start_over'),
          ('Temperature and pressure', 'ask_conditions'),
          ('Duration and frame interval', 'ask_length'),
          ('Timestep', 'ask_timestep'),
          ('Charge and multiplicity', 'ask_chemistry'),
          ('Seed and initial relaxation', 'ask_seed_and_relaxation'),
          ('Long-range cutoff', 'ask_cutoff'),
          ('Output directory', 'ask_directory'),
          ('← Back to review', None))
EDIT_CHOICES = tuple(label for label, _ in _EDITS)


class _Draft:
    """Answers of one guided setup; each group can be asked again from the review."""

    def __init__(self, console, old):
        from .so3lr_cli import (DEFAULT_MD_CYCLES, DEFAULT_MD_STEPS_PER_CYCLE, DEFAULT_SAVE_BUFFER,
                                DEFAULT_SEED, DEFAULT_TEMPERATURE, DEFAULT_TIMESTEP)
        self.console, self.old = console, old
        self.default_seed, self.default_temperature = DEFAULT_SEED, DEFAULT_TEMPERATURE
        self.default_dt, self.default_buffer = Decimal(str(DEFAULT_TIMESTEP)), DEFAULT_SAVE_BUFFER
        self.default_frame = self.default_dt * DEFAULT_MD_STEPS_PER_CYCLE / 1000
        self.default_duration = self.default_frame * DEFAULT_MD_CYCLES

    # -- groups ---------------------------------------------------------
    def ask_system(self):
        from .setup_physics import long_range_settings
        console, old = self.console, self.old
        while True:
            _stage(console, 1, 'Structure')
            self.structure = _select_structure(console, old)
            _stage(console, 2, 'Dynamics')
            while True:
                self.ensemble = _ENSEMBLES[_choose(console, 'Dynamics',
                    ('NVT · Nosé–Hoover', 'NVT · Langevin', 'NVE', 'NPT'),
                    _ENSEMBLES.index(old.get('ensemble', 'nvt')) + 1)]
                if self.ensemble != 'npt' or self.structure.periodic:
                    break
                _say(console, 'NPT requires a fully periodic, positive, axis-aligned orthorhombic cell.', theme.DANGER)
            self.model = _select_model(console, old.get('model', DEFAULT_MODEL))
            try:
                self.proposal = long_range_settings(self.structure.periodic, self.structure.minimum_cell_height,
                                                    self.model, ensemble=self.ensemble)
                return
            except ValueError as exc:
                _say(console, exc, theme.DANGER)
                _say(console, 'Choose another structure or model to continue.')

    def ask_conditions(self):
        console, old = self.console, self.old
        self.temperature = _positive(console, 'Initial temperature (K)' if self.ensemble == 'nve' else 'Temperature (K)',
                                     getattr(self, 'temperature', old.get('temperature', self.default_temperature)))
        if self.structure.has_momenta:
            keep = _choose(console, 'The input contains velocities',
                           (f'Initialize velocities at {self.temperature:g} K',
                            f'Retain input velocities ({self.structure.input_temperature:.1f} K '
                            'before center-of-mass removal)'),
                           2 if getattr(self, 'kept_momenta', old.get('keep_input_momenta')) else 1) == 1
            self.structure = replace(self.structure, keep_input_momenta=keep)
            self.kept_momenta = keep
        previous = getattr(self, 'pressure', None) or old.get('pressure') or DEFAULT_PRESSURE
        self.pressure = None
        if self.ensemble == 'npt':
            self.pressure = _positive(console, 'Pressure (bar)', previous)

    def ask_length(self):
        console, old = self.console, self.old
        while True:
            self.duration = _positive(console, 'Duration (ps)',
                                      getattr(self, 'duration', old.get('duration', self.default_duration)))
            self.frame = _positive(console, 'Save a trajectory frame every (ps)',
                                   getattr(self, 'frame', old.get('frame', self.default_frame)))
            try:
                self._plan()
                return
            except ValueError as exc:
                _say(console, exc, theme.DANGER)

    def ask_timestep(self):
        self.dt = _positive(self.console, 'Timestep (fs)', self.dt)
        try:
            self._plan()
        except ValueError as exc:  # the frame interval no longer fits the timestep
            _say(self.console, exc, theme.DANGER)
            self.ask_length()

    def ask_chemistry(self):
        console, old, structure = self.console, self.old, self.structure
        same_source = old.get('source') == str(structure.path)
        charge_default = getattr(self, 'charge', old['charge'] if same_source and 'charge' in old
                                 else (structure.input_charge or 0))
        spin_default = getattr(self, 'spin', old['spin'] if same_source and 'spin' in old
                               else (structure.input_unpaired or 0))
        recorded = [label for value, label in ((structure.input_charge, 'total charge'),
                                               (structure.input_unpaired, 'multiplicity')) if value is not None]
        if recorded and not same_source and not hasattr(self, 'charge'):
            _say(console, f'Default {" and ".join(recorded)} read from the structure file.', theme.MUTED)
        while True:
            self.charge = _prompt('Total charge (e)', default=charge_default, type=int)
            self.spin = _prompt('Unpaired electrons (multiplicity minus one)', default=spin_default,
                                type=click.IntRange(min=0))
            problem = electron_count_problem(structure.electrons, self.charge, self.spin)
            if not problem:
                return
            _say(console, problem, theme.DANGER)
            if _confirm('Keep these values anyway?', default=False):
                return
            charge_default, spin_default = self.charge, self.spin

    def ask_seed_and_relaxation(self):
        self.seed = _prompt('Random seed', default=self.seed, type=click.IntRange(min=0))
        self.relax = _confirm('Relax the geometry before MD?', default=self.relax)

    def ask_cutoff(self):
        from .setup_physics import long_range_settings
        self._announce_pme()
        while True:
            cutoff = _positive(self.console, 'Long-range cutoff (Å)', self.cutoff)
            try:
                long_range_settings(self.structure.periodic, self.structure.minimum_cell_height, self.model,
                                    cutoff=float(cutoff), ensemble=self.ensemble)
                self.cutoff = cutoff
                return
            except ValueError as exc:
                _say(self.console, exc, theme.DANGER)

    def ask_directory(self):
        default = getattr(self, 'directory', None)
        default = str(default) if default is not None else _unused_directory(self.old.get('directory', 'so3lr-run'))
        while True:
            answer = _prompt('New output directory', default=default, type=str)
            try:
                destination = _entered_path(answer).absolute()
                if destination.exists() or destination.is_symlink():
                    raise ValueError('That path already exists. Choose a new output directory.')
                if not destination.parent.is_dir():
                    raise ValueError('The parent directory must already exist.')
                self.directory = destination.resolve()
                return
            except (OSError, ValueError, RuntimeError) as exc:
                _say(self.console, exc, theme.DANGER)

    def use_advanced_defaults(self):
        old, structure = self.old, self.structure
        self.dt = Decimal(str(old.get('dt', self.default_dt)))
        self.seed = old.get('seed', self.default_seed)
        self.relax = old.get('relax', True)
        same_input = (old.get('source') == str(structure.path) and old.get('model') == self.model
                      and old.get('isolated', False) == structure.isolated)
        self.cutoff = Decimal(str(old['cutoff'])) if same_input and 'cutoff' in old else \
            Decimal(str(self.proposal['lr_cutoff']))

    def _announce_pme(self):
        proposal = self.proposal
        if 'kspace_electrostatics' in proposal:
            _say(self.console, f'Proposed for this cell: {proposal["lr_cutoff"]:g} Å real-space cutoff, '
                 'PME electrostatics and PME dispersion. '
                 f'Electrostatics: {proposal["kspace_smearing"]:g} Å smearing, '
                 f'{proposal["kspace_spacing"]:g} Å mesh spacing; '
                 f'dispersion: {proposal["kspace_dispersion_smearing"]:g} Å smearing, '
                 f'{proposal["kspace_dispersion_spacing"]:g} Å mesh spacing. '
                 'These are starting settings; convergence depends on the system.')

    def _plan(self):
        frames, steps = frame_plan(self.duration, self.frame, getattr(self, 'dt', self.default_dt))
        self.frames, self.steps_per_frame = frames, steps
        exact = Decimal(frames) * self.frame
        if exact != self.duration:
            _say(self.console, f'Duration rounded up to {exact:g} ps ({frames:,} whole frames).', theme.MUTED)

    # -- the full first pass ------------------------------------------------
    def collect(self):
        self.ask_system()
        self.use_advanced_defaults()
        self.ask_conditions()
        self.ask_length()
        self.ask_chemistry()
        self._announce_pme()
        _say(self.console, f'Defaults: {float(self.dt):g} fs timestep · seed {self.seed} · relaxation '
             f'{"on" if self.relax else "off"} · {float(self.cutoff):g} Å cutoff. '
             'Change them with Edit settings in the review.', theme.MUTED)
        self.ask_directory()
        return self

    def edit(self, choice):
        """Ask one group again; return False when the whole setup restarts."""
        method = _EDITS[choice][1]
        if method == 'start_over':
            return False
        if method is not None:
            getattr(self, method)()
        return True

    # -- results ------------------------------------------------------------
    def result(self):
        from .setup_physics import long_range_settings
        structure = self.structure
        long_range = long_range_settings(structure.periodic, structure.minimum_cell_height, self.model,
                                         cutoff=float(self.cutoff), ensemble=self.ensemble)
        settings = dict(input_file='structure.extxyz', output_file='trajectory.extxyz', log_file='run.log',
                        restart_save_path='restart.npz', overwrite=False, model=self.model,
                        engine='auto', precision=DEFAULT_PRECISION, ensemble=self.ensemble,
                        md_T=float(self.temperature), md_dt=float(self.dt), md_cycles=self.frames,
                        md_steps=self.steps_per_frame, save_buffer=min(self.default_buffer, self.frames),
                        total_charge=self.charge, num_unpaired_electrons=self.spin, seed=self.seed,
                        relax_before_run=self.relax)
        settings.update(long_range)
        if self.ensemble == 'nve':
            settings['init_T'] = float(self.temperature)
        if self.pressure is not None:
            settings['md_P'] = float(self.pressure)
        validate_md_counts(settings)
        return structure, self.directory, settings

    def memory(self):
        """Answers to prefill a fresh setup after 'start over'."""
        structure = self.structure
        return dict(source=str(structure.path), frame_index=structure.frame_index,
                    keep_input_momenta=structure.keep_input_momenta, isolated=structure.isolated,
                    ensemble=self.ensemble, model=self.model, temperature=str(self.temperature),
                    pressure=str(self.pressure) if self.pressure is not None else None,
                    dt=str(self.dt), duration=str(self.duration), frame=str(self.frame),
                    charge=self.charge, spin=self.spin, seed=self.seed, relax=self.relax,
                    cutoff=str(self.cutoff), directory=str(self.directory))


def _collect(console, previous):
    return _Draft(console, previous or {}).collect()


def review_view(structure, destination, settings):
    """Render the actual review without reading, writing or starting a run."""
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text
    from .formatting import home_path, precision_label
    from .prompts import terminal_text
    precision = precision_label(settings['precision'])
    steps = settings['md_cycles'] * settings['md_steps']
    system = [('Structure', f'{home_path(structure.path)} ({structure.atoms} atoms; {structure.formula})'),
              ('Boundary', 'Gas phase · input cell removed from exported copy' if structure.isolated
               else 'Periodic' if structure.periodic else 'Gas phase'),
              *([('Cell lengths', ', '.join(f'{length:.6g}' for length in structure.cell_lengths) + ' Å')]
                if structure.periodic else []),
              ('Source frame', f'{structure.frame_index + 1} of {structure.frame_count}'),
              ('Charge / unpaired electrons', f"{settings['total_charge']} / {settings['num_unpaired_electrons']}")]
    dynamics = [('Model', settings['model']), ('Dynamics', settings['ensemble'].upper()),
                ('Initial temperature' if settings['ensemble'] == 'nve' else 'Target temperature',
                 f"{settings['md_T']:g} K")]
    if 'md_P' in settings:
        dynamics.append(('Target pressure', f"{settings['md_P']:g} bar"))
    if structure.has_momenta:
        dynamics.append(('Initial velocities',
                         f'Retain input ({structure.input_temperature:.1f} K before center-of-mass removal)'
                         if structure.keep_input_momenta else f"Initialize at {settings['md_T']:g} K"))
    dynamics += [('Run length', f"{steps:,} steps · {settings['md_dt']:g} fs · "
                  f"{steps * settings['md_dt'] / 1000:g} ps"),
                 ('Trajectory frames', f"Every {settings['md_steps'] * settings['md_dt'] / 1000:g} ps "
                  f"({settings['md_steps']:,} steps) · {settings['md_cycles']:,} frames"),
                 ('Checkpoint cadence', f"Every {settings['save_buffer']} cycles · "
                  f"{settings['save_buffer'] * settings['md_steps']:,} steps · "
                  f"{settings['save_buffer'] * settings['md_steps'] * settings['md_dt'] / 1000:g} ps; also at completion"),
                 ('Seed / initial relaxation', f"{settings['seed']} / {'yes' if settings['relax_before_run'] else 'no'}"),
                 ('Long-range cutoff', f"{settings['lr_cutoff']:g} Å")]
    if settings.get('kspace_electrostatics') == 'pme':
        dynamics.append(('Electrostatics', f"PME · {settings['kspace_smearing']:g} Å smearing · "
                         f"{settings['kspace_spacing']:g} Å mesh spacing"))
    if settings.get('kspace_dispersion') == 'pme':
        dynamics.append(('Dispersion', f"PME · {settings['kspace_dispersion_smearing']:g} Å smearing · "
                         f"{settings['kspace_dispersion_spacing']:g} Å mesh spacing"))
    dynamics.append(('Engine / precision', f'auto (checked at launch) / {precision}'))
    produced = ['trajectory.extxyz', 'run.log', 'restart.npz']
    if settings.get('relax_before_run'):
        produced.insert(1, 'trajectory_opt.xyz')
    output = [('Output directory', home_path(destination)),
              ('Files', 'structure.extxyz, settings.yaml, run.sh; the run adds ' + ', '.join(produced))]
    # No border: paths and values must stay copyable as plain text.
    table = Table.grid(padding=(0, 2))
    table.add_column(style=theme.MUTED, no_wrap=True)
    table.add_column(overflow='fold')
    for index, (heading, rows) in enumerate((('System', system), ('Dynamics', dynamics), ('Output', output))):
        table.add_row(Text(heading, style=theme.ACCENT_BOLD), '')
        for label, value in rows:
            table.add_row(Text(label), Text(terminal_text(value)))
    return Group(Text(''), _stage_rule(3, 'Review and start'), table,
                 Text('Start now saves structure.extxyz, settings.yaml and run.sh, then runs MD in this terminal.'),
                 Text('Save for later writes those files only. Start later with sh run.sh from the output directory.',
                      style=theme.MUTED))


def _review(console, structure, destination, settings):
    console.print(review_view(structure, destination, settings))


def launch_setup():
    """Export an MD run; return its absolute YAML path only for Start MD now.

    The caller must run the normal settings pathway with the bundle as its
    working directory. A save-only or cancelled wizard returns None.
    """
    if not _interactive():
        raise click.ClickException('Guided setup needs an interactive input and output terminal. '
                                   'Use so3lr --settings settings.yaml for noninteractive runs.')
    from rich.console import Console
    from rich.text import Text
    context = click.get_current_context(silent=True)
    console = Console(file=sys.stdout, markup=False, highlight=False,
                      no_color='NO_COLOR' in os.environ or (context is not None and context.color is False))
    from .prompts import terminal_text
    from .formatting import ascii_only_console
    ascii_only = ascii_only_console(console)
    title = Text(terminal_text(f'{theme.BRAND} SO3LR TAURI', ascii_only), style=theme.ACCENT_BOLD)
    title.append(terminal_text(' · New MD run', ascii_only), style=theme.STRONG)
    console.print(title)
    _say(console, 'Choose a structure, set the dynamics, then review and start or save.', theme.MUTED)
    console.print(theme.controls('↑/↓: move · Enter: select · Esc: cancel setup · Ctrl+C: cancel anywhere',
                                 ascii_only))
    try:
        draft = _collect(console, None)
        while True:
            structure, destination, settings = draft.result()
            _review(console, structure, destination, settings)
            action = _choose(console, 'Next action · Enter starts MD now', REVIEW_ACTIONS)
            if action == 2:
                if not draft.edit(_choose(console, 'Edit settings', EDIT_CHOICES)):
                    draft = _collect(console, draft.memory())
                continue
            if action == 3:
                _say(console, 'Setup cancelled. No files written.')
                return None
            try:
                settings_path = export_run(destination, structure, settings)
            except (OSError, ValueError) as exc:
                _say(console, f'Could not save the run: {exc}', theme.DANGER)
                continue
            from .formatting import home_path
            _say(console, f'{theme.status_glyph("completed")} Saved run: {home_path(destination)}', theme.SUCCESS)
            if action == 0:
                _say(console, 'Starting MD now in this terminal.')
                return settings_path
            _say(console, 'Start later with sh run.sh from that directory, in an environment with SO3LR installed.')
            return None
    except (click.Abort, KeyboardInterrupt, EOFError):
        _say(console, 'Setup cancelled. No files written.')
        return None
