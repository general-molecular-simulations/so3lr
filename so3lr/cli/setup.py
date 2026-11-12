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

from so3lr.md_settings_validation import DEFAULT_PRECISION, DEFAULT_PRESSURE, validate_md_counts
from so3lr.model_registry import DEFAULT_MODEL, canonical_models, model_aliases


_SCHEDULER_VARIABLES = ('SLURM_JOB_ID', 'SLURM_JOBID', 'PBS_JOBID', 'LSB_JOBID',
                        'JOB_ID', 'COBALT_JOBID')
_ENSEMBLES = ('nvt', 'nvt-langevin', 'nve', 'npt')
_PRESETS = (('Ala4 · ac.xyz', 'ac.xyz'), ('DHA · dha.xyz', 'dha.xyz'))
REVIEW_ACTIONS = ('Save setup and start MD now', 'Save setup for later', 'Back', 'Cancel')
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
    npt_compatible: bool
    digest: str
    frame_index: int = 0
    frame_count: int = 1
    cell_lengths: tuple = ()
    minimum_cell_height: float | None = None
    has_momenta: bool = False
    input_temperature: float | None = None
    keep_input_momenta: bool = True
    isolated: bool = False


def _digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _as_isolated(structure):
    """Describe an explicitly requested transformation of the exported copy."""
    return replace(structure, periodic=False, npt_compatible=False,
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
    npt_ok = periodic
    height = None
    if periodic:
        height = float(np.min(cell_heights(cell))) if np.linalg.matrix_rank(cell) == 3 else 0.0
    has_momenta = first.has('momenta') and bool(np.any(first.get_momenta()))
    return Structure(path, len(first), first.get_chemical_formula(),
                     periodic, npt_ok, _digest(path), frame_index or 0, count,
                     tuple(float(length) for length in first.cell.lengths()), height,
                     has_momenta, float(first.get_temperature()) if has_momenta else None)


def _decimal(value, label):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f'{label} must be a finite positive number.') from None
    if not number.is_finite() or number <= 0:
        raise ValueError(f'{label} must be a finite positive number.')
    return number


def duration_cycles(duration_ps, dt_fs, steps_per_cycle):
    """Convert user units exactly; never round a requested duration silently."""
    duration = _decimal(duration_ps, 'Duration')
    dt = _decimal(dt_fs, 'Timestep')
    if isinstance(steps_per_cycle, bool) or not isinstance(steps_per_cycle, int) or steps_per_cycle < 1:
        raise ValueError('Steps per cycle must be a positive integer.')
    cycle_ps = dt * steps_per_cycle / Decimal(1000)
    cycles = duration / cycle_ps
    if cycles != cycles.to_integral_value():
        raise ValueError(f'Duration must be an exact multiple of {cycle_ps:g} ps '
                         f'({steps_per_cycle} steps × {dt:g} fs). Change the duration or cadence.')
    return int(cycles)


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


def _say(console, text, style=''):
    from rich.text import Text
    from .prompts import terminal_text
    console.print(Text(terminal_text(text), style=style))


def _stage(console, number, title):
    _say(console, f'{number}/3 · {title}', 'bold yellow')


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
    """Return only presets belonging to this source checkout, never to cwd."""
    checkout = Path(__file__).resolve().parents[2]
    if not (checkout / 'pyproject.toml').is_file():
        return []
    data = checkout / 'tests' / 'test_data'
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
        answer = click.prompt('Structure path (.xyz or .extxyz)', default=default, type=str)
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
            _say(console, f'Cannot list this directory: {exc}', 'red')
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
            structure = _pick_frame(console, source, old)
        except (ValueError, OSError, IndexError, AssertionError) as exc:
            _say(console, exc, 'red')
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
        _say(console, 'Cell lengths: ' + ', '.join(f'{length:.6g}' for length in structure.cell_lengths) + ' Å')
        if structure.frame_count > 1:
            _say(console, f'Copying frame {structure.frame_index + 1} of {structure.frame_count}; source stays unchanged.')
        return structure


def _positive(console, label, default):
    while True:
        answer = click.prompt(label, default=str(default), type=str)
        try:
            value = _decimal(answer, label)
            if not float(value) > 0 or not math.isfinite(float(value)):
                raise ValueError(f'{label} is outside the supported numeric range.')
            return value
        except (ValueError, OverflowError) as exc:
            _say(console, exc, 'red')


def _collect(console, previous):
    # Read authoritative CLI defaults only after entering the wizard. There
    # is no runtime import from the MD implementation and no duplicated table.
    from .so3lr_cli import (DEFAULT_MD_CYCLES, DEFAULT_MD_STEPS_PER_CYCLE, DEFAULT_SAVE_BUFFER,
                            DEFAULT_SEED, DEFAULT_TEMPERATURE, DEFAULT_TIMESTEP)
    from .setup_physics import long_range_settings
    old = previous or {}
    while True:
        _stage(console, 1, 'Structure')
        structure = _select_structure(console, old)
        _stage(console, 2, 'Dynamics')
        while True:
            ensemble = _ENSEMBLES[_choose(console, 'Dynamics',
                ('NVT · Nosé–Hoover', 'NVT · Langevin', 'NVE', 'NPT'),
                _ENSEMBLES.index(old.get('ensemble', 'nvt')) + 1)]
            if ensemble != 'npt' or structure.npt_compatible:
                break
            _say(console, 'NPT requires a fully periodic, positive, axis-aligned orthorhombic cell.', 'red')
        model = _select_model(console, old.get('model', DEFAULT_MODEL))
        try:
            proposal = long_range_settings(structure.periodic, structure.minimum_cell_height, model,
                                           ensemble=ensemble)
            break
        except ValueError as exc:
            _say(console, exc, 'red')
            _say(console, 'Choose another structure or model to continue.')
    temperature = _positive(console, 'Initial temperature (K)' if ensemble == 'nve' else 'Temperature (K)',
                            old.get('temperature', DEFAULT_TEMPERATURE))
    if structure.has_momenta:
        keep = _choose(console, 'The input contains velocities',
                       (f'Initialize velocities at {temperature:g} K',
                        f'Retain input velocities ({structure.input_temperature:.1f} K before center-of-mass removal)'),
                       2 if old.get('keep_input_momenta') else 1) == 1
        structure = replace(structure, keep_input_momenta=keep)
    pressure = None
    if ensemble == 'npt':
        pressure = _positive(console, 'Pressure (bar)', old.get('pressure') or DEFAULT_PRESSURE)
    default_ps = Decimal(str(DEFAULT_TIMESTEP)) * DEFAULT_MD_CYCLES * DEFAULT_MD_STEPS_PER_CYCLE / 1000
    while True:
        dt = _positive(console, 'Timestep (fs)', old.get('dt', DEFAULT_TIMESTEP))
        duration = _positive(console, 'Duration (ps)', old.get('duration', default_ps))
        cadence = click.prompt('Steps per cycle (one trajectory frame per cycle)',
                               default=old.get('cadence', DEFAULT_MD_STEPS_PER_CYCLE),
                               type=click.IntRange(min=1))
        try:
            cycles = duration_cycles(duration, dt, cadence)
            break
        except ValueError as exc:
            _say(console, exc, 'red')
    charge = click.prompt('Total charge (e)', default=old.get('charge', 0), type=int)
    spin = click.prompt('Unpaired electrons (multiplicity minus one)', default=old.get('spin', 0),
                        type=click.IntRange(min=0))
    seed = click.prompt('Random seed', default=old.get('seed', DEFAULT_SEED), type=click.IntRange(min=0))
    relax = click.confirm('Relax the geometry before MD?', default=old.get('relax', True))
    if 'kspace_electrostatics' in proposal:
        _say(console, f'Proposed for this cell: {proposal["lr_cutoff"]:g} Å real-space cutoff, '
             'PME electrostatics and PME dispersion. '
             f'Electrostatics: {proposal["kspace_smearing"]:g} Å smearing, '
             f'{proposal["kspace_spacing"]:g} Å mesh spacing; '
             f'dispersion: {proposal["kspace_dispersion_smearing"]:g} Å smearing, '
             f'{proposal["kspace_dispersion_spacing"]:g} Å mesh spacing. '
             'These are starting settings; convergence depends on the system.')
    same_input = (old.get('source') == str(structure.path) and old.get('model') == model
                  and old.get('isolated', False) == structure.isolated)
    cutoff_default = old.get('cutoff', proposal['lr_cutoff']) if same_input else proposal['lr_cutoff']
    while True:
        cutoff = _positive(console, 'Long-range cutoff (Å)', cutoff_default)
        try:
            long_range = long_range_settings(structure.periodic, structure.minimum_cell_height, model,
                                             cutoff=float(cutoff), ensemble=ensemble)
            break
        except ValueError as exc:
            _say(console, exc, 'red')
    while True:
        answer = click.prompt('New output directory', default=old.get('directory', 'so3lr-run'), type=str)
        try:
            destination = _entered_path(answer).absolute()
            if destination.exists() or destination.is_symlink():
                raise ValueError('That path already exists. Choose a new output directory.')
            if not destination.parent.is_dir():
                raise ValueError('The parent directory must already exist.')
            destination = destination.resolve()
            break
        except (OSError, ValueError, RuntimeError) as exc:
            _say(console, exc, 'red')
    settings = dict(input_file='structure.extxyz', output_file='trajectory.extxyz', log_file='run.log',
                    restart_save_path='restart.npz', overwrite=False, model=model,
                    engine='auto', precision=DEFAULT_PRECISION, ensemble=ensemble,
                    md_T=float(temperature), md_dt=float(dt), md_cycles=cycles,
                    md_steps=cadence, save_buffer=min(DEFAULT_SAVE_BUFFER, cycles),
                    total_charge=charge, num_unpaired_electrons=spin, seed=seed,
                    relax_before_run=relax)
    settings.update(long_range)
    if ensemble == 'nve':
        settings['init_T'] = float(temperature)
    if pressure is not None:
        settings['md_P'] = float(pressure)
    validate_md_counts(settings)
    draft = dict(source=str(structure.path), frame_index=structure.frame_index,
                 keep_input_momenta=structure.keep_input_momenta, isolated=structure.isolated,
                 ensemble=ensemble, model=model, temperature=str(temperature),
                 pressure=str(pressure) if pressure is not None else None, dt=str(dt), duration=str(duration),
                 cadence=cadence, charge=charge, spin=spin, seed=seed, relax=relax,
                 cutoff=str(cutoff), directory=str(destination))
    return structure, destination, settings, draft


def review_view(structure, destination, settings):
    """Render the actual review without reading, writing or starting a run."""
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text
    from .formatting import precision_label
    from .prompts import terminal_text
    precision = precision_label(settings['precision'])
    rows = [('Structure', f'{structure.path} ({structure.atoms} atoms; {structure.formula})'),
            ('Boundary', 'Gas phase · input cell removed from exported copy' if structure.isolated
             else 'Periodic' if structure.periodic else 'Gas phase'),
            ('Cell lengths', ', '.join(f'{length:.6g}' for length in structure.cell_lengths) + ' Å'),
            ('Source frame', f'{structure.frame_index + 1} of {structure.frame_count}'),
            ('Model', settings['model']), ('Dynamics', settings['ensemble'].upper()),
            ('Initial temperature' if settings['ensemble'] == 'nve' else 'Target temperature',
             f"{settings['md_T']:g} K"),
            ('Run length', f"{settings['md_cycles'] * settings['md_steps']:,} steps · "
             f"{settings['md_dt']:g} fs · "
             f"{settings['md_cycles'] * settings['md_steps'] * settings['md_dt'] / 1000:g} ps"),
            ('Trajectory cadence', f"Every {settings['md_steps']} steps"),
            ('Checkpoint cadence', f"Every {settings['save_buffer']} cycles · "
             f"{settings['save_buffer'] * settings['md_steps']:,} steps · "
             f"{settings['save_buffer'] * settings['md_steps'] * settings['md_dt'] / 1000:g} ps; also at completion"),
            ('Charge / unpaired electrons', f"{settings['total_charge']} / {settings['num_unpaired_electrons']}"),
            ('Seed / initial relaxation', f"{settings['seed']} / {settings['relax_before_run']}"),
            ('Long-range cutoff', f"{settings['lr_cutoff']:g} Å"),
            ('Engine / precision', f'auto (checked at launch) / {precision}'),
            ('Output directory', str(destination)),
            ('Files', 'structure.extxyz, settings.yaml, run.sh, trajectory.extxyz, run.log, restart.npz')]
    if structure.has_momenta:
        rows.insert(5, ('Initial velocities',
                    f'Retain input ({structure.input_temperature:.1f} K before center-of-mass removal)'
                    if structure.keep_input_momenta else f"Initialize at {settings['md_T']:g} K"))
    if 'md_P' in settings:
        rows.insert(5, ('Target pressure', f"{settings['md_P']:g} bar"))
    if settings.get('kspace_electrostatics') == 'pme':
        rows.insert(-3, ('Electrostatics', f"PME · {settings['kspace_smearing']:g} Å smearing · "
                         f"{settings['kspace_spacing']:g} Å mesh spacing"))
    if settings.get('kspace_dispersion') == 'pme':
        rows.insert(-3, ('Dispersion', f"PME · {settings['kspace_dispersion_smearing']:g} Å smearing · "
                         f"{settings['kspace_dispersion_spacing']:g} Å mesh spacing"))
    table = Table.grid(padding=(0, 2))
    table.add_column(style='dim')
    table.add_column(overflow='fold')
    for label, value in rows:
        table.add_row(Text(label), Text(terminal_text(value)))
    return Group(Text('3/3 · Review and start', style='bold yellow'), table,
                 Text('Start now saves structure.extxyz, settings.yaml and run.sh, then runs MD in this terminal.'),
                 Text('Save for later writes those files only. Start later with sh run.sh from the output directory.', style='dim'))


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
    context = click.get_current_context(silent=True)
    console = Console(file=sys.stdout, markup=False, highlight=False,
                      no_color='NO_COLOR' in os.environ or (context is not None and context.color is False))
    _say(console, 'SO3LR TAURI · New MD run', 'bold yellow')
    _say(console, 'Choose a structure, set the dynamics, then review and start or save.')
    _say(console, '↑/↓ and Enter select · Esc cancels a menu · Ctrl+C cancels anywhere', 'dim')
    previous = None
    try:
        while True:
            structure, destination, settings, previous = _collect(console, previous)
            _review(console, structure, destination, settings)
            action = _choose(console, 'Next action · Enter starts MD now', REVIEW_ACTIONS)
            if action == 2:
                continue
            if action == 3:
                _say(console, 'Setup cancelled. No files written.')
                return None
            try:
                settings_path = export_run(destination, structure, settings)
            except (OSError, ValueError) as exc:
                _say(console, f'Could not save the run: {exc}', 'red')
                continue
            _say(console, f'Saved run: {destination}', 'green')
            if action == 0:
                _say(console, 'Starting MD now in this terminal.')
                return settings_path
            _say(console, 'Start later with sh run.sh from that directory, in an environment with SO3LR installed.')
            return None
    except (click.Abort, KeyboardInterrupt, EOFError):
        _say(console, 'Setup cancelled. No files written.')
        return None
