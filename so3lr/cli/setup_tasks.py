"""Guided flows for `so3lr start` beyond MD: optimization, dataset evaluation,
watching runs, and pointers to the expert commands.

Like MD setup, nothing is written before an explicit review choice, and no
model or JAX import happens while drafting.
"""
from dataclasses import dataclass
from pathlib import Path
import shlex

import click

from . import theme
from .setup import (_choose, _confirm, _entered_path, _positive, _prompt, _review_actions_for, _say,
                    _select_model, _select_structure, _stage, _stage_rule, _unused_directory,
                    electron_count_problem, export_run)
from so3lr.md_settings_validation import DEFAULT_PRECISION
from so3lr.model_registry import DEFAULT_MODEL

TASKS = (('Run molecular dynamics', 'md'),
         ('Optimize a geometry', 'opt'),
         ('Evaluate a dataset (energies and forces)', 'eval'),
         ('Watch runs (so3lr show)', 'show'),
         ('Other tasks: fine-tuning, training, k-space tuning', 'other'))
TASK_CHOICES = tuple(label for label, _ in TASKS)
DOCS_URL = 'https://so3lr.github.io/usage/cli/'

_EVAL_LAUNCHER = '''#!/bin/sh
set -eu
run_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$run_dir"
exec {command}
'''


@dataclass(frozen=True)
class Action:
    """What the CLI should run after setup: a saved command, or the run viewer."""
    kind: str                 # 'eval' or 'show'
    argv: tuple = ()
    directory: Path | None = None


def _charge_and_spin(console, structure):
    charge = _prompt('Total charge (e)', default=structure.input_charge or 0, type=int)
    spin = _prompt('Unpaired electrons (multiplicity minus one)', default=structure.input_unpaired or 0,
                   type=click.IntRange(min=0))
    problem = electron_count_problem(structure.electrons, charge, spin)
    if problem:
        _say(console, problem, theme.DANGER)
        if not _confirm('Keep these values anyway?', default=False):
            return _charge_and_spin(console, structure)
    return charge, spin


def _new_directory(console, default):
    while True:
        answer = _prompt('New output directory', default=_unused_directory(default), type=str)
        try:
            destination = _entered_path(answer).absolute()
            if destination.exists() or destination.is_symlink():
                raise ValueError('That path already exists. Choose a new output directory.')
            if not destination.parent.is_dir():
                raise ValueError('The parent directory must already exist.')
            return destination.resolve()
        except (OSError, ValueError, RuntimeError) as exc:
            _say(console, exc, theme.DANGER)


def _rows_view(number, title, rows, notes):
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text
    from .prompts import terminal_text
    table = Table.grid(padding=(0, 2))
    table.add_column(style=theme.MUTED, no_wrap=True)
    table.add_column(overflow='fold')
    for label, value in rows:
        table.add_row(Text(label), Text(terminal_text(value)))
    return Group(Text(''), _stage_rule(number, title), table,
                 *(Text(note, style=theme.MUTED if index else '') for index, note in enumerate(notes)))


# -- geometry optimization -----------------------------------------------------

def optimization_setup(console):
    """Draft, review and export an optimization bundle; return its settings path to run now."""
    from .formatting import home_path
    from .setup_physics import long_range_settings
    _stage(console, 1, 'Structure')
    structure = _select_structure(console, {})
    _stage(console, 2, 'Optimization')
    model = _select_model(console, DEFAULT_MODEL)
    while True:
        try:
            long_range = long_range_settings(structure.periodic, structure.minimum_cell_height, model,
                                             ensemble='nvt')
            break
        except ValueError as exc:
            _say(console, exc, theme.DANGER)
            model = _select_model(console, model)
    force = _positive(console, 'Stop when the largest force is below (eV/Å)', '0.05')
    budget = _prompt('Maximum optimization steps', default=1000, type=click.IntRange(min=10))
    charge, spin = _charge_and_spin(console, structure)
    destination = _new_directory(console, 'so3lr-opt')
    steps_per_frame = 10
    settings = dict(operation='optimization', input_file='structure.extxyz',
                    output_file='optimization.extxyz', log_file='run.log', overwrite=False,
                    model=model, precision=DEFAULT_PRECISION, total_charge=charge,
                    num_unpaired_electrons=spin, force_convergence=float(force),
                    min_steps=steps_per_frame, min_cycles=-(-budget // steps_per_frame))
    settings.update(long_range)
    rows = [('Structure', f'{home_path(structure.path)} ({structure.atoms} atoms; {structure.formula})'),
            ('Boundary', 'Periodic' if structure.periodic else 'Gas phase'),
            ('Model', model),
            ('Stops at', f'largest force ≤ {float(force):g} eV/Å, or after {settings["min_cycles"] * steps_per_frame:,} steps'),
            ('Charge / unpaired electrons', f'{charge} / {spin}'),
            ('Long-range cutoff', f"{settings['lr_cutoff']:g} Å"),
            ('Output directory', home_path(destination)),
            ('Files', 'structure.extxyz, settings.yaml, run.sh; the run adds optimization.extxyz, run.log')]
    console.print(_rows_view(3, 'Review and start', rows, (
        'Start now saves the files, then optimizes in this terminal.',
        'Save for later writes those files only. Start later with sh run.sh from the output directory.')))
    action = _choose(console, 'Next action · Enter starts now', _review_actions_for('optimization'))
    if action == 2:
        _say(console, 'Setup cancelled. No files written.')
        return None
    path = export_run(destination, structure, settings,
                      header='# Generated by so3lr start: geometry optimization.\n')
    _say(console, f'{theme.status_glyph("completed")} Saved: {home_path(destination)}', theme.SUCCESS)
    if action == 0:
        _say(console, 'Starting the optimization now in this terminal.')
        return path
    _say(console, 'Start later with sh run.sh from that directory, in an environment with SO3LR installed.')
    return None


# -- dataset evaluation ----------------------------------------------------------

def _dataset_prompt(console):
    while True:
        answer = _prompt('Dataset path (.xyz or .extxyz with reference energies/forces)', type=str)
        try:
            path = _entered_path(answer).resolve()
            if not path.is_file():
                raise ValueError(f'Choose an existing file: {path}')
            if path.suffix.lower() not in ('.xyz', '.extxyz'):
                raise ValueError('Choose an .xyz or .extxyz dataset.')
            return path
        except (OSError, ValueError, RuntimeError) as exc:
            _say(console, exc, theme.DANGER)


def eval_setup(console):
    """Draft, review and save an evaluation launcher; return the command to run now."""
    from .formatting import home_path
    _stage(console, 1, 'Dataset')
    dataset = _dataset_prompt(console)
    _stage(console, 2, 'Model')
    model = _select_model(console, DEFAULT_MODEL)
    _say(console, 'Use 12 Å for periodic data; gas-phase molecules need a long cutoff such as 100 Å.',
         theme.MUTED)
    cutoff = _positive(console, 'Long-range cutoff (Å)', '12')
    destination = _new_directory(console, 'so3lr-eval')
    argv = ('eval', '--datafile', str(dataset), '--model', model, '--lr-cutoff', f'{float(cutoff):g}',
            '--save-to', 'predictions.extxyz', '--log-file', 'eval.log')
    rows = [('Dataset', home_path(dataset)), ('Model', model), ('Long-range cutoff', f'{float(cutoff):g} Å'),
            ('Output directory', home_path(destination)),
            ('Files', 'run.sh; the run adds predictions.extxyz and eval.log (the dataset is not copied)')]
    console.print(_rows_view(3, 'Review and start', rows, (
        'Start now saves run.sh, then evaluates in this terminal.',
        'Save for later writes run.sh only. Start later with sh run.sh from the output directory.')))
    action = _choose(console, 'Next action · Enter starts now', _review_actions_for('evaluation'))
    if action == 2:
        _say(console, 'Setup cancelled. No files written.')
        return None
    destination.mkdir()
    launcher = destination / 'run.sh'
    with launcher.open('x') as handle:
        handle.write(_EVAL_LAUNCHER.format(command='so3lr ' + ' '.join(shlex.quote(part) for part in argv)))
    launcher.chmod(0o755)
    _say(console, f'{theme.status_glyph("completed")} Saved: {home_path(destination)}', theme.SUCCESS)
    if action == 0:
        _say(console, 'Starting the evaluation now in this terminal.')
        return Action('eval', argv, destination)
    _say(console, 'Start later with sh run.sh from that directory, in an environment with SO3LR installed.')
    return None


# -- watching runs and expert commands --------------------------------------------

def show_setup(console):
    while True:
        answer = _prompt('Folder or journal to watch', default='.', type=str)
        path = _entered_path(answer)
        if path.exists():
            return Action('show', (str(path),))
        _say(console, f'No such file or folder: {path}', theme.DANGER)


def other_tasks(console):
    """Expert workflows keep their own commands; show where to start."""
    from rich.table import Table
    from rich.text import Text
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True, style=theme.STRONG)
    table.add_column(style=theme.MUTED)
    for command, purpose in (
            ('so3lr finetune --datafile data.extxyz --workdir finetune', 'Fine-tune a model on your data'),
            ('so3lr train --config config.yaml', 'Train a model from a configuration file'),
            ('so3lr tune-ewald --input box.xyz --kspace-electrostatics pme', 'Tune k-space (Ewald/PME/P3M) parameters'),
            ('so3lr eval --datafile data.extxyz', 'Evaluate without the guided setup')):
        table.add_row(Text(command), Text(purpose))
    from rich.rule import Rule
    console.print(Text(''))
    console.print(Rule(Text('Other tasks', style=theme.ACCENT_BOLD), align='left', style=theme.FAINT))
    console.print(table)
    _say(console, f'Every command explains its options with --help. Guide: {DOCS_URL}', theme.MUTED)
    return None
