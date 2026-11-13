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
from .prompts import Back
from .setup import (_choose, _confirm, _entered_path, _positive, _prompt, _review_actions_for, _say,
                    _select_model, _select_structure, _stage, _stage_rule, _unused_directory,
                    electron_count_problem, export_run, run_steps)
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
    """Draft, review and export an optimization bundle; return its settings path to run now.

    Esc/Backspace in a menu returns to the previous step; from the first step,
    to the task menu.
    """
    from .formatting import home_path
    from .setup_physics import long_range_settings
    state = {'model': DEFAULT_MODEL, 'force': '0.05', 'budget': 1000}

    def structure():
        _stage(console, 1, 'Structure')
        state['structure'] = _select_structure(console, {})

    def model():
        _stage(console, 2, 'Optimization')
        state['model'] = _select_model(console, state['model'])
        try:
            state['long_range'] = long_range_settings(state['structure'].periodic,
                                                      state['structure'].minimum_cell_height,
                                                      state['model'], ensemble='nvt')
        except ValueError as exc:
            _say(console, exc, theme.DANGER)
            raise Back() from None

    def details():
        state['force'] = _positive(console, 'Stop when the largest force is below (eV/Å)', state['force'])
        state['budget'] = _prompt('Maximum optimization steps', default=state['budget'], type=click.IntRange(min=10))
        state['charge'], state['spin'] = _charge_and_spin(console, state['structure'])
        state['destination'] = _new_directory(console, 'so3lr-opt')

    def review():
        structure, force, budget = state['structure'], state['force'], state['budget']
        steps_per_frame = 10
        settings = dict(operation='optimization', input_file='structure.extxyz',
                        output_file='optimization.extxyz', log_file='run.log', overwrite=False,
                        model=state['model'], precision=DEFAULT_PRECISION, total_charge=state['charge'],
                        num_unpaired_electrons=state['spin'], force_convergence=float(force),
                        min_steps=steps_per_frame, min_cycles=-(-budget // steps_per_frame))
        settings.update(state['long_range'])
        destination = state['destination']
        rows = [('Structure', f'{home_path(structure.path)} ({structure.atoms} atoms; {structure.formula})'),
                ('Boundary', 'Periodic' if structure.periodic else 'Gas phase'),
                ('Model', state['model']),
                ('Stops at', f'largest force ≤ {float(force):g} eV/Å, '
                             f'or after {settings["min_cycles"] * steps_per_frame:,} steps'),
                ('Charge / unpaired electrons', f"{state['charge']} / {state['spin']}"),
                ('Long-range cutoff', f"{settings['lr_cutoff']:g} Å"),
                ('Output directory', home_path(destination)),
                ('Files', 'structure.extxyz, settings.yaml, run.sh; the run adds optimization.extxyz, run.log')]
        console.print(_rows_view(3, 'Review and start', rows, (
            'Start now saves the files, then optimizes in this terminal.',
            'Save for later writes those files only. Start later with sh run.sh from the output directory.')))
        action = _choose(console, 'Next action · Enter starts now', _review_actions_for('optimization'))
        if action == 2:
            _say(console, 'Setup cancelled. No files written.')
            state['result'] = None
            return
        path = export_run(destination, structure, settings,
                          header='# Generated by so3lr start: geometry optimization.\n')
        _say(console, f'{theme.status_glyph("completed")} Saved: {home_path(destination)}', theme.SUCCESS)
        if action == 0:
            _say(console, 'Starting the optimization now in this terminal.')
            state['result'] = path
            return
        _say(console, 'Start later with sh run.sh from that directory, in an environment with SO3LR installed.')
        state['result'] = None

    run_steps(structure, model, details, review)
    return state['result']


# -- dataset evaluation ----------------------------------------------------------

def _dataset_prompt(console, default=None):
    while True:
        answer = _prompt('Dataset path (.xyz or .extxyz with reference energies/forces)',
                         default=None if default is None else str(default), type=str)
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
    state = {'model': DEFAULT_MODEL, 'cutoff': '12', 'dataset': None}

    def dataset():
        _stage(console, 1, 'Dataset')
        state['dataset'] = _dataset_prompt(console, state['dataset'])

    def model():
        _stage(console, 2, 'Model')
        state['model'] = _select_model(console, state['model'])

    def details():
        _say(console, 'Use 12 Å for periodic data; gas-phase molecules need a long cutoff such as 100 Å.',
             theme.MUTED)
        state['cutoff'] = _positive(console, 'Long-range cutoff (Å)', state['cutoff'])
        state['destination'] = _new_directory(console, 'so3lr-eval')

    def review():
        destination = state['destination']
        argv = ('eval', '--datafile', str(state['dataset']), '--model', state['model'],
                '--lr-cutoff', f"{float(state['cutoff']):g}", '--save-to', 'predictions.extxyz',
                '--log-file', 'eval.log')
        rows = [('Dataset', home_path(state['dataset'])), ('Model', state['model']),
                ('Long-range cutoff', f"{float(state['cutoff']):g} Å"),
                ('Output directory', home_path(destination)),
                ('Files', 'run.sh; the run adds predictions.extxyz and eval.log (the dataset is not copied)')]
        console.print(_rows_view(3, 'Review and start', rows, (
            'Start now saves run.sh, then evaluates in this terminal.',
            'Save for later writes run.sh only. Start later with sh run.sh from the output directory.')))
        action = _choose(console, 'Next action · Enter starts now', _review_actions_for('evaluation'))
        if action == 2:
            _say(console, 'Setup cancelled. No files written.')
            state['result'] = None
            return
        destination.mkdir()
        launcher = destination / 'run.sh'
        with launcher.open('x') as handle:
            handle.write(_EVAL_LAUNCHER.format(command='so3lr ' + ' '.join(shlex.quote(part) for part in argv)))
        launcher.chmod(0o755)
        _say(console, f'{theme.status_glyph("completed")} Saved: {home_path(destination)}', theme.SUCCESS)
        if action == 0:
            _say(console, 'Starting the evaluation now in this terminal.')
            state['result'] = Action('eval', argv, destination)
            return
        _say(console, 'Start later with sh run.sh from that directory, in an environment with SO3LR installed.')
        state['result'] = None

    run_steps(dataset, model, details, review)
    return state['result']


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
