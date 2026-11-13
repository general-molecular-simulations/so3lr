"""Guided setup exports real CLI inputs without loading or running the model."""
import io
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import click
from click.testing import CliRunner
import pytest
import yaml

from so3lr.cli import setup


@pytest.fixture
def structure_file(tmp_path):
    path = tmp_path / 'water [sample].extxyz'
    path.write_text('3\nProperties=species:S:1:pos:R:3:vel:R:3 charge=0 pbc="F F F" note="keep [blue] literal"\n'
                    'O 0 0 0 0.1 0.2 0.3\nH 0.96 0 0 0.4 0.5 0.6\nH -0.24 0.93 0 0.7 0.8 0.9\n')
    return path


MD = '1'  # first menu: Run molecular dynamics


def answers(source, output, action='2', *, ensemble='1', duration='5', frame='0.05', task=True):
    # Task, path source, path, dynamics, model, temperature, duration, frame
    # interval, charge, unpaired electrons, output directory, review action.
    return '\n'.join([*([MD] if task else []), str(len(setup.available_presets()) + 2), str(source), ensemble,
                      '1', '300', duration, frame, '0', '0', str(output), action]) + '\n'


def edit(group, *values):
    """Review action 'Edit settings…', one group, then its answers."""
    return ['3', str(setup.EDIT_CHOICES.index(group) + 1), *values]


@click.command()
def wizard():
    result = setup.launch_setup()
    click.echo(f'RETURN:{result}')


class _TTY(io.StringIO):
    def isatty(self):
        return True


def test_automatic_launch_requires_both_ttys_and_no_batch_environment():
    assert setup.should_launch(_TTY(), _TTY(), {'TERM': 'xterm'})
    assert not setup.should_launch(io.StringIO(), _TTY(), {})
    assert not setup.should_launch(_TTY(), io.StringIO(), {})
    assert not setup.should_launch(_TTY(), _TTY(), {'TERM': 'dumb'})
    for variable in setup._SCHEDULER_VARIABLES:
        assert not setup.should_launch(_TTY(), _TTY(), {variable: '123'})


def test_explicit_setup_rejects_noninteractive_streams():
    result = CliRunner().invoke(wizard)
    assert result.exit_code == 1
    assert 'interactive input and output terminal' in result.output


def test_save_exports_user_units_and_preserves_source_metadata(tmp_path, structure_file, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'new run [literal]'
    result = CliRunner().invoke(wizard, input=answers(structure_file, destination))
    assert result.exit_code == 0, result.output
    assert 'RETURN:None' in result.output
    assert '[literal]' in result.output
    assert (destination / 'structure.extxyz').read_bytes() == structure_file.read_bytes()
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert exported['md_dt'] == 0.5
    assert exported['md_cycles'] == 100
    assert exported['md_steps'] == 100
    from so3lr.cli.so3lr_cli import DEFAULT_SAVE_BUFFER
    assert exported['save_buffer'] == min(DEFAULT_SAVE_BUFFER, 100)  # the CLI default, bounded by the run
    assert exported['lr_cutoff'] == 100.0
    assert exported['model'] == 'so3lr-tauri-m-2606'
    assert exported['overwrite'] is False
    assert set(p.name for p in destination.iterdir()) == {'structure.extxyz', 'settings.yaml', 'run.sh'}
    # The exact exported file must survive the existing CLI unit conversion.
    from so3lr.cli.so3lr_cli import load_md_settings
    normalized = load_md_settings(destination / 'settings.yaml')
    assert normalized['md_dt'] == 0.0005
    assert normalized['model_path'] == exported['model']
    assert normalized['md_cycles'] * normalized['md_steps'] * normalized['md_dt'] == 5.0


def test_npt_setup_proposes_margin_and_retries_an_unsafe_custom_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    source = tmp_path / 'periodic.xyz'
    source.write_text('2\nLattice="24 0 0 0 24 0 0 0 24" pbc="T T T"\nH 0 0 0\nH 0 0 .74\n')
    destination = tmp_path / 'npt-margin'
    # NPT with defaults, then edit the cutoff: 12 Å is unsafe, Enter accepts the proposal.
    text = '\n'.join([MD, str(len(setup.available_presets()) + 2), str(source),
                      '4', '1', '300', '1', '0.01', '0.005', '0', '0', str(destination),
                      *edit('Long-range cutoff', '12', ''), '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert '3% NPT contraction margin' in ' '.join(result.output.split())
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert exported['ensemble'] == 'npt'
    assert exported['lr_cutoff'] == 10.8
    assert exported['kspace_spacing'] == 1.0
    assert '1 Å mesh spacing' in ' '.join(result.output.split())


@pytest.mark.parametrize('action', ['1', ''])
def test_run_locally_is_first_and_default_and_returns_saved_absolute_settings_path(tmp_path, structure_file, monkeypatch, action):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'local'
    result = CliRunner().invoke(wizard, input=answers(structure_file, destination, action=action))
    assert result.exit_code == 0, result.output
    assert f'RETURN:{destination / "settings.yaml"}' in result.output
    assert 'Starting MD now in this terminal.' in result.output


def test_eof_cancels_and_writes_nothing(tmp_path, monkeypatch):
    # An explicit Cancel on the review screen: test_complete_draft_and_cancel_never_import_the_model.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    result = CliRunner().invoke(wizard, input='')
    assert result.exit_code == 0, result.output
    assert 'Setup cancelled' in result.output
    assert not list(tmp_path.iterdir())


def test_multiframe_input_is_rejected_without_selecting_a_frame(tmp_path, structure_file):
    multi = tmp_path / 'trajectory.xyz'
    multi.write_bytes(structure_file.read_bytes() * 2)
    with pytest.raises(ValueError, match='multiple frames'):
        setup.inspect_structure(multi)


def test_mixed_axis_periodicity_is_rejected(tmp_path, structure_file):
    periodic = tmp_path / 'periodic.extxyz'
    periodic.write_text(structure_file.read_text().replace('pbc="F F F"',
                        'pbc="T T F" Lattice="30 0 0 0 30 0 0 0 30"'))
    with pytest.raises(ValueError, match='mixed-axis periodic boundary conditions'):
        setup.inspect_structure(periodic)


@pytest.mark.parametrize('cell', [
    '30 0 0 1e-12 30 0 0 0 30',  # a tilted cell, however slightly: runtime rejects every nonzero off-diagonal
    '0 0 0 0 30 0 0 0 30',  # a nonpositive diagonal
])
def test_all_periodic_setup_rejects_unsupported_cells_before_ensemble_selection(tmp_path, cell):
    source = tmp_path / 'invalid-cell.xyz'
    source.write_text(f'2\nLattice="{cell}" pbc="T T T"\nH 0 0 0\nH 0 0 0.74\n')
    with pytest.raises(ValueError, match='Periodic MD requires a positive, axis-aligned orthorhombic cell'):
        setup.inspect_structure(source)


def test_invalid_structure_retries_before_dynamics(tmp_path, structure_file, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    invalid = tmp_path / 'invalid.xyz'
    invalid.write_text('1\n\nNotAnElement 0 0 0\n')
    message = 'Unrecognized element or property'
    with pytest.raises(ValueError, match=message):
        setup.inspect_structure(invalid)
    path_choice = str(len(setup.available_presets()) + 2)
    destination = tmp_path / 'valid-run'
    text = '\n'.join([MD, path_choice, str(invalid)]) + '\n' + answers(structure_file, destination, task=False)
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert message in result.output
    assert result.output.count('2/3 · Dynamics') == 1
    assert (destination / 'structure.extxyz').read_bytes() == structure_file.read_bytes()


def test_npt_choice_on_vacuum_retries_the_ensemble(tmp_path, structure_file, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    text = answers(structure_file, tmp_path / 'nvt')
    text = text.replace(str(structure_file) + '\n1\n', str(structure_file) + '\n4\n1\n', 1)
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'NPT requires a fully periodic' in result.output
    assert yaml.safe_load((tmp_path / 'nvt/settings.yaml').read_text())['ensemble'] == 'nvt'


def test_export_refuses_existing_directory_and_changed_input(tmp_path, structure_file):
    inspected = setup.inspect_structure(structure_file)
    existing = tmp_path / 'existing'
    existing.mkdir()
    marker = existing / 'keep'
    marker.write_text('untouched')
    with pytest.raises(FileExistsError):
        setup.export_run(existing, inspected, {})
    assert marker.read_text() == 'untouched'
    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'must-not-create')
    with pytest.raises(FileExistsError):
        setup.export_run(dangling, inspected, {})
    assert not (tmp_path / 'must-not-create').exists()
    structure_file.write_text(structure_file.read_text().replace('O 0 0 0', 'O 0.1 0 0'))
    with pytest.raises(ValueError, match='changed during setup'):
        setup.export_run(tmp_path / 'changed', inspected, {})
    assert not (tmp_path / 'changed').exists()


@pytest.mark.parametrize(('rewrite', 'symlink'), [(False, True), (True, False)])  # copy and ASE-rewrite branches
def test_export_preserves_structure_created_concurrently(tmp_path, structure_file, monkeypatch, rewrite, symlink):
    inspected = replace(setup.inspect_structure(structure_file), keep_input_momenta=not rewrite)
    destination = tmp_path / 'racing-run'
    unrelated = tmp_path / 'unrelated.xyz'
    unrelated.write_text('another process owns this content')
    original_mkdir = Path.mkdir

    def racing_mkdir(path, *args, **kwargs):
        original_mkdir(path, *args, **kwargs)
        if path == destination:
            target = path / 'structure.extxyz'
            if symlink:
                target.symlink_to(unrelated)
            else:
                target.write_bytes(unrelated.read_bytes())

    monkeypatch.setattr(Path, 'mkdir', racing_mkdir)
    with pytest.raises(FileExistsError):
        setup.export_run(destination, inspected, {})
    assert unrelated.read_text() == 'another process owns this content'
    assert (destination / 'structure.extxyz').read_bytes() == unrelated.read_bytes()
    assert (destination / 'structure.extxyz').is_symlink() is symlink
    assert not (destination / 'settings.yaml').exists()
    assert not (destination / 'run.sh').exists()


def test_launcher_handles_spaces_and_shell_metacharacters(tmp_path, structure_file):
    destination = tmp_path / 'run with spaces [x] $(touch injected)'
    setup.export_run(destination, setup.inspect_structure(structure_file), {'md_dt': 0.5})
    assert os.access(destination / 'run.sh', os.X_OK)
    binary = tmp_path / 'bin'
    binary.mkdir()
    fake = binary / 'so3lr'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
    fake.chmod(0o755)
    result = subprocess.run(['sh', str(destination / 'run.sh')], cwd=tmp_path,
                            env={**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH']},
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [str(destination), '--settings', 'settings.yaml']
    assert not (tmp_path / 'injected').exists()


def test_complete_draft_and_cancel_never_import_the_model(tmp_path, structure_file):
    code = '''
import sys
import click
from click.testing import CliRunner
from so3lr.cli import setup
assert not {'ase', 'jax', 'flax'}.intersection(sys.modules)  # importing the wizard stays light
setup._interactive = lambda: True
@click.command()
def command():
    setup.launch_setup()
result = CliRunner().invoke(command, input=sys.stdin.read())
assert result.exit_code == 0, result.output
assert 'Setup cancelled' in result.output
assert 'jax' not in sys.modules
assert 'flax' not in sys.modules
assert 'so3lr.cli.so3lr_md' not in sys.modules
'''
    destination = tmp_path / 'cancel-without-model'
    subprocess.run([sys.executable, '-c', code],
                   input=answers(structure_file, destination, action='4'), text=True, check=True)
    assert not destination.exists()


@pytest.mark.parametrize(('command', 'fails'), [('start', False), ('init', True)])
def test_local_run_uses_bundle_directory_and_restores_cwd(tmp_path, monkeypatch, fails, command):
    from so3lr.cli import so3lr_cli
    bundle = tmp_path / 'run bundle'
    bundle.mkdir()
    settings_path = bundle / 'settings.yaml'
    settings_path.write_text('input_file: structure.extxyz\n')
    original = Path.cwd()
    captured = []
    monkeypatch.setattr(setup, 'launch_setup', lambda: settings_path)
    def run(path):
        captured.append((Path.cwd(), path))
        if fails:
            raise RuntimeError('deliberate run failure')
    monkeypatch.setattr(so3lr_cli, '_run_settings_file', run)
    result = CliRunner().invoke(so3lr_cli.cli, [command])
    assert result.exit_code == (1 if fails else 0)
    assert captured == [(bundle, settings_path)]
    assert Path.cwd() == original


@pytest.mark.parametrize(('sequence', 'expected'), [
    ('\x1bOA', 'up'), ('\x1b[6~', 'page_down'),
    ('\x1b', 'back'), ('\x7f', 'back'), ('\x08', 'back'), ('\x03', 'cancel'),
])
def test_native_key_decoder(sequence, expected):
    from so3lr.cli.prompts import decode_key
    assert decode_key(sequence, final=True) == (expected, '')


def test_key_reader_handles_split_arrows_and_coalesced_keys():
    from so3lr.cli.prompts import _Keys, decode_key
    assert decode_key('\x1b[') == (None, '\x1b[')
    chunks = iter(['\x1b', '[B\x1b[A\r'])
    keys = _Keys(lambda: next(chunks), lambda timeout: True)
    assert [keys.next(), keys.next(), keys.next()] == ['down', 'up', 'enter']
    with pytest.raises(click.Abort):
        _Keys(lambda: '\x1b', lambda timeout: False).next()


@pytest.mark.parametrize('prefix', ['\x1b[', '\x1bO'])
def test_incomplete_arrow_prefix_does_not_cancel_setup_or_become_a_shortcut(prefix):
    from so3lr.cli.prompts import _Keys, decode_key
    assert decode_key(prefix) == (None, prefix)
    assert decode_key(prefix, final=True, literal=True) == ('ignore', '')
    chunks = iter([prefix, '\r'])
    assert _Keys(lambda: next(chunks), lambda timeout: False).next() == 'enter'


def test_selection_preview_matches_menu_and_keeps_long_literal_entries_on_one_line():
    from rich.console import Console
    from rich.cells import cell_len
    from so3lr.cli.prompts import selection_view
    output = io.StringIO()
    choices = ['[red]file\x1b[2J-' + 'long' * 30] + [f'Frame {index}' for index in range(99)]
    Console(file=output, width=80, color_system=None).print(
        selection_view('Choose a file', choices, selected=0, height=24))
    text = output.getvalue()
    assert '[red]file\\x1b[2J' in text and '\x1b' not in text
    assert len(text.splitlines()) <= 24
    assert all(cell_len(line) <= 80 for line in text.splitlines())
    assert 'more' in text


def test_missing_pasted_path_retries_in_place_and_accepts_quoted_paths(tmp_path, structure_file, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'quoted output'
    text = answers(structure_file, destination)
    text = text.replace(str(structure_file), f'{tmp_path / "missing.xyz"}\n"{structure_file}"', 1)
    text = text.replace(str(destination), f'"{destination}"', 1)
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'Choose an existing structure file' in result.output
    assert (destination / 'settings.yaml').is_file()
    assert result.output.count('1/3 · Structure') == 1


def test_null_path_is_an_actionable_validation_error():
    with pytest.raises(ValueError, match='null characters'):
        setup._entered_path('bad\x00name.xyz')


@pytest.mark.parametrize('model', ['so3lr-tauri-l-2606', 'so3lr'])  # a current and a legacy release
def test_model_picker_keeps_current_releases_first_and_legacy_hidden(monkeypatch, model):
    from rich.console import Console
    current = ('so3lr-tauri-m-2606', 'so3lr-tauri-s-2606', 'so3lr-tauri-l-2606')
    legacy = [name for name in setup.canonical_models() if name not in current]
    calls = []
    def choose(console, title, choices, default=1):
        calls.append((title, choices, default))
        if title == 'Model (canonical release)':
            assert len(choices) == 4
            assert choices[:3] == [f'M (default) · {current[0]}', f'S (faster) · {current[1]}',
                                  f'L (recommended for reactions) · {current[2]}']
            assert choices[-1] == 'Legacy models…'
            assert default == 1
            assert all(name not in ' '.join(choices) for name in legacy if name != 'so3lr')
            return current.index(model) if model in current else 3
        assert title == 'Legacy models'
        assert choices == legacy + ['← Back to current models']
        return legacy.index(model)
    monkeypatch.setattr(setup, '_choose', choose)
    assert setup._select_model(Console(file=io.StringIO())) == model
    assert len(calls) == (1 if model in current else 2)


def test_legacy_model_picker_can_go_back_and_remembers_previous_release(monkeypatch):
    from rich.console import Console
    previous = 'so3lr-tauri-l-2602'
    calls = []
    def choose(console, title, choices, default=1):
        calls.append(title)
        if len(calls) == 1:
            assert default == 4
            return 3
        if len(calls) == 2:
            assert title == 'Legacy models'
            assert choices[default - 1] == previous
            return len(choices) - 1
        assert title == 'Model (canonical release)'
        return 0
    monkeypatch.setattr(setup, '_choose', choose)
    assert setup._select_model(Console(file=io.StringIO()), previous) == setup.DEFAULT_MODEL
    assert len(calls) == 3


def test_esc_in_the_legacy_model_menu_returns_to_current_models(monkeypatch):
    from rich.console import Console
    from so3lr.cli.prompts import Back
    titles = []
    def choose(console, title, choices, default=1):
        titles.append(title)
        if len(titles) == 1:
            return len(choices) - 1  # Legacy models…
        if len(titles) == 2:
            raise Back()  # Esc/Backspace
        return 1
    monkeypatch.setattr(setup, '_choose', choose)
    assert setup._select_model(Console(file=io.StringIO())) == 'so3lr-tauri-s-2606'
    assert titles == ['Model (canonical release)', 'Legacy models', 'Model (canonical release)']


def test_browser_lists_xyz_files_and_navigates_up_and_back(tmp_path, structure_file, monkeypatch):
    from rich.console import Console
    folder = tmp_path / 'folder [literal]'
    folder.mkdir()
    target = folder / 'chosen.xyz'
    target.write_bytes(structure_file.read_bytes())
    (folder / 'notes.txt').write_text('not a structure')
    calls = iter(['folder [literal]/', '↑ Up one directory', 'folder [literal]/', 'chosen.xyz'])
    def choose(console, title, choices, default=1):
        assert 'notes.txt' not in choices
        return choices.index(next(calls))
    monkeypatch.setattr(setup, '_choose', choose)
    console = Console(file=io.StringIO())
    assert setup.browse_structure(console, tmp_path) == target
    monkeypatch.setattr(setup, '_choose', lambda *args, **kwargs: 1)
    assert setup.browse_structure(console, tmp_path) is None


def test_multiframe_picker_exports_only_explicit_selection(tmp_path, structure_file, monkeypatch):
    from ase.io import read
    from rich.console import Console
    source = tmp_path / 'frames.xyz'
    source.write_bytes(structure_file.read_bytes() + structure_file.read_bytes().replace(b'O 0 0 0', b'O 0.25 0 0'))
    original = source.read_bytes()
    monkeypatch.setattr(setup, '_choose', lambda *args, **kwargs: 1)
    selected = setup._pick_frame(Console(file=io.StringIO()), source, {})
    assert (selected.frame_index, selected.frame_count) == (1, 2)
    destination = tmp_path / 'one-frame'
    setup.export_run(destination, selected, {})
    exported = read(destination / 'structure.extxyz', index=':')
    assert len(exported) == 1
    assert exported[0].positions[0, 0] == 0.25
    assert exported[0].info['note'] == 'keep [blue] literal'
    assert source.read_bytes() == original


def test_example_structures_report_actual_frames_pbc_and_cell():
    presets = {path.name: path for _, path in setup.available_presets()}
    ala4 = setup.inspect_structure(presets['ala4.xyz'])
    assert (ala4.atoms, ala4.frame_count, ala4.periodic) == (42, 1, False)
    water = setup.inspect_structure(presets['water_64.xyz'])
    assert (water.atoms, water.periodic, water.has_momenta) == (192, True, False)
    assert water.minimum_cell_height == pytest.approx(12.4)
    dha = setup.inspect_structure(presets['dha.xyz'])
    assert (dha.atoms, dha.periodic) == (56, False)
    water = setup.inspect_structure(Path(__file__).parent / 'test_data/water_64.xyz')
    assert water.atoms == 192 and water.periodic and water.has_momenta
    assert water.minimum_cell_height == pytest.approx(12.4)
    assert water.input_temperature == pytest.approx(1008.8347683)


def test_manual_water_input_reviews_pme_and_explicit_velocity_choice(tmp_path, monkeypatch):
    # Retaining the input velocities: test_editing_temperature_keeps_this_drafts_pressure_and_velocity_choice.
    from ase.io import read
    from so3lr.cli.setup_physics import long_range_settings
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    source = Path(__file__).parent / 'test_data/water_64.xyz'
    original = source.read_bytes()
    destination = tmp_path / 'water'
    # Manual path, NPT, default model, target T, initialize velocities, pressure,
    # duration, frame interval, charge, spin, directory; then edit the cutoff and start.
    text = '\n'.join([MD, str(len(setup.available_presets()) + 2), str(source),
                      '4', '1', '300', '1', '1',
                      '0.01', '0.005', '0', '0', str(destination),
                      *edit('Long-range cutoff', '5.5'), '1']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert '1008.8 K before center-of-mass removal' in result.output
    settings = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert settings['lr_cutoff'] == 5.5
    # The exported long-range block is the reviewed proposal (values: test_setup_physics).
    proposal = long_range_settings(True, 12.4, settings['model'], cutoff=5.5, ensemble='npt')
    assert proposal['kspace_electrostatics'] == proposal['kspace_dispersion'] == 'pme'
    assert {key: settings[key] for key in proposal} == proposal
    assert settings['md_T'] == 300
    exported = read(destination / 'structure.extxyz')
    assert not exported.has('momenta')
    assert source.read_bytes() == original


@pytest.mark.skipif(os.name == 'nt', reason='POSIX pseudo-terminal integration')
@pytest.mark.parametrize(('keys', 'expected'), [(b'\x1b[B\x1b[B\x1b[A\r\x1b[B\r', '1,1'), (b'\x1b', 'cancel')])
def test_real_terminal_arrow_selection_and_escape_restore_input_mode(keys, expected):
    import pty
    import select
    import termios
    import time
    master, slave = pty.openpty()
    code = '''
import click, termios
from rich.console import Console
from so3lr.cli.prompts import choose
before = termios.tcgetattr(0)
try:
    console = Console(force_terminal=True)
    first = choose(console, 'Pick a preset', ['Ala4', 'DHA', 'Browse files'])
    second = choose(console, 'Pick dynamics', ['NVT', 'NVE'])
    result = f'{first},{second}'
except click.Abort:
    result = 'cancel'
after = termios.tcgetattr(0)
# macOS sets PENDIN itself when canonical input resumes. It records pending
# input bookkeeping, not a change to the input mode we must restore.
before[3] &= ~getattr(termios, 'PENDIN', 0)
after[3] &= ~getattr(termios, 'PENDIN', 0)
print(f'RESULT:{result};RESTORED:{before == after}')
'''
    process = subprocess.Popen([sys.executable, '-c', code], stdin=slave, stdout=slave, stderr=slave,
                               env={**os.environ, 'TERM': 'xterm-256color'})
    output, sent = b'', False
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                output += os.read(master, 65536)
            if not sent and not termios.tcgetattr(slave)[3] & termios.ICANON:
                # Begin when character input is ready; queue the next menu's
                # keypresses too, to catch reads or flushes that lose typeahead.
                os.write(master, keys)
                sent = True
            if process.poll() is not None:
                while select.select([master], [], [], 0)[0]:
                    output += os.read(master, 65536)
                break
        else:
            pytest.fail(f'Terminal selection timed out: {output!r}')
        assert process.returncode == 0, output
        assert f'RESULT:{expected};RESTORED:True'.encode() in output
        assert b'\r\n' in output  # newline output processing stays enabled
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)


def test_wizard_defaults_to_recorded_charge_and_rejects_impossible_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    source = tmp_path / 'hydroxide.extxyz'
    source.write_text('2\nProperties=species:S:1:pos:R:3 charge=-1 multiplicity=1 pbc="F F F"\n'
                      'O 0 0 0\nH 0.97 0 0\n')
    destination = tmp_path / 'run'
    # Path, NVT, model, T, dt, duration, cadence; accept recorded charge -1 and
    # multiplicity 1; then try charge 0 with no unpaired electron, refuse to
    # keep it, and fall back to the recorded values.
    text = '\n'.join([MD, str(len(setup.available_presets()) + 2), str(source), '1', '1', '300', '5', '0.05',
                      '0', '', 'n', '-1', '0', str(destination), '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    output = ' '.join(result.output.split())
    assert 'Total charge (e) [-1]:' in output
    assert 'Charge +0 leaves 9 electrons (an odd number), which cannot have 0 unpaired electrons.' in output
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert (exported['total_charge'], exported['num_unpaired_electrons']) == (-1, 0)


def test_example_preset_exports_a_single_frame_without_a_frame_menu(tmp_path, monkeypatch):
    presets = setup.available_presets()
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    preset = str(1 + [path.name for _, path in presets].index('ala4.xyz'))
    destination = tmp_path / 'ala4'
    # 0.01 ps at one frame per 0.005 ps: two frames, fewer than the default checkpoint cadence.
    text = '\n'.join([MD, preset, '1', '1', '300', '0.01', '0.005', '0', '0', str(destination), '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'choose the frame to copy' not in result.output
    assert 'Example · Ala4 peptide' in result.output
    assert (destination / 'structure.extxyz').read_bytes() == presets[int(preset) - 1][1].read_bytes()
    settings = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert settings['lr_cutoff'] == 100 and 'kspace_electrostatics' not in settings  # gas phase
    assert settings['md_cycles'] == 2 and settings['save_buffer'] == 2  # checkpoint cadence bounded by the run


def test_default_output_directory_skips_existing_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert setup._unused_directory('so3lr-run') == 'so3lr-run'
    (tmp_path / 'so3lr-run').mkdir()
    (tmp_path / 'so3lr-run-2').write_text('')
    assert setup._unused_directory('so3lr-run') == 'so3lr-run-3'
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'runs' / 'water').mkdir()
    assert setup._unused_directory(str(Path('runs') / 'water')) == str(Path('runs') / 'water-2')


def test_frame_plan_keeps_one_frame_and_whole_timesteps():
    # Ordinary rounding up: test_rounded_duration_is_announced_and_exported_exactly.
    assert setup.frame_plan('0.001', '0.05', '1') == (1, 50)
    with pytest.raises(ValueError, match='whole number of 0.5 fs timesteps'):
        setup.frame_plan('5', '0.0003', '0.5')


def test_rounded_duration_is_announced_and_exported_exactly(tmp_path, structure_file, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'rounded'
    result = CliRunner().invoke(wizard, input=answers(structure_file, destination, duration='5.01'))
    assert result.exit_code == 0, result.output
    assert 'Duration rounded up to 5.05 ps (101 whole frames).' in result.output
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert (exported['md_cycles'], exported['md_steps']) == (101, 100)


def test_first_pass_uses_advanced_defaults_and_review_edits_one_group(tmp_path, structure_file, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'edited'
    text = '\n'.join([MD, str(len(setup.available_presets()) + 2), str(structure_file), '1', '1', '300', '5', '0.05',
                      '0', '0', str(destination),
                      *edit('Seed and initial relaxation', '7', 'n'),
                      *edit('Timestep', '0.25'),
                      *edit('Temperature and pressure', '310'),
                      '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    output = ' '.join(result.output.split())
    assert 'Defaults: 0.5 fs timestep · seed 52 · relaxation on · 100 Å cutoff.' in output
    assert 'Random seed' not in result.output.split('Edit settings')[0]
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert (exported['seed'], exported['relax_before_run']) == (7, False)
    # A 0.25 fs timestep keeps the 0.05 ps frame interval: twice the steps per frame.
    assert (exported['md_dt'], exported['md_steps'], exported['md_cycles']) == (0.25, 200, 100)
    assert exported['md_T'] == 310


def test_start_over_keeps_previous_answers_as_defaults(tmp_path, structure_file, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'again'
    first = answers(structure_file, destination, action='X').split('\n')[:-2]  # drop the action
    # Start over: keep the previous structure, then accept every remembered answer.
    text = '\n'.join([*first, *edit('Structure, dynamics and model (start over)'),
                      '1', '', '', '', '', '', '', '', '', '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'Keep previous' in result.output
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert exported['md_cycles'] == 100 and exported['md_T'] == 300


def test_editing_temperature_keeps_this_drafts_pressure_and_velocity_choice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    source = Path(__file__).parent / 'test_data/water_64.xyz'
    destination = tmp_path / 'water'
    # NPT, retain velocities, 500 bar; then edit only the temperature, pressing
    # Enter for the velocity choice and the pressure.
    text = '\n'.join([MD, str(len(setup.available_presets()) + 2), str(source),
                      '4', '1', '300', '2', '500', '0.01', '0.005', '0', '0', str(destination),
                      *edit('Temperature and pressure', '310', '', ''), '2']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert (exported['md_T'], exported['md_P']) == (310, 500)
    from ase.io import read
    assert read(destination / 'structure.extxyz').has('momenta')
    assert (destination / 'structure.extxyz').read_bytes() == source.read_bytes()  # retained: copied unchanged
