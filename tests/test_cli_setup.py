"""Guided setup exports real CLI inputs without loading or running the model."""
import io
from contextlib import contextmanager
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


def answers(source, output, action='2', *, ensemble='1', dt='0.5', duration='5', cadence='100'):
    return '\n'.join([str(len(setup.available_presets()) + 2), str(source), ensemble, '1', '300', dt, duration, cadence,
                      '0', '0', '52', 'y', '100', str(output), action]) + '\n'


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
    text = ' '.join(result.output.split())
    assert 'Start now saves structure.extxyz, settings.yaml and run.sh, then runs MD in this terminal.' in text
    assert 'Save for later writes those files only.' in text
    assert all(stage in text for stage in ('1/3 · Structure', '2/3 · Dynamics', '3/3 · Review and start'))
    assert 'Starting MD now in this terminal.' not in result.output
    assert '[literal]' in result.output
    assert 'Checkpoint cadence' in result.output
    assert 'mixed (float32 model; float64 dynamics)' in ' '.join(result.output.split())
    assert '2.5 ps; also at completion' in ' '.join(result.output.split())
    assert (destination / 'structure.extxyz').read_bytes() == structure_file.read_bytes()
    exported = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert exported['md_dt'] == 0.5
    assert exported['md_cycles'] == 100
    assert exported['md_steps'] == 100
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


@pytest.mark.parametrize('action', ['1', ''])
def test_run_locally_is_first_and_default_and_returns_saved_absolute_settings_path(tmp_path, structure_file, monkeypatch, action):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'local'
    result = CliRunner().invoke(wizard, input=answers(structure_file, destination, action=action))
    assert result.exit_code == 0, result.output
    assert f'RETURN:{destination / "settings.yaml"}' in result.output
    assert 'Starting MD now in this terminal.' in result.output
    assert '1. Save setup and start MD now' in result.output
    assert '2. Save setup for later' in result.output
    assert 'Next action · Enter starts MD now' in result.output


@pytest.mark.parametrize('action', ['4', ''])
def test_cancel_or_eof_writes_nothing(tmp_path, structure_file, monkeypatch, action):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'cancelled'
    # EOF at the very first prompt or an explicit Cancel on the review screen.
    text = answers(structure_file, destination, action=action) if action else ''
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'Setup cancelled' in result.output
    assert not destination.exists()


def test_multiframe_input_is_rejected_without_selecting_a_frame(tmp_path, structure_file):
    multi = tmp_path / 'trajectory.xyz'
    multi.write_bytes(structure_file.read_bytes() * 2)
    with pytest.raises(ValueError, match='multiple frames'):
        setup.inspect_structure(multi)


def test_npt_requires_full_periodicity_and_a_positive_orthorhombic_cell(tmp_path, structure_file):
    assert not setup.inspect_structure(structure_file).npt_compatible
    periodic = tmp_path / 'periodic.extxyz'
    periodic.write_text(structure_file.read_text().replace('pbc="F F F"',
                        'pbc="T T T" Lattice="30 0 0 0 30 0 0 0 30"'))
    assert setup.inspect_structure(periodic).npt_compatible
    periodic.write_text(periodic.read_text().replace('pbc="T T T"', 'pbc="T T F"'))
    with pytest.raises(ValueError, match='mixed-axis periodic boundary conditions'):
        setup.inspect_structure(periodic)


def test_npt_choice_on_vacuum_retries_the_ensemble(tmp_path, structure_file, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    text = answers(structure_file, tmp_path / 'nvt')
    text = text.replace(str(structure_file) + '\n1\n', str(structure_file) + '\n4\n1\n', 1)
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert 'NPT requires a fully periodic' in result.output
    assert yaml.safe_load((tmp_path / 'nvt/settings.yaml').read_text())['ensemble'] == 'nvt'


def test_nondivisible_duration_is_not_rounded():
    assert setup.duration_cycles('5', '0.5', 100) == 100
    with pytest.raises(ValueError, match='exact multiple of 0.05 ps'):
        setup.duration_cycles('5.01', '0.5', 100)
    for invalid in ('nan', 'inf', '0', '-1'):
        with pytest.raises(ValueError, match='finite positive'):
            setup.duration_cycles(invalid, '0.5', 100)


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


def test_wizard_import_and_structure_inspection_never_import_jax(structure_file):
    code = '''
import sys
from so3lr.cli.setup import inspect_structure
assert 'ase' not in sys.modules
assert 'jax' not in sys.modules
inspect_structure(sys.argv[1])
assert 'jax' not in sys.modules
assert 'flax' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code, str(structure_file)], check=True)


def test_complete_draft_and_cancel_never_import_the_model(tmp_path, structure_file):
    code = '''
import sys
import click
from click.testing import CliRunner
from so3lr.cli import setup
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


@pytest.mark.parametrize('flag', ['--help', '-h'])
def test_public_init_help_never_launches_the_wizard(monkeypatch, flag):
    from so3lr.cli.so3lr_cli import cli
    def unexpected():
        raise AssertionError('Help must not start setup')
    monkeypatch.setattr(setup, 'launch_setup', unexpected)
    result = CliRunner().invoke(cli, ['init', flag])
    assert result.exit_code == 0, result.output
    assert 'settings.yaml' in result.output


@pytest.mark.parametrize('fails', [False, True])
def test_local_run_uses_bundle_directory_and_restores_cwd(tmp_path, monkeypatch, fails):
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
    result = CliRunner().invoke(so3lr_cli.cli, ['init'])
    assert result.exit_code == (1 if fails else 0)
    assert captured == [(bundle, settings_path)]
    assert Path.cwd() == original


@pytest.mark.parametrize(('sequence', 'expected'), [
    ('\x1b[A', 'up'), ('\x1b[B', 'down'), ('\x1bOA', 'up'),
    ('\xe0P', 'down'), ('\r', 'enter'), ('\n', 'enter'),
    ('\x1b[5~', 'page_up'), ('\x1b[6~', 'page_down'),
    ('\x1b', 'cancel'), ('\x03', 'cancel'),
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


def test_arrow_selector_moves_highlight_and_commits_on_enter(monkeypatch):
    from rich.console import Console
    from so3lr.cli import prompts
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    monkeypatch.setattr(prompts, 'arrow_input_available', lambda console: True)
    @contextmanager
    def keys():
        chunks = iter(['\x1b[B\x1b[B\x1b[A\r'])
        yield prompts._Keys(lambda: next(chunks), lambda timeout: False)
    monkeypatch.setattr(prompts, 'terminal_keys', keys)
    assert prompts.choose(console, 'Choose a structure', ['Ala4', 'DHA', 'Browse files']) == 1
    assert 'Choose a structure: DHA' in console.file.getvalue()


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


@pytest.mark.parametrize('model', setup.canonical_models())
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


def test_legacy_model_picker_escape_cancels_the_wizard(monkeypatch):
    from rich.console import Console
    def choose(console, title, choices, default=1):
        if title == 'Model (canonical release)':
            return len(choices) - 1
        raise click.Abort()
    monkeypatch.setattr(setup, '_choose', choose)
    with pytest.raises(click.Abort):
        setup._select_model(Console(file=io.StringIO()))


def test_legacy_selection_exports_the_exact_canonical_model(tmp_path, structure_file, monkeypatch):
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    destination = tmp_path / 'legacy-run'
    text = answers(structure_file, destination)
    # Open Legacy models, then choose the original SO3LR release.
    text = text.replace(f'{structure_file}\n1\n1\n300\n',
                        f'{structure_file}\n1\n4\n4\n300\n', 1)
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert settings['model'] == 'so3lr'


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


def test_presets_are_checkout_relative_and_absent_in_an_installed_package(tmp_path, monkeypatch):
    presets = setup.available_presets()
    assert [path.name for _, path in presets] == ['ac.xyz', 'dha.xyz']
    monkeypatch.chdir(tmp_path)
    assert setup.available_presets() == presets
    monkeypatch.setattr(setup, '__file__', str(tmp_path / 'site-packages/so3lr/cli/setup.py'))
    assert setup.available_presets() == []


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


def test_velocity_reinitialization_is_explicit_and_source_is_unchanged(tmp_path, structure_file):
    from ase.io import read
    source = tmp_path / 'with-momenta.xyz'
    source.write_text(structure_file.read_text().replace(':vel:R:3', ':momenta:R:3'))
    original = source.read_bytes()
    structure = setup.inspect_structure(source)
    assert structure.has_momenta
    assert structure.input_temperature > 0
    retain = tmp_path / 'retain'
    setup.export_run(retain, structure, {})
    assert (retain / 'structure.extxyz').read_bytes() == original
    initialize = tmp_path / 'initialize'
    setup.export_run(initialize, replace(structure, keep_input_momenta=False), {})
    assert not read(initialize / 'structure.extxyz').has('momenta')
    assert source.read_bytes() == original


def test_example_structures_report_actual_frames_pbc_and_cell():
    presets = {path.name: path for _, path in setup.available_presets()}
    ala4 = setup.inspect_structure(presets['ac.xyz'], frame_index=0)
    assert (ala4.atoms, ala4.frame_count, ala4.periodic) == (42, 100, False)
    dha = setup.inspect_structure(presets['dha.xyz'])
    assert dha.atoms == 56 and dha.periodic
    assert dha.cell_lengths == pytest.approx((999.900024,) * 3)
    water = setup.inspect_structure(Path(__file__).parent / 'test_data/water_64.xyz')
    assert water.atoms == 192 and water.periodic and water.has_momenta
    assert water.minimum_cell_height == pytest.approx(12.4)
    assert water.input_temperature == pytest.approx(1008.8347683)


@pytest.mark.parametrize('retain', [False, True])
def test_manual_water_input_reviews_pme_and_explicit_velocity_choice(tmp_path, monkeypatch, retain):
    from ase.io import read
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    source = Path(__file__).parent / 'test_data/water_64.xyz'
    original = source.read_bytes()
    destination = tmp_path / 'water'
    # Manual path, NPT, default model, target T, velocity decision, pressure,
    # timestep, duration, cadence, charge, spin, seed, relax, cutoff, save.
    text = '\n'.join([str(len(setup.available_presets()) + 2), str(source),
                      '4', '1', '300', '2' if retain else '1', '1',
                      '0.5', '0.01', '10', '0', '0', '52', 'n', '5.5', str(destination), '1']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert '1008.8 K before center-of-mass removal' in result.output
    assert 'PME electrostatics and PME dispersion' in ' '.join(result.output.split())
    settings = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert settings['lr_cutoff'] == 5.5
    assert settings['kspace_electrostatics'] == settings['kspace_dispersion'] == 'pme'
    assert settings['kspace_smearing'] == settings['kspace_dispersion_smearing'] == 2.0
    assert settings['md_T'] == 300
    exported = read(destination / 'structure.extxyz')
    assert exported.has('momenta') == retain
    assert source.read_bytes() == original


def test_dha_isolation_is_chosen_and_only_changes_exported_copy(tmp_path, monkeypatch):
    from ase.io import read
    monkeypatch.setattr(setup, '_interactive', lambda: True)
    source = dict((path.name, path) for _, path in setup.available_presets())['dha.xyz']
    original = source.read_bytes()
    destination = tmp_path / 'isolated-dha'
    text = '\n'.join(['2', '2', '1', '1', '300', '0.5', '0.01', '10',
                      '0', '0', '52', 'n', '100', str(destination), '1']) + '\n'
    result = CliRunner().invoke(wizard, input=text)
    assert result.exit_code == 0, result.output
    assert '999.9' in result.output
    assert 'input cell removed from exported copy' in result.output
    exported = read(destination / 'structure.extxyz')
    assert not exported.pbc.any()
    assert not exported.cell.any()
    assert exported.positions == pytest.approx(read(source).positions)
    assert source.read_bytes() == original
    settings = yaml.safe_load((destination / 'settings.yaml').read_text())
    assert settings['lr_cutoff'] == 100
    assert 'kspace_electrostatics' not in settings


@pytest.mark.skipif(os.name == 'nt', reason='POSIX pseudo-terminal integration')
@pytest.mark.parametrize(('keys', 'expected'), [(b'\x1b[B\r\x1b[B\r', '1,1'), (b'\x1b', 'cancel')])
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
