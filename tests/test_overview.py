"""The folder overview renders journal states without reading or altering runs."""
from datetime import datetime, timezone
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from rich.console import Console

from so3lr.cli.overview import render_overview


def reader(path, **overrides):
    state = dict(path=Path(path), status='Running', start_step=1000, total_steps=100,
                 samples=[dict(step=1050, segment_step=50, temperature_K=298.5, energy_eV=-38.123456)],
                 sample_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    state.update(overrides)
    return SimpleNamespace(state=SimpleNamespace(**state))


def render(readers, width=80, **kwargs):
    output = io.StringIO()
    kwargs.setdefault('now', datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc))
    Console(file=output, width=width, color_system=None).print(
        render_overview(readers, width=width, **kwargs))
    return output.getvalue()


def test_overview_reports_all_states_and_segment_progress():
    readers = [reader('/runs/ethanol/run.events.jsonl'),
               reader('/runs/ala4/run.events.jsonl', status='Completed'),
               reader('/runs/dha/run.events.jsonl', status='Failed'),
               reader('/runs/new/run.events.jsonl', status='Started', samples=[])]
    text = render(readers)
    assert '4 runs' in text
    assert '1 running' in text and '1 completed' in text and '1 failed' in text
    assert '1 waiting for MD' in text
    assert 'Segment steps' in text and '50/100' in text and '50%' in text
    assert '1,050' not in text
    assert 'restart offsets excluded' in text
    assert 'T [K]' in text and '298.5' in text
    assert 'Age' in text and '30s' in text
    wide = render(readers, width=100)
    assert 'Total E [eV]' in wide and '-38.123456' in wide
    assert 'Folder: /runs' in text
    assert all(name in text for name in ('ethanol', 'ala4', 'dha', 'new'))
    assert '←/→: individual runs' in text


def test_missing_segment_uses_restart_offset_without_inventing_completion():
    item = reader('/runs/restarted/run.events.jsonl',
                  samples=[dict(step=1100, temperature_K=300)], status='Running')
    text = render([item], width=100, now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert '100/100' in text and '100%' in text
    assert 'Stale?' in text and '0 completed' in text
    assert 'current job status unknown' in text
    assert 'Total E [eV]' in text
    assert '304d' in text
    assert item.state.status == 'Running'


def test_missing_or_malformed_values_remain_unavailable():
    items = [reader('/runs/malformed/run.events.jsonl', status='Unknown producer state',
                    start_step='bad', total_steps=True, sample_timestamp='bad',
                    samples=[dict(step='bad', segment_step=float('nan'),
                                  temperature_K=float('inf'), energy_eV=None,
                                  potential_energy_eV=-12345.0)]),
             reader('/runs/waiting/run.events.jsonl', status='No lifecycle event yet', samples=[]),
             reader('/runs/bad-row/run.events.jsonl', status=None, samples=[None])]
    text = render(items, width=100)
    assert '1 other' in text and '2 unknown' in text
    assert '-12345' not in text  # Do not substitute potential for total energy.
    assert 'nan' not in text.lower() and 'inf' not in text.lower()
    assert text.count('-') >= 10


def test_names_disambiguate_journals_in_one_directory_and_repeated_directory_names():
    text = render([reader('/root/shared/first.events.jsonl'),
                   reader('/root/shared/second.events.jsonl'),
                   reader('/root/a/same/run.events.jsonl'),
                   reader('/root/b/same/run.events.jsonl')], width=120)
    assert 'shared/first' in text and 'shared/second' in text
    assert 'a/same' in text and 'b/same' in text


@pytest.mark.parametrize('width', [40, 52, 80, 100, 120])
def test_overview_stays_inside_terminal_width_and_height(width):
    readers = [reader(f'/runs/long-run-name-{index:02d}/run.events.jsonl') for index in range(64)]
    text = render(readers, width=width, height=24)
    assert len(text.splitlines()) <= 24
    assert all(cell_len(line) <= width for line in text.splitlines())
    assert 'Overview page 1/4' in text
    if width >= 80:
        assert 'Age' in text and 'T [K]' in text
        assert 'q: close' in text
    assert 'Age' in text


def test_paging_wraps_and_full_snapshot_includes_every_reader():
    readers = [reader(f'/runs/run-{index:03d}/run.events.jsonl') for index in range(64)]
    first = render(readers, height=24)
    second = render(readers, height=24, page=1)
    last = render(readers, height=24, page=-1)
    assert 'Runs 1-16 of 64' in first and 'run-015' in first and 'run-016' not in first
    assert 'Runs 17-32 of 64' in second and 'run-016' in second
    assert 'Overview page 4/4' in last and 'run-063' in last
    assert render(readers, height=24, page=4) == first
    snapshot = render(readers)
    assert 'Overview page 1/1' in snapshot
    assert all(f'run-{index:03d}' in snapshot for index in range(64))


def test_all_128_discovered_runs_fit_one_unbounded_snapshot():
    readers = [reader(f'/runs/run-{index:03d}/run.events.jsonl') for index in range(128)]
    text = render(readers)
    assert '128 runs' in text and 'Runs 1-128 of 128' in text
    assert all(f'run-{index:03d}' in text for index in range(128))


def test_static_snapshot_includes_all_runs_without_interactive_or_page_hints():
    items = [reader(f'/runs/run-{index:03d}/run.events.jsonl') for index in range(128)]
    text = render(items, static=True, height=2, page=12)
    assert 'snapshot' in text and '128 runs' in text
    assert all(f'run-{index:03d}' in text for index in range(128))
    assert 'Esc/q' not in text and 'page' not in text and 'individual runs' not in text
    assert 'Age' in text


@pytest.mark.parametrize('width', [32, 40, 80, 120])
def test_age_remains_visible_and_origin_is_available_at_standard_width(width):
    item = reader('/runs/ethanol/run.events.jsonl', host='mel2024',
                  scheduler={'SLURM_JOB_ID': '5450148'})
    text = render([item], width=width, height=24)
    assert 'Age' in text and '30s' in text
    if width >= 80:
        assert 'Host / Slurm' in text and 'mel2024 / 5450148' in text
    assert all(cell_len(line) <= width for line in text.splitlines())
    assert len(text.splitlines()) <= 24


def test_stale_unknown_and_terminal_states_remain_distinct():
    items = [reader('/runs/stale/run.events.jsonl'),
             reader('/runs/completed/run.events.jsonl', status='Completed'),
             reader('/runs/unknown/run.events.jsonl', status='No lifecycle event yet', sample_timestamp=None)]
    text = render(items, width=100, now=datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert '1 stale?' in text and '1 completed' in text and '1 unknown' in text
    assert '31d' in text and 'Stale?' in text
    assert 'current job status unknown' in text
    assert items[0].state.status == 'Running'
    assert 'Failed' not in text


def test_final_console_width_and_small_height_are_respected():
    items = [reader(f'/runs/sample-{index}/run.events.jsonl') for index in range(20)]
    output = io.StringIO()
    Console(file=output, width=80, color_system=None).print(
        render_overview(items, width=120, height=24))
    text = output.getvalue()
    assert 'Age' in text and 'Total E [eV]' not in text
    assert max(map(cell_len, text.splitlines())) <= 80
    for height in (3, 6, 12):
        compact = render(items, height=height)
        assert len(compact.splitlines()) <= height
        assert 'q: close' in compact


def test_literal_names_and_ascii_output_do_not_emit_terminal_controls():
    items = [reader('/runs/[red]mol\x1b[2J/first-α.events.jsonl', status='[bold]\x07'),
             reader('/runs/[red]mol\x1b[2J/second-β.events.jsonl')]
    text = render(items, width=120, ascii_only=True)
    assert text.isascii()
    assert '\x1b' not in text and '\x07' not in text
    assert '\\x1b' in text and '[red]' in text
    assert 'Left/Right: individual runs' in text


def test_rendering_does_not_poll_read_or_change_state(monkeypatch):
    item = reader('/no/such/path/run.events.jsonl')
    original = dict(item.state.samples[0])
    def forbidden(*args, **kwargs):
        raise AssertionError('Rendering must not perform file I/O')
    item.poll = item.snapshot = forbidden
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(Path, 'stat', forbidden)
    assert '50/100' in render([item])
    assert item.state.samples[0] == original


def test_read_errors_are_visible_without_changing_reported_simulation_state():
    item = reader('/runs/cached/run.events.jsonl')
    item.error = 'Permission denied'
    text = render([item], height=24)
    assert 'Read error' in text
    assert '1 journal read error; cached values may be outdated.' in text
    assert '1 read errors' in text and '0 failed' in text
    assert '50/100' in text  # Explicitly marked cached data remains useful.
    assert item.state.status == 'Running'
    assert len(text.splitlines()) <= 24


def test_mixed_optimization_overview_uses_its_own_iterations_and_explicit_mode():
    optimization = reader('/runs/relax/run.events.jsonl', status='Started',
                          info=dict(operation='optimization', opt_step=24, opt_total_steps=200,
                                    opt_energy_eV=-9999.5, opt_force_max_eVA=0.03))
    md = reader('/runs/md/run.events.jsonl')
    text = render([optimization, md], height=24)
    assert 'Mode' in text and 'OPT' in text and 'MD' in text
    assert '24/200' in text and '12%' in text
    assert '50/100' in text and '50%' in text
    assert 'optimization iteration' in text
    opt_row = next(line for line in text.splitlines() if 'relax' in line)
    assert '298.5' not in opt_row and '-38.123456' not in opt_row
    assert '-9999.5' not in text  # Opt potential is not MD total energy.
    assert len(text.splitlines()) <= 24
    assert all(cell_len(line) <= 80 for line in text.splitlines())
    assert 'OPT:' in render([optimization, md], width=60)


def test_force_converged_optimization_does_not_invent_a_percentage():
    item = reader('/runs/relax/run.events.jsonl', status='Completed',
                  info=dict(operation='optimization', opt_step=24, opt_total_steps=None))
    text = render([item])
    row = next(line for line in text.splitlines() if 'relax' in line and 'Completed' in line)
    assert '24' in row and '%' not in row
    assert '100%' not in text


def test_empty_overview_and_unknown_state_are_renderable():
    text = render([])
    assert 'No run journals found.' in text and '0 runs' in text
    assert 'q: close' in text


def test_overview_import_is_numerically_cheap():
    code = "from so3lr.cli import overview; import sys; assert not {'jax', 'flax', 'ase'} & sys.modules.keys()"
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('ascii_only', [False, True])
def test_wide_overview_draws_progress_bars_and_narrow_keeps_percentages(ascii_only):
    readers = [reader('/runs/half/run.events.jsonl'),
               reader('/runs/new/run.events.jsonl', status='Started', samples=[])]
    wide = render(readers, width=110, ascii_only=ascii_only)
    assert 'Progress' in wide
    half = next(line for line in wide.splitlines() if 'half' in line)
    assert ('###' if ascii_only else '━━━') in half and '50%' in half
    new = next(line for line in wide.splitlines() if ' new ' in line)
    assert '%' not in new
    narrow = render(readers, width=80, ascii_only=ascii_only)
    assert 'Progress' not in narrow and '50%' in narrow


def test_overview_folder_abbreviates_home(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    readers = [reader(tmp_path / 'runs' / name / 'run.events.jsonl') for name in ('a', 'b')]
    assert 'Folder: ~/runs' in render(readers)


def test_interrupted_runs_are_not_counted_as_failures():
    readers = [reader('/runs/a/run.events.jsonl', status='Failed', message='KeyboardInterrupt: interrupted'),
               reader('/runs/b/run.events.jsonl', status='Failed', message='Terminated by SIGTERM'),
               reader('/runs/c/run.events.jsonl', status='Failed', message='RuntimeError: diverged')]
    text = render(readers)
    assert '1 failed' in text and '2 interrupted' in text
    assert text.count('Interrupted') == 2
