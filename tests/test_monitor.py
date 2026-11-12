"""The monitor is an offline observer with bounded, incremental journal reads."""
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from so3lr.cli import monitor, theme
from so3lr.cli.formatting import show_command


def event(seq, kind, **data):
    return {'schema': 1, 'run_id': 'run-example', 'seq': seq,
            'timestamp': '2026-01-01T00:00:00Z', 'elapsed_s': float(seq),
            'type': kind, 'data': data}


def write(path, events):
    path.write_text(''.join(json.dumps(item) + '\n' for item in events))


def samples(count=20):
    return [event(0, 'run_started', settings={
        'model': 'so3lr-tauri-m-2606', 'ensemble': 'nvt', 'md_T': 300,
        'input_file': '[red]ethanol.xyz', 'output_file': '[red]trajectory.xyz',
        'log_file': 'run.log', 'precision': 'mixed',
    }), event(1, 'info', atoms=9, engine='standard'),
        event(2, 'md_started', start_step=1000, total_steps=count * 100, dt_fs=0.5),
        *[event(index + 3, 'sample', step=1000 + index * 100, segment_step=index * 100,
                time_ps=index * 0.05, temperature_K=280 + index,
                potential_energy_eV=-39 + index * 0.002,
                energy_eV=-38.8 + index * 0.001,
                conserved_energy_eV=-38.7 + index * 1e-6)
          for index in range(count + 1)]]


def render(reader, width=80, **kwargs):
    from rich.console import Console
    output = io.StringIO()
    console = Console(file=output, width=width, color_system=None)
    kwargs.setdefault('now', datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc))
    console.print(monitor.render_monitor([reader], width=width, **kwargs))
    return output.getvalue()


def test_partial_appends_malformed_lines_and_incremental_offset(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    first = json.dumps(event(0, 'run_started', settings={})).encode() + b'\n'
    second = json.dumps(event(1, 'sample', step=0, time_ps=0, temperature_K=300)).encode() + b'\n'
    path.write_bytes(first + b'not json\n' + second[:17])
    reader = monitor.JournalReader(path)
    read = reader.poll()
    assert read == path.stat().st_size
    assert reader.state.malformed == 1 and reader.state.sample_count == 0
    offset = reader.offset
    assert reader.poll() == 0 and reader.offset == offset
    with path.open('ab') as stream:
        stream.write(second[17:])
    assert reader.poll() == len(second) - 17
    assert reader.state.sample_count == 1 and reader.pending == b''
    assert reader.poll() == 0 and reader.state.sample_count == 1


def test_unknown_schema_bad_data_and_duplicates_are_skipped(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    unsupported = event(1, 'sample', step=0, time_ps=0)
    unsupported['schema'] = 2
    valid = event(2, 'sample', step=0, time_ps=0)
    write(path, [[], event(0, 'run_started', settings={}), unsupported,
                 valid, valid, event(3, 'sample', step='bad', time_ps=1)])
    reader = monitor.JournalReader(path)
    reader.snapshot()
    assert reader.state.unsupported == 1
    assert reader.state.malformed == 2
    assert reader.state.sample_count == 1


def test_large_partial_line_is_discarded_without_unbounded_buffer(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, 'MAX_LINE_BYTES', 64)
    path = tmp_path / 'run.events.jsonl'
    path.write_bytes(b'x' * 100)
    reader = monitor.JournalReader(path)
    reader.poll()
    assert reader.pending == b'' and reader.discarding
    with path.open('ab') as stream:
        stream.write(b'x' * 100 + b'\n')
    reader.poll()
    assert reader.pending == b'' and not reader.discarding
    assert reader.state.malformed == 1


def test_reader_handles_replacement_and_truncation(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    assert reader.state.sample_count == 21
    write(path, [event(0, 'run_started', settings={})])
    reader.poll()
    assert reader.state.sample_count == 0
    replacement = tmp_path / 'replacement'
    write(replacement, [event(0, 'run_failed', message='replaced')])
    replacement.replace(path)
    reader.poll()
    assert reader.state.status == 'Failed'


def test_bounded_history_retains_actual_extrema_and_endpoints(tmp_path):
    state = monitor.RunState(tmp_path / 'large.events.jsonl')
    for index in range(10_000):
        value = 1e6 if index == 200 else -1e6 if index == 201 else index % 300
        volume = 1e7 if index == 202 else 1 if index == 203 else 1000 + index % 250
        state.consume(event(index, 'sample', step=index, time_ps=index / 1000,
                            temperature_K=value, potential_energy_eV=-value, volume_A3=volume))
        assert len(state.samples) <= monitor.MAX_HISTORY
    assert state.sample_count == 10_000
    assert state.samples[0]['step'] == 0
    assert state.samples[-1]['step'] == 9999
    assert min(row['temperature_K'] for row in state.samples) == -1e6
    assert max(row['temperature_K'] for row in state.samples) == 1e6
    assert min(row['volume_A3'] for row in state.samples) == 1
    assert max(row['volume_A3'] for row in state.samples) == 1e7


def test_explicit_status_is_not_inferred_from_age_or_completed_steps(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, now=datetime(2026, 11, 11, tzinfo=timezone.utc))
    assert 'Last reported state: Running' in text
    assert 'stale?' in text and 'Job status unknown' in text
    assert 'Completed' not in text and 'Failed' not in text
    reader.state.consume(event(100, 'error', message='warning-like diagnostic'))
    assert reader.state.status == 'Running'
    assert 'ERROR: warning-like diagnostic' in render(reader)
    assert reader.state.unsupported == 0
    reader.state.consume(event(101, 'run_completed'))
    assert 'Last reported state: Completed' in render(reader)
    assert reader.state.phase == 'Complete'
    assert reader.state.detail == ''


def test_freshness_uses_bounded_median_cadence_and_last_event_not_process_liveness(tmp_path):
    reader = monitor.JournalReader(tmp_path / 'run.events.jsonl')
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    def consume(seq, kind, seconds, **data):
        item = event(seq, kind, **data)
        item['timestamp'] = (origin + timedelta(seconds=seconds)).isoformat()
        reader.state.consume(item)
    consume(0, 'run_started', 0, settings={}, host='mel2024',
            scheduler={'SLURM_JOB_ID': '5450148'})
    consume(1, 'md_started', 0, start_step=0, total_steps=1000, dt_fs=0.5)
    for index in range(140):
        consume(index + 2, 'sample', index * 60, step=index, time_ps=index * 0.0005)
    assert len(reader.state.sample_intervals) == 128
    last = reader.state.event_timestamp
    assert monitor.stale_after_seconds(reader.state) == 1200
    assert monitor.observation_status(reader, last + timedelta(seconds=1200)) == 'Running'
    assert monitor.observation_status(reader, last + timedelta(seconds=1201)) == 'Stale?'
    assert reader.state.status == 'Running'
    text = render(reader, now=last + timedelta(hours=2))
    assert 'Host mel2024' in text and 'Slurm job 5450148' in text
    assert 'Job status unknown' in text
    # A new checkpoint or phase is evidence of a newer update, even without a sample.
    consume(200, 'phase', 140 * 60 + 4000, name='Saving')
    assert monitor.observation_status(reader, reader.state.event_timestamp) == 'Running'
    consume(201, 'run_completed', 140 * 60 + 4001)
    assert monitor.observation_status(reader, origin + timedelta(days=300)) == 'Completed'


def test_freshness_handles_missing_timestamps_and_loading_without_inventing_death(tmp_path):
    reader = monitor.JournalReader(tmp_path / 'run.events.jsonl')
    started = event(0, 'run_started', settings={})
    started['timestamp'] = 'not a timestamp'
    reader.state.consume(started)
    text = render(reader, height=10)
    assert 'Age unknown' in text and 'job status unknown' in text
    assert monitor.observation_status(reader, datetime.now(timezone.utc)) == 'Started'
    assert 'Failed' not in text and 'dead' not in text
    reader.state.consume(event(1, 'phase', name='Compiling'))
    now = datetime(2026, 1, 1, 0, 10, 1, tzinfo=timezone.utc)
    assert monitor.observation_status(reader, now) == 'Stale?'
    reader.state.sample_intervals.extend([1, 1, 999999, float('nan'), -1])
    assert monitor.stale_after_seconds(reader.state) == 600
    reader.state.consume(event(2, 'sample', step=0, time_ps=0))
    invalid = event(3, 'phase', name='Saving')
    invalid['timestamp'] = None
    reader.state.consume(invalid)
    assert 'Age unknown' in render(reader)


def test_unread_history_is_not_mistaken_for_current_or_stale_running_state(tmp_path):
    from so3lr.cli.overview import render_overview
    from rich.console import Console
    path = tmp_path / 'run.events.jsonl'
    initial = samples(1)
    write(path, initial + [event(100, 'run_completed')])
    prefix = ''.join(json.dumps(item) + '\n' for item in initial).encode()
    reader = monitor.JournalReader(path)
    reader.poll(max_bytes=len(prefix))
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    assert reader.state.status == 'Running'
    assert monitor.observation_status(reader, now) == 'Reading'
    assert 'reading history' in render(reader, now=now)
    output = io.StringIO()
    Console(file=output, width=100, color_system=None).print(render_overview([reader], now=now))
    assert '1 reading history' in output.getvalue() and 'Stale?' not in output.getvalue()
    reader.snapshot()
    assert monitor.observation_status(reader, now) == 'Completed'


def test_plot_connects_samples_without_changing_numeric_readout():
    from rich.console import Console
    output = io.StringIO()
    rows = [{'time_ps': 0, 'temperature_K': 100},
            {'time_ps': 1, 'temperature_K': 300}]
    Console(file=output, width=40, color_system=None).print(
        monitor._plot(rows, 'temperature_K', 'Temperature', 'K', 40, ascii_only=True))
    assert output.getvalue().count('*') > 10
    assert '300.0' in output.getvalue()
    assert rows == [{'time_ps': 0, 'temperature_K': 100}, {'time_ps': 1, 'temperature_K': 300}]


def test_target_guide_uses_a_solid_terminal_row_with_exact_caption():
    # Solid guides use character-row resolution; the caption retains the
    # exact value when two nearby targets occupy the same terminal row.
    samples = [{'time_ps': 0, 'temperature_K': 0},
               {'time_ps': 0, 'temperature_K': 1}]
    low, high, pixels = -0.07, 1.07, 28
    targets = [high - pixel / (pixels - 1) * (high - low) for pixel in (8, 11)]
    top = monitor._plot(samples, 'temperature_K', 'Temperature', 'K', 40, target=targets[0])
    bottom = monitor._plot(samples, 'temperature_K', 'Temperature', 'K', 40, target=targets[1])
    # Heading is followed by plot rows; pixel8 and pixel11 both occupy row2.
    top_row, bottom_row = top.renderables[3].plain, bottom.renderables[3].plain
    assert '────' in top_row and top_row == bottom_row
    assert f'Thermostat target {targets[0]:g} K' in top.renderables[-1].plain
    assert f'Thermostat target {targets[1]:g} K' in bottom.renderables[-1].plain
    ascii_plot = monitor._plot(samples, 'temperature_K', 'Temperature', 'K', 40,
                               target=targets[1], ascii_only=True)
    assert '-' in ascii_plot.renderables[3].plain


def test_render_uses_segment_time_literal_paths_and_distinct_energy_fields(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    for field, title in monitor.ENERGIES.items():
        text = render(reader, energy=field)
        assert title in text
        assert '[red]trajectory.xyz' in text
        assert 'current segment' in text
        assert 'step counter 3,000' in text
        assert 'Segment 2,000/2,000 steps' in text
        assert all(len(line) <= 80 for line in text.splitlines())
    assert 'Saved checkpoint' not in render(reader)
    reader.state.consume(event(200, 'checkpoint', step=3000, path='[red]restart.npz'))
    assert '[red]restart.npz' in render(reader)


@pytest.mark.parametrize('width', [80, 100, 120])
@pytest.mark.parametrize('constrained', [False, True])
def test_colored_chart_rows_never_wrap_away_from_their_axes(tmp_path, monkeypatch, width, constrained):
    from rich.console import Console
    from rich.cells import cell_len

    path = tmp_path / 'run.events.jsonl'
    write(path, samples(100))
    reader = monitor.JournalReader(path)
    reader.snapshot()
    monkeypatch.setenv('TERM', 'dumb' if constrained else 'xterm-256color')
    # In Rich 15, force_terminal + TERM=dumb may constrain a requested 100/120
    # column console to 80 unless height is also explicit. Exercise the actual
    # Console options instead of assuming the constructor width was honored.
    console = Console(record=True, width=width, file=io.StringIO(),
                      force_terminal=True, color_system='truecolor', no_color=False)
    console.print(monitor.render_monitor([reader], width=width))
    lines = console.export_text(clear=False).splitlines()
    chart_rows = [line for line in lines if '│' in line]
    assert len(chart_rows) == 14
    assert all(line.count('│') == 1 for line in chart_rows)
    temperature_line = next(index for index, line in enumerate(lines) if 'Temperature [K]' in line)
    energy_line = next(index for index, line in enumerate(lines) if 'Total energy [eV]' in line)
    assert energy_line > temperature_line + 7
    assert all('│' in line for line in lines if any('\u2800' <= char <= '\u28ff' for char in line))
    assert all(cell_len(line) <= console.width for line in lines)
    assert not any(line.strip() == '5' for line in lines)


def test_ascii_plot_and_no_missing_property_substitution(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples(1))
    reader = monitor.JournalReader(path)
    reader.snapshot()
    for row in reader.state.samples:
        row.pop('conserved_energy_eV', None)
    text = render(reader, energy='conserved_energy_eV', ascii_only=True)
    assert text.isascii()
    assert 'No samples for this property yet.' in text
    assert 'Conserved energy' in text


def test_compact_view_has_controls_in_24_rows(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, height=24)
    assert len(text.splitlines()) <= 24
    assert 'q: detach' in text


@pytest.mark.parametrize('width', [32, 50, 60, 80])
@pytest.mark.parametrize('completed', [False, True])
def test_ascii_paged_controls_fit_after_arrow_names_expand(tmp_path, width, completed):
    path = tmp_path / 'run.events.jsonl'
    entries = samples(1)
    if completed:
        entries.append(event(100, 'run_completed', message='Completed'))
    write(path, entries)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, width=width, height=24, ascii_only=True)
    assert text.isascii() and '\\x0a' not in text
    assert len(text.splitlines()) <= 24
    assert all(len(line) <= width for line in text.splitlines())
    assert ('q: close' if completed else 'q: detach') in text.splitlines()[-1]


def test_cycle_throughput_and_checkpoint_age_are_explicit(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    reader.state.samples[-1]['seconds_per_step'] = 0.0014
    reader.state.consume(event(200, 'checkpoint', step=3000, path='restart.npz'))
    now = datetime(2026, 1, 1, 0, 0, 12, tzinfo=timezone.utc)
    text = render(reader, now=now)
    assert 'Cycle throughput 30.857 ns/day · 1.56e-04 s/atom/step' in text
    assert '(excludes startup/I/O)' in text
    assert 'mixed (float32 model; float64 dynamics)' in text
    assert 'restart.npz · age 12s' in text
    compact = render(reader, now=now, height=24)
    assert 'Cycle 30.857 ns/day · 1.56e-04 s/atom/step' in compact
    assert 'Checkpoint step 3,000 · age 12s' in compact
    assert len(compact.splitlines()) <= 24


def test_throughput_omits_unknown_atom_normalization(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    reader.state.samples[-1]['seconds_per_step'] = 0.0014
    reader.state.info.pop('atoms')
    for height in (None, 24):
        text = render(reader, height=height)
        assert '30.857 ns/day' in text
        assert 's/atom/step' not in text


@pytest.mark.parametrize('width', [80, 100, 120])
def test_npt_volume_is_below_temperature_and_energy(tmp_path, width):
    path = tmp_path / 'npt.events.jsonl'
    events = samples()
    events[0]['data']['settings']['ensemble'] = 'npt'
    for item in events:
        if item['type'] == 'sample':
            item['data']['volume_A3'] = 1000 + item['data']['time_ps'] * 2
    write(path, events)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    assert reader.state.samples[-1]['volume_A3'] == 1002
    for height in (None, 80):
        text = render(reader, width=width, height=height)
        assert text.index('Temperature [K]') < text.index('Total energy [eV]') < text.index('Volume [Å³]')
        assert 'Volume [Å³]  1002.000' in text
        assert sum('│' in line for line in text.splitlines()) == 21
        assert all(len(line) <= width for line in text.splitlines())
    ascii_text = render(reader, width=width, ascii_only=True)
    assert 'Volume [A^3]' in ascii_text and ascii_text.isascii()


def test_npt_missing_volume_is_explicit_and_nvt_does_not_add_a_volume_plot(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    assert 'Volume [' not in render(reader)
    reader.state.settings['ensemble'] = 'NPT'
    text = render(reader)
    assert 'Volume [Å³]\nNo samples for this property yet.' in text
    assert 'Volume [Å³]  0' not in text


@pytest.mark.parametrize('height', [18, 24])
def test_short_npt_terminal_pages_preserve_complete_plots_and_quit(tmp_path, height):
    path = tmp_path / 'run.events.jsonl'
    events = samples()
    events[0]['data']['settings']['ensemble'] = 'npt'
    for item in events:
        if item['type'] == 'sample':
            item['data']['volume_A3'] = 1000 + item['data']['time_ps']
    write(path, events)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    pages = [render(reader, height=height, page=page) for page in range(3)]
    combined = '\n'.join(pages)
    for title in ('Temperature [K]', 'Total energy [eV]', 'Volume [Å³]'):
        assert title in combined
    for text in pages:
        assert len(text.splitlines()) <= height
        assert '↑/↓: plot page' in text and 'q: detach' in text
        chart_count = sum(title in text for title in ('Temperature [K]', 'Total energy [eV]', 'Volume [Å³]'))
        plot_rows = sum('│' in line for line in text.splitlines())
        # Every chart on a page gets the same, readable height.
        assert plot_rows % chart_count == 0 and plot_rows // chart_count >= 3
        assert sum(line.lstrip().startswith('ps ') for line in text.splitlines()) == chart_count
    if height == 24:
        assert 'Temperature [K]' in pages[0] and 'Total energy [eV]' in pages[0]
        assert 'Plot page 1/2' in pages[0]
        assert 'Volume [Å³]' in pages[1] and 'Plot page 2/2' in pages[1]


def test_very_short_terminal_keeps_exit_and_explains_missing_plots(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, height=10)
    assert len(text.splitlines()) <= 10
    assert 'Enlarge terminal' in text and 'q: detach' in text


def test_discovery_is_bounded_and_does_not_follow_directory_links(tmp_path):
    root = tmp_path / 'runs'
    root.mkdir()
    top = root / 'top.events.jsonl'
    write(top, [])
    deep = root / 'a' / 'b'
    deep.mkdir(parents=True)
    write(deep / 'deep.events.jsonl', [])
    external = tmp_path / 'external'
    external.mkdir()
    write(external / 'external.events.jsonl', [])
    (root / 'linked').symlink_to(external, target_is_directory=True)
    assert monitor.discover_journals(root, max_depth=0) == [top]
    assert len(monitor.discover_journals(root)) == 2
    assert len(monitor.discover_journals(root, max_files=1)) == 1


def test_discovery_order_is_stable_when_writers_touch_journals(tmp_path):
    paths = [tmp_path / name for name in ('z.events.jsonl', 'a.events.jsonl', 'm.events.jsonl')]
    for index, path in enumerate(paths):
        write(path, samples(1))
        os.utime(path, (100 + index, 100 + index))
    expected = sorted(paths)
    assert monitor.discover_journals(tmp_path) == expected
    for index, path in enumerate(reversed(paths)):
        os.utime(path, (10000 + index, 10000 + index))
        assert monitor.discover_journals(tmp_path) == expected


@pytest.mark.parametrize('name', ['run.log', 'trajectory.xyz', 'other.jsonl'])
def test_explicit_nonjournal_file_has_actionable_neighbor_hint(tmp_path, name):
    path = tmp_path / name
    path.write_text('original content\n')
    journal = tmp_path / 'run.abc.events.jsonl'
    write(journal, samples(1))
    with pytest.raises(ValueError, match=r'Expected a \*\.events\.jsonl journal') as exc:
        monitor.discover_journals(path)
    assert f'so3lr show {journal}' in str(exc.value)
    assert path.read_text() == 'original content\n'
    journal.unlink()
    with pytest.raises(ValueError) as exc:
        monitor.discover_journals(path)
    assert f'so3lr show {tmp_path}' in str(exc.value)


def test_non_tty_show_is_one_plain_snapshot(tmp_path, monkeypatch):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples(1))
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', output)
    monkeypatch.setattr(monitor.time, 'sleep', lambda *_: pytest.fail('Should not wait'))
    monitor.show(path)
    assert '\x1b' not in output.getvalue()
    assert 'Temperature' in output.getvalue()
    assert 'snapshot' in output.getvalue()
    assert 'Esc/q' not in output.getvalue() and 'e: energy' not in output.getvalue()


def test_show_existing_parent_folder_opens_all_run_overview(tmp_path, monkeypatch):
    for name in ('ala4', 'dha'):
        folder = tmp_path / name
        folder.mkdir()
        write(folder / 'run.events.jsonl', samples(1))
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', output)
    monitor.show(tmp_path, once=True)
    text = output.getvalue()
    assert 'Run overview' in text and '2 runs' in text
    assert 'ala4' in text and 'dha' in text
    assert 'Age' in text and 'snapshot' in text
    assert 'Esc/q' not in text and 'Overview page' not in text
    assert '\x1b' not in text


@pytest.mark.parametrize('close_key', ['q', 'escape'])
def test_show_cycles_overview_and_runs_and_returns_directly(tmp_path, monkeypatch, close_key):
    from rich.console import Console
    import rich.live

    paths = [tmp_path / 'alpha.events.jsonl', tmp_path / 'beta.events.jsonl']
    for path in paths:
        write(path, samples(1))
    monkeypatch.setattr(monitor, 'discover_journals', lambda *_: paths)
    monkeypatch.setenv('TERM', 'xterm')
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr(sys, 'stdout', Terminal())
    monkeypatch.setattr(sys, 'stdin', Terminal())
    frames = []
    class Live:
        def __init__(self, **kwargs):
            self.console = kwargs['console']
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def stop(self):
            pass
        def update(self, view, **kwargs):
            output = io.StringIO()
            Console(file=output, width=100).print(view)
            frames.append(output.getvalue())
    class Keyboard:
        def __enter__(self):
            self.keys = iter(['next', 'e', 'next', 'next', 'previous', 'o', close_key])
            return self
        def __exit__(self, *args):
            pass
        def read(self, *_):
            return next(self.keys)
    monkeypatch.setattr(rich.live, 'Live', Live)
    monkeypatch.setattr(monitor, '_Keyboard', Keyboard)
    monitor.show(tmp_path)
    assert len(frames) == 7
    assert 'Run overview' in frames[0]
    assert 'Run 1/2' in frames[1] and 'Total energy' in frames[1]
    assert 'Conserved energy' in frames[2]
    assert 'Run 2/2' in frames[3]
    assert 'Run overview' in frames[4]
    assert 'Run 2/2' in frames[5]
    assert 'Run overview' in frames[6]


def test_once_is_ansi_free_even_in_a_terminal(tmp_path, monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    path = tmp_path / 'run.events.jsonl'
    write(path, samples(1))
    output = Terminal()
    monkeypatch.setattr(sys, 'stdout', output)
    monitor.show(path, once=True)
    assert '\x1b' not in output.getvalue()
    assert 'snapshot' in output.getvalue()
    assert 'Esc/q' not in output.getvalue() and 'Plot page' not in output.getvalue()


@pytest.mark.parametrize('kind', ['run_completed', 'run_failed'])
def test_valid_once_snapshot_returns_normally_regardless_recorded_outcome(tmp_path, monkeypatch, kind):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples(1) + [event(100, kind, message='recorded outcome')])
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', output)
    assert monitor.show(path, once=True) is None
    assert 'recorded outcome' in output.getvalue()
    assert 'Esc/q' not in output.getvalue() and 'Plot page' not in output.getvalue()


@pytest.mark.parametrize('width,height', [(32, 3), (32, 6), (32, 10), (40, 24), (80, 24)])
def test_compact_view_keeps_age_and_recorded_origin(tmp_path, width, height):
    path = tmp_path / 'run.events.jsonl'
    entries = samples(1)
    entries[0]['data'].update(host='mel2024', scheduler={'SLURM_JOB_ID': '5450148'})
    write(path, entries)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, width=width, height=height)
    assert 'Age 30s' in text
    if height >= 6:
        assert 'Host mel2024' in text
    if width >= 40:
        assert '5450148' in text
    assert len(text.splitlines()) <= height


@pytest.mark.parametrize('width', [32, 80])
@pytest.mark.parametrize('ascii_only', [False, True])
def test_folder_overview_always_shows_escape_close_hint(tmp_path, width, ascii_only):
    from rich.console import Console
    from so3lr.cli.overview import render_overview
    reader = monitor.JournalReader(tmp_path / 'run.events.jsonl')
    output = io.StringIO()
    Console(file=output, width=width, color_system=None).print(
        render_overview([reader], width=width, height=2, ascii_only=ascii_only))
    assert output.getvalue().splitlines()[-1].startswith('Esc/q: close')


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf')])
def test_bad_intervals_are_rejected(tmp_path, interval):
    with pytest.raises(ValueError, match='positive'):
        monitor.show(tmp_path, interval=interval)


def test_time_axis_includes_intermediate_numbered_ticks():
    ticks = monitor._time_ticks(0, 5, 60)
    assert [label for _, label in ticks] == ['0', '1.25', '2.5', '3.75', '5']
    assert [position for position, _ in ticks] == [0, 15, 30, 44, 59]
    for ascii_only in (False, True):
        plot = monitor._plot([{'time_ps': 0, 'temperature_K': 300},
                              {'time_ps': 5, 'temperature_K': 301}],
                             'temperature_K', 'Temperature', 'K', 80, ascii_only=ascii_only)
        axis, labels = plot.renderables[-2:]
        assert axis.plain.count('+' if ascii_only else '┬') == (6 if ascii_only else 5)
        assert labels.plain.split() == ['ps', '0', '1.25', '2.5', '3.75', '5']


@pytest.mark.parametrize('columns', [8, 12, 20, 30])
@pytest.mark.parametrize('right', [0.000003, 5, 123456789])
def test_time_ticks_adapt_without_overlapping_labels(columns, right):
    ticks = monitor._time_ticks(0, right, columns)
    previous_end = -2
    for position, label in ticks:
        start = max(0, min(columns - len(label), position - len(label) // 2))
        assert start > previous_end + 1
        assert start + len(label) <= columns
        previous_end = start + len(label) - 1
    assert ticks[0][1] == '0'
    assert len(ticks) <= 5


def test_time_zero_axis_has_no_invented_elapsed_time():
    plot = monitor._plot([{'time_ps': 0, 'energy_eV': -39}], 'energy_eV', 'Total energy', 'eV', 80)
    assert plot.renderables[-1].plain.split() == ['ps', '0']
    assert monitor._time_ticks(0, 0, 60) == [(0, '0')]


def test_total_energy_reference_uses_elapsed_time_not_half_the_sample_count(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    entries = [event(0, 'run_started', settings={}),
               *[event(index + 1, 'sample', step=index, time_ps=time_ps, energy_eV=value,
                       potential_energy_eV=value, conserved_energy_eV=value)
                 for index, (time_ps, value) in enumerate([(0, 100), (1, 100), (2, 100), (8, -10), (10, -20)])]]
    write(path, entries)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    assert reader.state.energy_mean.reference() == (-15, False)
    text = render(reader, energy='energy_eV')
    assert 'Reference · latter half: -15.000 eV' in text
    assert 'Target ' not in text
    assert 'Reference · latter half: -15.000 eV' in render(reader, energy='potential_energy_eV')
    assert 'Reference · latter half: -15.000000 eV' in render(reader, energy='conserved_energy_eV')
    assert 'Reference | latter half: -15.000 eV' in render(reader, energy='energy_eV', ascii_only=True)


def test_energy_mean_survives_plot_decimation_without_extrema_bias(tmp_path):
    state = monitor.RunState(tmp_path / 'long.events.jsonl')
    values = [20 if index % 7 == 0 else 0 for index in range(5003)]
    for index, value in enumerate(values):
        state.consume(event(index, 'sample', step=index, time_ps=index,
                            energy_eV=value, potential_energy_eV=value - 100,
                            conserved_energy_eV=value - 90, temperature_K=value + 280,
                            volume_A3=value + 1000))
    reference, approximate = state.energy_mean.reference()
    expected = sum(values[2501:]) / len(values[2501:])
    retained = [row['energy_eV'] for row in state.samples if row['time_ps'] >= 2501]
    assert len(state.samples) <= monitor.MAX_HISTORY < len(values)
    assert abs(sum(retained) / len(retained) - expected) > 0.05
    assert abs(reference - expected) < 0.02
    assert approximate
    assert len(state.energy_mean.bins) <= monitor.MAX_MEAN_BINS
    assert not state.energy_mean.exact
    assert sum(item.count for item in state.energy_mean.bins.values()) == len(values)
    assert sum(item.total for item in state.energy_mean.bins.values()) == sum(values)
    for key, offset in (('potential_energy_eV', -100), ('conserved_energy_eV', -90),
                        ('temperature_K', 280), ('volume_A3', 1000)):
        value, estimated = state.reference(key)
        assert estimated and abs(value - (expected + offset)) < 0.02
        assert len(state.property_means[key].bins) <= monitor.MAX_MEAN_BINS
    plot = monitor._state_plot(state, 'energy_eV', 'Total energy', 'eV', 80, None, False)
    assert '≈' in plot.renderables[0].plain and '(binned)' in plot.renderables[0].plain


def test_whole_time_bins_still_give_exact_sample_mean(monkeypatch):
    monkeypatch.setattr(monitor, 'MAX_MEAN_BINS', 8)
    mean = monitor._LatterHalfMean()
    for index in range(32):
        mean.add(index, float(index))
    # The midpoint15.5 lies between bins12..15 and16..19.
    assert mean.reference() == (23.5, False)
    assert len(mean.bins) <= 8


def test_reference_resets_per_md_segment_and_omits_missing_latter_half(tmp_path):
    state = monitor.RunState(tmp_path / 'run.events.jsonl')
    state.consume(event(0, 'sample', step=0, time_ps=0, energy_eV=-39, temperature_K=300))
    state.consume(event(1, 'sample', step=1, time_ps=1))
    assert state.energy_mean.reference() == (None, False)
    assert state.reference('temperature_K') == (None, False)
    state.consume(event(2, 'md_started', start_step=100, total_steps=20, dt_fs=0.5))
    assert state.energy_mean.reference() == (None, False)
    assert state.reference('temperature_K') == (None, False)
    state.consume(event(3, 'sample', step=100, time_ps=0, energy_eV=-10))
    assert state.energy_mean.reference() == (-10, False)


def test_attached_total_energy_reference_and_checkpoint_fit_short_terminal(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    reader.state.samples[-1]['seconds_per_step'] = 0.0014
    live = monitor.LiveProgress(completed=1000, total=2000, step=3000, dt_fs=0.5, elapsed=1, eta=1, keyboard=True)
    reader.state.consume(event(200, 'checkpoint', step=3000, path='restart.npz'))
    text = render(reader, energy='energy_eV', height=24, attached=True, live=live)
    assert 'Temperature [K]' in text and 'Total energy [eV]' in text
    assert 'Reference · latter half' in text
    assert 'Checkpoint' in text and 'q: hide plots' in text
    assert 's/atom/step' in text and 'MD elapsed' in text
    assert len(text.splitlines()) <= 24


@pytest.mark.parametrize('height', [None, 24])
def test_completed_attached_display_reopens_with_show_and_has_no_stale_controls(tmp_path, height):
    directory = tmp_path / 'run with spaces'
    directory.mkdir()
    path = directory / 'run.events.jsonl'
    write(path, [*samples(), event(200, 'run_completed')])
    reader = monitor.JournalReader(path)
    reader.snapshot()
    live = monitor.LiveProgress(completed=2000, total=2000, running=False, keyboard=True)
    text = render(reader, height=height, attached=True, live=live)
    assert 'View saved graphs: so3lr show' in text
    assert 'q:' not in text and 'Ctrl+C:' not in text and 'e: energy' not in text
    assert 'watch' not in text
    assert "'" + str(directory) + "'" in text.replace('\n', '')
    if height is not None:
        assert len(text.splitlines()) <= height


@pytest.mark.parametrize('height', [None, 24])
@pytest.mark.parametrize('status', ['Completed', 'Failed'])
def test_opted_out_attached_display_does_not_offer_unsaved_graphs(tmp_path, height, status):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    reader.state.status = status
    reader.state.info['_journal_enabled'] = False
    text = render(reader, height=height, attached=True)
    assert 'Saved graphs unavailable: SO3LR_JOURNAL=0. Text log retained.' in text.replace('\n', '')
    assert 'so3lr show' not in text
    assert 'Esc/q:' not in text


@pytest.mark.parametrize('keyboard', [False, True])
def test_opted_out_attached_plots_do_not_advertise_a_nonexistent_journal(tmp_path, keyboard):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    reader.state.info.update(_journal_enabled=False)
    text = render(reader, attached=True, live=monitor.LiveProgress(completed=1, total=2, keyboard=keyboard))
    assert 'Temperature' in text and 'Total energy' in text
    assert 'so3lr show' not in text and 'Journal' not in text
    assert str(path) not in text


def test_import_does_not_load_model_backends():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import so3lr.cli.monitor; "
        "assert not any(m.split('.')[0] in ('jax', 'flax', 'jax_md') for m in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal restoration')
def test_keyboard_restores_terminal_on_failure(monkeypatch):
    import termios
    import tty
    saved, restored = object(), []
    monkeypatch.setattr(sys, 'stdin', type('Input', (), {'fileno': lambda self: 42})())
    monkeypatch.setattr(termios, 'tcgetattr', lambda fd: saved)
    monkeypatch.setattr(tty, 'setcbreak', lambda fd: None)
    monkeypatch.setattr(termios, 'tcsetattr', lambda fd, when, state: restored.append((fd, state)))
    with pytest.raises(RuntimeError):
        with monitor._Keyboard():
            raise RuntimeError('reader failed')
    assert restored == [(42, saved)]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX keyboard byte stream')
@pytest.mark.parametrize('chunks, expected', [
    ([b'\x1b[Ceq'], ['next', 'e', 'q']),
    ([b'\x1b', b'[', b'Dq'], ['', '', 'previous', 'q']),
    ([b'\x1bOA\x1bOB\x1bOCeq'], ['up', 'down', 'next', 'e', 'q']),
    ([b'\x1b', b'O', b'Dq'], ['', '', 'previous', 'q']),
    ([b'\x1b[1;5Aq'], ['', 'q']),  # unsupported modifier sequence is one key
    ([b'\x1bOPq'], ['', 'q']),  # an unsupported SS3 key must not become O
])
def test_keyboard_preserves_batched_keys_and_split_arrows(monkeypatch, chunks, expected):
    import select
    chunks = list(chunks)
    keyboard = monitor._Keyboard()
    keyboard.fd, keyboard.pending = 42, ''
    monkeypatch.setattr(select, 'select', lambda *args: ([42], [], []) if chunks else ([], [], []))
    monkeypatch.setattr(os, 'read', lambda *args: chunks.pop(0))
    assert [keyboard.read(timeout=0) for _ in expected] == expected
    assert keyboard.pending == ''


@pytest.mark.skipif(os.name == 'nt', reason='POSIX keyboard byte stream')
@pytest.mark.parametrize('prefix, expected', [(b'\x1b', 'escape'), (b'\x1b[', ''), (b'\x1bO', '')])
def test_keyboard_times_out_standalone_escape_without_leaking_prefix_shortcuts(monkeypatch, prefix, expected):
    import select
    from so3lr.cli.prompts import ESCAPE_TIMEOUT

    chunks, waits, now = [prefix], [], [0.0]
    keyboard = monitor._Keyboard()
    keyboard.fd = 42
    def ready(read, write, errors, timeout):
        waits.append(timeout)
        return ([42], [], []) if chunks else ([], [], [])
    monkeypatch.setattr(select, 'select', ready)
    monkeypatch.setattr(os, 'read', lambda *args: chunks.pop(0))
    monkeypatch.setattr(monitor.time, 'monotonic', lambda: now[0])
    assert keyboard.read(timeout=0) == ''
    now[0] = ESCAPE_TIMEOUT / 2
    assert keyboard.read(timeout=0) == ''
    now[0] = ESCAPE_TIMEOUT * 2
    assert keyboard.read(timeout=0) == expected
    assert keyboard.pending == ''
    chunks.append(b'q')
    assert keyboard.read(timeout=0) == 'q'
    assert waits and not any(waits)  # attached polling never waits for a key


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal byte stream')
def test_real_pty_application_arrows_preserve_shortcuts_and_restore_input(monkeypatch):
    import pty
    import termios

    master, slave = pty.openpty()
    saved = termios.tcgetattr(slave)
    stream = os.fdopen(os.dup(slave), 'r')
    monkeypatch.setattr(sys, 'stdin', stream)
    try:
        with pytest.raises(RuntimeError, match='restore after input'):
            with monitor._Keyboard() as keyboard:
                os.write(master, b'\x1bOA\x1bOCeq')
                assert [keyboard.read(timeout=0.2) for _ in range(4)] == ['up', 'next', 'e', 'q']
                os.write(master, b'\x1b')
                assert keyboard.read(timeout=0.2) == ''
                os.write(master, b'O')
                assert keyboard.read(timeout=0.2) == ''
                os.write(master, b'Dq')
                assert keyboard.read(timeout=0.2) == 'previous'
                assert keyboard.read(timeout=0.2) == 'q'
                os.write(master, b'\x1b')
                assert keyboard.read(timeout=0.2) == ''
                assert keyboard.read(timeout=0.3) == 'escape'
                raise RuntimeError('restore after input')
        restored = termios.tcgetattr(slave)
        pending = getattr(termios, 'PENDIN', 0)
        assert saved[:3] == restored[:3] and saved[4:] == restored[4:]
        assert saved[3] & ~pending == restored[3] & ~pending
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize('surface', ['show', 'prompt'])
@pytest.mark.parametrize('kind', ['keyboard', 'termination'])
def test_live_start_interruption_restores_cursor_before_context_entry(tmp_path, monkeypatch, surface, kind):
    from rich.console import Console
    from rich.live import Live
    from so3lr.cli import prompts
    from so3lr.cli.terminal import TerminationSignal
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    output = Terminal()
    monkeypatch.setattr(sys, 'stdout', output)
    monkeypatch.setattr(sys, 'stdin', Terminal())
    monkeypatch.setenv('TERM', 'xterm')
    closed = []
    class Keyboard:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
    monkeypatch.setattr(monitor, '_Keyboard', Keyboard)
    original_start, original_refresh = Live.start, Live.refresh
    seen = []
    exception = KeyboardInterrupt() if kind == 'keyboard' else TerminationSignal(15)
    def refresh(self):
        if not seen:
            seen.append(self)
            raise exception
        return original_refresh(self)
    monkeypatch.setattr(Live, 'refresh', refresh)
    monkeypatch.setattr(Live, 'start', lambda self, refresh=False: original_start(self, refresh=True))
    def run():
        if surface == 'show':
            monitor.show(tmp_path)
        else:
            prompts.choose(Console(file=output), 'Model', ['M', 'S'])
    if surface == 'show' and kind == 'keyboard':
        run()  # the viewer intentionally treats Ctrl+C as detach
    else:
        with pytest.raises(type(exception)) as raised:
            run()
        assert raised.value is exception
    assert '\x1b[?25l' in output.getvalue() and '\x1b[?25h' in output.getvalue()
    assert seen and not seen[0]._started
    if surface == 'show':
        assert closed == [True]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal mode side effects')
def test_live_cursor_restored_when_start_interrupts_before_render_hook(monkeypatch):
    from rich.console import Console
    from rich.live import Live
    from so3lr.cli.terminal import live_display
    monkeypatch.setenv('TERM', 'xterm')
    output = io.StringIO()
    console = Console(file=output, force_terminal=True)
    original_cursor = console.show_cursor
    interruption = KeyboardInterrupt()
    def cursor(show=True):
        original_cursor(show)
        if not show:
            raise interruption
    monkeypatch.setattr(console, 'show_cursor', cursor)
    live = Live('Starting', console=console, auto_refresh=False)
    with pytest.raises(KeyboardInterrupt) as caught:
        with live_display(live):
            pytest.fail('interrupted start must not enter the body')
    assert caught.value is interruption
    assert '\x1b[?25l' in output.getvalue() and '\x1b[?25h' in output.getvalue()
    assert not live._started


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal mode side effects')
@pytest.mark.parametrize('owner', ['show', 'attached'])
def test_keyboard_cleanup_is_owned_before_entry_returns(tmp_path, monkeypatch, owner):
    import pty
    import termios
    from so3lr.cli import presentation as ui
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    stream = os.fdopen(os.dup(slave), 'r')
    monkeypatch.setattr(sys, 'stdin', stream)
    monkeypatch.setattr(sys, 'stdout', Terminal())
    monkeypatch.setattr(sys, 'stderr', Terminal())
    monkeypatch.setenv('TERM', 'xterm')
    original_enter = monitor._Keyboard.__enter__
    def enter(self):
        original_enter(self)
        raise KeyboardInterrupt
    monkeypatch.setattr(monitor._Keyboard, '__enter__', enter)
    ui.configure(display='rich')
    try:
        if owner == 'show':
            monitor.show(tmp_path)
        else:
            with pytest.raises(KeyboardInterrupt):
                with ui.run_display({'log_file': str(tmp_path/'run.log')}) as reporter:
                    reporter.start_md(0, 10, .5)
        after = termios.tcgetattr(slave)
        pending = getattr(termios, 'PENDIN', 0)
        assert before[:3] == after[:3] and before[4:] == after[4:]
        assert before[3] & ~pending == after[3] & ~pending
    finally:
        ui.reset()
        stream.close()
        os.close(master)
        os.close(slave)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signal and terminal ownership')
@pytest.mark.parametrize('surface', ['show', 'prompt'])
@pytest.mark.parametrize('signal_name', ['SIGTERM', 'SIGHUP'])
def test_interactive_viewer_and_prompt_restore_terminal_before_termination(tmp_path, surface, signal_name):
    import fcntl
    import pty
    import select
    import signal
    import struct
    import termios
    import time
    code = '''
from contextlib import contextmanager
import fcntl, json, pathlib, sys, termios
from rich.console import Console
from so3lr.cli import monitor, prompts
fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
before = termios.tcgetattr(sys.stdin.fileno())
def check_restored():
    after = termios.tcgetattr(sys.stdin.fileno())
    pending = getattr(termios, 'PENDIN', 0)
    restored = (before[:3] == after[:3] and before[4:] == after[4:]
                and before[3] & ~pending == after[3] & ~pending)
    pathlib.Path('terminal.json').write_text(json.dumps({'restored': restored}))
if sys.argv[1] == 'show':
    original_exit, original_read = monitor._Keyboard.__exit__, monitor._Keyboard.read
    def leave(self, *args):
        try:
            return original_exit(self, *args)
        finally:
            check_restored()
    def read(self, timeout=.1):
        pathlib.Path('ready').write_text('ready')
        return original_read(self, timeout)
    monitor._Keyboard.__exit__, monitor._Keyboard.read = leave, read
    monitor.show('.')
else:
    original_keys, original_next = prompts.terminal_keys, prompts._Keys.next
    @contextmanager
    def keys():
        try:
            with original_keys() as value:
                yield value
        finally:
            check_restored()
    def next_key(self):
        pathlib.Path('ready').write_text('ready')
        return original_next(self)
    prompts.terminal_keys, prompts._Keys.next = keys, next_key
    prompts.choose(Console(), 'Select model', ['M', 'S', 'L'])
'''
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 80, 0, 0))
    process = subprocess.Popen([sys.executable, '-c', code, surface], cwd=tmp_path,
                               stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
                               env=dict(os.environ, TERM='xterm', COLUMNS='80', LINES='24'))
    capture, sent = bytearray(), False
    deadline = time.monotonic() + 10
    def read():
        try:
            chunk = os.read(master, 65536)
        except OSError:
            return False
        capture.extend(chunk)
        return bool(chunk)
    try:
        while process.poll() is None:
            assert time.monotonic() < deadline, 'terminal termination cleanup timed out'
            if select.select([master], [], [], .05)[0] and not read():
                process.wait(timeout=2)
                break
            if not sent and (tmp_path/'ready').exists():
                process.send_signal(getattr(signal, signal_name))
                sent = True
        while time.monotonic() < deadline and select.select([master], [], [], .05)[0] and read():
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)
    assert sent and process.returncode == -getattr(signal, signal_name)
    assert json.loads((tmp_path/'terminal.json').read_text())['restored']
    assert b'\x1b[?25l' in capture and b'\x1b[?25h' in capture
    assert b'Traceback' not in capture


def test_incomplete_unknown_escape_cannot_grow_without_bound(monkeypatch):
    import select
    chunks = [b'\x1b[' + b'1' * 30, b'1' * 32, b'q']
    keyboard = monitor._Keyboard()
    keyboard.fd, keyboard.pending = 42, ''
    monkeypatch.setattr(select, 'select', lambda *args: ([42], [], []) if chunks else ([], [], []))
    monkeypatch.setattr(os, 'read', lambda *args: chunks.pop(0))
    assert keyboard.read(timeout=0) == ''
    assert keyboard.read(timeout=0) == ''
    assert keyboard.pending == ''
    assert keyboard.read(timeout=0) == 'q'


def test_solid_guide_preserves_trace_and_never_colors_guide_as_data():
    rows = [{'time_ps': 0, 'energy_eV': 0}, {'time_ps': 1, 'energy_eV': 1}]
    bare = monitor._plot(rows, 'energy_eV', 'Total energy', 'eV', 60)
    guided = monitor._plot(rows, 'energy_eV', 'Total energy', 'eV', 60, reference=0.5)
    # Bounds are unchanged; 0.5 maps to pixel14 (the third dot of row3).
    trace_style = theme.series_style('energy_eV')
    base = bare.renderables[4].plain.split('│')[1]
    guide = guided.renderables[4]
    trace = guide.plain.split('│')[1]
    crosses = False
    for index, (old, new) in enumerate(zip(base, trace)):
        if old != ' ':
            assert new == old
            crosses = True
        else:
            assert new == '─'
            offset = guide.plain.index('│') + 1 + index
            assert not any(span.start <= offset < span.end and str(span.style) == trace_style
                           for span in guide.spans)
    assert crosses
    assert any(theme.FAINT == str(span.style) for span in guide.spans)
    assert any(trace_style == str(span.style) for span in guide.spans)
    ascii_plot = monitor._plot(rows, 'energy_eV', 'Total energy', 'eV', 60, reference=0.5, ascii_only=True)
    assert '---' in ascii_plot.renderables[4].plain
    assert '*' in ascii_plot.renderables[4].plain


@pytest.mark.parametrize('width', [32, 60, 80, 100, 120])
@pytest.mark.parametrize('ascii_only', [False, True])
@pytest.mark.parametrize('height', [None, 20, 23, 24])
def test_all_stacked_plot_axes_align_with_wide_values_and_missing_tail(tmp_path, width, ascii_only, height):
    path = tmp_path / 'wide.events.jsonl'
    entries = samples()
    entries[0]['data']['settings']['ensemble'] = 'npt'
    for item in entries:
        if item['type'] == 'sample':
            data = item['data']
            data['volume_A3'] = 12345678901.123 + data['time_ps']
            data['conserved_energy_eV'] = -123456789.123456 + data['time_ps'] * 0.0001
            if data['time_ps'] > 0.8:
                data['temperature_K'] = None
    write(path, entries)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    views = [render(reader, width=width, height=height, page=page, ascii_only=ascii_only,
                    energy='conserved_energy_eV') for page in range(3 if height else 1)]
    positions, right_edges, time_labels = set(), set(), set()
    for text in views:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            axis = '|' if ascii_only else '│'
            if axis in line and line.index(axis) > 0:
                prefix = line.partition(axis)[0].strip()
                try:
                    if prefix:
                        float(prefix)
                except ValueError:
                    continue
                positions.add(line.index(axis))
                right_edges.add(len(line))
            if line.lstrip().startswith('ps '):
                time_labels.add(line)
        if height is not None:
            assert len(lines) <= height
            assert 'q' in lines[-1]
    assert len(positions) == 1
    assert right_edges == {width}
    assert len(time_labels) == 1  # All fields share the full recorded time span.


def test_nve_temperature_reference_is_observed_not_a_thermostat_target(tmp_path):
    path = tmp_path / 'nve.events.jsonl'
    entries = samples()
    entries[0]['data']['settings']['ensemble'] = 'nve'
    write(path, entries)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader)
    assert 'Total energy [eV]' in text
    assert 'Thermostat target' not in text
    assert 'Reference · latter half: 295.0 K' in text
    assert 'Reference · latter half: -38.785 eV' in text


def test_langevin_target_and_unavailable_conserved_energy_are_explicit(tmp_path):
    path = tmp_path / 'langevin.events.jsonl'
    entries = samples()
    entries[0]['data']['settings']['ensemble'] = 'nvt-langevin'
    for item in entries:
        if item['type'] == 'sample':
            item['data']['conserved_energy_eV'] = None
    write(path, entries)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, energy='conserved_energy_eV')
    assert 'Thermostat target 300 K' in text
    assert 'Not defined for Langevin dynamics (stochastic thermostat).' in text
    assert 'Conserved energy [eV]  0' not in text
    assert 'Reference · latter half:' not in text


def test_loading_and_failed_short_view_show_recorded_details_without_fake_samples(tmp_path):
    path = tmp_path / 'loading.events.jsonl'
    write(path, [event(0, 'run_started', settings={'model': 'so3lr-tauri-m-2606'}),
                 event(1, 'phase', name='Compiling', detail='Warming up the first MD cycle')])
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, height=20)
    assert 'Compiling' in text
    assert 'Warming up the first MD cycle' in text
    assert '│' not in text and 'Reference' not in text
    reader.state.consume(event(2, 'run_failed', message='Structure has no atoms'))
    failed = render(reader, height=20)
    assert 'Structure has no atoms' in failed
    assert 'q: close viewer' in failed
    assert 'job continues' not in failed


@pytest.mark.parametrize('height', [None, 20, 24])
@pytest.mark.parametrize('ascii_only', [False, True])
def test_optimization_view_shows_recorded_force_criterion_without_md_graphs(tmp_path, height, ascii_only):
    path = tmp_path / 'opt.events.jsonl'
    write(path, [event(0, 'run_started', settings={'model': 'so3lr-tauri-m-2606', 'precision': 'mixed',
                                                'output_file': '[literal]optimized.xyz'}),
                 event(1, 'info', operation='optimization', opt_step=200, opt_total_steps=200,
                       opt_energy_eV=-39.12, opt_force_max_eVA=0.03, opt_force_target_eVA=0.01),
                 event(2, 'phase', name='Complete'), event(3, 'run_completed')])
    reader = monitor.JournalReader(path)
    reader.snapshot()
    text = render(reader, height=height, ascii_only=ascii_only)
    assert 'Optimization' in text and 'latest recorded cycle' in text
    assert '200/200 steps' in text
    assert '-39.120000 eV' in text
    assert '0.03 eV/' in text and '0.01 eV/' in text
    assert 'Step budget reached; force criterion not met' in text
    assert 'Temperature' not in text and 'Time [ps]' not in text and 'MD samples' not in text
    assert 'q: close viewer' in text and 'e: energy' not in text
    if height is not None:
        assert len(text.splitlines()) <= height
    if ascii_only:
        assert text.isascii()
    reader.state.info['opt_force_max_eVA'] = 0.005
    assert 'Force criterion met' in render(reader, height=height)


def test_optimization_marker_clears_when_md_begins(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    write(path, [event(0, 'run_started', settings={}),
                 event(1, 'info', operation='optimization', opt_step=5)])
    reader = monitor.JournalReader(path)
    reader.snapshot()
    assert 'Optimization' in render(reader)
    reader.state.consume(event(2, 'info', operation='md'))
    reader.state.consume(event(3, 'md_started', start_step=1000, total_steps=200, dt_fs=0.5))
    reader.state.consume(event(4, 'sample', step=1000, segment_step=0, time_ps=0, temperature_K=290, energy_eV=-39))
    text = render(reader)
    assert 'Optimization' not in text
    assert 'Temperature [K]' in text and 'Total energy [eV]' in text
    assert 'Segment 0/200 steps' in text and 'step counter 1,000' in text


def test_deeply_nested_json_is_skipped_and_later_records_still_load(tmp_path):
    path = tmp_path / 'nested.events.jsonl'
    nested = '[' * 10_000 + '0' + ']' * 10_000 + '\n'
    path.write_text(nested + ''.join(json.dumps(item) + '\n' for item in samples(2)))
    reader = monitor.JournalReader(path)
    reader.snapshot()
    assert reader.state.malformed == 1
    assert reader.state.sample_count == 3
    with path.open('a') as stream:
        stream.write(json.dumps(event(20, 'run_completed')) + '\n')
    reader.poll()
    assert reader.state.status == 'Completed' and reader.state.malformed == 1


@pytest.mark.parametrize('key,values', [
    ('energy_eV', [-39.10001, -39.10002]),
    ('conserved_energy_eV', [-39.100000001, -39.100000002]),
    ('temperature_K', [300.001, 300.002]),
    ('energy_eV', [1e-100, 2e-100]),
])
@pytest.mark.parametrize('width', [32, 80])
def test_axis_labels_resolve_small_recorded_variations(key, values, width):
    rows = [{'time_ps': index, key: value} for index, value in enumerate(values)]
    plot = monitor._plot(rows, key, 'Property', 'unit', width)
    labels = [plot.renderables[index].plain.split('│')[0].strip() for index in (1, 4, 7)]
    assert len(set(labels)) == 3
    numeric = list(map(float, labels))
    assert numeric[0] > numeric[1] > numeric[2]
    assert all(len(row.plain) <= width for row in plot.renderables[1:10])


class StrictAsciiOutput(io.StringIO):
    encoding = 'ascii'

    def write(self, value):
        value.encode('ascii')
        return super().write(value)


@pytest.mark.parametrize('operation,height', [('md', 24), ('optimization', 10)])
def test_ascii_paged_views_never_add_a_unicode_ellipsis(tmp_path, operation, height):
    from rich.console import Console
    folder = tmp_path / ('long-molecular-run-' * 8)
    folder.mkdir()
    path = folder / 'run.events.jsonl'
    write(path, samples(2))
    reader = monitor.JournalReader(path)
    reader.snapshot()
    if operation == 'optimization':
        reader.state.info.update(operation=operation, opt_step=20, opt_total_steps=100,
                                 opt_force_max_eVA=0.03, opt_force_target_eVA=0.01)
    output = StrictAsciiOutput()
    Console(file=output, width=80, height=height, color_system=None).print(
        monitor.render_monitor([reader], width=80, height=height, ascii_only=True))
    assert output.getvalue().isascii()
    assert 'q:' in output.getvalue()
    assert len(output.getvalue().splitlines()) <= height


@pytest.mark.parametrize('height', [1, 3, 6, 10, 12])
@pytest.mark.parametrize('status', ['Running', 'Completed', 'Failed'])
def test_short_optimization_views_keep_controls_and_outcome(tmp_path, height, status):
    path = tmp_path / 'optimization.events.jsonl'
    write(path, [event(0, 'run_started', settings={'model': 'so3lr-tauri-m-2606', 'precision': 'mixed'}),
                 event(1, 'info', operation='optimization', opt_step=20, opt_total_steps=100,
                       opt_energy_eV=-39, opt_force_max_eVA=0.03, opt_force_target_eVA=0.01),
                 event(2, 'phase', name='Relaxing')])
    reader = monitor.JournalReader(path)
    reader.snapshot()
    reader.state.status = status
    if status == 'Failed':
        reader.state.message = 'Model evaluation failed'
    text = render(reader, height=height)
    assert len(text.splitlines()) <= height
    assert text.splitlines()[-1].startswith('Esc/q:')
    if height >= 3:
        assert monitor.optimization_status(reader.state.info, status) in text


def test_completed_attached_paged_view_lists_saved_files(tmp_path, monkeypatch):
    # Paths are shown relative to the working directory, which is the run
    # directory for guided runs and run.sh launchers.
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'run.events.jsonl'
    write(path, [*samples(), event(200, 'run_completed')])
    reader = monitor.JournalReader(path)
    reader.snapshot()
    reader.state.settings.update(output_file='trajectory.extxyz', log_file='run.log')
    reader.state.consume(event(201, 'checkpoint', step=2000, path=str(tmp_path.resolve() / 'restart.npz')))
    text = render(reader, height=24, attached=True)
    assert 'Saved: trajectory.extxyz · run.log · restart.npz' in text
    assert 'Checkpoint step 2,000' in text
    assert len(text.splitlines()) <= 24


def test_show_command_abbreviates_home_and_stays_shell_safe(monkeypatch, tmp_path):
    home = tmp_path / 'home'
    monkeypatch.setenv('HOME', str(home))
    assert show_command(home / 'runs' / 'water 300K') == "~/'runs/water 300K'"
    assert show_command(home / 'runs') == '~/runs'
    assert show_command(home) == '~'
    assert show_command(tmp_path / 'home2' / 'x') == str(tmp_path / 'home2' / 'x')


@pytest.mark.parametrize('attached', [False, True])
def test_paged_views_never_exceed_the_terminal_when_guide_captions_wrap(tmp_path, attached):
    path = tmp_path / 'run.events.jsonl'
    write(path, samples())
    reader = monitor.JournalReader(path)
    reader.snapshot()
    for width in (40, 44, 48, 52, 60, 80):
        for height in range(12, 40):
            text = render(reader, width=width, height=height, attached=attached)
            assert len(text.splitlines()) <= height, (width, height)


def test_show_reports_only_path_errors_as_bad_paths(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from so3lr.cli.so3lr_cli import cli
    log = tmp_path / 'run.log'
    log.write_text('')
    result = CliRunner().invoke(cli, ['show', str(log)])
    assert result.exit_code == 2 and 'Invalid value for PATH' in result.output

    def broken(*args, **kwargs):
        raise ValueError('a rendering bug')
    monkeypatch.setattr(monitor, 'show', broken)
    result = CliRunner().invoke(cli, ['show', str(tmp_path)])
    assert result.exit_code == 1 and 'Invalid value for PATH' not in result.output
    assert isinstance(result.exception, ValueError)


def test_render_cache_follows_new_samples(tmp_path):
    path = tmp_path / 'run.events.jsonl'
    entries = samples()
    write(path, entries)
    reader = monitor.JournalReader(path)
    reader.snapshot()
    first = render(reader, height=24)
    assert render(reader, height=24) == first  # served from the cache between samples
    last = max(item['seq'] for item in entries)
    reader.state.consume(event(last + 1, 'sample', step=99_000, segment_step=2000, time_ps=9.0,
                               temperature_K=999.0, energy_eV=-1.0))
    updated = render(reader, height=24)
    assert '999.0' in updated and updated != first
