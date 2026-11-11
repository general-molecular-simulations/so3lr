"""Reporting follows accepted MD states and successful writes in both drivers."""
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from jax_md import partition, space

from so3lr.cli import so3lr_md
from so3lr.cli.so3lr_cli import YAML_DEFAULTS


class Recorder:
    observing = True

    def __init__(self):
        self.starts, self.samples, self.saved = [], [], []
        self.info = []

    def phase(self, *args, **kwargs):
        pass

    def update_info(self, **kwargs):
        self.info.append(kwargs)

    def start_md(self, *args):
        self.starts.append(args)

    def progress(self, step, **values):
        self.samples.append((step, values))

    def checkpoint(self, step, path):
        # Check the completed file, rather than trusting the callback's ordering.
        with np.load(path, allow_pickle=False) as state:
            self.saved.append((step, int(state['step'])))


def _energy(position, **kwargs):
    return 0.5 * jnp.sum(position ** 2)


class HarmonicFlash:
    P = 1
    device_mesh = None

    def energy_forces_staged(self, position):
        return _energy(position), -position

    def energy_forces_strain_staged(self, position, **kwargs):
        return (*self.energy_forces_staged(position), jnp.zeros((), position.dtype))

    def set_box(self, box):
        pass


@pytest.fixture(params=['standard', 'flash'])
def driver(request, tmp_path, monkeypatch):
    """Use real integrators/output, with a cheap deterministic harmonic force."""
    recorder = Recorder()
    monkeypatch.setattr(so3lr_md, 'get_reporter', lambda: recorder)
    input_file = tmp_path / 'input.xyz'
    write(input_file, Atoms('HH', positions=[[0.1, 0., 0.], [0.8, 0., 0.]]))
    settings = dict(YAML_DEFAULTS, input_file=str(input_file), ensemble='nve',
                    precision='float64', model_path='so3lr-tauri-m-2606', lr_cutoff=100.,
                    md_dt=0.0005, md_T=300., md_P=None, md_steps=2, md_cycles=3,
                    save_buffer=2, relax_before_run=False, output_format='extxyz',
                    output_file=str(tmp_path / 'out.xyz'), log_file=str(tmp_path / 'out.log'),
                    restart_save_path=str(tmp_path / 'restart.npz'))
    if request.param == 'standard':
        def model(*args, **kwargs):
            neighbors = partition.neighbor_list(space.free()[0], 1., r_cutoff=100.,
                                                disable_cell_list=True,
                                                format=partition.NeighborListFormat.Dense)
            return neighbors, neighbors, _energy
        monkeypatch.setattr(so3lr_md, 'load_model', lambda *a, **k: SimpleNamespace(
            cutoff=4.5, long_range_cutoff=k.get('lr_cutoff', 100.)))
        monkeypatch.setattr(so3lr_md, 'process_model', model)
        run, module = so3lr_md.perform_md, so3lr_md
    else:
        from so3lr.flash import md as flash_md
        monkeypatch.setattr(flash_md, 'get_reporter', lambda: recorder)
        monkeypatch.setattr(flash_md, 'flash_problem', lambda *a, **k: None)
        monkeypatch.setattr(flash_md, 'build_engine', lambda *a, **k: (HarmonicFlash(), True))
        run, module = flash_md.perform_md_flash, flash_md
    return run, module, settings, recorder


def test_progress_tracks_current_invocation_and_successful_checkpoints(driver):
    run, _, settings, recorder = driver
    run(settings)
    assert recorder.starts == [(0, 6, 0.5)]
    assert [step for step, _ in recorder.samples] == [0, 2, 4, 6]
    assert [v['time_per_step'] for _, v in recorder.samples[:2]] == [None, None]
    assert all(v['time_per_step'] > 0 for _, v in recorder.samples[2:])
    for _, values in recorder.samples:
        assert values['energy'] == pytest.approx(values['kinetic_energy'] + float(values['potential_energy']))
        assert values['conserved_energy'] == pytest.approx(values['energy'])
        assert 'volume' not in values  # A vacuum engine's computational box is not a physical cell.
    assert recorder.saved == [(4, 2), (6, 3)]

    recorder.starts.clear()
    recorder.samples.clear()
    recorder.saved.clear()
    settings['restart_load_path'] = settings['restart_save_path']
    settings['md_cycles'] = 2
    run(settings, restart=True)
    assert recorder.starts == [(6, 4, 0.5)]
    assert [step for step, _ in recorder.samples] == [6, 8, 10]
    assert [v['time_per_step'] for _, v in recorder.samples[:2]] == [None, None]
    assert recorder.saved == [(10, 5)]


@pytest.mark.parametrize('operation', ['write_to_extxyz', 'save_restart_checkpoints'])
def test_failed_write_never_reports_checkpoint(driver, monkeypatch, operation):
    run, module, settings, recorder = driver

    def fail(*args, **kwargs):
        raise OSError('test disk write failed')

    monkeypatch.setattr(module, operation, fail)
    with pytest.raises(OSError, match='test disk write failed'):
        run(settings)
    assert recorder.saved == []


def test_checkpoint_report_uses_actual_npz_filename(driver):
    run, _, settings, recorder = driver
    settings['restart_save_path'] = settings['restart_save_path'].removesuffix('.npz')
    run(settings)
    assert recorder.saved == [(4, 2), (6, 3)]


def test_checkpoint_report_uses_writer_returned_path(driver, monkeypatch):
    run, _, settings, recorder = driver
    save = so3lr_md.save_state

    def relocated(state, box, cycle, path_to_save, **kwargs):
        return save(state, box, cycle, str(path_to_save) + '.relocated', **kwargs)

    monkeypatch.setattr(so3lr_md, 'save_state', relocated)
    run(settings)
    assert not Path(settings['restart_save_path']).exists()
    assert Path(settings['restart_save_path'] + '.relocated.npz').exists()
    assert recorder.saved == [(4, 2), (6, 3)]


def test_library_reporting_skips_observer_only_cell_work(monkeypatch):
    from so3lr.cli.presentation import _NullReporter

    class DeviceBox:
        def __array__(self, *args, **kwargs):
            raise AssertionError('A fixed-cell library call should not transfer the box for telemetry')

    rows = []
    monkeypatch.setattr(so3lr_md.logger, 'info', rows.append)
    so3lr_md.report_md_row(_NullReporter(), 2, 1., -2., -1., 300., .1,
                          ensemble='nve', box=DeviceBox())
    assert rows == ['2\t-1.000\t1.000\t-2.000\t-1.000\t300.0\t1.00e-01']


def test_standard_rejected_cycle_does_not_advance_progress(driver, monkeypatch):
    run, module, settings, recorder = driver
    if module is not so3lr_md:
        pytest.skip('neighbour-list retry belongs to the standard driver')
    check = so3lr_md.check_overflow
    calls = 0

    def overflow_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        sr, lr, sr_overflow, lr_overflow = check(*args, **kwargs)
        return sr, lr, sr_overflow or calls == 1, lr_overflow

    monkeypatch.setattr(so3lr_md, 'check_overflow', overflow_once)
    run(settings)
    assert calls == 4
    assert [step for step, _ in recorder.samples] == [0, 2, 4, 6]


def test_cell_metrics_preserve_log_format_and_use_lattice_rows():
    box = np.array([[10., 1., 0.], [0., 11., 2.], [0., 0., 12.]])
    values = {}
    row = so3lr_md.format_md_row(100, 1., -2., -1., 300., 0.1, box=box, metrics=values)
    assert row == so3lr_md.format_md_row(100, 1., -2., -1., 300., 0.1, box=box)
    assert values['box'] == box.T.tolist()
    assert values['volume'] == pytest.approx(1320.)


def test_float32_energy_metric_uses_the_existing_log_sum():
    ke, pe = jnp.float32(100.1), jnp.float32(-100000.1)
    values = {}
    row = so3lr_md.format_md_row(50, ke, pe, ke + pe, 300., .01, metrics=values)
    assert row.split('\t')[1] == '-99900.000'
    assert f'{float(ke) + float(pe):.3f}' == '-99900.002'  # previous journal value
    assert values['energy'] == float(ke + pe)
    assert f'{values["energy"]:.3f}' == row.split('\t')[1]


def test_float32_driver_energy_matches_log_rows(driver, monkeypatch):
    run, module, settings, recorder = driver
    settings['precision'] = 'float32'
    original_energy = _energy
    # A constant energy offset keeps the harmonic forces/integrator unchanged
    # while making the float32 sum round differently from host float addition.
    def shifted_energy(position, **kwargs):
        return original_energy(position, **kwargs) - jnp.asarray(100000.1, position.dtype)
    monkeypatch.setitem(globals(), '_energy', shifted_energy)
    run(settings)
    rows = []
    for line in Path(settings['log_file']).read_text().splitlines():
        fields = line.split(' - INFO - ')[-1].split('\t')
        if len(fields) == 7 and fields[0].isdigit():
            rows.append(fields)
    assert len(rows) == len(recorder.samples) == 4
    for row, (step, values) in zip(rows, recorder.samples):
        assert int(row[0]) == step
        assert row[1] == f'{values["energy"]:.3f}'
        assert values['energy'] == values['conserved_energy']
    if module is so3lr_md:
        assert any(abs(values['energy'] - (values['kinetic_energy'] + float(values['potential_energy']))) > .0005
                   for _, values in recorder.samples)


@pytest.mark.parametrize('box,volume,box_column', [
    (10., 1000., '10.000'),
    (np.array(10.), 1000., '10.000'),
    (np.array([10.]), 1000., '10.000'),
    (np.array([10., 11., 12.]), 1320., '(10.000,11.000,12.000)'),
    (np.diag([10., 11., 12.]), 1320., '(10.000,11.000,12.000)'),
])
def test_volume_metrics_accept_supported_box_representations(box, volume, box_column):
    values = {}
    row = so3lr_md.format_md_row(0, 1., -2., -1., 300., 0.,
                               npt_state=object(), box=box, metrics=values)
    assert row.split('\t')[6] == box_column
    assert np.asarray(values['box']).shape == (3, 3)
    assert values['volume'] == pytest.approx(volume)
    assert np.linalg.det(values['box']) == pytest.approx(volume)


def test_npt_reported_volumes_match_saved_trajectory(driver):
    run, _, settings, recorder = driver
    atoms = read(settings['input_file'])
    atoms.set_cell([30., 30., 30.])
    atoms.set_pbc(True)
    write(settings['input_file'], atoms)
    settings.update(ensemble='npt', md_P=1., lr_cutoff=10.)
    run(settings)

    assert [step for step, _ in recorder.samples] == [0, 2, 4, 6]
    assert recorder.samples[0][1]['volume'] == pytest.approx(27000.)
    for _, values in recorder.samples:
        assert values['volume'] > 0
        assert values['volume'] == pytest.approx(abs(np.linalg.det(values['box'])))
    frames = read(settings['output_file'], index=':')
    assert len(frames) == 3
    assert [frame.get_volume() for frame in frames] == pytest.approx(
        [values['volume'] for _, values in recorder.samples[1:]])


def test_langevin_samples_do_not_label_energy_as_conserved(driver):
    run, _, settings, recorder = driver
    settings['ensemble'] = 'nvt-langevin'
    run(settings)
    assert all(values['conserved_energy'] is None for _, values in recorder.samples)
    log = Path(settings['log_file']).read_text()
    assert log.count('For NVT-Langevin, H is the total energy E, not a conserved quantity.') == 1
    assert 'Step\tE [eV]\tKE\tPE\tH\tTemp [K]\ttime/step [s]' in log


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_pressure_is_shared_between_log_and_metrics(monkeypatch, dtype):
    class FlexibleState:
        pass

    calls = []
    tensor = np.diag(np.asarray([0.01, 0.02, 0.03], dtype=dtype))
    monkeypatch.setattr(so3lr_md.jax_md.simulate, 'NPTNoseHooverFlexState', FlexibleState)

    def pressure(state):
        calls.append(state)
        return tensor

    monkeypatch.setattr(so3lr_md.jax_md.simulate, 'pressure_tensor_flex', pressure)
    metrics = {}
    state = FlexibleState()
    row = so3lr_md.format_md_row(100, 1., -2., -1., 300., 0.1,
                               state, np.array([10., 11., 12.]), metrics=metrics)
    assert calls == [state]
    assert metrics['pressure_diagonal'] == pytest.approx([0.01, 0.02, 0.03])
    # Match the previous device-side conversion, including printed rounding.
    pressure_bar = np.diag(np.asarray(jnp.asarray(tensor) / so3lr_md.units.metal_unit_system()['pressure']))
    assert row.split('\t')[6] == '({:.1f},{:.1f},{:.1f})'.format(*pressure_bar)


@pytest.mark.parametrize('target,steps', [(None, [0, 3, 6]), (1e-10, [0, 3, 6]),
                                        (1e6, [0, 3])])
def test_optimization_reports_existing_log_values_and_stopping_step(driver, target, steps):
    _, module, settings, recorder = driver
    settings.update(min_cycles=2, min_steps=3, force_convergence=target)
    if module is so3lr_md:
        so3lr_md.perform_min(settings)
    else:
        # perform_min normally sets up the logger/precision before dispatching.
        so3lr_md.setup_logger(settings['log_file'])
        resolved = so3lr_md.setup_run(dict(settings))
        module.perform_min_flash(resolved, read(settings['input_file']))

    updates = [values for values in recorder.info if 'opt_step' in values]
    assert [values['opt_step'] for values in updates] == steps
    assert updates[0]['operation'] == 'optimization'
    assert updates[0]['opt_total_steps'] == 6
    if target is None:
        assert 'opt_force_target_eVA' not in updates[0]
    else:
        assert updates[0]['opt_force_target_eVA'] == target
    assert recorder.samples == []  # Optimization is not a trajectory in simulation time.
    lines = Path(settings['log_file']).read_text().splitlines()
    if target == 1e-10:
        assert any('force_convergence=1e-10 eV/Å' in line for line in lines)
    rows = [line.split(' - INFO - ', 1)[-1].split('\t') for line in lines]
    rows = [row for row in rows if len(row) == 3 and row[0].isdigit()]
    assert [int(row[0]) for row in rows] == steps
    for row, values in zip(rows, updates):
        assert row[1].strip() == f"{values['opt_energy_eV']:.3f}"
        assert row[2].strip() == f"{values['opt_force_max_eVA']:.3f}"
    frames = read(settings['output_file'], index=':')
    assert len(frames) == len(steps)
