"""Regression tests for lossless MD trajectory and restart output."""

import h5py
import jax.numpy as jnp
from jax_md import simulate
import numpy as np
import pytest
import stat
from ase import Atoms
from ase.io import read

from so3lr.cli import so3lr_md
from so3lr.cli.so3lr_md import (
    _validate_md_output_write_policy,
    box_for_trajectory,
    cell_from_atoms_boundary_conditions,
    init_hdf5_store,
    load_state,
    output_position_array,
    positions_to_cartesian,
    save_restart_checkpoints,
    save_state,
    write_to_extxyz,
    write_to_hdf5,
)
from so3lr.md_settings_validation import validate_positive_md_settings


@pytest.mark.parametrize('indices', [None, [2, 0]])
def test_hdf_dipole_of_three_atoms_is_molecular(tmp_path, indices):
    destination = tmp_path / 'water.hdf5'
    positions = [np.arange(9, dtype=np.float32).reshape(3, 3)]
    dipole = np.array([1., 2., 3.], dtype=np.float32)
    store = init_hdf5_store(
        destination, batch_size=1,
        num_atoms=3 if indices is None else len(indices),
        num_box_entries=3, observables=['dipole_vec'],
    )

    write_to_hdf5(
        store, [np.zeros((3, 3))], positions, [np.zeros(3)],
        {'dipole_vec': [dipole]}, indices=indices,
    )

    with h5py.File(destination) as handle:
        assert handle['dipole_vec'].shape == (1, 1, 3)
        np.testing.assert_array_equal(handle['dipole_vec'][0, 0], dipole)


@pytest.mark.parametrize('output_format', ['hdf5', 'extxyz'])
def test_embedding_output_keeps_features_and_atom_selection(tmp_path, output_format):
    destination = tmp_path / f'embeddings.{output_format}'
    positions = [np.arange(9, dtype=np.float32).reshape(3, 3)]
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = [2, 0]
    if output_format == 'hdf5':
        store = init_hdf5_store(
            destination, batch_size=1, num_atoms=2, num_box_entries=3,
            observables=['x'], num_features=4,
        )
        write_to_hdf5(
            store, [np.zeros((3, 3))], positions, [np.zeros(3)],
            {'x': [embeddings]}, indices=indices,
        )
        with h5py.File(destination) as handle:
            actual = handle['x'][0]
            actual_positions = handle['positions'][0]
    else:
        write_to_extxyz(
            destination, Atoms('HOH'), None, [], positions,
            {'x': [embeddings]}, indices=indices,
            fractional_coordinates=False,
        )
        atoms = read(destination)
        actual = atoms.arrays['x']
        actual_positions = atoms.positions
    np.testing.assert_array_equal(actual, embeddings[indices])
    np.testing.assert_array_equal(actual_positions, positions[0][indices])


def test_nonperiodic_atoms_with_bounding_cell_remain_free():
    atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[10.0, 1.0, 0.0], [0.0, 11.0, 2.0], [0.0, 0.0, 12.0]],
        pbc=False,
    )
    assert cell_from_atoms_boundary_conditions(atoms, 12.0, None) is None


def test_mixed_axis_pbc_fails_clearly():
    atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([10.0, 11.0, 12.0]),
        pbc=[True, True, False],
    )
    with pytest.raises(ValueError, match="Mixed-axis periodic boundary conditions"):
        cell_from_atoms_boundary_conditions(atoms, 12.0, None)


def test_fully_periodic_atoms_keep_their_cell():
    cell = np.diag([10.0, 11.0, 12.0])
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    np.testing.assert_allclose(
        cell_from_atoms_boundary_conditions(atoms, 12.0, None), cell
    )


def test_cartesian_extxyz_output_is_not_scaled_twice(tmp_path):
    box = jnp.array([10.0, 20.0, 30.0])
    fractional = jnp.array([[0.1, 0.2, 0.3]])
    cartesian = output_position_array(
        fractional, box, fractional_coordinates=True,
        output_cartesian_positions=True,
    )

    path = tmp_path / "cartesian.extxyz"
    template = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.diag(box), pbc=True)
    write_to_extxyz(
        path,
        template,
        boxes=[np.asarray(box_for_trajectory(box))],
        momenta=[],
        positions=[np.asarray(cartesian)],
        fractional_coordinates=False,
    )

    frame = read(path)
    np.testing.assert_allclose(frame.positions, [[1.0, 4.0, 9.0]])
    np.testing.assert_allclose(frame.cell.array, np.diag([10.0, 20.0, 30.0]))


def test_sheared_cell_round_trips_through_extxyz(tmp_path):
    # JAX-MD uses column-vector affine transforms; ASE stores cell vectors as
    # rows.  A deliberately non-symmetric cell catches a missing transpose.
    jax_box = jnp.array([
        [10.0, 2.0, 1.0],
        [0.0, 8.0, 1.5],
        [0.0, 0.0, 6.0],
    ])
    fractional = jnp.array([[0.2, 0.3, 0.4]])
    expected_cartesian = np.asarray(
        positions_to_cartesian(fractional, jax_box, True)
    )
    ase_cell = np.asarray(box_for_trajectory(jax_box))

    path = tmp_path / "sheared.extxyz"
    template = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=ase_cell, pbc=True)
    write_to_extxyz(
        path,
        template,
        boxes=[ase_cell],
        momenta=[],
        positions=[np.asarray(fractional)],
        fractional_coordinates=True,
    )

    frame = read(path)
    np.testing.assert_allclose(frame.cell.array, ase_cell)
    np.testing.assert_allclose(frame.positions, expected_cartesian)


def test_hdf5_accepts_partial_tail_without_padding_and_keeps_dtype(tmp_path):
    path = tmp_path / "trajectory.hdf5"
    store = init_hdf5_store(
        path,
        batch_size=3,
        num_atoms=1,
        num_box_entries=3,
        exist_ok=True,
        dtype=np.float64,
        fractional_coordinates=False,
    )

    values = [1.123456789012345 + i for i in range(5)]

    def _write(indices):
        positions = [np.full((1, 3), values[i], dtype=np.float64) for i in indices]
        momenta = [np.full((1, 3), -values[i], dtype=np.float64) for i in indices]
        boxes = [np.array([10.0 + i, 11.0, 12.0], dtype=np.float64) for i in indices]
        write_to_hdf5(store, momenta, positions, boxes)

    _write(range(3))
    _write(range(3, 5))  # final buffer is shorter than batch_size

    with h5py.File(path, "r") as h5f:
        assert h5f["positions"].shape == (5, 1, 3)
        assert h5f["momenta"].shape == (5, 1, 3)
        assert h5f["box"].shape == (5, 3)
        assert h5f["positions"].dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(h5f["positions"][:, 0, 0], values)
        assert h5f.attrs["trajectory_layout"] == "frame-major-v1"
        assert not bool(h5f.attrs["fractional_coordinates"])


def test_hdf5_preserves_full_cell_matrix(tmp_path):
    path = tmp_path / "trajectory.hdf5"
    store = init_hdf5_store(
        path,
        batch_size=4,
        num_atoms=1,
        num_box_entries=(3, 3),
        exist_ok=True,
    )
    jax_box = jnp.array([
        [10.0, 2.0, 1.0],
        [0.0, 8.0, 1.5],
        [0.0, 0.0, 6.0],
    ])
    expected = np.asarray(box_for_trajectory(jax_box))
    write_to_hdf5(
        store,
        momenta=[np.zeros((1, 3))],
        positions=[np.zeros((1, 3))],
        boxes=[expected],
    )

    with h5py.File(path, "r") as h5f:
        assert h5f["box"].shape == (1, 3, 3)
        np.testing.assert_allclose(h5f["box"][0], expected)
        assert h5f.attrs["box_convention"] == "ase-cell-vectors-as-rows"


def test_hdf5_validates_whole_batch_before_resizing(tmp_path):
    path = tmp_path / "trajectory.hdf5"
    store = init_hdf5_store(
        path,
        batch_size=2,
        num_atoms=1,
        num_box_entries=3,
        exist_ok=True,
    )

    with pytest.raises(ValueError, match="same number of frames"):
        store.append({
            "positions": np.zeros((2, 1, 3)),
            "box": np.zeros((1, 3)),
        })

    with h5py.File(path, "r") as h5f:
        assert h5f["positions"].shape[0] == 0
        assert h5f["box"].shape[0] == 0


def test_hdf5_restart_refuses_to_truncate_existing_trajectory(tmp_path):
    output = tmp_path / "trajectory.hdf5"
    original = b"published trajectory bytes"
    output.write_bytes(original)

    with pytest.raises(RuntimeError, match="Refusing to truncate"):
        _validate_md_output_write_policy(
            output_file=output,
            output_format="hdf5",
            overwrite=False,
            restart_load_path=tmp_path / "restart.npz",
        )

    assert output.read_bytes() == original


def test_extxyz_restart_may_append_to_existing_trajectory(tmp_path):
    output = tmp_path / "trajectory.extxyz"
    output.write_text("existing frame\n", encoding="utf-8")

    _validate_md_output_write_policy(
        output_file=output,
        output_format="extxyz",
        overwrite=False,
        restart_load_path=tmp_path / "restart.npz",
    )


def test_nonrestart_output_still_requires_explicit_overwrite(tmp_path):
    output = tmp_path / "trajectory.extxyz"
    output.write_text("existing frame\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        _validate_md_output_write_policy(
            output_file=output,
            output_format="extxyz",
            overwrite=False,
            restart_load_path=None,
        )


def _nve_state(position):
    position = jnp.asarray(position)
    return simulate.NVEState(
        position=position,
        momentum=jnp.zeros_like(position),
        force=jnp.zeros_like(position),
        mass=jnp.ones((position.shape[0], 1)),
    )


def test_final_restart_overwrites_buffer_checkpoint_with_final_state(tmp_path):
    path = tmp_path / "restart.npz"
    box = jnp.array([10.0, 10.0, 10.0])

    save_restart_checkpoints(
        _nve_state([[1.0, 0.0, 0.0]]), box, 3,
        ensemble="nve", fractional_coordinates=True,
        restart_save_path=str(path),
    )
    save_restart_checkpoints(
        _nve_state([[5.0, 0.0, 0.0]]), box, 5,
        ensemble="nve", fractional_coordinates=True,
        restart_save_path=str(path),
    )

    with np.load(path, allow_pickle=False) as restart:
        assert int(restart["step"]) == 5
        np.testing.assert_allclose(restart["position"], [[5.0, 0.0, 0.0]])


def test_sheared_cell_restart_coordinate_round_trip(tmp_path):
    jax_box = jnp.array([
        [10.0, 2.0, 1.0],
        [0.0, 8.0, 1.5],
        [0.0, 0.0, 6.0],
    ])
    fractional = jnp.array([[0.2, 0.3, 0.4]])
    first = tmp_path / "fractional.npz"
    second = tmp_path / "cartesian.npz"

    save_state(
        _nve_state(fractional), jax_box, 7, str(first), ensemble="nve",
        fractional_coordinates=True,
    )
    cart_state, loaded_box, cycle = load_state(
        str(first), ensemble="nve", fractional_coordinates=False,
        precision=jnp.float32,
    )
    np.testing.assert_allclose(
        cart_state.position,
        positions_to_cartesian(fractional, jax_box, True),
        atol=1e-6,
    )
    np.testing.assert_allclose(loaded_box, jax_box)
    assert int(cycle) == 7

    save_state(
        cart_state, loaded_box, cycle, str(second), ensemble="nve",
        fractional_coordinates=False,
    )
    round_trip, round_trip_box, _ = load_state(
        str(second), ensemble="nve", fractional_coordinates=True,
        precision=jnp.float32,
    )
    np.testing.assert_allclose(round_trip.position, fractional, atol=1e-6)
    np.testing.assert_allclose(round_trip_box, jax_box)


def test_final_pysages_checkpoint_uses_latest_result(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(
        so3lr_md,
        "save_pysages_state",
        lambda result, path: saved.append((result, path)),
        raising=False,
    )
    final_result = object()
    path = str(tmp_path / "pysages.pkl")
    save_restart_checkpoints(
        None, None, 5,
        ensemble="nvt", fractional_coordinates=False,
        pysages_result=final_result,
        restart_pysages_save_path=path,
    )
    assert saved == [(final_result, path)]


@pytest.mark.parametrize("key", [
    "md_cycles", "md_steps", "save_buffer", "kspace_npt_cycles",
])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_positive_md_settings_reject_invalid_values(key, value):
    with pytest.raises(ValueError, match=key):
        validate_positive_md_settings({key: value})


@pytest.mark.parametrize("key", ["md_steps", "save_buffer", "kspace_npt_cycles"])
def test_perform_md_api_validates_counts_before_io(key, tmp_path):
    settings = {
        "input_file": str(tmp_path / "missing.xyz"),
        "output_file": str(tmp_path / "trajectory.xyz"),
        "md_cycles": 1,
        "md_steps": 1,
        "save_buffer": 1,
        "kspace_npt_cycles": 1,
    }
    settings[key] = 0
    with pytest.raises(ValueError, match=key):
        so3lr_md.perform_md(settings)


def _write_extxyz_frame(path, x, **kwargs):
    return write_to_extxyz(
        path, Atoms('H'), boxes=None, momenta=None,
        positions=[np.array([[x, 0., 0.]])],
        fractional_coordinates=False, **kwargs,
    )


@pytest.mark.parametrize('restart', [False, True])
def test_extxyz_overwrite_replaces_first_batch_then_appends(tmp_path, restart):
    path = tmp_path / 'trajectory.xyz'
    _write_extxyz_frame(path, 99.)
    append = _validate_md_output_write_policy(
        output_file=path, output_format='extxyz', overwrite=True,
        restart_load_path='restart.npz' if restart else None,
    )
    assert append is False
    _write_extxyz_frame(path, 1., append=append, overwrite=True)
    _write_extxyz_frame(path, 2., append=True, overwrite=True)
    frames = read(path, index=':')
    assert [frame.positions[0, 0] for frame in frames] == [1., 2.]


def test_extxyz_restart_without_overwrite_keeps_prior_frames(tmp_path):
    path = tmp_path / 'trajectory.xyz'
    _write_extxyz_frame(path, 1.)
    append = _validate_md_output_write_policy(
        output_file=path, output_format='extxyz', overwrite=False,
        restart_load_path='restart.npz',
    )
    _write_extxyz_frame(path, 2., append=append)
    assert [a.positions[0, 0] for a in read(path, index=':')] == [1., 2.]


@pytest.mark.parametrize('failure, expected', [
    (OSError, SystemExit), (KeyboardInterrupt, KeyboardInterrupt),
])
def test_extxyz_failed_first_batch_preserves_old_output(
    tmp_path, monkeypatch, failure, expected,
):
    path = tmp_path / 'trajectory.xyz'
    path.write_bytes(b'previous trajectory')

    def broken_write(stream, *args, **kwargs):
        stream.write('incomplete frame')
        raise failure('interrupted serialization')

    monkeypatch.setattr(so3lr_md.ase.io, 'write', broken_write)
    with pytest.raises(expected):
        _write_extxyz_frame(path, 1., append=False, overwrite=True)
    assert path.read_bytes() == b'previous trajectory'
    assert list(tmp_path.iterdir()) == [path]


def test_extxyz_empty_first_batch_preserves_old_output(tmp_path):
    path = tmp_path / 'trajectory.xyz'
    path.write_bytes(b'previous trajectory')
    write_to_extxyz(
        path, Atoms('H'), None, None, [], append=False, overwrite=True,
    )
    assert path.read_bytes() == b'previous trajectory'


def test_extxyz_first_batch_refuses_late_existing_output(tmp_path):
    path = tmp_path / 'trajectory.xyz'
    path.write_bytes(b'other run')
    with pytest.raises(SystemExit):
        _write_extxyz_frame(path, 1., append=False, overwrite=False)
    assert path.read_bytes() == b'other run'
    assert list(tmp_path.iterdir()) == [path]


def test_perform_md_does_not_append_when_restart_is_not_enabled(tmp_path, monkeypatch):
    path = tmp_path / 'trajectory.xyz'
    path.write_bytes(b'old trajectory')
    monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: None)
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        so3lr_md.perform_md({
            'output_file': str(path), 'output_format': 'extxyz',
            'restart_load_path': 'unused.npz',
        }, restart=False)
    assert path.read_bytes() == b'old trajectory'


def test_direct_perform_md_configures_requested_precision(monkeypatch):
    calls = []
    monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="npt_coupling='anisotropic'"):
        so3lr_md.perform_md({
            'precision': 'float64', 'matmul_precision': 'high',
            'ensemble': 'npt', 'npt_coupling': 'anisotropic',
        })
    assert calls == [{'precision': 'float64', 'matmul_precision': 'high'}]


@pytest.mark.parametrize('pbc, solver, mode, error', [
    (False, None, 'free', None),
    (True, None, 'periodic', None),
    ([True, True, False], None, None, 'Mixed-axis periodic'),
    (False, 'pme', None, 'requires a periodic cell'),
])
def test_optimization_honors_input_boundary_conditions(monkeypatch, pbc, solver, mode, error):
    atoms = Atoms('H', positions=[[1., 2., 3.]], cell=[20., 21., 22.], pbc=pbc)
    monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: None)
    monkeypatch.setattr(so3lr_md, 'read', lambda path: atoms.copy())
    monkeypatch.setattr(so3lr_md, 'validate_and_log_input', lambda *args, **kwargs: None)
    monkeypatch.setattr(so3lr_md, 'atoms_to_jnp', lambda atoms, **kwargs: {'positions': atoms.positions})

    class GeometryChecked(Exception):
        pass

    def inspect_box(shift_displacement, positions, cell, **kwargs):
        assert shift_displacement == mode
        if mode == 'free':
            assert cell is None
        else:
            np.testing.assert_array_equal(cell, atoms.cell.array)
        raise GeometryChecked

    monkeypatch.setattr(so3lr_md, 'handle_box', inspect_box)
    with pytest.raises(ValueError if error else GeometryChecked, match=error):
        so3lr_md.perform_min({
            'input_file': 'geometry.xyz', 'lr_cutoff': 12.,
            'kspace_electrostatics': solver,
        })


@pytest.mark.parametrize('output_format, destination', [
    ('extxyz', 'trajectory.xyz'), ('hdf5', 'trajectory_opt.xyz'),
])
def test_optimization_checks_actual_output_before_loading(
    tmp_path, monkeypatch, output_format, destination,
):
    actual = tmp_path / destination
    actual.write_bytes(b'previous output')
    monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: None)
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        so3lr_md.perform_min({
            'output_file': str(tmp_path / ('trajectory.hdf5' if output_format == 'hdf5' else destination)),
            'output_format': output_format,
        })
    assert actual.read_bytes() == b'previous output'


@pytest.mark.parametrize('output_format', ['extxyz', 'hdf5'])
def test_pre_md_relaxation_has_separate_output(tmp_path, monkeypatch, output_format):
    calls = []
    monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: None)
    monkeypatch.setattr(so3lr_md, 'perform_min', lambda settings, **kwargs: calls.append(('opt', dict(settings))))
    monkeypatch.setattr(so3lr_md, 'perform_md', lambda settings, *args, **kwargs: calls.append(('md', dict(settings))))
    output = tmp_path / ('trajectory.xyz' if output_format == 'extxyz' else 'trajectory.hdf5')
    settings = {'output_file': str(output), 'output_format': output_format}
    so3lr_md.run(settings)
    assert calls[0][1]['output_file'] == str(tmp_path / 'trajectory_opt.xyz')
    assert calls[0][1]['output_format'] == 'extxyz'
    assert calls[1][1] == settings
    assert settings['output_file'] == str(output)


def test_pre_md_relaxation_checks_md_destination_before_optimization(tmp_path, monkeypatch):
    path = tmp_path / 'trajectory.xyz'
    path.write_bytes(b'old trajectory')
    monkeypatch.setattr(so3lr_md, 'setup_jax_globals', lambda **kwargs: None)

    def forbidden(*args, **kwargs):
        pytest.fail('optimization must not run before checking the MD destination')

    monkeypatch.setattr(so3lr_md, 'perform_min', forbidden)
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        so3lr_md.run({'output_file': str(path), 'output_format': 'extxyz'})
    assert path.read_bytes() == b'old trajectory'


@pytest.mark.parametrize('failure_point, failure', [
    ('write', OSError), ('write', KeyboardInterrupt), ('flush', OSError),
])
def test_hdf5_failed_append_rolls_back_frame_counts(tmp_path, monkeypatch, failure_point, failure):
    path = tmp_path / 'trajectory.hdf5'
    store = init_hdf5_store(path, batch_size=1, num_atoms=1, num_box_entries=3)
    original_write = h5py.Dataset.__setitem__
    original_flush = h5py.File.flush
    failed = False

    def fail_once():
        nonlocal failed
        if not failed:
            failed = True
            raise failure('injected output failure')

    def write(dataset, key, value):
        if failure_point == 'write' and dataset.name == '/momenta':
            fail_once()
        return original_write(dataset, key, value)

    def flush(handle):
        if failure_point == 'flush':
            fail_once()
        return original_flush(handle)

    monkeypatch.setattr(h5py.Dataset, '__setitem__', write)
    monkeypatch.setattr(h5py.File, 'flush', flush)
    with pytest.raises(failure):
        store.append({
            'positions': np.zeros((1, 1, 3)),
            'momenta': np.zeros((1, 1, 3)), 'box': np.zeros((1, 3)),
        })
    assert store.i == 0
    with h5py.File(path) as handle:
        assert all(dataset.shape[0] == 0 for dataset in handle.values())


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
def test_checkpoint_serialization_failure_preserves_previous_file(tmp_path, monkeypatch, failure):
    path = tmp_path / 'restart.npz'
    path.write_bytes(b'previous checkpoint')

    def broken_save(stream, **kwargs):
        stream.write(b'incomplete checkpoint')
        raise failure('interrupted serialization')

    monkeypatch.setattr(so3lr_md.np, 'savez', broken_save)
    with pytest.raises(failure):
        save_state(_nve_state([[0., 0., 0.]]), np.ones(3), 1, str(path), ensemble='nve')
    assert path.read_bytes() == b'previous checkpoint'
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_retains_npz_suffix_behavior(tmp_path):
    save_state(_nve_state([[0., 0., 0.]]), np.ones(3), 1, str(tmp_path / 'restart'), ensemble='nve')
    assert list(tmp_path.iterdir()) == [tmp_path / 'restart.npz']
    with np.load(tmp_path / 'restart.npz', allow_pickle=False) as checkpoint:
        assert int(checkpoint['step']) == 1


@pytest.mark.parametrize('kind', ['extxyz', 'checkpoint'])
@pytest.mark.parametrize('existing', [False, True])
def test_staged_outputs_preserve_file_permissions(tmp_path, kind, existing):
    path = tmp_path / ('trajectory.xyz' if kind == 'extxyz' else 'restart.npz')
    reference = tmp_path / 'reference'
    reference.write_bytes(b'')
    expected_mode = stat.S_IMODE(reference.stat().st_mode)
    if existing:
        path.write_bytes(b'previous output')
        path.chmod(0o640)
        expected_mode = 0o640
    if kind == 'extxyz':
        _write_extxyz_frame(path, 1., append=False, overwrite=True)
    else:
        save_state(_nve_state([[0., 0., 0.]]), np.ones(3), 1, str(path), ensemble='nve')
    assert stat.S_IMODE(path.stat().st_mode) == expected_mode


@pytest.mark.parametrize('kind', ['extxyz-first', 'extxyz-append', 'hdf5', 'checkpoint', 'md-guard'])
@pytest.mark.parametrize('dangling', [False, True])
def test_output_writers_refuse_final_component_symlinks(tmp_path, kind, dangling):
    target = tmp_path / 'target'
    if not dangling:
        target.write_bytes(b'untouched target')
    link = tmp_path / 'output.npz'
    link.symlink_to(target)
    with pytest.raises(SystemExit if kind.startswith('extxyz') else RuntimeError):
        if kind.startswith('extxyz'):
            _write_extxyz_frame(link, 1., append=kind == 'extxyz-append', overwrite=True)
        elif kind == 'hdf5':
            init_hdf5_store(link, batch_size=1, num_atoms=1, num_box_entries=3, exist_ok=True)
        elif kind == 'checkpoint':
            save_state(_nve_state([[0., 0., 0.]]), np.ones(3), 1, str(link), ensemble='nve')
        else:
            _validate_md_output_write_policy(
                output_file=link, output_format='extxyz', overwrite=True,
                restart_load_path='restart.npz',
            )
    assert link.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert target.read_bytes() == b'untouched target'
