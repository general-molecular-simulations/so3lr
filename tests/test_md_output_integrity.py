"""Regression tests for lossless MD trajectory and restart output."""

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from jax_md import simulate

from so3lr.cli import so3lr_md
from so3lr.md_settings_validation import validate_md_counts
from so3lr.pbc import lattice_rows
from so3lr.cli.so3lr_md import (
    _validate_md_output_write_policy,
    cell_from_atoms_boundary_conditions,
    check_box_for_neighbour_lists,
    init_hdf5_store,
    load_state,
    output_position_array,
    positions_to_cartesian,
    save_restart_checkpoints,
    save_state,
    write_to_extxyz,
    write_to_hdf5,
)

# JAX-MD uses column-vector affine transforms; ASE stores cell vectors as rows.
# A deliberately non-symmetric cell catches a missing transpose.
SHEARED_BOX = np.array([[10.0, 2.0, 1.0], [0.0, 8.0, 1.5], [0.0, 0.0, 6.0]])


def _nve_state(position):
    position = jnp.asarray(position)
    return simulate.NVEState(
        position=position, momentum=jnp.zeros_like(position),
        force=jnp.zeros_like(position), mass=jnp.ones((position.shape[0], 1)),
    )


def _write_extxyz_frame(path, x, **kwargs):
    return write_to_extxyz(
        path, Atoms('H'), boxes=None, momenta=None,
        positions=[np.array([[x, 0., 0.]])],
        fractional_coordinates=False, **kwargs,
    )


# --- HDF5 trajectory -------------------------------------------------------

def test_hdf5_accepts_partial_tail_without_padding_and_keeps_dtype(tmp_path):
    path = tmp_path / 'trajectory.hdf5'
    store = init_hdf5_store(
        path, batch_size=3, num_atoms=1, num_box_entries=3, exist_ok=True,
        dtype=np.float64, fractional_coordinates=False,
    )
    values = [1.123456789012345 + i for i in range(5)]

    def _write(indices):
        write_to_hdf5(
            store,
            [np.full((1, 3), -values[i]) for i in indices],
            [np.full((1, 3), values[i]) for i in indices],
            [np.array([10.0 + i, 11.0, 12.0]) for i in indices],
        )

    _write(range(3))
    _write(range(3, 5))  # final buffer is shorter than batch_size

    with h5py.File(path, 'r') as h5f:
        assert h5f['positions'].shape == (5, 1, 3)
        assert h5f['momenta'].shape == (5, 1, 3)
        assert h5f['box'].shape == (5, 3)
        assert h5f['positions'].dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(h5f['positions'][:, 0, 0], values)
        assert h5f.attrs['trajectory_layout'] == 'frame-major-v1'
        assert not bool(h5f.attrs['fractional_coordinates'])


def test_hdf5_validates_whole_batch_before_resizing(tmp_path):
    path = tmp_path / 'trajectory.hdf5'
    store = init_hdf5_store(path, batch_size=2, num_atoms=1, num_box_entries=3)
    with pytest.raises(ValueError, match='same n_frames'):
        store.append({'positions': np.zeros((2, 1, 3)), 'box': np.zeros((1, 3))})
    with h5py.File(path, 'r') as h5f:
        assert h5f['positions'].shape[0] == h5f['box'].shape[0] == 0


@pytest.mark.parametrize('indices', [None, [2, 0]])
def test_hdf_dipole_of_three_atoms_is_molecular(tmp_path, indices):
    path = tmp_path / 'water.hdf5'
    dipole = np.array([1., 2., 3.], dtype=np.float32)
    store = init_hdf5_store(
        path, batch_size=1, num_atoms=3 if indices is None else len(indices),
        num_box_entries=3, observables=['dipole_vec'],
    )
    write_to_hdf5(
        store, [np.zeros((3, 3))], [np.arange(9.).reshape(3, 3)], [np.zeros(3)],
        {'dipole_vec': [dipole]}, indices=indices,
    )
    with h5py.File(path) as h5f:
        assert h5f['dipole_vec'].shape == (1, 1, 3)
        np.testing.assert_array_equal(h5f['dipole_vec'][0, 0], dipole)


def test_pysages_restart_needs_so3lr_restart(tmp_path):
    """A PySAGES state without an SO3LR restart is refused before anything runs (and
    without PySAGES installed); it used to reach an unset restart_data_pysages
    (UnboundLocalError) after setup."""
    settings = {
        'do_enhanced_sampling': True,
        'restart_pysages_load_path': str(tmp_path / 'restart_pysages.pkl'),   # no restart_load_path
        'input_file': str(tmp_path / 'in.xyz'),
        'output_file': str(tmp_path / 'out.xyz'),
        'ensemble': 'nvt',
    }
    with pytest.raises(ValueError, match='restart_pysages_load_path needs an SO3LR restart'):
        so3lr_md.perform_md(settings)


@pytest.mark.parametrize('output_format', ['hdf5', 'extxyz'])
def test_embedding_output_keeps_features_and_atom_selection(tmp_path, output_format):
    path = tmp_path / f'embeddings.{output_format}'
    positions = [np.arange(9, dtype=np.float32).reshape(3, 3)]
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = [2, 0]
    if output_format == 'hdf5':
        store = init_hdf5_store(
            path, batch_size=1, num_atoms=2, num_box_entries=3,
            observables=['x'], num_features=4,
        )
        write_to_hdf5(store, [np.zeros((3, 3))], positions, [np.zeros(3)],
                      {'x': [embeddings]}, indices=indices)
        with h5py.File(path) as h5f:
            actual, actual_positions = h5f['x'][0], h5f['positions'][0]
    else:
        write_to_extxyz(path, Atoms('HOH'), None, [], positions, {'x': [embeddings]},
                        indices=indices, fractional_coordinates=False)
        atoms = read(path)
        actual, actual_positions = atoms.arrays['x'], atoms.positions
    np.testing.assert_array_equal(actual, embeddings[indices])
    np.testing.assert_array_equal(actual_positions, positions[0][indices])


def test_hdf5_preserves_full_sheared_cell_matrix(tmp_path):
    path = tmp_path / 'trajectory.hdf5'
    store = init_hdf5_store(path, batch_size=4, num_atoms=1, num_box_entries=(3, 3))
    expected = np.asarray(lattice_rows(SHEARED_BOX))
    write_to_hdf5(store, [np.zeros((1, 3))], [np.zeros((1, 3))], [expected])
    with h5py.File(path, 'r') as h5f:
        assert h5f['box'].shape == (1, 3, 3)
        np.testing.assert_allclose(h5f['box'][0], expected)
        assert h5f.attrs['box_convention'] == 'ase-cell-vectors-as-rows'


# --- extxyz trajectory -----------------------------------------------------

def test_cartesian_extxyz_output_is_not_scaled_twice(tmp_path):
    box = jnp.array([10.0, 20.0, 30.0])
    cartesian = output_position_array(
        jnp.array([[0.1, 0.2, 0.3]]), box,
        fractional_coordinates=True, output_cartesian_positions=True,
    )
    path = tmp_path / 'cartesian.extxyz'
    template = Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=np.diag(box), pbc=True)
    write_to_extxyz(path, template, boxes=[np.asarray(lattice_rows(box))],
                    momenta=[], positions=[np.asarray(cartesian)],
                    fractional_coordinates=False)
    frame = read(path)
    np.testing.assert_allclose(frame.positions, [[1.0, 4.0, 9.0]])
    np.testing.assert_allclose(frame.cell.array, np.diag([10.0, 20.0, 30.0]))


def test_sheared_cell_round_trips_through_extxyz(tmp_path):
    fractional = jnp.array([[0.2, 0.3, 0.4]])
    ase_cell = np.asarray(lattice_rows(SHEARED_BOX))
    path = tmp_path / 'sheared.extxyz'
    template = Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=ase_cell, pbc=True)
    write_to_extxyz(path, template, boxes=[ase_cell], momenta=[],
                    positions=[np.asarray(fractional)], fractional_coordinates=True)
    frame = read(path)
    np.testing.assert_allclose(frame.cell.array, ase_cell)
    np.testing.assert_allclose(
        frame.positions, positions_to_cartesian(fractional, SHEARED_BOX, True))


# --- Boundary conditions ---------------------------------------------------

@pytest.mark.parametrize('pbc, periodic', [(False, False), (True, True)])
def test_input_pbc_decides_periodicity(pbc, periodic):
    cell = np.diag([30.0, 31.0, 32.0]) if periodic else [[10., 1., 0.], [0., 11., 2.], [0., 0., 12.]]
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=pbc)
    result = cell_from_atoms_boundary_conditions(atoms, 12.0, None)
    if periodic:
        np.testing.assert_allclose(result, cell)
    else:
        assert result is None  # a bounding cell with pbc=False stays free


@pytest.mark.parametrize('box', [np.diag([23.0, 30.0, 30.0]), np.array([23.0, 30.0, 30.0]), np.array(23.0)])
def test_md_box_check_takes_every_box_shape(box):
    """The NPT box (lengths, a semi-isotropic matrix or a scalar) is checked after
    every cycle, and a box from a restart file before the run."""
    with pytest.raises(ValueError, match='at least twice the'):
        check_box_for_neighbour_lists(box, 12.0, 'NPT box')
    check_box_for_neighbour_lists(box * 2, 12.0, 'NPT box')


def test_mixed_axis_pbc_fails_clearly():
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=np.diag([10.0, 11.0, 12.0]),
                  pbc=[True, True, False])
    with pytest.raises(ValueError, match='mixed-axis periodic'):
        cell_from_atoms_boundary_conditions(atoms, 12.0, None)


# --- Output overwrite / restart policies -----------------------------------

@pytest.mark.parametrize('output_format, restart, match', [
    ('hdf5', True, 'Refusing to truncate'),      # HDF5 cannot append on restart
    ('extxyz', False, 'Refusing to overwrite'),  # fresh runs need --overwrite
])
def test_existing_output_is_refused_without_overwrite(tmp_path, output_format, restart, match):
    output = tmp_path / f'trajectory.{output_format}'
    output.write_bytes(b'published trajectory bytes')
    with pytest.raises(RuntimeError, match=match):
        _validate_md_output_write_policy(
            output_file=output, output_format=output_format, overwrite=False,
            restart_load_path=tmp_path / 'restart.npz' if restart else None,
        )
    assert output.read_bytes() == b'published trajectory bytes'


def test_extxyz_restart_without_overwrite_appends_to_prior_frames(tmp_path):
    path = tmp_path / 'trajectory.xyz'
    _write_extxyz_frame(path, 1.)
    append = _validate_md_output_write_policy(
        output_file=path, output_format='extxyz', overwrite=False,
        restart_load_path='restart.npz',
    )
    assert append is True
    _write_extxyz_frame(path, 2., append=append)
    assert [a.positions[0, 0] for a in read(path, index=':')] == [1., 2.]


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
    assert [f.positions[0, 0] for f in read(path, index=':')] == [1., 2.]


def test_pre_md_relaxation_checks_md_destination_before_optimization(tmp_path, monkeypatch):
    path = tmp_path / 'trajectory.xyz'
    path.write_bytes(b'old trajectory')
    monkeypatch.setattr(so3lr_md, 'perform_min', lambda *a, **k: pytest.fail('relaxed first'))
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        so3lr_md.run({'output_file': str(path), 'output_format': 'extxyz'})
    assert path.read_bytes() == b'old trajectory'


@pytest.mark.parametrize('key', ['md_cycles', 'md_steps', 'save_buffer', 'kspace_npt_cycles'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_md_counts_must_be_positive_integers(key, value):
    with pytest.raises(ValueError, match=key):
        validate_md_counts({key: value})


def test_perform_md_api_validates_counts_before_io(tmp_path):
    with pytest.raises(ValueError, match='md_steps'):
        so3lr_md.perform_md({
            'input_file': str(tmp_path / 'missing.xyz'),
            'output_file': str(tmp_path / 'trajectory.xyz'),
            'md_cycles': 1, 'md_steps': 0, 'save_buffer': 1,
        })


# --- Checkpoints -----------------------------------------------------------

def test_final_restart_overwrites_buffer_checkpoint_with_final_state(tmp_path):
    path = tmp_path / 'restart.npz'
    for step, x in ((3, 1.0), (5, 5.0)):
        save_restart_checkpoints(
            _nve_state([[x, 0.0, 0.0]]), jnp.full(3, 10.0), step, ensemble='nve',
            fractional_coordinates=True, restart_save_path=str(path),
        )
    with np.load(path, allow_pickle=False) as restart:
        assert int(restart['step']) == 5
        np.testing.assert_allclose(restart['position'], [[5.0, 0.0, 0.0]])


def test_sheared_cell_restart_coordinate_round_trip(tmp_path):
    fractional = jnp.array([[0.2, 0.3, 0.4]])
    first, second = tmp_path / 'fractional.npz', tmp_path / 'cartesian.npz'
    save_state(_nve_state(fractional), SHEARED_BOX, 7, str(first), ensemble='nve',
               fractional_coordinates=True)
    cart_state, box, cycle = load_state(str(first), ensemble='nve',
                                        fractional_coordinates=False, precision=jnp.float32)
    np.testing.assert_allclose(
        cart_state.position, positions_to_cartesian(fractional, SHEARED_BOX, True), atol=1e-6)
    np.testing.assert_allclose(box, SHEARED_BOX)
    assert int(cycle) == 7

    save_state(cart_state, box, cycle, str(second), ensemble='nve', fractional_coordinates=False)
    round_trip, round_trip_box, _ = load_state(str(second), ensemble='nve',
                                               fractional_coordinates=True, precision=jnp.float32)
    np.testing.assert_allclose(round_trip.position, fractional, atol=1e-6)
    np.testing.assert_allclose(round_trip_box, SHEARED_BOX)


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


@pytest.mark.parametrize('kind', ['extxyz', 'checkpoint'])
@pytest.mark.parametrize('dangling', [False, True])
def test_output_writers_write_through_symlinks(tmp_path, kind, dangling):
    target = tmp_path / 'scratch' / 'output.npz'
    target.parent.mkdir()
    if not dangling:
        target.write_bytes(b'previous output')
    link = tmp_path / 'output.npz'
    link.symlink_to(target)
    if kind == 'extxyz':
        _write_extxyz_frame(link, 1., append=False, overwrite=True)
    else:
        save_state(_nve_state([[0., 0., 0.]]), np.ones(3), 1, str(link), ensemble='nve')
    assert link.is_symlink()
    assert target.read_bytes() not in (b'', b'previous output')
