"""The MD state's precision (md_precision, float64 by default) around a float32 model, on the
neighbour-list engine as on flash."""
import pathlib
import subprocess

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from subprocess_env import so3lr_subprocess_env
from so3lr.jaxmd_utils import neighbor_list_featurizer, to_jax_md
from so3lr.md_settings_validation import canonical_precision, state_precision

ACETONE = pathlib.Path(__file__).resolve().parent / 'test_data' / 'acetone.xyz'


@pytest.fixture
def x64():
    """float64 arrays for the test, then back to the previous setting (the CLI turns x64 on itself)."""
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', previous)


def test_precision_helpers():
    assert canonical_precision(None) == 'float32' and canonical_precision(' Float64 ') == 'float64'
    with pytest.raises(ValueError):
        canonical_precision('mixed')
    assert state_precision('float32', {}) == 'float64'
    assert state_precision('float32', {'md_precision': 'float32'}) == 'float32'
    assert state_precision('float32', {'flash_mixed': False}) == 'float32'  # the older form
    assert state_precision('float32', {'md_precision': 'float64', 'flash_mixed': False}) == 'float64'
    assert state_precision('float64', {'md_precision': 'float32'}) == 'float64'


def test_featurizer_forms_displacements_before_the_cast(x64):
    """Far from the origin, float32 positions lose what the displacement needs."""
    R = jnp.array([[400.0, 0.0, 0.0], [400.123456789, 0.0, 0.0]], dtype=jnp.float64)
    nbr = jnp.array([[0, 1], [1, 0]])
    displacement, _ = space.free()
    graph = neighbor_list_featurizer(displacement, jnp.array([1, 1]), precision=jnp.float32,
                                     fractional_coordinates=False)(R, nbr, nbr)
    expected = (R[nbr[0]] - R[nbr[1]]).astype(jnp.float32)
    cast_first = R.astype(jnp.float32)[nbr[0]] - R.astype(jnp.float32)[nbr[1]]
    assert graph.edges.dtype == jnp.float32 and graph.positions.dtype == jnp.float32
    np.testing.assert_array_equal(graph.edges, expected)
    assert not np.array_equal(cast_first, expected)


class _Stub:
    """Per-atom energies in float32 whose float32 sum loses the 1."""
    cutoff = long_range_cutoff = 3.0

    def __call__(self, graph, has_aux=False):
        energies = jnp.array([1e8, 1.0, -1e8], jnp.float32) + 0.0 * jnp.sum(graph.edges)
        return (energies, {}) if has_aux else energies


def test_energy_is_summed_in_the_state_dtype(x64):
    displacement, _ = space.free()
    _, _, energy_fn = to_jax_md(_Stub(), displacement, None, species=jnp.array([1, 1, 1]),
                                disable_cell_list=True, fractional_coordinates=False, precision=jnp.float32)
    R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float64)
    nbr = jnp.array([[0, 1, 2, 0], [1, 0, 0, 2]])
    energy, forces = jax.value_and_grad(energy_fn)(R, neighbor=nbr, neighbor_lr=nbr)
    assert energy.dtype == jnp.float64 and float(energy) == 1.0
    assert forces.dtype == jnp.float64


def _so3lr(tmp_path, *args):
    run = subprocess.run(['so3lr', *args, '--input', str(ACETONE), '--engine', 'standard', '--no-relax',
                          '--md-cycles', '2', '--md-steps', '2'],
                         cwd=tmp_path, capture_output=True, text=True, env=so3lr_subprocess_env(), timeout=900)
    assert run.returncode == 0, f'so3lr {args[0]} failed:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}'


@pytest.mark.slow
def test_neighbour_list_engine_state_trajectory_and_restart_dtypes(tmp_path):
    """Default: float64 state, trajectories in the model's float32. A float32-state checkpoint
    (md_precision float32) continues in another ensemble with the float64 state (the loaded
    arrays are cast)."""
    _so3lr(tmp_path, 'nve', '--output', str(tmp_path / 'a.hdf5'), '--restart-save', str(tmp_path / 'a.npz'))
    assert np.load(tmp_path / 'a.npz')['position'].dtype == np.float64
    with h5py.File(tmp_path / 'a.hdf5') as f:
        assert f['positions'].dtype == np.float32

    _so3lr(tmp_path, 'nve', '--md-precision', 'float32', '--output', str(tmp_path / 'b.xyz'),
           '--restart-save', str(tmp_path / 'b.npz'))
    assert np.load(tmp_path / 'b.npz')['position'].dtype == np.float32

    _so3lr(tmp_path, 'nvt-langevin', '--output', str(tmp_path / 'c.xyz'),
           '--restart-load', str(tmp_path / 'b.npz'), '--restart-save', str(tmp_path / 'c.npz'))
    c = np.load(tmp_path / 'c.npz')
    assert c['position'].dtype == np.float64 and c['momentum'].dtype == np.float64


def test_restart_into_another_ensemble_gives_jax_arrays_in_the_state_dtype(tmp_path, x64):
    """load_state casts before it raises, so the caller gets device arrays in the run's dtype
    (numpy positions raise on the neighbour lists' padding index)."""
    from so3lr.cli.so3lr_md import RestartInNewEnsembleError, load_state
    path = tmp_path / 'nvt.npz'
    f32 = np.float32
    np.savez(path, position=np.zeros((2, 3), f32), momentum=np.zeros((2, 3), f32), mass=np.ones(2, f32),
             box=np.zeros(1, f32), ensemble='nvt', step=0)
    with pytest.raises(RestartInNewEnsembleError) as err:
        load_state(path_to_load=str(path), ensemble='nve', precision=jnp.float64)
    for key in ('position', 'momentum', 'mass', 'box'):
        value = err.value.loaded_state[key]
        assert isinstance(value, jax.Array) and value.dtype == jnp.float64, key
