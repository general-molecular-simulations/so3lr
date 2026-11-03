"""The MD state's precision (md_precision, float64 by default) around a float32 model, on the
neighbour-list engine as on flash, in so3lr eval and the jax-md potential; float64 needs x64."""
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
    assert state_precision('float32', {'flash_mixed': None}) == 'float64'  # unset, not off
    with pytest.raises(ValueError):
        state_precision('float64', {'md_precision': 'flaot32'})


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


# so3lr eval and the ASE calculator: `precision` is the model's dtype there too.

MODEL = 'so3lr-tauri-s-2606'


def _eval_batch(frames, tmp_path):
    """frames as one padded batch of so3lr eval's model inputs (read back from an extxyz)."""
    from ase.io import write
    from so3lr.mlff.data import AseDataLoaderSparse
    from so3lr.mlff.utils import jraph_utils
    from so3lr.model_config import load_model_config
    from so3lr.model_registry import resolve_model
    write(tmp_path / 'frames.extxyz', frames)
    cutoff = load_model_config(resolve_model(MODEL).dir)['model']['cutoff']
    data, stats = AseDataLoaderSparse(str(tmp_path / 'frames.extxyz')).load(
        cutoff=cutoff, cutoff_lr=12.0, calculate_neighbors_lr=True)
    n = len(data)
    (batch,) = jraph_utils.dynamically_batch_with_lr(
        data, n_node=stats['max_num_of_nodes'] * n + 1, n_edge=stats['max_num_of_edges'] * n + 1,
        n_graph=n + 1, n_pairs=stats['max_num_of_pairs'] * n + 1)
    return jraph_utils.graph_to_batch_fn(*batch)


def test_eval_model_forms_displacements_before_the_cast(tmp_path, x64):
    """so3lr eval's model in float32 around float64 positions: 400 A from the origin the forces are
    the molecule's at the origin, as in MD. Cast to float32 first, the displacements lose that."""
    from ase.io import read
    from so3lr.base_calculator import make_so3lr
    from so3lr.mlff.calculators.potential import cast_float_params
    near = read(ACETONE)
    far = near.copy()
    far.positions += 400.0
    inputs = _eval_batch([near, far], tmp_path)
    n = len(near)
    model = jax.jit(make_so3lr(model=MODEL, precision=jnp.float32))

    out = model(inputs)
    forces, energy = np.asarray(out['forces']), np.asarray(out['energy'])
    # the structures' energies summed over their atoms in float64 too, not inside the model
    assert forces.dtype == energy.dtype == np.float64
    assert abs(energy[0] - energy[1]) < 1e-4, energy
    cast_first = np.asarray(model(cast_float_params(inputs, jnp.float32))['forces'])
    moved, moved_cast_first = (np.abs(f[:n] - f[n:2 * n]).max() for f in (forces, cast_first))
    assert moved < 1e-5 < moved_cast_first, (moved, moved_cast_first)



def test_eval_precision_is_the_models_and_predictions_are_saved_unrounded(tmp_path, x64):
    """so3lr eval in float32 computes in float32 (it computed in float64 whatever the flag), and in
    float64 it saves So3lrCalculator's float64 energy, not one rounded to 6 digits."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
    from so3lr import So3lrCalculator
    from so3lr.cli.so3lr_eval import evaluate_so3lr_on
    atoms = read(ACETONE)
    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((len(atoms), 3)))
    write(tmp_path / 'data.extxyz', atoms)
    saved = {}
    for precision in ('float32', 'float64'):
        evaluate_so3lr_on(str(tmp_path / 'data.extxyz'), model_path=MODEL, precision=precision,
                          targets='energy,forces', engine='standard', save_to=str(tmp_path / f'{precision}.extxyz'))
        saved[precision] = read(tmp_path / f'{precision}.extxyz')
    f32, f64 = (saved[p].arrays['forces_so3lr'] for p in ('float32', 'float64'))
    assert 0 < np.abs(f32 - f64).max() < 1e-3

    atoms.calc = So3lrCalculator(model=MODEL, lr_cutoff=12.0, dtype=np.float64)
    assert abs(saved['float64'].info['energy_so3lr'] - atoms.get_potential_energy()) < 1e-8


def test_jax_md_potential_computes_in_its_dtype(x64):
    """So3lrPotential(dtype=float32) through to_jax_md around float64 positions: a float32 model on
    float64 displacements (the float32 parameters were promoted, so it computed in float64)."""
    from ase.io import read
    from so3lr import So3lrPotential, to_jax_md
    atoms = read(ACETONE)
    R = jnp.asarray(atoms.positions, jnp.float64)
    displacement, _ = space.free()

    def forces(dtype):
        neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
            So3lrPotential(model=MODEL, dtype=dtype), displacement, None, species=jnp.asarray(atoms.numbers),
            disable_cell_list=True, fractional_coordinates=False)
        return jax.grad(energy_fn)(R, neighbor=neighbor_fn.allocate(R).idx, neighbor_lr=neighbor_fn_lr.allocate(R).idx)

    f32, f64 = forces(jnp.float32), forces(jnp.float64)
    assert f32.dtype == jnp.float64
    assert 0 < np.abs(f32 - f64).max() < 1e-3


def test_float64_without_x64_raises():
    """JAX truncates float64 to float32 without x64, with a warning only: the ASE calculator and the
    potential refuse it before loading the model."""
    from so3lr import So3lrCalculator, So3lrPotential
    with jax.enable_x64(False):
        for make in (So3lrCalculator, So3lrPotential):
            with pytest.raises(ValueError, match='float64 needs'):
                make(model=MODEL, dtype=np.float64)


def test_ase_calculator_forms_displacements_before_the_cast(x64):
    """So3lrCalculator in float32 (its default) around float64 positions, as in so3lr's MD: 400 A
    from the origin it gives the forces at the origin, in float64; without x64 (float32) it does not."""
    from ase.io import read
    from so3lr import So3lrCalculator
    near = read(ACETONE)
    far = near.copy()
    far.positions += 400.0

    def forces(**kwargs):
        calc = So3lrCalculator(model=MODEL, **kwargs)
        return [calc.get_forces(atoms) for atoms in (near, far)]

    f_near, f_far = forces()
    with jax.enable_x64(False):
        g_near, g_far = forces()
    assert f_near.dtype == np.float64
    moved, moved_float32 = np.abs(f_near - f_far).max(), np.abs(g_near - g_far).max()
    assert moved < 1e-5 < moved_float32, (moved, moved_float32)
