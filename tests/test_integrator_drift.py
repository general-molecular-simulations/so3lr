"""Test integrator drift across ensembles (NVE, NVT, NPT iso, NPT flex).

Asserts that the drift of each integrator's conserved quantity stays below
thresholds established from measurement runs (scripts/measure_drift.py). The
NPT flex thresholds verify the integrator bug fix (fractional coordinate
conversion) holds.

Thresholds are relative to |H0| and have ~3x safety margin over measured
values on water_64:
- f64: all ensembles ~3.0e-4 → threshold 1.0e-3
- f32: NVT/NPT iso ~3.0e-4 → threshold 1.0e-3
- f32: NPT flex ~5-8e-3 → threshold 2.5e-2
The assertion is made per atom, max|H - H0| / N < threshold * |H0| / N: the
same bound, reported in meV/atom so a failure reads as a physical number.

The anisotropic flex mode is not tested here: the CLI and create_npt_nhc_fn
reject it (tests/test_npt_coupling_safety.py).

The step and the invariant are jitted (nothing inside the potential is), and
the invariant is sampled every INVARIANT_EVERY steps. Sampling can only miss
a peak between samples, so the check is never stricter than sampling every
step was.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import pytest

jax.config.update('jax_enable_x64', True)

from ase.io import read
from jax_md import space, simulate, units, partition

from so3lr.jaxmd_utils import neighbor_list_featurizer
from so3lr.mlff.calculators.potential import PotentialSparse
from so3lr.model_registry import resolve_model

PACKAGE_DIR = pathlib.Path(__file__).parent.parent.resolve()
# 'sv2j_b128_l3nnn_2_nano_lr_full_2' was the dev name for the current v2 M lineage.
# Resolve via the registry so the test is runnable from a clean checkout/install.
_SPEC = resolve_model('so3lr-tauri-m')
MODEL_DIR, MODEL_FROM_FILE = _SPEC.dir, _SPEC.from_file

N_STEPS = 100
INVARIANT_EVERY = 10
DT_FS = 0.5e-3  # 0.5 fs in ps (metal units)


def _setup_system(dtype):
    """Load water_64 system and create energy function."""
    mol = read(PACKAGE_DIR / 'tests' / 'test_data' / 'water_64.xyz')
    cell = jnp.array(np.diag(mol.get_cell()), dtype=dtype)
    species = jnp.array(mol.get_atomic_numbers())
    masses = jnp.array(mol.get_masses(), dtype=dtype)

    displacement, shift = space.periodic_general(cell, fractional_coordinates=True)
    positions = jnp.array(mol.get_positions(), dtype=dtype) / cell

    mlff_potential = PotentialSparse.create_from_workdir(
        MODEL_DIR,
        from_file=MODEL_FROM_FILE,
        long_range_kwargs=dict(
            cutoff_lr=10.0,
            dispersion_energy_cutoff_lr_damping=2.0,
            neighborlist_format_lr='ordered_sparse',
        ),
        dtype=dtype,
    )

    neighbor_fn = partition.neighbor_list(
        displacement, cell, r_cutoff=mlff_potential.cutoff,
        capacity_multiplier=1.25, buffer_size_multiplier=1.25,
        fractional_coordinates=True,
        format=partition.NeighborListFormat(1),
        disable_cell_list=True,
    )
    neighbor_fn_lr = partition.neighbor_list(
        displacement, cell, r_cutoff=mlff_potential.long_range_cutoff,
        capacity_multiplier=1.25, buffer_size_multiplier=1.25,
        fractional_coordinates=True,
        format=partition.NeighborListFormat(2),
        disable_cell_list=True,
    )

    featurizer = neighbor_list_featurizer(
        displacement, species, total_charge=0., precision=dtype,
        fractional_coordinates=True,
    )

    def energy_fn(R, neighbor, neighbor_lr, **kwargs):
        graph = featurizer(R, neighbor, neighbor_lr, **kwargs)
        return mlff_potential(graph).sum()

    nbrs = neighbor_fn.allocate(positions, box=cell)
    nbrs_lr = neighbor_fn_lr.allocate(positions, box=cell)

    return energy_fn, shift, positions, cell, masses, species, nbrs, nbrs_lr


@pytest.fixture(scope='module')
def water_system():
    """Build the water_64 system (model, neighbour lists) once per dtype."""
    cache = {}

    def get(dtype):
        key = jnp.dtype(dtype).name
        if key not in cache:
            cache[key] = _setup_system(dtype)
        return cache[key]

    return get


def _measure_drift(init_fn, apply_fn, invariant_fn, init_args, init_kwargs,
                   step_kwargs, invariant_kwargs, n_steps):
    """Run simulation and return (max |H - H0|, H0)."""
    key = jax.random.PRNGKey(42)
    state = init_fn(key, *init_args, **init_kwargs)
    step = jax.jit(apply_fn)
    invariant = jax.jit(invariant_fn)

    H0 = float(invariant(state, **invariant_kwargs))
    H_max_dev = 0.0

    for i in range(1, n_steps + 1):
        state = step(state, **step_kwargs)
        if i % INVARIANT_EVERY == 0 or i == n_steps:
            H_i = float(invariant(state, **invariant_kwargs))
            H_max_dev = max(H_max_dev, abs(H_i - H0))

    return H_max_dev, H0


def _run_ensemble(ensemble, dtype, system):
    """Run a single ensemble and return (max |H - H0|, H0, n_atoms)."""
    energy_fn, shift, positions, cell, masses, _, nbrs, nbrs_lr = system
    n_atoms = positions.shape[0]

    # x64 is enabled globally (see top of file), so metal_unit_system() returns
    # float64 scalars. Cast every scalar quantity to the system dtype so the
    # float32 run is genuinely single precision -- otherwise a float64 kT / tau
    # upcasts the Nose-Hoover chain state while the momenta stay float32, which
    # makes the chain's lax.scan carry dtypes disagree.
    unit_system = units.metal_unit_system()
    dt = jnp.asarray(DT_FS * unit_system['time'], dtype)
    kT = jnp.asarray(300.0 * unit_system['temperature'], dtype)
    P = jnp.asarray(1.0 * unit_system['pressure'], dtype)
    thermostat_kwargs = {'tau': jnp.asarray(dt * 100, dtype)}
    barostat_kwargs = {'tau': jnp.asarray(dt * 1000, dtype)}

    # Guard against precision mixing regressing (float64 scalars + float32
    # system would upcast the thermostat chain and break its lax.scan carry).
    expected_dtype = jnp.dtype(dtype)
    assert dt.dtype == expected_dtype
    assert kT.dtype == expected_dtype
    assert P.dtype == expected_dtype
    assert thermostat_kwargs['tau'].dtype == expected_dtype

    nbrs_idx = nbrs.idx
    nbrs_lr_idx = nbrs_lr.idx

    if ensemble == 'NVE':
        # Plain velocity-Verlet: conserved quantity is the total energy
        # E = PE + KE (the jax-md fork has no nve_invariant). kT only seeds the
        # initial Maxwell-Boltzmann momenta (init_fn's required arg); it does
        # not enter the dynamics. Passing it to nve() would be silently dropped
        # into **sim_kwargs, so it must go to init_fn instead.
        init_fn, apply_fn = simulate.nve(energy_fn, shift, dt=dt)
        return (*_measure_drift(
            init_fn, apply_fn,
            lambda state, **kw: energy_fn(state.position, **kw)
            + simulate.kinetic_energy(state),
            init_args=(positions,),
            init_kwargs=dict(kT=kT, mass=masses,
                             neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                             box=cell),
            step_kwargs=dict(neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                             box=cell),
            invariant_kwargs=dict(neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                                  box=cell),
            n_steps=N_STEPS,
        ), n_atoms)

    elif ensemble == 'NVT':
        init_fn, apply_fn = simulate.nvt_nose_hoover(
            energy_fn, shift, dt=dt, kT=kT, tau=dt * 100,
        )
        return (*_measure_drift(
            init_fn, apply_fn,
            lambda state, **kw: simulate.nvt_nose_hoover_invariant(
                energy_fn, state, kT=kT, **kw),
            init_args=(positions,),
            init_kwargs=dict(mass=masses,
                             neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                             box=cell, kT=kT),
            step_kwargs=dict(neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                             box=cell, kT=kT),
            invariant_kwargs=dict(neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                                  box=cell),
            n_steps=N_STEPS,
        ), n_atoms)

    elif ensemble == 'NPT_iso':
        init_fn, apply_fn = simulate.npt_nose_hoover(
            energy_fn, shift, dt=dt, pressure=P, kT=kT,
            barostat_kwargs=barostat_kwargs,
            thermostat_kwargs=thermostat_kwargs,
        )
        return (*_measure_drift(
            init_fn, apply_fn,
            lambda state, **kw: simulate.npt_nose_hoover_invariant(
                energy_fn, state, pressure=P, kT=kT, **kw),
            init_args=(positions, cell),
            init_kwargs=dict(mass=masses,
                             neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                             kT=kT),
            step_kwargs=dict(kT=kT, pressure=P,
                             neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx),
            invariant_kwargs=dict(neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx),
            n_steps=N_STEPS,
        ), n_atoms)

    else:
        # NPT flex variants
        coupling_map = {
            'NPT_flex_iso': 'isotropic',
            'NPT_flex_semi_iso': 'semi-isotropic',
        }
        coupling = coupling_map[ensemble]
        init_fn, apply_fn = simulate.npt_nose_hoover_flex(
            energy_fn, shift, dt=dt, pressure=P, kT=kT, coupling=coupling,
            barostat_kwargs=barostat_kwargs,
            thermostat_kwargs=thermostat_kwargs,
        )
        return (*_measure_drift(
            init_fn, apply_fn,
            lambda state, **kw: simulate.npt_nose_hoover_flex_invariant(
                energy_fn, state, pressure=P, kT=kT, **kw),
            init_args=(positions, cell),
            init_kwargs=dict(mass=masses,
                             neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx,
                             kT=kT),
            step_kwargs=dict(kT=kT, pressure=P,
                             neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx),
            invariant_kwargs=dict(neighbor=nbrs_idx, neighbor_lr=nbrs_lr_idx),
            n_steps=N_STEPS,
        ), n_atoms)


ALL_ENSEMBLES = ['NVE', 'NVT', 'NPT_iso', 'NPT_flex_iso', 'NPT_flex_semi_iso']


def _assert_drift_below(ensemble, precision, threshold, max_dev, H0, n_atoms):
    """max|H - H0| / N < threshold * |H0| / N: the relative threshold per atom."""
    per_atom = max_dev / n_atoms
    bound = threshold * abs(H0) / n_atoms
    assert per_atom < bound, (
        f"{ensemble} {precision}: max |H - H0| = {per_atom * 1e3:.4f} meV/atom exceeds "
        f"{bound * 1e3:.4f} meV/atom (= {threshold:.1e} x |H0| = {abs(H0):.2f} eV over "
        f"{n_atoms} atoms; relative drift {max_dev / abs(H0) if H0 else float('inf'):.2e})"
    )


@pytest.mark.parametrize('ensemble', ALL_ENSEMBLES)
def test_integrator_drift_f64(ensemble, water_system):
    """All ensembles should have a drift below 1e-3 x |H0| in f64."""
    max_dev, H0, n_atoms = _run_ensemble(ensemble, jnp.float64, water_system(jnp.float64))
    _assert_drift_below(ensemble, 'f64', 1.0e-3, max_dev, H0, n_atoms)


@pytest.mark.parametrize('ensemble', ALL_ENSEMBLES)
def test_integrator_drift_f32(ensemble, water_system):
    """NPT flex has looser threshold in f32 due to matrix precision loss."""
    max_dev, H0, n_atoms = _run_ensemble(ensemble, jnp.float32, water_system(jnp.float32))
    threshold = 2.5e-2 if 'flex' in ensemble else 1.0e-3
    _assert_drift_below(ensemble, 'f32', threshold, max_dev, H0, n_atoms)
