"""Test integrator drift across all ensembles (NVT, NPT iso, NPT flex).

Asserts that max_rel_drift stays below thresholds established from
measurement runs (scripts/measure_drift.py). The NPT flex thresholds
verify the integrator bug fix (fractional coordinate conversion) holds.

Thresholds have ~3x safety margin over measured values on water_64:
- f64: all ensembles ~3.0e-4 → threshold 1.0e-3
- f32: NVT/NPT iso ~3.0e-4 → threshold 1.0e-3
- f32: NPT flex ~5-8e-3 → threshold 2.5e-2
"""
import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import pytest

jax.config.update('jax_enable_x64', True)

from ase.io import read
from jax_md import space, simulate, units, partition

from so3lr.cli.so3lr_md import neighbor_list_featurizer_custom
from so3lr.mlff.calculators.potential import PotentialSparse

PACKAGE_DIR = pathlib.Path(__file__).parent.parent.resolve()
MODEL_DIR = PACKAGE_DIR / 'sv2j_b128_l3nnn_2_nano_lr_full_2'

N_STEPS = 100
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
        from_file=False,
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

    featurizer = neighbor_list_featurizer_custom(
        displacement, species, total_charge=0., precision=dtype,
        fractional_coordinates=True,
    )

    def energy_fn(R, neighbor, neighbor_lr, **kwargs):
        graph = featurizer(R, neighbor, neighbor_lr, **kwargs)
        return mlff_potential(graph).sum()

    nbrs = neighbor_fn.allocate(positions, box=cell)
    nbrs_lr = neighbor_fn_lr.allocate(positions, box=cell)

    return energy_fn, shift, positions, cell, masses, species, nbrs, nbrs_lr


def _measure_drift(init_fn, apply_fn, invariant_fn, init_args, init_kwargs,
                   step_kwargs, invariant_kwargs, n_steps):
    """Run simulation and return max_rel_drift."""
    key = jax.random.PRNGKey(42)
    state = init_fn(key, *init_args, **init_kwargs)

    H0 = float(invariant_fn(state, **invariant_kwargs))
    H_max_dev = 0.0

    for _ in range(n_steps):
        state = apply_fn(state, **step_kwargs)
        H_i = float(invariant_fn(state, **invariant_kwargs))
        dev = abs(H_i - H0)
        if dev > H_max_dev:
            H_max_dev = dev

    max_rel_drift = H_max_dev / abs(H0) if H0 != 0 else float('inf')
    return max_rel_drift


def _run_ensemble(ensemble, dtype):
    """Run a single ensemble and return max_rel_drift."""
    energy_fn, shift, positions, cell, masses, _, nbrs, nbrs_lr = _setup_system(dtype)

    unit_system = units.metal_unit_system()
    dt = DT_FS * unit_system['time']
    kT = 300.0 * unit_system['temperature']
    P = 1.0 * unit_system['pressure']
    thermostat_kwargs = {'tau': dt * 100}
    barostat_kwargs = {'tau': dt * 1000}

    nbrs_idx = nbrs.idx
    nbrs_lr_idx = nbrs_lr.idx

    if ensemble == 'NVT':
        init_fn, apply_fn = simulate.nvt_nose_hoover(
            energy_fn, shift, dt=dt, kT=kT, tau=dt * 100,
        )
        return _measure_drift(
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
        )

    elif ensemble == 'NPT_iso':
        init_fn, apply_fn = simulate.npt_nose_hoover(
            energy_fn, shift, dt=dt, pressure=P, kT=kT,
            barostat_kwargs=barostat_kwargs,
            thermostat_kwargs=thermostat_kwargs,
        )
        return _measure_drift(
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
        )

    else:
        # NPT flex variants
        coupling_map = {
            'NPT_flex_iso': 'isotropic',
            'NPT_flex_semi_iso': 'semi-isotropic',
            'NPT_flex_aniso': 'anisotropic',
        }
        coupling = coupling_map[ensemble]
        init_fn, apply_fn = simulate.npt_nose_hoover_flex(
            energy_fn, shift, dt=dt, pressure=P, kT=kT, coupling=coupling,
            barostat_kwargs=barostat_kwargs,
            thermostat_kwargs=thermostat_kwargs,
        )
        return _measure_drift(
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
        )


ALL_ENSEMBLES = ['NVT', 'NPT_iso', 'NPT_flex_iso', 'NPT_flex_semi_iso', 'NPT_flex_aniso']


@pytest.mark.parametrize('ensemble', ALL_ENSEMBLES)
def test_integrator_drift_f64(ensemble):
    """All ensembles should have max_rel_drift < 1e-3 in f64."""
    max_rel_drift = _run_ensemble(ensemble, jnp.float64)
    assert max_rel_drift < 1.0e-3, (
        f"{ensemble} f64: max_rel_drift={max_rel_drift:.2e} exceeds 1.0e-3"
    )


@pytest.mark.parametrize('ensemble', ALL_ENSEMBLES)
def test_integrator_drift_f32(ensemble):
    """NPT flex has looser threshold in f32 due to matrix precision loss."""
    max_rel_drift = _run_ensemble(ensemble, jnp.float32)
    threshold = 2.5e-2 if 'flex' in ensemble else 1.0e-3
    assert max_rel_drift < threshold, (
        f"{ensemble} f32: max_rel_drift={max_rel_drift:.2e} exceeds {threshold:.1e}"
    )
