"""Test flexible cell NPT (anisotropic/semi-isotropic/isotropic coupling)."""
import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import pytest

from ase.io import read
from jax_md import space, simulate, units

from so3lr.mlff.calculators.potential import PotentialSparse
from so3lr.model_registry import resolve_model

jax.config.update('jax_enable_x64', True)

PACKAGE_DIR = pathlib.Path(__file__).parent.parent.resolve()
# 'sv2j_b128_l3nnn_2_nano_lr_full_2' was the dev name for the current v2 M lineage.
# Resolve via the registry so the test is runnable from a clean checkout/install.
_SPEC = resolve_model('so3lr-m')
MODEL_DIR, MODEL_FROM_FILE = _SPEC.dir, _SPEC.from_file


def _setup_system():
    """Load water_64 system and create energy function."""
    mol = read(PACKAGE_DIR / 'tests' / 'test_data' / 'water_64.xyz')
    cell = jnp.array(np.diag(mol.get_cell()))
    species = jnp.array(mol.get_atomic_numbers())
    masses = jnp.array(mol.get_masses())

    # Use fractional coordinates
    displacement, shift = space.periodic_general(cell, fractional_coordinates=True)
    positions = jnp.array(mol.get_positions()) / cell

    mlff_potential = PotentialSparse.create_from_workdir(
        MODEL_DIR,
        from_file=MODEL_FROM_FILE,
        long_range_kwargs=dict(
            cutoff_lr=10.0,
            dispersion_energy_cutoff_lr_damping=2.0,
            neighborlist_format_lr='ordered_sparse'
        ),
        dtype=jnp.float64
    )

    from jax_md import partition
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

    from so3lr.jaxmd_utils import neighbor_list_featurizer
    featurizer = neighbor_list_featurizer(
        displacement, species, total_charge=0., precision=jnp.float64,
        fractional_coordinates=True,
    )

    def energy_fn(R, neighbor, neighbor_lr, **kwargs):
        graph = featurizer(R, neighbor, neighbor_lr, **kwargs)
        return mlff_potential(graph).sum()

    nbrs = neighbor_fn.allocate(positions, box=cell)
    nbrs_lr = neighbor_fn_lr.allocate(positions, box=cell)

    return energy_fn, shift, positions, cell, masses, species, nbrs, nbrs_lr


@pytest.mark.parametrize('coupling', ['isotropic', 'semi-isotropic', 'anisotropic'])
def test_npt_flex_coupling_modes(coupling):
    """Test that NPT flex runs without error for each coupling mode."""
    energy_fn, shift, positions, cell, masses, species, nbrs, nbrs_lr = _setup_system()

    # Unit conversion (jax-md metal units)
    unit_system = units.metal_unit_system()
    dt = 0.5e-3 * unit_system['time']  # 0.5 fs in ps -> jax-md units
    kT = 300.0 * unit_system['temperature']
    P = 1.0 * unit_system['pressure']

    init_fn, apply_fn = simulate.npt_nose_hoover_flex(
        energy_fn,
        shift,
        dt=dt,
        pressure=P,
        kT=kT,
        coupling=coupling,
        barostat_kwargs={'tau': dt * 1000},
        thermostat_kwargs={'tau': dt * 100},
    )

    key = jax.random.PRNGKey(42)
    state = init_fn(
        key, positions, box=cell, mass=masses,
        neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, kT=kT,
    )

    # Verify state structure
    assert state.position.shape == positions.shape
    assert state.velocity.shape == positions.shape
    assert state.box_position.shape == (3, 3)
    assert state.box_velocity.shape == (3, 3)
    assert state.stress_tensor.shape == (3, 3)

    # Check momentum property works
    mom = state.momentum
    assert mom.shape == positions.shape
    np.testing.assert_allclose(mom, state.velocity * state.mass, atol=1e-12)

    # Run 2 steps
    for _ in range(2):
        state = apply_fn(
            state,
            neighbor=nbrs.idx,
            neighbor_lr=nbrs_lr.idx,
            kT=kT,
            pressure=P,
        )

    # Box should still be valid
    box_out = simulate.npt_box_flex(state)
    assert box_out.shape == (3, 3)
    vol = jnp.linalg.det(box_out)
    assert vol > 0, f"Volume should be positive, got {vol}"

    # Coupling constraints should be respected
    diag = jnp.diag(state.box_velocity)
    off_diag = state.box_velocity - jnp.diag(diag)

    if coupling == 'isotropic':
        # All diagonal elements should be equal
        np.testing.assert_allclose(diag[0], diag[1], atol=1e-10)
        np.testing.assert_allclose(diag[0], diag[2], atol=1e-10)
        # Off-diagonals should be zero
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)
    elif coupling == 'semi-isotropic':
        # xy should be equal, z independent
        np.testing.assert_allclose(diag[0], diag[1], atol=1e-10)
        # xy should differ from z
        assert not np.allclose(diag[0], diag[2], atol=1e-10)
        # Off-diagonals should be zero
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)
    elif coupling == 'anisotropic':
        # All three diagonal elements should differ
        assert not np.allclose(diag[0], diag[1], atol=1e-10)
        assert not np.allclose(diag[1], diag[2], atol=1e-10)
        assert not np.allclose(diag[0], diag[2], atol=1e-10)


def test_npt_flex_energy_conservation():
    """Test that the extended Hamiltonian is approximately conserved."""
    energy_fn, shift, positions, cell, masses, species, nbrs, nbrs_lr = _setup_system()

    unit_system = units.metal_unit_system()
    dt = 0.5e-3 * unit_system['time']
    kT = 300.0 * unit_system['temperature']
    P = 1.0 * unit_system['pressure']

    init_fn, apply_fn = simulate.npt_nose_hoover_flex(
        energy_fn,
        shift,
        dt=dt,
        pressure=P,
        kT=kT,
        coupling='anisotropic',
        barostat_kwargs={'tau': dt * 1000},
        thermostat_kwargs={'tau': dt * 100},
    )

    key = jax.random.PRNGKey(123)
    state = init_fn(
        key, positions, box=cell, mass=masses,
        neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, kT=kT,
    )

    # Compute initial invariant
    H0 = simulate.npt_nose_hoover_flex_invariant(
        energy_fn, state, pressure=P, kT=kT,
        neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx,
    )

    # Run 5 steps
    for _ in range(5):
        state = apply_fn(
            state,
            neighbor=nbrs.idx,
            neighbor_lr=nbrs_lr.idx,
            kT=kT,
            pressure=P,
        )

    H1 = simulate.npt_nose_hoover_flex_invariant(
        energy_fn, state, pressure=P, kT=kT,
        neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx,
    )

    # Extended Hamiltonian should be approximately conserved
    # (not exactly, due to finite dt and NHC thermostat)
    rel_drift = abs(float(H1 - H0)) / abs(float(H0))
    assert rel_drift < 0.05, (
        f"Extended Hamiltonian drifted by {rel_drift*100:.2f}%: "
        f"H0={float(H0):.6f}, H1={float(H1):.6f}"
    )
