import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pathlib
import pytest

from ase.io import read
from jax_md import space, quantity
from mlff.mdx.potential import MLFFPotentialSparse
from so3lr import jaxmd_utils

lr_cutoff = 12.0
lr_cutoff_damp = 2.0

capacity_multiplier = 1.25
buffer_size_multiplier_sr = 1.25
buffer_size_multiplier_lr = 1.25
minimum_cell_size_multiplier_sr = 1.0


def test_jax_md_import():
    import jax_md


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_molecules_with_box(name: str):
    jax.config.update('jax_enable_x64', True)

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    # targets
    target_predictions = np.load(package_dir / f'tests/test_data/{name}_jaxmd.npz')

    # read molecule
    mol = read(
        package_dir / f'tests/test_data/{name}.xyz'
    )

    mol.set_cell([999.999, 999.999, 999.999])
    current_center = mol.get_center_of_mass()
    desired_center = mol.get_cell().sum(axis=0) / 2
    translation_vector = desired_center - current_center
    mol.translate(translation_vector)

    species = jnp.array(mol.get_atomic_numbers())
    atomic_masses = jnp.array(mol.get_masses())
    box = jnp.array([999.999, 999.999, 999.999])
    R0 = jnp.array(mol.get_positions()) / box[0]

    # define mlff potential
    mlff_potential = MLFFPotentialSparse.create_from_ckpt_dir(
        package_dir / 'so3lr' / 'params',
        from_file=True,
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,  # this controls whether elec and disp modules use force shifting and damping.
            dispersion_energy_cutoff_lr_damping=lr_cutoff_damp,
            neighborlist_format_lr='ordered_sparse'
        ),
        dtype=jnp.float64
    )

    displacement, shift = space.periodic_general(
        box,
        fractional_coordinates=True
    )

    neighbor_fn, neighbor_fn_lr, energy_fn = jaxmd_utils.to_jax_md(
        potential=mlff_potential,
        displacement_or_metric=displacement,
        box_size=box,
        species=species,
        capacity_multiplier=capacity_multiplier,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        minimum_cell_size_multiplier_sr=minimum_cell_size_multiplier_sr,
        disable_cell_list=True
    )

    energy_fn = jax.jit(energy_fn)
    force_fn = jax.jit(quantity.force(energy_fn))

    nbrs = neighbor_fn.allocate(R0, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(R0, box=box)

    energy = energy_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
    forces = force_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        atol=1e-6
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces,
        atol=1e-4
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_molecules_free(name: str):
    jax.config.update('jax_enable_x64', True)

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    # targets
    target_predictions = np.load(package_dir / f'tests/test_data/{name}_jaxmd.npz')

    # read molecule
    mol = read(
        package_dir / f'tests/test_data/{name}.xyz'
    )

    species = jnp.array(mol.get_atomic_numbers())
    box = None
    R0 = jnp.array(mol.get_positions())

    # define mlff potential
    mlff_potential = MLFFPotentialSparse.create_from_ckpt_dir(
        package_dir / 'so3lr' / 'params',
        from_file=True,
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,  # this controls whether elec and disp modules use force shifting and damping.
            dispersion_energy_cutoff_lr_damping=lr_cutoff_damp,
            neighborlist_format_lr='ordered_sparse'
        ),
        dtype=jnp.float64
    )

    displacement, shift = space.free()

    neighbor_fn, neighbor_fn_lr, energy_fn = jaxmd_utils.to_jax_md(
        potential=mlff_potential,
        displacement_or_metric=displacement,
        box_size=box,
        species=species,
        capacity_multiplier=capacity_multiplier,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        minimum_cell_size_multiplier_sr=minimum_cell_size_multiplier_sr,
        disable_cell_list=True
    )

    energy_fn = jax.jit(energy_fn)
    force_fn = jax.jit(quantity.force(energy_fn))

    nbrs = neighbor_fn.allocate(R0, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(R0, box=box)

    energy = energy_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
    forces = force_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        atol=1e-6
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces,
        atol=1e-4
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))


def test_water():
    jax.config.update('jax_enable_x64', True)

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    # targets
    target_predictions = np.load(package_dir / f'tests/test_data/water_jaxmd.npz')

    # read molecule
    mol = read(
        package_dir / f'tests/test_data/water_64.xyz'
    )
    mol.wrap()

    # Repeat the cell to match reference predictions.
    mol = mol.repeat([2, 2, 2])

    species = jnp.array(mol.get_atomic_numbers())
    atomic_masses = jnp.array(mol.get_masses())
    box = jnp.array(jnp.diag(mol.get_cell().T))
    R0 = jnp.array(mol.get_positions()) / box  # box can have different side lengths

    # define mlff potential
    mlff_potential = MLFFPotentialSparse.create_from_ckpt_dir(
        package_dir / 'so3lr' / 'params',
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,  # this controls whether elec and disp modules use force shifting and damping.
            dispersion_energy_cutoff_lr_damping=lr_cutoff_damp,
            neighborlist_format_lr='ordered_sparse'
        ),
        dtype=jnp.float64
        # determines the output dtype. When float64 is passed all calculations are performed in float64 and
        # final result is casted to dtype.
    )

    displacement, shift = space.periodic_general(
        box,
        fractional_coordinates=True
    )

    neighbor_fn, neighbor_fn_lr, energy_fn = jaxmd_utils.to_jax_md(
        potential=mlff_potential,
        displacement_or_metric=displacement,
        box_size=box,
        species=species,
        capacity_multiplier=capacity_multiplier,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        minimum_cell_size_multiplier_sr=minimum_cell_size_multiplier_sr
    )

    energy_fn = jax.jit(energy_fn)
    force_fn = jax.jit(quantity.force(energy_fn))

    nbrs = neighbor_fn.allocate(R0, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(R0, box=box)

    energy = energy_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
    forces = force_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        atol=1e-6,
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces,
        atol=5e-4
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_with_solar_potential(name: str):
    from so3lr import So3lrPotential

    jax.config.update('jax_enable_x64', True)

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    # targets
    target_predictions = np.load(package_dir / f'tests/test_data/{name}_jaxmd.npz')

    # read molecule
    mol = read(
        package_dir / f'tests/test_data/{name}.xyz'
    )

    species = jnp.array(mol.get_atomic_numbers())
    box = None
    R0 = jnp.array(mol.get_positions())

    displacement, shift = space.free()

    neighbor_fn, neighbor_fn_lr, energy_fn = jaxmd_utils.to_jax_md(
        potential=So3lrPotential(
            dtype=np.float64
        ),
        displacement_or_metric=displacement,
        box_size=box,
        species=species,
        capacity_multiplier=capacity_multiplier,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        minimum_cell_size_multiplier_sr=minimum_cell_size_multiplier_sr,
        disable_cell_list=True
    )

    energy_fn = jax.jit(energy_fn)
    force_fn = jax.jit(quantity.force(energy_fn))

    nbrs = neighbor_fn.allocate(R0, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(R0, box=box)

    energy = energy_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
    forces = force_fn(R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        atol=1e-6
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces,
        atol=1e-4
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))
