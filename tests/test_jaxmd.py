import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pathlib
import pytest

from ase.io import read
from jax_md import space, quantity
from so3lr.mlff.calculators.potential import PotentialSparse
from so3lr import jaxmd_utils
from so3lr.model_registry import resolve_model

from model_references import TOLERANCES, V2_MODELS, reference_path

# Original/legacy SO3LR model, resolved via the registry so the test is
# decoupled from the on-disk layout (the model now lives under models/so3lr).
MODEL_DIR = resolve_model('so3lr').dir

lr_cutoff = 12.0
lr_cutoff_damp = 2.0

capacity_multiplier = 1.25
buffer_size_multiplier_sr = 1.25
buffer_size_multiplier_lr = 1.25
minimum_cell_size_multiplier_sr = 1.0


def test_water(x64):
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
    mlff_potential = PotentialSparse.create_from_workdir(
        MODEL_DIR,
        from_file=True,
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


@pytest.mark.parametrize('model', V2_MODELS)
def test_with_solar_potential_v2(model: str, x64):
    """jax-md energy/forces for bundled v2 models against the saved ASE references
    (tests/test_ase.py::test_v2_ase_calculator), across the two code paths."""
    from so3lr import So3lrPotential

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(reference_path(model))

    mol = read(
        package_dir / 'tests/test_data/dha.xyz'
    )

    species = jnp.array(mol.get_atomic_numbers())
    box = None
    R0 = jnp.array(mol.get_positions())

    displacement, shift = space.free()

    neighbor_fn, neighbor_fn_lr, energy_fn = jaxmd_utils.to_jax_md(
        potential=So3lrPotential(
            model=model,
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

    nbrs = neighbor_fn.allocate(R0, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(R0, box=box)

    # One compile for both: free space, so the gradient is already Cartesian.
    energy, grad = jax.jit(jax.value_and_grad(energy_fn))(
        R0, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
    forces = -grad

    npt.assert_allclose(energy, target_predictions['energy'], rtol=0, atol=TOLERANCES['energy'])
    npt.assert_allclose(forces, target_predictions['forces'], rtol=0, atol=TOLERANCES['forces'])

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))


@pytest.mark.parametrize('name', ['dha'])
def test_with_solar_potential(name: str, x64):
    """jax-md energy/forces for the legacy model in vacuum against saved references,
    through the public So3lrPotential entry point."""
    from so3lr import So3lrPotential

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
            model="so3lr",  # reference .npz generated with the legacy (v1) model
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
