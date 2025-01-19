import numpy as np
import pathlib
import pytest
import jax
import jax.numpy as jnp

from ase.io import read
from jax_md import space
from jax_md import quantity
from so3lr import to_jax_md
from so3lr import So3lrPotential
from so3lr import So3lrCalculator


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_ase_example(name: str):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    # atoms = Atoms(...)
    # This block mimics structure without pbc.
    atoms = read(package_dir / f'tests/test_data/{name}.xyz')
    atoms.cell = None
    atoms.pbc = False

    calc = So3lrCalculator(
        calculate_stress=False,
        dtype=np.float32
    )
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print('Energy and forces in ASE')
    print('Energy = ', energy)
    print('Forces = ', forces)


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_jax_md_example(name):

    # atoms = Atoms(...)
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    # atoms = Atoms(...)
    # This block mimics structure without pbc.
    atoms = read(package_dir / f'tests/test_data/{name}.xyz')
    atoms.cell = None
    atoms.pbc = False

    assert np.asarray(
        atoms.get_pbc()
    ).all().item() is False, "Readme example assumes no box. See `examples/` folder for simulations in box."

    positions = jnp.array(atoms.get_positions())
    atomic_numbers = jnp.array(atoms.get_atomic_numbers())

    # We assume there is no box.
    box = None
    displacement, shift = space.free()

    neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
        potential=So3lrPotential(),
        displacement_or_metric=displacement,
        box_size=box,
        species=atomic_numbers,
        capacity_multiplier=1.25,
        buffer_size_multiplier_sr=1.25,
        buffer_size_multiplier_lr=1.25,
        minimum_cell_size_multiplier_sr=1.0,
        disable_cell_list=True,
        fractional_coordinates=False
    )

    # Energy and force functions.
    energy_fn = jax.jit(energy_fn)
    force_fn = jax.jit(quantity.force(energy_fn))

    # Initialize the short and long-range neighbor lists.
    nbrs = neighbor_fn.allocate(
        positions,
        box=box
    )
    nbrs_lr = neighbor_fn_lr.allocate(
        positions,
        box=box
    )
    energy = energy_fn(
        positions,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )
    forces = force_fn(
        positions,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )
    print('Energy and forces in JAX-MD')
    print('Energy = ', np.array(energy))
    print('Forces = ', np.array(forces))
