import numpy as np
import numpy.testing as npt
import pathlib
import pytest
import jax
from ase.io import read
from mlff.md.calculator_sparse import mlffCalculatorSparse


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_molecules(name: str):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_disable_jit', True)

    target_predictions = np.load(package_dir / f'tests/test_data/{name}_ase.npz')

    atoms = read(package_dir / f'tests/test_data/{name}.xyz')

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=package_dir / 'sup_gems' / 'sup_gems_params',
        lr_cutoff=12.,
        dispersion_energy_lr_cutoff_damping=2.,
        from_file=True,
        dtype=np.float64
    )

    atoms.calc = calc

    # for ASE we remove the cell
    atoms.cell = None
    atoms.pbc = False

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        atol=1e-6
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces,
        atol=1e-6
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))


def test_water():
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_disable_jit', True)

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/water_ase.npz')

    atoms = read(
        package_dir / f'tests/test_data/water_64.xyz'
    ) * [2, 2, 2]

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=package_dir / 'sup_gems' / 'sup_gems_params',
        lr_cutoff=12.,
        dispersion_energy_lr_cutoff_damping=2.,
        from_file=True,
        dtype=np.float64
    )

    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        atol=1e-6,
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces,
        atol=1e-6
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))