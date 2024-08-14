import numpy as np
import numpy.testing as npt
import pathlib
import pytest
import jax
from ase.io import read
from mlff.md.calculator_sparse import mlffCalculatorSparse


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_molecules(name: str):
    jax.config.update('jax_enable_x64', True)

    # target_predictions = np.load(
    #     io.BytesIO(
    #         pkgutil.get_data(
    #             __name__, f'test_data/{name}_ase.npz')
    #     )
    # )

    # atoms = read(
    #     pkgutil.get_data(
    #         __name__, f'test_data/{name}.xyz')
    # )

    target_predictions = np.load(f'test_data/{name}_ase.npz')

    atoms = read(f'test_data/{name}.xyz')

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=pathlib.Path(__file__).parent.parent.resolve() / 'sup_gems' / 'sup_gems_params',
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
        energy
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(forces))


def test_water():
    jax.config.update('jax_enable_x64', True)

    # target_predictions = np.load(
    #     io.BytesIO(
    #         pkgutil.get_data(
    #             __name__, f'test_data/{name}_ase.npz')
    #     )
    # )

    # atoms = read(
    #     pkgutil.get_data(
    #         __name__, f'test_data/{name}.xyz')
    # )

    target_predictions = np.load(f'test_data/water_ase.npz')

    atoms = read(f'test_data/water_64.xyz') * [2, 2, 2]

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=pathlib.Path(__file__).parent.parent.resolve() / 'sup_gems' / 'sup_gems_params',
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
        energy
    )

    npt.assert_allclose(
        target_predictions['forces'],
        forces
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(forces))