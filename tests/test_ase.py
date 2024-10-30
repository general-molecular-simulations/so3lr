import numpy as np
import numpy.testing as npt
import pathlib
import pytest
from ase.io import read
from mlff.md.calculator_sparse import mlffCalculatorSparse


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_molecules(name: str):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/{name}_ase.npz')

    atoms = read(package_dir / f'tests/test_data/{name}.xyz')

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=package_dir / 'so3lr' / 'params',
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
        atol=1e-4
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))


def test_water():

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/water_ase.npz')

    atoms = read(
        package_dir / f'tests/test_data/water_64.xyz'
    ) * [2, 2, 2]

    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=package_dir / 'so3lr' / 'params',
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
        atol=5e-4
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['energy'], np.zeros_like(energy))

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(target_predictions['forces'], np.zeros_like(forces))


@pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
def test_so3lr_ase_calculator(name: str):
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/{name}_ase.npz')

    atoms = read(package_dir / f'tests/test_data/{name}.xyz')

    calc = So3lrCalculator(
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
        atol=1e-4
    )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(
            target_predictions['energy'],
            np.zeros_like(energy)
        )

    with npt.assert_raises(AssertionError):
        npt.assert_allclose(
            target_predictions['forces'],
            np.zeros_like(forces)
        )


@pytest.mark.parametrize('calculate_stress', [True, False])
def test_so3lr_ase_calculator_stress(calculate_stress: bool):
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    atoms = read(
        package_dir / f'tests/test_data/water_64.xyz'
    ) * [2, 2, 2]

    calc = So3lrCalculator(
        dtype=np.float64,
        calculate_stress=calculate_stress
    )

    atoms.calc = calc

    if calculate_stress is True:
        stress = atoms.get_stress()

        # Stress is in Voigt notation.
        npt.assert_equal(
            actual=stress.shape,
            desired=(6, )
        )
    else:
        with npt.assert_raises(NotImplementedError):
            atoms.get_stress()
