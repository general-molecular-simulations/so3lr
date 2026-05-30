import numpy as np
import numpy.testing as npt
import pathlib
import pytest
from ase.io import read
from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse


@pytest.mark.parametrize('name', ['dha'])
def test_molecules(name: str):
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/{name}_ase.npz')

    atoms = read(package_dir / f'tests/test_data/{name}.xyz')

    calc = AseCalculatorSparse.create_from_workdir(
        workdir=package_dir / 'so3lr' / 'params',
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
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
        atol=1e-5,
        rtol=1e-6
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

    calc = AseCalculatorSparse.create_from_workdir(
        workdir=package_dir / 'so3lr' / 'params',
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        from_file=True,
        dtype=np.float64
    )

    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        rtol=1e-6,
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


@pytest.mark.parametrize('name', ['dha'])
def test_so3lr_ase_calculator(name: str):
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/{name}_ase.npz')

    atoms = read(package_dir / f'tests/test_data/{name}.xyz')

    # Reference predictions were generated with the legacy (v1) model, so pin to
    # it explicitly; the no-arg default is now so3lr-m (see test_default_model_*).
    calc = So3lrCalculator(
        model="so3lr",
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
        atol=2e-5,
        rtol=1e-6
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
        model="so3lr",
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


def test_so3lr_s_water():
    """Regression check for so3lr-s on a periodic water box (PBC / long-range)."""
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/water_so3lr_s_ase.npz')

    atoms = read(
        package_dir / f'tests/test_data/water_64.xyz'
    ) * [2, 2, 2]

    calc = So3lrCalculator(
        model="so3lr-s",
        dtype=np.float64
    )

    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        rtol=1e-6,
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


@pytest.mark.parametrize('name', ['dha'])
def test_default_model_is_so3lr_m(name: str):
    """The no-arg default must equal model='so3lr-m' and differ from legacy 'so3lr'."""
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()
    atoms = read(package_dir / f'tests/test_data/{name}.xyz')
    atoms.cell = None
    atoms.pbc = False

    def predict(model):
        a = atoms.copy()
        a.calc = So3lrCalculator(model=model, dtype=np.float64)
        return a.get_potential_energy(), a.get_forces()

    default_energy, default_forces = predict(None)        # no-arg default
    m_energy, m_forces = predict("so3lr-m")
    legacy_energy, _ = predict("so3lr")

    # Outputs must be finite.
    assert np.isfinite(default_energy)
    assert np.all(np.isfinite(default_forces))

    # The bare default must be exactly so3lr-m.
    npt.assert_allclose(default_energy, m_energy, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(default_forces, m_forces, rtol=1e-10, atol=1e-10)

    # so3lr-m must be a genuinely different model from the legacy v1 model.
    with npt.assert_raises(AssertionError):
        npt.assert_allclose(m_energy, legacy_energy, atol=1e-4)


@pytest.mark.parametrize('name', ['dha'])
def test_so3lr_s_ase_calculator(name: str):
    """Regression check for the bundled so3lr-s model against saved references."""
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/{name}_so3lr_s_ase.npz')

    atoms = read(package_dir / f'tests/test_data/{name}.xyz')

    calc = So3lrCalculator(
        model="so3lr-s",
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
        atol=2e-5,
        rtol=1e-6
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
