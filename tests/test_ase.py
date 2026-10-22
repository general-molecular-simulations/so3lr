import numpy as np
import numpy.testing as npt
import pathlib
import pytest
from ase.io import read
from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse
from so3lr.model_registry import resolve_model

from model_references import TOLERANCES, V2_MODELS, reference_path

# Original/legacy SO3LR model, resolved via the registry so the test is
# decoupled from the on-disk layout (the model now lives under models/so3lr).
MODEL_DIR = resolve_model('so3lr').dir


def test_bounding_cell_does_not_change_free_molecule():
    """A pbc=False molecule with a tight bounding cell is still free."""
    package_dir = pathlib.Path(__file__).parent.parent.resolve()
    atoms = read(package_dir / 'tests/test_data/dha.xyz')
    atoms.cell = None
    atoms.pbc = False
    atoms.calc = AseCalculatorSparse.create_from_workdir(
        workdir=MODEL_DIR, lr_cutoff=12., dispersion_energy_cutoff_lr_damping=2.,
        from_file=True, dtype=np.float64,
    )
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()

    boxed = atoms.copy()
    boxed.center(vacuum=1.0)
    boxed.pbc = False
    boxed.calc = atoms.calc

    # Tolerances cover float32 when x64 is off; the minimum-image error this
    # guards against is ~1e-2 eV and ~1e-2 eV/A.
    npt.assert_allclose(boxed.get_potential_energy(), energy, atol=1e-4)
    npt.assert_allclose(boxed.get_forces(), forces, atol=1e-3)


def test_water():
    """Periodic water box against saved references, and its energy decomposition.

    The components are read from the same evaluation (requesting observables
    does not change the energy, see test_observables.py), so the 1536-atom box
    is evaluated once for both checks.
    """
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/water_ase.npz')

    atoms = read(
        package_dir / f'tests/test_data/water_64.xyz'
    ) * [2, 2, 2]

    calc = AseCalculatorSparse.create_from_workdir(
        workdir=MODEL_DIR,
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        from_file=True,
        dtype=np.float64,
        has_aux=True,
        observables=['zbl_repulsion', 'nn_energy', 'electrostatic_energy', 'dispersion_energy']
    )

    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # The reference is float64; without x64 the run is float32 (~1e-6 relative),
    # so the tolerance must not depend on whether an earlier test enabled x64.
    npt.assert_allclose(
        target_predictions['energy'],
        energy,
        rtol=5e-6,
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

    # The energy components sum to the total for the periodic box too.
    results = calc.results['aux']
    sum_components = sum(
        np.sum(results[key])
        for key in ('nn_energy', 'electrostatic_energy', 'dispersion_energy', 'zbl_repulsion')
    )
    npt.assert_allclose(
        sum_components,
        energy,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Energy components don't sum to total for water box"
    )


@pytest.mark.parametrize('name', ['dha'])
def test_so3lr_ase_calculator(name: str):
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/{name}_ase.npz')

    atoms = read(package_dir / f'tests/test_data/{name}.xyz')

    # Reference predictions were generated with the legacy (v1) model, so pin to
    # it explicitly; the no-arg default is so3lr-tauri-m-2606 (see
    # tests/test_model_registry.py::test_defaults).
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
    """Stress only when asked for. A contract test, so the 192-atom cell suffices:
    below twice the cutoff the calculator uses explicit images, and stress values
    are checked against finite differences in test_ase_small_cell.py."""
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    atoms = read(
        package_dir / f'tests/test_data/water_64.xyz'
    )

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
    """Regression check for so3lr-tauri-s-2602 on a periodic water box.

    Pins the exact release the saved references were generated from -- a
    floating alias here would break whenever a new release is registered.
    """
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_predictions = np.load(package_dir / f'tests/test_data/water_so3lr_s_ase.npz')

    atoms = read(
        package_dir / f'tests/test_data/water_64.xyz'
    ) * [2, 2, 2]

    calc = So3lrCalculator(
        model="so3lr-tauri-s-2602",
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


def test_shipped_default_has_a_reference():
    """Changing DEFAULT_MODEL must come with a saved reference for the new default."""
    from so3lr.model_registry import DEFAULT_MODEL
    assert DEFAULT_MODEL in V2_MODELS


@pytest.mark.parametrize('model', V2_MODELS)
def test_v2_ase_calculator(model: str, x64):
    """Regression check for bundled v2 models against saved references
    (tests/model_references.py). Needs x64: the tight bounds hold in float64 only."""
    from so3lr import So3lrCalculator

    package_dir = pathlib.Path(__file__).parent.parent.resolve()
    target = np.load(reference_path(model))

    atoms = read(package_dir / 'tests/test_data/dha.xyz')
    atoms.cell = None
    atoms.pbc = False
    # Energy and forces from the plain calculator users run; c6_ratios, a head of
    # the v2 models, from a second one that returns it as an observable.
    atoms.calc = So3lrCalculator(model=model, dtype=np.float64)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms_obs = atoms.copy()
    atoms_obs.calc = So3lrCalculator(model=model, dtype=np.float64, has_aux=True,
                                     observables=['c6_ratios'])
    atoms_obs.get_potential_energy()
    c6_ratios = np.asarray(atoms_obs.calc.results['aux']['c6_ratios'])

    for key, value in (('energy', energy), ('forces', forces), ('c6_ratios', c6_ratios)):
        npt.assert_allclose(value, target[key], rtol=0, atol=TOLERANCES[key],
                            err_msg=f'{model} {key}')
    # The references are not degenerate.
    assert abs(target['energy']) > 1.0 and np.abs(target['forces']).max() > 0.1
    assert np.abs(target['c6_ratios']).max() > 0.1


def test_calculate_hessian_is_forwarded_by_make_ase_calculator():
    """make_ase_calculator accepted calculate_hessian but never forwarded it, so
    the public So3lrCalculator(calculate_hessian=True) silently produced no
    hessian. The existing hessian tests call AseCalculatorSparse.create_from_workdir
    directly, which is why this path was never covered.
    """
    from ase import Atoms
    from so3lr.ase_utils import make_ase_calculator

    water = Atoms('OHH',
                  positions=[[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.]],
                  pbc=False)
    water.calc = make_ase_calculator(calculate_hessian=True, dtype=np.float64)
    _ = water.get_potential_energy()

    assert 'hessian' in water.calc.results, (
        'calculate_hessian=True produced no hessian'
    )
    assert water.calc.results['hessian'].shape == (3, 3, 3, 3)
