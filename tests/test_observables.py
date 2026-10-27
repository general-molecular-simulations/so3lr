"""
Test extraction of different observables.

This test file validates that individual observables
(nn_energy, electrostatic_energy, dispersion_energy, partial_charges, etc.)
can be extracted from the SO3LR model and that the energy components
sum to the total energy. The same decomposition for a periodic water box is
checked in test_ase.py::test_water, alongside that box's reference values.

One molecule (dha) is enough: these are properties of the observable
plumbing, not of a particular molecule.
"""

import numpy as np
import numpy.testing as npt
import pathlib
import pytest
from ase.io import read
from so3lr.mlff.calculators.ase_calculator import AseCalculatorSparse
from so3lr.model_registry import resolve_model

# Original/legacy SO3LR model, resolved via the registry so the test is
# decoupled from the on-disk layout (the model now lives under models/so3lr).
MODEL_DIR = resolve_model('so3lr').dir


# List of available observables
OBSERVABLES = [
    'zbl_repulsion',
    'nn_energy',
    'electrostatic_energy',
    'dispersion_energy',
    'partial_charges',
    'hirshfeld_ratios',
]

# Additional observables that may be available depending on model configuration
OPTIONAL_OBSERVABLES = [
    'electrostatic_energy_kspace',
    'electrostatic_energy_undamped',
    'dispersion_energy_undamped',
]


def get_package_dir():
    """Return the package directory path."""
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture
def package_dir():
    """Fixture providing the package directory path."""
    return get_package_dir()


class TestObservablesExtraction:
    """Test class for observable extraction."""

    @pytest.mark.parametrize('name', ['dha'])
    def test_energy_components_sum_to_total_molecules(self, name: str, package_dir):
        """
        Test that extracted energy components sum approximately to total energy
        for molecular systems (non-periodic), and that the per-atom observables
        from the same evaluation have their basic properties: neutral partial
        charges, non-negative Hirshfeld ratios, attractive dispersion (damped and
        undamped), undamped electrostatics of the damped sign, and one embedding
        row per atom.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        # Create calculator with observables output
        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['zbl_repulsion', 'nn_energy', 'electrostatic_energy', 'dispersion_energy',
                         'dispersion_energy_undamped', 'electrostatic_energy_undamped',
                         'partial_charges', 'hirshfeld_ratios', 'x']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        # Get total energy
        total_energy = atoms.get_potential_energy()

        # Access the auxiliary outputs containing energy components
        results = calc.results.get('aux', {})

        # Check that energy components were extracted
        assert 'nn_energy' in results, "nn_energy not found in observables"
        assert 'electrostatic_energy' in results, "electrostatic_energy not found in observables"
        assert 'dispersion_energy' in results, "dispersion_energy not found in observables"
        assert 'zbl_repulsion' in results, "zbl_repulsion not found in observables"

        # Get individual energy components (these are per-atom, so sum them)
        nn_energy = np.sum(results['nn_energy'])
        electrostatic_energy = np.sum(results['electrostatic_energy'])
        dispersion_energy = np.sum(results['dispersion_energy'])
        zbl_repulsion = np.sum(results['zbl_repulsion'])
        # Sum of components
        sum_components = nn_energy + electrostatic_energy + dispersion_energy + zbl_repulsion

        # The sum of components should approximate the total energy
        # Note: There might be small numerical differences due to different
        # computational paths
        print(f"\nMolecule: {name}")
        print(f"Total energy: {total_energy:.6f} eV")
        print(f"NN energy: {nn_energy:.6f} eV")
        print(f"Electrostatic energy: {electrostatic_energy:.6f} eV")
        print(f"Dispersion energy: {dispersion_energy:.6f} eV")
        print(f"ZBL repulsion energy: {zbl_repulsion:.6f} eV")
        print(f"Sum of components: {sum_components:.6f} eV")

        # Allow for some numerical tolerance
        npt.assert_allclose(
            sum_components,
            total_energy,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Energy components don't sum to total for {name}"
        )

        # For neutral molecules, total charge should be close to 0
        npt.assert_allclose(
            np.sum(results['partial_charges']),
            0.0,
            atol=1e-5,
            err_msg=f"Partial charges don't sum to zero for neutral molecule {name}"
        )

        # Hirshfeld ratios are volume ratios: one per atom, non-negative
        hirshfeld_ratios = results['hirshfeld_ratios']
        assert hirshfeld_ratios.shape[0] == len(atoms), \
            f"Expected {len(atoms)} Hirshfeld ratios, got {hirshfeld_ratios.shape[0]}"
        assert np.all(hirshfeld_ratios >= 0), "Hirshfeld ratios should be non-negative"

        # Dispersion is attractive, damped or not
        dispersion_energy_undamped = np.sum(results['dispersion_energy_undamped'])
        assert dispersion_energy <= 0, \
            f"Dispersion energy should be negative (attractive), got {dispersion_energy}"
        assert dispersion_energy_undamped <= 0, \
            f"Undamped dispersion energy should be negative (attractive), got {dispersion_energy_undamped}"

        # Damping changes the magnitude of the electrostatic energy, not its sign
        electrostatic_energy_undamped = np.sum(results['electrostatic_energy_undamped'])
        assert np.sign(electrostatic_energy) == np.sign(electrostatic_energy_undamped) or \
               np.isclose(electrostatic_energy, 0, atol=1e-10), \
            "Damped and undamped electrostatic energies should have the same sign"

        # Atom embeddings: one feature row per atom
        x = results['x']
        assert x.ndim == 2, f"Expected 2D embeddings, got shape {x.shape}"
        assert x.shape[0] == len(atoms), (
            f"Expected {len(atoms)} atom embeddings, got {x.shape[0]}"
        )
        assert x.shape[1] > 0, "Feature dimension must be non-zero"


class TestConsistencyBetweenCalculators:
    """Test consistency between different calculator configurations."""

    @pytest.mark.parametrize('name', ['dha'])
    def test_energy_and_forces_same_with_and_without_observables(self, name: str, package_dir):
        """
        Test that requesting observables, the atom embeddings 'x' included, changes
        neither the total energy nor the forces.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        # Calculator without observables
        calc_no_obs = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64
        )

        # Calculator with observables
        calc_with_obs = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy', 'electrostatic_energy', 'dispersion_energy', 'x']
        )

        atoms_no_obs = atoms.copy()
        atoms_no_obs.calc = calc_no_obs
        atoms_no_obs.cell = None
        atoms_no_obs.pbc = False

        atoms_with_obs = atoms.copy()
        atoms_with_obs.calc = calc_with_obs
        atoms_with_obs.cell = None
        atoms_with_obs.pbc = False

        energy_no_obs = atoms_no_obs.get_potential_energy()
        energy_with_obs = atoms_with_obs.get_potential_energy()
        forces_no_obs = atoms_no_obs.get_forces()
        forces_with_obs = atoms_with_obs.get_forces()

        print(f"\nMolecule: {name}")
        print(f"Energy without observables: {energy_no_obs:.6f} eV")
        print(f"Energy with observables: {energy_with_obs:.6f} eV")
        print(f"Max force difference: {np.max(np.abs(forces_no_obs - forces_with_obs)):.2e}")

        npt.assert_allclose(
            energy_no_obs,
            energy_with_obs,
            rtol=1e-10,
            err_msg=f"Energy differs with observables extraction for {name}"
        )

        npt.assert_allclose(
            forces_no_obs,
            forces_with_obs,
            atol=1e-8,
            err_msg=f"Forces differ with observables extraction for {name}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
