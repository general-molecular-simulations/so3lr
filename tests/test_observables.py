"""
Test extraction of different observables.

This test file validates that individual observables
(nn_energy, electrostatic_energy, dispersion_energy, partial_charges, etc.)
can be extracted from the SO3LR model and that the energy components
sum to the total energy.
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
MODEL_DIR = resolve_model('so3lr')[0]


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

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_energy_components_sum_to_total_molecules(self, name: str, package_dir):
        """
        Test that extracted energy components sum approximately to total energy
        for molecular systems (non-periodic).
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
            observables=['zbl_repulsion','nn_energy', 'electrostatic_energy', 'dispersion_energy']
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

    def test_energy_components_sum_to_total_water_box(self, package_dir):
        """
        Test that extracted energy components sum approximately to total energy
        for periodic water system.
        """
        # Repeat water box to ensure it's large enough for the 12 Å cutoff
        atoms = read(package_dir / f'tests/test_data/water_64.xyz') * [2, 2, 2]

        # Create calculator with observables output
        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['zbl_repulsion','nn_energy', 'electrostatic_energy', 'dispersion_energy']
        )

        atoms.calc = calc

        # Get total energy
        total_energy = atoms.get_potential_energy()

        # Access the auxiliary outputs
        results = calc.results.get('aux', {})

        # Get individual energy components
        nn_energy = np.sum(results['nn_energy'])
        electrostatic_energy = np.sum(results['electrostatic_energy'])
        dispersion_energy = np.sum(results['dispersion_energy'])
        zbl_repulsion = np.sum(results['zbl_repulsion'])

        sum_components = nn_energy + electrostatic_energy + dispersion_energy + zbl_repulsion

        print(f"\nWater box (64 molecules, repeated 2x2x2)")
        print(f"Total energy: {total_energy:.6f} eV")
        print(f"NN energy: {nn_energy:.6f} eV")
        print(f"Electrostatic energy: {electrostatic_energy:.6f} eV")
        print(f"Dispersion energy: {dispersion_energy:.6f} eV")
        print(f"ZBL repulsion energy: {zbl_repulsion:.6f} eV")
        print(f"Sum of components: {sum_components:.6f} eV")
        
        npt.assert_allclose(
            sum_components,
            total_energy,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Energy components don't sum to total for water box"
        )


class TestPartialChargesExtraction:
    """Test extraction and properties of partial charges."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_partial_charges_extraction(self, name: str, package_dir):
        """Test that partial charges can be extracted."""
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['partial_charges']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        # Trigger calculation
        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'partial_charges' in results, "partial_charges not found in observables"

        partial_charges = results['partial_charges']

        # Check shape - should have one charge per atom
        assert partial_charges.shape[0] == len(atoms), \
            f"Expected {len(atoms)} partial charges, got {partial_charges.shape[0]}"

        print(f"\nMolecule: {name}")
        print(f"Number of atoms: {len(atoms)}")
        print(f"Partial charges shape: {partial_charges.shape}")
        print(f"Total charge (sum): {np.sum(partial_charges):.6f}")
        print(f"Min charge: {np.min(partial_charges):.6f}")
        print(f"Max charge: {np.max(partial_charges):.6f}")

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_partial_charges_sum_to_zero_neutral(self, name: str, package_dir):
        """
        Test that partial charges sum approximately to zero for neutral molecules.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['partial_charges']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})
        partial_charges = results['partial_charges']
        total_charge = np.sum(partial_charges)

        print(f"\nMolecule: {name}")
        print(f"Total charge: {total_charge:.8f}")

        # For neutral molecules, total charge should be close to 0
        npt.assert_allclose(
            total_charge,
            0.0,
            atol=1e-5,
            err_msg=f"Partial charges don't sum to zero for neutral molecule {name}"
        )


class TestHirshfeldRatiosExtraction:
    """Test extraction and properties of Hirshfeld ratios."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_hirshfeld_ratios_extraction(self, name: str, package_dir):
        """Test that Hirshfeld ratios can be extracted and are positive."""
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['hirshfeld_ratios']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'hirshfeld_ratios' in results, "hirshfeld_ratios not found in observables"

        hirshfeld_ratios = results['hirshfeld_ratios']

        # Check shape
        assert hirshfeld_ratios.shape[0] == len(atoms), \
            f"Expected {len(atoms)} Hirshfeld ratios, got {hirshfeld_ratios.shape[0]}"

        # Hirshfeld ratios should be positive (they represent volume ratios)
        assert np.all(hirshfeld_ratios >= 0), \
            "Hirshfeld ratios should be non-negative"

        print(f"\nMolecule: {name}")
        print(f"Hirshfeld ratios shape: {hirshfeld_ratios.shape}")
        print(f"Min ratio: {np.min(hirshfeld_ratios):.6f}")
        print(f"Max ratio: {np.max(hirshfeld_ratios):.6f}")
        print(f"Mean ratio: {np.mean(hirshfeld_ratios):.6f}")


class TestUndampedObservables:
    """Test extraction of undamped energy observables."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_electrostatic_energy_undamped_extraction(self, name: str, package_dir):
        """
        Test that electrostatic_energy_undamped can be extracted.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['electrostatic_energy', 'electrostatic_energy_undamped']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'electrostatic_energy_undamped' in results, \
            "electrostatic_energy_undamped not found in observables"

        electrostatic_energy = np.sum(results['electrostatic_energy'])
        electrostatic_energy_undamped = np.sum(results['electrostatic_energy_undamped'])

        print(f"\nMolecule: {name}")
        print(f"Electrostatic energy (damped): {electrostatic_energy:.6f} eV")
        print(f"Electrostatic energy (undamped): {electrostatic_energy_undamped:.6f} eV")

        # Undamped should have larger magnitude (damping reduces the energy)
        # Both should have the same sign
        assert np.sign(electrostatic_energy) == np.sign(electrostatic_energy_undamped) or \
               np.isclose(electrostatic_energy, 0, atol=1e-10), \
            "Damped and undamped electrostatic energies should have the same sign"

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_dispersion_energy_undamped_extraction(self, name: str, package_dir):
        """
        Test that dispersion_energy_undamped can be extracted.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['dispersion_energy', 'dispersion_energy_undamped']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'dispersion_energy_undamped' in results, \
            "dispersion_energy_undamped not found in observables"

        dispersion_energy = np.sum(results['dispersion_energy'])
        dispersion_energy_undamped = np.sum(results['dispersion_energy_undamped'])

        print(f"\nMolecule: {name}")
        print(f"Dispersion energy (damped): {dispersion_energy:.6f} eV")
        print(f"Dispersion energy (undamped): {dispersion_energy_undamped:.6f} eV")

        # Both should be negative (attractive)
        assert dispersion_energy <= 0, \
            f"Dispersion energy should be negative (attractive), got {dispersion_energy}"
        assert dispersion_energy_undamped <= 0, \
            f"Undamped dispersion energy should be negative (attractive), got {dispersion_energy_undamped}"


class TestMultipleObservablesSimultaneous:
    """Test extracting multiple observables in a single calculation."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_all_observables_extraction(self, name: str, package_dir):
        """
        Test that all major observables can be extracted simultaneously.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        all_observables = [
            'nn_energy',
            'electrostatic_energy',
            'dispersion_energy',
            'partial_charges',
            'hirshfeld_ratios',
        ]

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=all_observables
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        print(f"\nMolecule: {name}")
        print(f"Available observables: {list(results.keys())}")

        for obs in all_observables:
            assert obs in results, f"{obs} not found in observables"
            print(f"  {obs}: shape={results[obs].shape}")


class TestConsistencyBetweenCalculators:
    """Test consistency between different calculator configurations."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_energy_same_with_and_without_observables(self, name: str, package_dir):
        """
        Test that total energy is the same whether or not observables are extracted.
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
            observables=['nn_energy', 'electrostatic_energy', 'dispersion_energy']
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

        print(f"\nMolecule: {name}")
        print(f"Energy without observables: {energy_no_obs:.6f} eV")
        print(f"Energy with observables: {energy_with_obs:.6f} eV")

        npt.assert_allclose(
            energy_no_obs,
            energy_with_obs,
            rtol=1e-10,
            err_msg=f"Energy differs with observables extraction for {name}"
        )


class TestForceConsistencyWithObservables:
    """Test that forces are consistent when extracting observables."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_forces_same_with_observables(self, name: str, package_dir):
        """
        Test that forces are the same whether or not observables are extracted.
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
            observables=['nn_energy', 'electrostatic_energy', 'dispersion_energy']
        )

        atoms_no_obs = atoms.copy()
        atoms_no_obs.calc = calc_no_obs
        atoms_no_obs.cell = None
        atoms_no_obs.pbc = False

        atoms_with_obs = atoms.copy()
        atoms_with_obs.calc = calc_with_obs
        atoms_with_obs.cell = None
        atoms_with_obs.pbc = False

        forces_no_obs = atoms_no_obs.get_forces()
        forces_with_obs = atoms_with_obs.get_forces()

        print(f"\nMolecule: {name}")
        print(f"Max force difference: {np.max(np.abs(forces_no_obs - forces_with_obs)):.2e}")

        npt.assert_allclose(
            forces_no_obs,
            forces_with_obs,
            atol=1e-8,
            err_msg=f"Forces differ with observables extraction for {name}"
        )


class TestEmbeddingsExtraction:
    """Test extraction and properties of atom embeddings ('x')."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_embeddings_shape(self, name: str, package_dir):
        """Test that 'x' embeddings have the correct per-atom shape."""
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['x']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'x' in results, "'x' embeddings not found in observables"

        x = results['x']
        assert x.ndim == 2, f"Expected 2D embeddings, got shape {x.shape}"
        assert x.shape[0] == len(atoms), (
            f"Expected {len(atoms)} atom embeddings, got {x.shape[0]}"
        )
        assert x.shape[1] > 0, "Feature dimension must be non-zero"

        print(f"\nMolecule: {name}")
        print(f"Number of atoms: {len(atoms)}")
        print(f"Embeddings shape: {x.shape}  (n_atoms, features)")

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_embeddings_consistent_across_runs(self, name: str, package_dir):
        """Test that 'x' embeddings are identical across two identical forward passes."""
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['x']
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()
        x1 = calc.results['aux']['x'].copy()

        # Reset and run again
        atoms.calc.reset()
        _ = atoms.get_potential_energy()
        x2 = calc.results['aux']['x'].copy()

        npt.assert_array_equal(x1, x2,
            err_msg=f"Embeddings differ between two identical runs for {name}")

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_energy_unchanged_when_requesting_embeddings(self, name: str, package_dir):
        """Test that requesting 'x' does not change the computed energy."""
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        calc_no_x = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
        )

        calc_with_x = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['x']
        )

        atoms_a = atoms.copy()
        atoms_a.calc = calc_no_x
        atoms_a.cell = None
        atoms_a.pbc = False

        atoms_b = atoms.copy()
        atoms_b.calc = calc_with_x
        atoms_b.cell = None
        atoms_b.pbc = False

        energy_no_x = atoms_a.get_potential_energy()
        energy_with_x = atoms_b.get_potential_energy()

        print(f"\nMolecule: {name}")
        print(f"Energy without 'x': {energy_no_x:.8f} eV")
        print(f"Energy with 'x':    {energy_with_x:.8f} eV")

        npt.assert_allclose(
            energy_no_x, energy_with_x, rtol=1e-10,
            err_msg=f"Energy changed when requesting embeddings for {name}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
