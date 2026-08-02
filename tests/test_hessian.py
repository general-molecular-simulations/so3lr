"""
Test Hessian calculation and validation against numerical Hessians.

This test file validates that Hessians can be extracted from the SO3LR model
using the calculate_hessian option, and that they match numerical finite
difference calculations. It also tests subsystem Hessians using output_atom_indices.
"""

import jax
jax.config.update("jax_enable_x64", True)

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


def get_package_dir():
    """Return the package directory path."""
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture
def package_dir():
    """Fixture providing the package directory path."""
    return get_package_dir()


class TestHessianShape:
    """Test that Hessian has the correct shape."""

    @pytest.mark.parametrize('name', ['acetone'])  # Only test on smallest molecule for speed
    def test_full_hessian_shape(self, name: str, package_dir):
        """
        Test that full Hessian has shape (n_atoms, 3, n_atoms, 3).
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        assert 'hessian' in calc.results, "Hessian not found in results"

        hessian = calc.results['hessian']

        print(f"\nMolecule: {name}")
        print(f"Number of atoms: {n_atoms}")
        print(f"Hessian shape: {hessian.shape}")

        # Full Hessian should have shape (n_atoms, 3, n_atoms, 3)
        expected_shape = (n_atoms, 3, n_atoms, 3)
        assert hessian.shape == expected_shape, \
            f"Expected Hessian shape {expected_shape}, got {hessian.shape}"

    @pytest.mark.parametrize('name', ['acetone'])  # Only test on smallest molecule for speed
    def test_subsystem_hessian_shape(self, name: str, package_dir):
        """
        Test that subsystem Hessian has shape (n_subset, 3, n_subset, 3).
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select a contiguous subset of atoms
        n_subset = min(5, n_atoms)
        output_atom_indices = list(range(n_subset))

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True,
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        assert 'hessian' in calc.results, "Hessian not found in results"

        hessian = calc.results['hessian']

        print(f"\nMolecule: {name}")
        print(f"Number of atoms: {n_atoms}")
        print(f"Subset size: {n_subset}")
        print(f"Hessian shape: {hessian.shape}")

        # Subsystem Hessian has shape (n_subset, 3, n_subset, 3)
        expected_shape = (n_subset, 3, n_subset, 3)
        assert hessian.shape == expected_shape, \
            f"Expected Hessian shape {expected_shape}, got {hessian.shape}"


class TestHessianSymmetry:
    """Test that Hessian has expected symmetry properties."""

    @pytest.mark.parametrize('name', ['acetone'])  # Only test on smallest molecule for speed
    def test_hessian_symmetry(self, name: str, package_dir):
        """
        Test that full Hessian is symmetric: H[i,a,j,b] = H[j,b,i,a].
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        hessian = calc.results['hessian']

        # Reshape to (n_atoms*3, n_atoms*3) for easier symmetry check
        hessian_2d = hessian.reshape(n_atoms * 3, n_atoms * 3)

        print(f"\nMolecule: {name}")
        print(f"Max asymmetry: {np.max(np.abs(hessian_2d - hessian_2d.T)):.2e}")

        npt.assert_allclose(
            hessian_2d,
            hessian_2d.T,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Hessian is not symmetric for {name}"
        )


class TestHessianNumerical:
    """Test Hessian against numerical (finite difference) Hessian."""

    def _compute_numerical_hessian(self, atoms, calc, n_displace=None, eps=1e-4):
        """
        Compute numerical Hessian using finite differences on forces.

        The Hessian is: H[i,a,j,b] = d^2E / (dR_i^a dR_j^b) = -dF_j^b / dR_i^a

        Args:
            atoms: ASE Atoms object
            calc: Calculator instance (reused for all displacements)
            n_displace: Number of atoms to displace. If None, displace all atoms.
            eps: Finite difference step size

        Returns:
            Numerical Hessian as numpy array of shape (n_displace, 3, n_atoms, 3)
        """
        n_atoms = len(atoms)
        if n_displace is None:
            n_displace = n_atoms
        numerical_hessian = np.zeros((n_displace, 3, n_atoms, 3))
        original_positions = atoms.get_positions().copy()

        for i in range(n_displace):
            for a in range(3):
                # Positive displacement
                positions_plus = original_positions.copy()
                positions_plus[i, a] += eps
                atoms.set_positions(positions_plus)
                atoms.calc = calc
                forces_plus = atoms.get_forces().copy()

                # Negative displacement
                positions_minus = original_positions.copy()
                positions_minus[i, a] -= eps
                atoms.set_positions(positions_minus)
                forces_minus = atoms.get_forces().copy()

                # Central difference: H[i,a,j,b] = -dF[j,b]/dR[i,a]
                numerical_hessian[i, a, :, :] = -(forces_plus - forces_minus) / (2 * eps)

        # Restore original positions
        atoms.set_positions(original_positions)
        return numerical_hessian

    @pytest.mark.parametrize('name', ['acetone'])
    def test_full_hessian_numerical(self, name: str, package_dir, full_test):
        """
        Test full Hessian against numerical Hessian.
        Without --full, only displaces first atom for a quick sanity check.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)
        atoms.cell = None
        atoms.pbc = False

        # Calculator for analytical Hessian
        calc_hessian = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True
        )

        # Calculator for numerical Hessian (reused)
        calc_numerical = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64
        )

        # Get analytical Hessian
        atoms.calc = calc_hessian
        _ = atoms.get_potential_energy()
        analytical_hessian = calc_hessian.results['hessian']

        # In short mode, only displace first atom; in full mode, displace all
        n_displace = n_atoms if full_test else 1

        # Compute numerical Hessian (reuses calc_numerical)
        numerical_hessian = self._compute_numerical_hessian(
            atoms, calc_numerical, n_displace=n_displace, eps=1e-4
        )

        print(f"\nMolecule: {name} ({'full' if full_test else 'short'} mode)")
        print(f"Displacing {n_displace}/{n_atoms} atoms")
        print(f"Analytical Hessian shape: {analytical_hessian[:n_displace].shape}")
        print(f"Numerical Hessian shape: {numerical_hessian.shape}")
        print(f"Max absolute difference: {np.max(np.abs(analytical_hessian[:n_displace] - numerical_hessian)):.2e}")
        print(f"Mean absolute difference: {np.mean(np.abs(analytical_hessian[:n_displace] - numerical_hessian)):.2e}")

        npt.assert_allclose(
            analytical_hessian[:n_displace],
            numerical_hessian,
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Hessian mismatch for {name}"
        )


class TestSubsystemHessianNumerical:
    """Test subsystem Hessian against numerical calculations."""

    def _compute_numerical_subsystem_hessian(self, atoms, calc, atom_indices,
                                              n_displace=None, eps=1e-4):
        """
        Compute numerical Hessian for a subset of atoms.

        Args:
            atoms: ASE Atoms object
            calc: Calculator instance (reused for all displacements)
            atom_indices: List of atom indices for which to compute Hessian rows
            n_displace: Number of subset atoms to displace. If None, displace all.
            eps: Finite difference step size

        Returns:
            Numerical Hessian as numpy array of shape (n_displace, 3, n_subset, 3)
        """
        n_subset = len(atom_indices)
        if n_displace is None:
            n_displace = n_subset
        numerical_hessian = np.zeros((n_displace, 3, n_subset, 3))
        original_positions = atoms.get_positions().copy()

        for idx, i in enumerate(atom_indices[:n_displace]):
            for a in range(3):
                # Positive displacement
                positions_plus = original_positions.copy()
                positions_plus[i, a] += eps
                atoms.set_positions(positions_plus)
                atoms.calc = calc
                forces_plus = atoms.get_forces()[atom_indices].copy()

                # Negative displacement
                positions_minus = original_positions.copy()
                positions_minus[i, a] -= eps
                atoms.set_positions(positions_minus)
                forces_minus = atoms.get_forces()[atom_indices].copy()

                # Central difference
                numerical_hessian[idx, a, :, :] = -(forces_plus - forces_minus) / (2 * eps)

        # Restore original positions
        atoms.set_positions(original_positions)
        return numerical_hessian

    @pytest.mark.parametrize('name', ['acetone'])
    def test_subsystem_hessian_numerical(self, name: str, package_dir, full_test):
        """
        Test subsystem Hessian against numerical Hessian.
        Without --full, only displaces first atom for a quick sanity check.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)
        atoms.cell = None
        atoms.pbc = False

        # Select first 5 atoms (or all if fewer)
        n_subset = min(5, n_atoms)
        output_atom_indices = list(range(n_subset))

        # Calculator for analytical Hessian
        calc_hessian = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True,
            output_atom_indices=output_atom_indices
        )

        # Calculator for numerical Hessian (reused)
        calc_numerical = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64
        )

        # Get analytical Hessian
        atoms.calc = calc_hessian
        _ = atoms.get_potential_energy()
        analytical_hessian = calc_hessian.results['hessian']

        # In short mode, only displace first atom; in full mode, displace all
        n_displace = n_subset if full_test else 1

        # Compute numerical Hessian (reuses calc_numerical)
        numerical_hessian = self._compute_numerical_subsystem_hessian(
            atoms, calc_numerical, output_atom_indices,
            n_displace=n_displace, eps=1e-4
        )

        print(f"\nMolecule: {name} ({'full' if full_test else 'short'} mode)")
        print(f"Displacing {n_displace}/{n_subset} subset atoms")
        print(f"Analytical Hessian shape: {analytical_hessian[:n_displace].shape}")
        print(f"Numerical Hessian shape: {numerical_hessian.shape}")
        print(f"Max absolute difference: {np.max(np.abs(analytical_hessian[:n_displace] - numerical_hessian)):.2e}")
        print(f"Mean absolute difference: {np.mean(np.abs(analytical_hessian[:n_displace] - numerical_hessian)):.2e}")

        npt.assert_allclose(
            analytical_hessian[:n_displace],
            numerical_hessian,
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Subsystem Hessian mismatch for {name}"
        )

    @pytest.mark.parametrize('name', ['acetone'])
    def test_subsystem_matches_full_hessian(self, name: str, package_dir):
        """
        Test that subsystem Hessian matches the corresponding block of full Hessian.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select first 5 atoms
        n_subset = min(5, n_atoms)
        output_atom_indices = list(range(n_subset))

        # Get full Hessian
        calc_full = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True
        )

        atoms_full = atoms.copy()
        atoms_full.calc = calc_full
        atoms_full.cell = None
        atoms_full.pbc = False
        _ = atoms_full.get_potential_energy()
        full_hessian = calc_full.results['hessian']

        # Get subsystem Hessian
        calc_sub = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True,
            output_atom_indices=output_atom_indices
        )

        atoms_sub = atoms.copy()
        atoms_sub.calc = calc_sub
        atoms_sub.cell = None
        atoms_sub.pbc = False
        _ = atoms_sub.get_potential_energy()
        sub_hessian = calc_sub.results['hessian']

        # Extract corresponding block from full Hessian (subset x subset)
        start_idx = output_atom_indices[0]
        end_idx = output_atom_indices[-1] + 1
        full_hessian_block = full_hessian[start_idx:end_idx, :, start_idx:end_idx, :]

        print(f"\nMolecule: {name}")
        print(f"Full Hessian shape: {full_hessian.shape}")
        print(f"Subsystem Hessian shape: {sub_hessian.shape}")
        print(f"Full Hessian block shape: {full_hessian_block.shape}")
        print(f"Max difference: {np.max(np.abs(sub_hessian - full_hessian_block)):.2e}")

        npt.assert_allclose(
            sub_hessian,
            full_hessian_block,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Subsystem Hessian doesn't match full Hessian block for {name}"
        )


class TestHessianConsistency:
    """Test consistency between Hessian and forces."""

    @pytest.mark.parametrize('name', ['acetone'])  # Only test on smallest molecule for speed
    def test_energy_unchanged_with_hessian(self, name: str, package_dir):
        """
        Test that energy is the same whether or not Hessian is calculated.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        # Calculator without Hessian
        calc_no_hess = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64
        )

        # Calculator with Hessian
        calc_with_hess = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True
        )

        atoms_no_hess = atoms.copy()
        atoms_no_hess.calc = calc_no_hess
        atoms_no_hess.cell = None
        atoms_no_hess.pbc = False

        atoms_with_hess = atoms.copy()
        atoms_with_hess.calc = calc_with_hess
        atoms_with_hess.cell = None
        atoms_with_hess.pbc = False

        energy_no_hess = atoms_no_hess.get_potential_energy()
        energy_with_hess = atoms_with_hess.get_potential_energy()

        print(f"\nMolecule: {name}")
        print(f"Energy without Hessian: {energy_no_hess:.6f} eV")
        print(f"Energy with Hessian: {energy_with_hess:.6f} eV")

        npt.assert_allclose(
            energy_no_hess,
            energy_with_hess,
            rtol=1e-10,
            err_msg=f"Energy differs with Hessian calculation for {name}"
        )

    @pytest.mark.parametrize('name', ['acetone'])  # Only test on smallest molecule for speed
    def test_forces_unchanged_with_hessian(self, name: str, package_dir):
        """
        Test that forces are the same whether or not Hessian is calculated.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')

        # Calculator without Hessian
        calc_no_hess = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64
        )

        # Calculator with Hessian
        calc_with_hess = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True
        )

        atoms_no_hess = atoms.copy()
        atoms_no_hess.calc = calc_no_hess
        atoms_no_hess.cell = None
        atoms_no_hess.pbc = False

        atoms_with_hess = atoms.copy()
        atoms_with_hess.calc = calc_with_hess
        atoms_with_hess.cell = None
        atoms_with_hess.pbc = False

        forces_no_hess = atoms_no_hess.get_forces()
        forces_with_hess = atoms_with_hess.get_forces()

        print(f"\nMolecule: {name}")
        print(f"Max force difference: {np.max(np.abs(forces_no_hess - forces_with_hess)):.2e}")

        npt.assert_allclose(
            forces_no_hess,
            forces_with_hess,
            atol=1e-8,
            err_msg=f"Forces differ with Hessian calculation for {name}"
        )


class TestDifferentSubsystemRanges:
    """Test Hessian with different subsystem ranges."""

    @pytest.mark.parametrize('name', ['acetone'])
    def test_middle_atoms_subset(self, name: str, package_dir):
        """
        Test Hessian for atoms in the middle of the molecule.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select middle atoms
        start = n_atoms // 4
        end = min(start + 5, n_atoms)
        output_atom_indices = list(range(start, end))

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True,
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        hessian = calc.results['hessian']
        n_subset = len(output_atom_indices)

        print(f"\nMolecule: {name}")
        print(f"Total atoms: {n_atoms}")
        print(f"Subset indices: {output_atom_indices}")
        print(f"Hessian shape: {hessian.shape}")

        expected_shape = (n_subset, 3, n_subset, 3)
        assert hessian.shape == expected_shape, \
            f"Expected Hessian shape {expected_shape}, got {hessian.shape}"

    @pytest.mark.parametrize('name', ['acetone'])
    def test_last_atoms_subset(self, name: str, package_dir):
        """
        Test Hessian for atoms at the end of the molecule.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select last atoms
        n_subset = min(5, n_atoms)
        start = n_atoms - n_subset
        output_atom_indices = list(range(start, n_atoms))

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True,
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        hessian = calc.results['hessian']

        print(f"\nMolecule: {name}")
        print(f"Total atoms: {n_atoms}")
        print(f"Subset indices: {output_atom_indices}")
        print(f"Hessian shape: {hessian.shape}")

        expected_shape = (n_subset, 3, n_subset, 3)
        assert hessian.shape == expected_shape, \
            f"Expected Hessian shape {expected_shape}, got {hessian.shape}"

    @pytest.mark.parametrize('name', ['acetone'])
    def test_single_atom_hessian(self, name: str, package_dir):
        """
        Test Hessian for a single atom.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select just one atom
        output_atom_indices = [n_atoms // 2]

        calc = AseCalculatorSparse.create_from_workdir(
            workdir=MODEL_DIR,
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            calculate_hessian=True,
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        hessian = calc.results['hessian']

        print(f"\nMolecule: {name}")
        print(f"Single atom index: {output_atom_indices[0]}")
        print(f"Hessian shape: {hessian.shape}")

        expected_shape = (1, 3, 1, 3)
        assert hessian.shape == expected_shape, \
            f"Expected Hessian shape {expected_shape}, got {hessian.shape}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
