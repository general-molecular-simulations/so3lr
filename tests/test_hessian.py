"""
Test Hessian calculation and validation against numerical Hessians.

This test file validates that Hessians can be extracted from the SO3LR model
using the calculate_hessian option, and that they match numerical finite
difference calculations. Subsystem Hessians (output_atom_indices) are checked
against blocks of the full Hessian, so the finite-difference check carries over.
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


class TestHessianSymmetry:
    """Test that Hessian has expected symmetry properties."""

    @pytest.mark.parametrize('name', ['acetone'])  # Only test on smallest molecule for speed
    def test_hessian_symmetry(self, name: str, package_dir):
        """
        Test that the full Hessian has shape (n_atoms, 3, n_atoms, 3) and is
        symmetric: H[i,a,j,b] = H[j,b,i,a].
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

        expected_shape = (n_atoms, 3, n_atoms, 3)
        assert hessian.shape == expected_shape, \
            f"Expected Hessian shape {expected_shape}, got {hessian.shape}"

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


# Contiguous atom ranges for subsystem Hessians, as functions of the atom count.
# Ranges that do not start at atom 0 exercise the offset in the slice update
# (calculate_hessian in ase_calculator.py), which a shape check cannot see.
SUBSETS = {
    'first': lambda n: list(range(min(5, n))),
    'middle': lambda n: list(range(n // 4, min(n // 4 + 5, n))),
    'last': lambda n: list(range(n - min(5, n), n)),
    'single': lambda n: [n // 2],
}


class TestSubsystemHessian:
    """Test subsystem Hessians against the full Hessian."""

    @pytest.mark.parametrize('subset', list(SUBSETS))
    @pytest.mark.parametrize('name', ['acetone'])
    def test_subsystem_matches_full_hessian(self, name: str, subset: str, package_dir):
        """
        Test that the subsystem Hessian equals the corresponding block of the
        full Hessian, for ranges at the start, middle and end, and a single atom.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        output_atom_indices = SUBSETS[subset](n_atoms)

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
        print(f"Subset indices: {output_atom_indices}")
        print(f"Subsystem Hessian shape: {sub_hessian.shape}")
        print(f"Max difference: {np.max(np.abs(sub_hessian - full_hessian_block)):.2e}")

        # Shape (n_subset, 3, n_subset, 3); assert_allclose also fails on a shape mismatch.
        n_subset = len(output_atom_indices)
        assert sub_hessian.shape == (n_subset, 3, n_subset, 3), \
            f"Expected Hessian shape {(n_subset, 3, n_subset, 3)}, got {sub_hessian.shape}"

        npt.assert_allclose(
            sub_hessian,
            full_hessian_block,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Subsystem Hessian doesn't match full Hessian block for {name}, {subset}"
        )


class TestHessianConsistency:
    """Test consistency between Hessian and forces."""

    @pytest.mark.parametrize('name', ['acetone'])  # Only test on smallest molecule for speed
    def test_energy_and_forces_unchanged_with_hessian(self, name: str, package_dir):
        """
        Test that energy and forces are the same whether or not Hessian is calculated.
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
        forces_no_hess = atoms_no_hess.get_forces()
        forces_with_hess = atoms_with_hess.get_forces()

        print(f"\nMolecule: {name}")
        print(f"Energy without Hessian: {energy_no_hess:.6f} eV")
        print(f"Energy with Hessian: {energy_with_hess:.6f} eV")
        print(f"Max force difference: {np.max(np.abs(forces_no_hess - forces_with_hess)):.2e}")

        npt.assert_allclose(
            energy_no_hess,
            energy_with_hess,
            rtol=1e-10,
            err_msg=f"Energy differs with Hessian calculation for {name}"
        )

        npt.assert_allclose(
            forces_no_hess,
            forces_with_hess,
            atol=1e-8,
            err_msg=f"Forces differ with Hessian calculation for {name}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
