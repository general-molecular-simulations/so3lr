"""
Test Hessian calculation and validation against numerical Hessians.

This test file validates that Hessians can be extracted from the SO3LR model
using the calculate_hessian option, and that they match numerical finite
difference calculations. Subsystem Hessians (output_atom_indices) are checked
against blocks of the full Hessian, so the finite-difference check carries over.

The full Hessian and a plain forces calculator are built once per module and
shared: every test below reads the same molecule with the same model settings.
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

# Only test on smallest molecule for speed
MOLECULE = 'acetone'


def get_package_dir():
    """Return the package directory path."""
    return pathlib.Path(__file__).parent.parent.resolve()


def _calculator(**kwargs):
    return AseCalculatorSparse.create_from_workdir(
        workdir=MODEL_DIR,
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        from_file=True,
        dtype=np.float64,
        **kwargs
    )


def _molecule():
    atoms = read(get_package_dir() / f'tests/test_data/{MOLECULE}.xyz')
    atoms.cell = None
    atoms.pbc = False
    return atoms


@pytest.fixture(scope='module')
def forces_calc():
    """Calculator without Hessian, for finite differences and consistency."""
    return _calculator()


@pytest.fixture(scope='module')
def full_hessian():
    """Energy, forces and full Hessian of the molecule, from one Hessian calculator."""
    atoms = _molecule()
    atoms.calc = _calculator(calculate_hessian=True)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert 'hessian' in atoms.calc.results, "Hessian not found in results"
    return dict(energy=energy, forces=forces, hessian=np.array(atoms.calc.results['hessian']))


class TestHessianSymmetry:
    """Test that Hessian has expected symmetry properties."""

    def test_hessian_symmetry(self, full_hessian):
        """
        Test that the full Hessian has shape (n_atoms, 3, n_atoms, 3) and is
        symmetric: H[i,a,j,b] = H[j,b,i,a].
        """
        n_atoms = len(_molecule())
        hessian = full_hessian['hessian']

        expected_shape = (n_atoms, 3, n_atoms, 3)
        assert hessian.shape == expected_shape, \
            f"Expected Hessian shape {expected_shape}, got {hessian.shape}"

        # Reshape to (n_atoms*3, n_atoms*3) for easier symmetry check
        hessian_2d = hessian.reshape(n_atoms * 3, n_atoms * 3)

        print(f"\nMolecule: {MOLECULE}")
        print(f"Max asymmetry: {np.max(np.abs(hessian_2d - hessian_2d.T)):.2e}")

        npt.assert_allclose(
            hessian_2d,
            hessian_2d.T,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Hessian is not symmetric for {MOLECULE}"
        )


class TestHessianNumerical:
    """Test Hessian against numerical (finite difference) Hessian."""

    @staticmethod
    def _compute_numerical_hessian(atoms, calc, displaced, eps=1e-4):
        """
        Compute rows of the numerical Hessian using finite differences on forces.

        The Hessian is: H[i,a,j,b] = d^2E / (dR_i^a dR_j^b) = -dF_j^b / dR_i^a

        Args:
            atoms: ASE Atoms object
            calc: Calculator instance (reused for all displacements)
            displaced: Indices of the atoms to displace (the rows i)
            eps: Finite difference step size

        Returns:
            Numerical Hessian as numpy array of shape (len(displaced), 3, n_atoms, 3)
        """
        n_atoms = len(atoms)
        numerical_hessian = np.zeros((len(displaced), 3, n_atoms, 3))
        original_positions = atoms.get_positions().copy()
        atoms.calc = calc

        for row, i in enumerate(displaced):
            for a in range(3):
                # Positive displacement
                positions_plus = original_positions.copy()
                positions_plus[i, a] += eps
                atoms.set_positions(positions_plus)
                forces_plus = atoms.get_forces().copy()

                # Negative displacement
                positions_minus = original_positions.copy()
                positions_minus[i, a] -= eps
                atoms.set_positions(positions_minus)
                forces_minus = atoms.get_forces().copy()

                # Central difference: H[i,a,j,b] = -dF[j,b]/dR[i,a]
                numerical_hessian[row, a, :, :] = -(forces_plus - forces_minus) / (2 * eps)

        # Restore original positions
        atoms.set_positions(original_positions)
        return numerical_hessian

    def test_full_hessian_numerical(self, full_hessian, forces_calc, full_test):
        """
        Test full Hessian against numerical Hessian.

        Without --full, only the first atom (a carbon) is displaced; --full adds a
        hydrogen. Every row comes from the same autodiff call, so more rows add
        cost rather than coverage.
        """
        atoms = _molecule()
        displaced = [0, 4] if full_test else [0]
        analytical_hessian = full_hessian['hessian'][displaced]

        numerical_hessian = self._compute_numerical_hessian(
            atoms, forces_calc, displaced, eps=1e-4
        )

        print(f"\nMolecule: {MOLECULE} ({'full' if full_test else 'short'} mode)")
        print(f"Displacing atoms {displaced} of {len(atoms)}")
        print(f"Max absolute difference: {np.max(np.abs(analytical_hessian - numerical_hessian)):.2e}")
        print(f"Mean absolute difference: {np.mean(np.abs(analytical_hessian - numerical_hessian)):.2e}")

        npt.assert_allclose(
            analytical_hessian,
            numerical_hessian,
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Hessian mismatch for {MOLECULE}"
        )


# Contiguous atom ranges of acetone (10 atoms; change them with MOLECULE) whose
# subsystem Hessian is checked. Ranges that do not start at atom 0 exercise the
# offset in the slice update (calculate_hessian in ase_calculator.py), which a
# shape check cannot see.
SUBSETS = {
    'first': [0, 1, 2, 3, 4],
    'middle': [2, 3, 4, 5, 6],
    'last': [5, 6, 7, 8, 9],
    'single': [5],
}


class TestSubsystemHessian:
    """Test subsystem Hessians against the full Hessian."""

    @pytest.mark.parametrize('subset', list(SUBSETS))
    def test_subsystem_matches_full_hessian(self, subset: str, full_hessian):
        """
        Test that the subsystem Hessian equals the corresponding block of the
        full Hessian, for ranges at the start, middle and end, and a single atom.
        """
        atoms = _molecule()
        output_atom_indices = SUBSETS[subset]

        # Get subsystem Hessian
        atoms.calc = _calculator(calculate_hessian=True, output_atom_indices=output_atom_indices)
        _ = atoms.get_potential_energy()
        sub_hessian = atoms.calc.results['hessian']

        # Extract corresponding block from full Hessian (subset x subset)
        start_idx = output_atom_indices[0]
        end_idx = output_atom_indices[-1] + 1
        full_hessian_block = full_hessian['hessian'][start_idx:end_idx, :, start_idx:end_idx, :]

        print(f"\nMolecule: {MOLECULE}")
        print(f"Subset indices: {output_atom_indices}")
        print(f"Subsystem Hessian shape: {sub_hessian.shape}")
        print(f"Max difference: {np.max(np.abs(sub_hessian - full_hessian_block)):.2e}")

        # assert_allclose broadcasts compatible shapes, so check the shape explicitly.
        n_subset = len(output_atom_indices)
        assert sub_hessian.shape == (n_subset, 3, n_subset, 3), \
            f"Expected Hessian shape {(n_subset, 3, n_subset, 3)}, got {sub_hessian.shape}"

        npt.assert_allclose(
            sub_hessian,
            full_hessian_block,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Subsystem Hessian doesn't match full Hessian block for {MOLECULE}, {subset}"
        )


class TestHessianConsistency:
    """Test consistency between Hessian and forces."""

    def test_energy_and_forces_unchanged_with_hessian(self, full_hessian, forces_calc):
        """
        Test that energy and forces are the same whether or not Hessian is calculated.
        """
        atoms = _molecule()
        atoms.calc = forces_calc
        energy_no_hess = atoms.get_potential_energy()
        forces_no_hess = atoms.get_forces()

        print(f"\nMolecule: {MOLECULE}")
        print(f"Energy without Hessian: {energy_no_hess:.6f} eV")
        print(f"Energy with Hessian: {full_hessian['energy']:.6f} eV")
        print(f"Max force difference: {np.max(np.abs(forces_no_hess - full_hessian['forces'])):.2e}")

        npt.assert_allclose(
            energy_no_hess,
            full_hessian['energy'],
            rtol=1e-10,
            err_msg=f"Energy differs with Hessian calculation for {MOLECULE}"
        )

        npt.assert_allclose(
            forces_no_hess,
            full_hessian['forces'],
            atol=1e-8,
            err_msg=f"Forces differ with Hessian calculation for {MOLECULE}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
