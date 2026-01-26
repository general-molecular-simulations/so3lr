"""
Test extraction and validation of observable gradients and Jacobians.

This test file validates that gradients and Jacobians of observables
can be extracted from the SO3LR model and that they match numerical
finite difference calculations.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import numpy.testing as npt
import pathlib
import pytest
from ase.io import read
from mlff.md.calculator_sparse import mlffCalculatorSparse


def get_package_dir():
    """Return the package directory path."""
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture
def package_dir():
    """Fixture providing the package directory path."""
    return get_package_dir()


class TestObservableGradients:
    """Test extraction of observable gradients."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_nn_energy_gradient_shape(self, name: str, package_dir):
        """
        Test that nn_energy gradients have the correct shape.
        The gradient of a scalar (atomic energy) w.r.t. position should be (3,).
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select a subset of atoms for gradient calculation
        output_atom_indices = list(range(min(3, n_atoms)))

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'nn_energy_grad' in results, "nn_energy_grad not found in observables"

        grad_dict = results['nn_energy_grad']

        print(f"\nMolecule: {name}")
        print(f"Number of atoms: {n_atoms}")
        print(f"Output atom indices: {output_atom_indices}")

        # Check that we have gradients for each requested atom
        for idx in output_atom_indices:
            assert idx in grad_dict, f"Gradient for atom {idx} not found"
            value, grad = grad_dict[idx]

            # value should be a scalar (the atomic energy)
            assert np.isscalar(value) or value.shape == (), \
                f"Expected scalar value for atom {idx}, got shape {np.array(value).shape}"

            # grad should be shape (3,) - gradient w.r.t. x, y, z position
            grad = np.array(grad)
            assert grad.shape == (3,), \
                f"Expected gradient shape (3,) for atom {idx}, got {grad.shape}"

            print(f"  Atom {idx}: value={float(value):.6f}, grad_shape={grad.shape}")

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_electrostatic_energy_gradient_shape(self, name: str, package_dir):
        """
        Test that electrostatic_energy gradients have the correct shape.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        output_atom_indices = list(range(min(3, n_atoms)))

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['electrostatic_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'electrostatic_energy_grad' in results, \
            "electrostatic_energy_grad not found in observables"

        grad_dict = results['electrostatic_energy_grad']

        for idx in output_atom_indices:
            assert idx in grad_dict, f"Gradient for atom {idx} not found"
            value, grad = grad_dict[idx]
            grad = np.array(grad)
            assert grad.shape == (3,), \
                f"Expected gradient shape (3,) for atom {idx}, got {grad.shape}"

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_dispersion_energy_gradient_shape(self, name: str, package_dir):
        """
        Test that dispersion_energy gradients have the correct shape.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        output_atom_indices = list(range(min(3, n_atoms)))

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['dispersion_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'dispersion_energy_grad' in results, \
            "dispersion_energy_grad not found in observables"

        grad_dict = results['dispersion_energy_grad']

        for idx in output_atom_indices:
            assert idx in grad_dict, f"Gradient for atom {idx} not found"
            value, grad = grad_dict[idx]
            grad = np.array(grad)
            assert grad.shape == (3,), \
                f"Expected gradient shape (3,) for atom {idx}, got {grad.shape}"

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_partial_charges_jacobian_shape(self, name: str, package_dir):
        """
        Test that partial_charges Jacobian has the correct shape.
        The Jacobian of all partial charges (n_atoms,) w.r.t. one atom position (3,)
        should be (n_atoms, 3).
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        output_atom_indices = list(range(min(3, n_atoms)))

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['partial_charges_jac'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        assert 'partial_charges_jac' in results, \
            "partial_charges_jac not found in observables"

        jac_dict = results['partial_charges_jac']

        print(f"\nMolecule: {name}")
        print(f"Number of atoms: {n_atoms}")

        for idx in output_atom_indices:
            assert idx in jac_dict, f"Jacobian for atom {idx} not found"
            value, jac = jac_dict[idx]
            value = np.array(value)
            jac = np.array(jac)

            # value is a scalar - the partial charge for atom idx
            assert value.shape == () or np.isscalar(value), \
                f"Expected scalar value for atom {idx}, got shape {value.shape}"

            # Jacobian should be shape (n_atoms, 3) - d(all charges)/d(atom_idx position)
            assert jac.shape == (n_atoms, 3), \
                f"Expected Jacobian shape ({n_atoms}, 3) for atom {idx}, got {jac.shape}"

            print(f"  Atom {idx}: value={float(value):.6f}, jac_shape={jac.shape}")


class TestOutputAtomIndices:
    """Test that gradients can be computed for specific atom subsets using output_atom_indices."""

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_single_atom_gradient(self, name: str, package_dir):
        """
        Test computing gradient for a single atom.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select just one atom (middle atom if possible)
        single_atom_idx = min(n_atoms // 2, n_atoms - 1)
        output_atom_indices = [single_atom_idx]

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})
        grad_dict = results['nn_energy_grad']

        print(f"\nMolecule: {name}")
        print(f"Total atoms: {n_atoms}, Selected atom: {single_atom_idx}")

        # Should only have gradient for the selected atom
        assert single_atom_idx in grad_dict, f"Gradient for atom {single_atom_idx} not found"
        assert len(grad_dict) == 1, f"Expected 1 gradient entry, got {len(grad_dict)}"

        value, grad = grad_dict[single_atom_idx]
        grad = np.array(grad)
        assert grad.shape == (3,), f"Expected gradient shape (3,), got {grad.shape}"

        print(f"  Atom {single_atom_idx}: value={float(value):.6f}, grad={grad}")

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_non_contiguous_atom_indices(self, name: str, package_dir):
        """
        Test computing gradients for non-contiguous atom indices (e.g., [0, 2, 4]).
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        # Select non-contiguous indices (every other atom, up to 3)
        output_atom_indices = [i for i in range(0, n_atoms, 2)][:3]

        if len(output_atom_indices) < 2:
            pytest.skip(f"Not enough atoms in {name} for non-contiguous test")

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})
        grad_dict = results['nn_energy_grad']

        print(f"\nMolecule: {name}")
        print(f"Total atoms: {n_atoms}")
        print(f"Selected non-contiguous indices: {output_atom_indices}")

        # Check we have gradients for all selected atoms
        for idx in output_atom_indices:
            assert idx in grad_dict, f"Gradient for atom {idx} not found"
            value, grad = grad_dict[idx]
            grad = np.array(grad)
            assert grad.shape == (3,), f"Expected gradient shape (3,) for atom {idx}, got {grad.shape}"
            print(f"  Atom {idx}: value={float(value):.6f}, grad_norm={np.linalg.norm(grad):.6f}")

    @pytest.mark.parametrize('name', ['atat', 'dha', 'bb'])
    def test_multiple_observables_same_atoms(self, name: str, package_dir):
        """
        Test computing gradients for multiple observables on the same atom subset.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        output_atom_indices = list(range(min(2, n_atoms)))

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy_grad', 'electrostatic_energy_grad', 'dispersion_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})

        print(f"\nMolecule: {name}")
        print(f"Output atom indices: {output_atom_indices}")

        for obs_name in ['nn_energy_grad', 'electrostatic_energy_grad', 'dispersion_energy_grad']:
            assert obs_name in results, f"{obs_name} not found in results"
            grad_dict = results[obs_name]

            print(f"\n  {obs_name}:")
            for idx in output_atom_indices:
                assert idx in grad_dict, f"Gradient for atom {idx} not found in {obs_name}"
                value, grad = grad_dict[idx]
                grad = np.array(grad)
                assert grad.shape == (3,), \
                    f"Expected gradient shape (3,) for {obs_name}, atom {idx}, got {grad.shape}"
                print(f"    Atom {idx}: value={float(value):.6f}, grad_norm={np.linalg.norm(grad):.6f}")

    @pytest.mark.parametrize('name', ['atat'])
    def test_gradients_differ_between_atoms(self, name: str, package_dir):
        """
        Test that gradients are different for different atoms (sanity check).
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)

        output_atom_indices = list(range(min(3, n_atoms)))

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        atoms.cell = None
        atoms.pbc = False

        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})
        grad_dict = results['nn_energy_grad']

        # Collect all gradients
        gradients = []
        for idx in output_atom_indices:
            _, grad = grad_dict[idx]
            gradients.append(np.array(grad))

        print(f"\nMolecule: {name}")
        print(f"Checking that gradients differ between atoms...")

        # Check that at least some gradients are different
        all_same = True
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                if not np.allclose(gradients[i], gradients[j], atol=1e-10):
                    all_same = False
                    break

        assert not all_same, "All gradients are identical - this is unexpected for different atoms"
        print("  ✓ Gradients differ between atoms as expected")


class TestObservableGradientsNumerical:
    """Test observable gradients against numerical (finite difference) gradients."""

    def _compute_numerical_gradient(self, atoms, calc, obs_name, atom_idx, eps=1e-4):
        """
        Compute numerical gradient of an observable using finite differences.

        Args:
            atoms: ASE Atoms object
            calc: Calculator instance (reused for all displacements)
            obs_name: Name of the observable (without _grad suffix)
            atom_idx: Index of atom to compute gradient for
            eps: Finite difference step size

        Returns:
            Numerical gradient as numpy array of shape (3,)
        """
        numerical_grad = np.zeros(3)
        original_positions = atoms.get_positions().copy()

        for dim in range(3):
            # Positive displacement
            positions_plus = original_positions.copy()
            positions_plus[atom_idx, dim] += eps
            atoms.set_positions(positions_plus)
            atoms.calc = calc
            _ = atoms.get_potential_energy()
            results_plus = calc.results.get('aux', {})
            value_plus = np.sum(results_plus[obs_name])

            # Negative displacement
            positions_minus = original_positions.copy()
            positions_minus[atom_idx, dim] -= eps
            atoms.set_positions(positions_minus)
            _ = atoms.get_potential_energy()
            results_minus = calc.results.get('aux', {})
            value_minus = np.sum(results_minus[obs_name])

            # Central difference
            numerical_grad[dim] = (value_plus - value_minus) / (2 * eps)

        # Restore original positions
        atoms.set_positions(original_positions)
        return numerical_grad

    @pytest.mark.parametrize('name', ['atat'])
    def test_nn_energy_gradient_numerical(self, name: str, package_dir):
        """
        Test nn_energy gradient against numerical gradient.
        Only testing on 'atat' to keep test time reasonable.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)
        atoms.cell = None
        atoms.pbc = False

        # Test gradient for first atom only (to keep test fast)
        test_atom_idx = 0
        output_atom_indices = [test_atom_idx]

        # Calculator for analytical gradient
        calc_with_grad = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        # Calculator for numerical gradient (reused)
        calc_numerical = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy']
        )

        # Get analytical gradient
        atoms.calc = calc_with_grad
        _ = atoms.get_potential_energy()
        results = calc_with_grad.results.get('aux', {})
        _, analytical_grad = results['nn_energy_grad'][test_atom_idx]
        analytical_grad = np.array(analytical_grad)

        # Compute numerical gradient (reuses calc_numerical)
        numerical_grad = self._compute_numerical_gradient(
            atoms, calc_numerical, 'nn_energy', test_atom_idx, eps=1e-4
        )

        print(f"\nMolecule: {name}, Atom: {test_atom_idx}")
        print(f"Analytical gradient: {analytical_grad}")
        print(f"Numerical gradient:  {numerical_grad}")
        print(f"Difference: {analytical_grad - numerical_grad}")
        print(f"Relative error: {np.abs(analytical_grad - numerical_grad) / (np.abs(numerical_grad) + 1e-10)}")

        # Note: The gradient computed is w.r.t. one atom's position, but affects
        # the sum of all atomic energies. We need a reasonable tolerance.
        npt.assert_allclose(
            analytical_grad,
            numerical_grad,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"nn_energy gradient mismatch for {name}, atom {test_atom_idx}"
        )

    @pytest.mark.parametrize('name', ['atat'])
    def test_electrostatic_energy_gradient_numerical(self, name: str, package_dir):
        """
        Test electrostatic_energy gradient against numerical gradient.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        atoms.cell = None
        atoms.pbc = False

        test_atom_idx = 0
        output_atom_indices = [test_atom_idx]

        # Calculator for analytical gradient
        calc_with_grad = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['electrostatic_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        # Calculator for numerical gradient (reused)
        calc_numerical = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['electrostatic_energy']
        )

        # Get analytical gradient
        atoms.calc = calc_with_grad
        _ = atoms.get_potential_energy()
        results = calc_with_grad.results.get('aux', {})
        _, analytical_grad = results['electrostatic_energy_grad'][test_atom_idx]
        analytical_grad = np.array(analytical_grad)

        # Compute numerical gradient (reuses calc_numerical)
        numerical_grad = self._compute_numerical_gradient(
            atoms, calc_numerical, 'electrostatic_energy', test_atom_idx, eps=1e-4
        )

        print(f"\nMolecule: {name}, Atom: {test_atom_idx}")
        print(f"Analytical gradient: {analytical_grad}")
        print(f"Numerical gradient:  {numerical_grad}")
        print(f"Difference: {analytical_grad - numerical_grad}")

        npt.assert_allclose(
            analytical_grad,
            numerical_grad,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"electrostatic_energy gradient mismatch for {name}, atom {test_atom_idx}"
        )

    @pytest.mark.parametrize('name', ['atat'])
    def test_dispersion_energy_gradient_numerical(self, name: str, package_dir):
        """
        Test dispersion_energy gradient against numerical gradient.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        atoms.cell = None
        atoms.pbc = False

        test_atom_idx = 0
        output_atom_indices = [test_atom_idx]

        # Calculator for analytical gradient
        calc_with_grad = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['dispersion_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        # Calculator for numerical gradient (reused)
        calc_numerical = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['dispersion_energy']
        )

        # Get analytical gradient
        atoms.calc = calc_with_grad
        _ = atoms.get_potential_energy()
        results = calc_with_grad.results.get('aux', {})
        _, analytical_grad = results['dispersion_energy_grad'][test_atom_idx]
        analytical_grad = np.array(analytical_grad)

        # Compute numerical gradient (reuses calc_numerical)
        numerical_grad = self._compute_numerical_gradient(
            atoms, calc_numerical, 'dispersion_energy', test_atom_idx, eps=1e-4
        )

        print(f"\nMolecule: {name}, Atom: {test_atom_idx}")
        print(f"Analytical gradient: {analytical_grad}")
        print(f"Numerical gradient:  {numerical_grad}")
        print(f"Difference: {analytical_grad - numerical_grad}")

        npt.assert_allclose(
            analytical_grad,
            numerical_grad,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"dispersion_energy gradient mismatch for {name}, atom {test_atom_idx}"
        )

    @pytest.mark.parametrize('name', ['atat'])
    def test_multiple_atoms_gradient(self, name: str, package_dir):
        """
        Test gradients for multiple atoms simultaneously.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)
        atoms.cell = None
        atoms.pbc = False

        # Test first 3 atoms or all if fewer
        output_atom_indices = list(range(min(3, n_atoms)))

        calc = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy_grad'],
            output_atom_indices=output_atom_indices
        )

        atoms.calc = calc
        _ = atoms.get_potential_energy()

        results = calc.results.get('aux', {})
        grad_dict = results['nn_energy_grad']

        print(f"\nMolecule: {name}")
        print(f"Testing atoms: {output_atom_indices}")

        # Calculator for numerical gradient (reused across all atoms)
        calc_numerical = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['nn_energy']
        )

        for atom_idx in output_atom_indices:
            _, analytical_grad = grad_dict[atom_idx]
            analytical_grad = np.array(analytical_grad)

            numerical_grad = self._compute_numerical_gradient(
                atoms, calc_numerical, 'nn_energy', atom_idx, eps=1e-4
            )

            print(f"  Atom {atom_idx}:")
            print(f"    Analytical: {analytical_grad}")
            print(f"    Numerical:  {numerical_grad}")

            npt.assert_allclose(
                analytical_grad,
                numerical_grad,
                rtol=1e-3,
                atol=1e-5,
                err_msg=f"nn_energy gradient mismatch for {name}, atom {atom_idx}"
            )


class TestObservableJacobiansNumerical:
    """Test observable Jacobians against numerical (finite difference) Jacobians."""

    def _compute_numerical_jacobian(self, atoms, calc, obs_name, atom_idx, eps=1e-4):
        """
        Compute numerical Jacobian of a vector observable using finite differences.
        Computes d(observable)/d(atom_idx position) for all atoms.

        Args:
            atoms: ASE Atoms object
            calc: Calculator instance (reused for all displacements)
            obs_name: Name of the observable (without _jac suffix)
            atom_idx: Index of atom to compute Jacobian for
            eps: Finite difference step size

        Returns:
            Numerical Jacobian as numpy array of shape (n_atoms, 3)
        """
        n_atoms = len(atoms)
        numerical_jac = np.zeros((n_atoms, 3))
        original_positions = atoms.get_positions().copy()

        for dim in range(3):
            # Positive displacement
            positions_plus = original_positions.copy()
            positions_plus[atom_idx, dim] += eps
            atoms.set_positions(positions_plus)
            atoms.calc = calc
            _ = atoms.get_potential_energy()
            results_plus = calc.results.get('aux', {})
            value_plus = np.array(results_plus[obs_name])

            # Negative displacement
            positions_minus = original_positions.copy()
            positions_minus[atom_idx, dim] -= eps
            atoms.set_positions(positions_minus)
            _ = atoms.get_potential_energy()
            results_minus = calc.results.get('aux', {})
            value_minus = np.array(results_minus[obs_name])

            # Central difference
            numerical_jac[:, dim] = (value_plus - value_minus) / (2 * eps)

        # Restore original positions
        atoms.set_positions(original_positions)
        return numerical_jac

    @pytest.mark.parametrize('name', ['atat'])
    def test_partial_charges_jacobian_numerical(self, name: str, package_dir):
        """
        Test partial_charges Jacobian against numerical Jacobian.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        n_atoms = len(atoms)
        atoms.cell = None
        atoms.pbc = False

        test_atom_idx = 0
        output_atom_indices = [test_atom_idx]

        # Calculator for analytical Jacobian
        calc_with_jac = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['partial_charges_jac'],
            output_atom_indices=output_atom_indices
        )

        # Calculator for numerical Jacobian (reused)
        calc_numerical = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['partial_charges']
        )

        # Get analytical Jacobian
        atoms.calc = calc_with_jac
        _ = atoms.get_potential_energy()
        results = calc_with_jac.results.get('aux', {})
        _, analytical_jac = results['partial_charges_jac'][test_atom_idx]
        analytical_jac = np.array(analytical_jac)

        # Compute numerical Jacobian (reuses calc_numerical)
        numerical_jac = self._compute_numerical_jacobian(
            atoms, calc_numerical, 'partial_charges', test_atom_idx, eps=1e-4
        )

        print(f"\nMolecule: {name}, Atom: {test_atom_idx}")
        print(f"Analytical Jacobian shape: {analytical_jac.shape}")
        print(f"Numerical Jacobian shape:  {numerical_jac.shape}")
        print(f"Max absolute difference: {np.max(np.abs(analytical_jac - numerical_jac)):.2e}")
        print(f"Mean absolute difference: {np.mean(np.abs(analytical_jac - numerical_jac)):.2e}")

        npt.assert_allclose(
            analytical_jac,
            numerical_jac,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"partial_charges Jacobian mismatch for {name}, atom {test_atom_idx}"
        )

    @pytest.mark.parametrize('name', ['atat'])
    def test_hirshfeld_ratios_jacobian_numerical(self, name: str, package_dir):
        """
        Test hirshfeld_ratios Jacobian against numerical Jacobian.
        """
        atoms = read(package_dir / f'tests/test_data/{name}.xyz')
        atoms.cell = None
        atoms.pbc = False

        test_atom_idx = 0
        output_atom_indices = [test_atom_idx]

        # Calculator for analytical Jacobian
        calc_with_jac = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['hirshfeld_ratios_jac'],
            output_atom_indices=output_atom_indices
        )

        # Calculator for numerical Jacobian (reused)
        calc_numerical = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=package_dir / 'so3lr' / 'params',
            lr_cutoff=12.,
            dispersion_energy_cutoff_lr_damping=2.,
            from_file=True,
            dtype=np.float64,
            has_aux=True,
            observables=['hirshfeld_ratios']
        )

        # Get analytical Jacobian
        atoms.calc = calc_with_jac
        _ = atoms.get_potential_energy()
        results = calc_with_jac.results.get('aux', {})
        _, analytical_jac = results['hirshfeld_ratios_jac'][test_atom_idx]
        analytical_jac = np.array(analytical_jac)

        # Compute numerical Jacobian (reuses calc_numerical)
        numerical_jac = self._compute_numerical_jacobian(
            atoms, calc_numerical, 'hirshfeld_ratios', test_atom_idx, eps=1e-4
        )

        print(f"\nMolecule: {name}, Atom: {test_atom_idx}")
        print(f"Analytical Jacobian shape: {analytical_jac.shape}")
        print(f"Numerical Jacobian shape:  {numerical_jac.shape}")
        print(f"Max absolute difference: {np.max(np.abs(analytical_jac - numerical_jac)):.2e}")

        npt.assert_allclose(
            analytical_jac,
            numerical_jac,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"hirshfeld_ratios Jacobian mismatch for {name}, atom {test_atom_idx}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
