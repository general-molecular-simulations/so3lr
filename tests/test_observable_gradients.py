"""
Test observable gradients and Jacobians against finite differences.

``<observable>_grad`` and ``<observable>_jac`` are derivatives with respect to
the position of each atom in ``output_atom_indices``, computed by writing that
atom's position back into the full position array
(``calculate_obs_grads`` in so3lr/mlff/calculators/ase_calculator.py). A wrong
index or mask there still returns arrays of the right shape, so the check is
against central finite differences of the plain observables, for a
non-contiguous selection of atoms.
"""

import jax
jax.config.update("jax_enable_x64", True)

import copy
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

# Scalar observables (summed over atoms) checked through '<name>_grad', and
# per-atom observables checked through '<name>_jac'.
GRAD_OBSERVABLES = ['nn_energy', 'electrostatic_energy', 'dispersion_energy']
JAC_OBSERVABLES = ['partial_charges', 'hirshfeld_ratios']

# Non-contiguous on purpose: a mask that assumed a contiguous range, or dropped
# the index altogether, would put atom 0's derivative under atom 2.
OUTPUT_ATOM_INDICES = [0, 2]

# Finite-difference step and tolerances of the earlier per-observable tests.
EPS = 1e-4
RTOL = 1e-3
ATOL = 1e-5


def get_package_dir():
    """Return the package directory path."""
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture
def package_dir():
    """Fixture providing the package directory path."""
    return get_package_dir()


def _calculator(observables, **kwargs):
    return AseCalculatorSparse.create_from_workdir(
        workdir=MODEL_DIR,
        lr_cutoff=12.,
        dispersion_energy_cutoff_lr_damping=2.,
        from_file=True,
        dtype=np.float64,
        has_aux=True,
        observables=observables,
        **kwargs
    )


def _observables_at(atoms, calc, positions):
    """The calculator's observables for `atoms` moved to `positions`."""
    atoms.set_positions(positions)
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    # A copy: the same calculator is called again for the next displacement.
    return copy.deepcopy(calc.results['aux'])


@pytest.mark.parametrize('name', ['dha'])
def test_gradients_and_jacobians_match_finite_differences(name: str, package_dir):
    """Every derivative observable, for every selected atom, against central differences.

    ``<name>_grad`` is the gradient of the observable summed over atoms;
    ``<name>_jac`` is the (n_atoms, 3) Jacobian of all per-atom values with
    respect to the selected atom's position.
    """
    atoms = read(package_dir / f'tests/test_data/{name}.xyz')
    atoms.cell = None
    atoms.pbc = False
    n_atoms = len(atoms)
    positions = atoms.get_positions().copy()

    calc_derivatives = _calculator(
        [f'{o}_grad' for o in GRAD_OBSERVABLES] + [f'{o}_jac' for o in JAC_OBSERVABLES],
        output_atom_indices=OUTPUT_ATOM_INDICES,
    )
    analytical = _observables_at(atoms, calc_derivatives, positions)

    # Plain observables: the reference values and the finite differences.
    calc_plain = _calculator(GRAD_OBSERVABLES + JAC_OBSERVABLES)
    reference = _observables_at(atoms, calc_plain, positions)

    numerical = {idx: {o: np.zeros((n_atoms, 3) if o in JAC_OBSERVABLES else 3)
                       for o in GRAD_OBSERVABLES + JAC_OBSERVABLES}
                 for idx in OUTPUT_ATOM_INDICES}
    for idx in OUTPUT_ATOM_INDICES:
        for dim in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[idx, dim] += EPS
            minus[idx, dim] -= EPS
            obs_plus = _observables_at(atoms, calc_plain, plus)
            obs_minus = _observables_at(atoms, calc_plain, minus)
            for o in GRAD_OBSERVABLES:
                numerical[idx][o][dim] = (
                    np.sum(obs_plus[o]) - np.sum(obs_minus[o])) / (2 * EPS)
            for o in JAC_OBSERVABLES:
                numerical[idx][o][:, dim] = (
                    np.array(obs_plus[o]) - np.array(obs_minus[o])) / (2 * EPS)
    atoms.set_positions(positions)

    for o in GRAD_OBSERVABLES + JAC_OBSERVABLES:
        key = f'{o}_grad' if o in GRAD_OBSERVABLES else f'{o}_jac'
        assert key in analytical, f"{key} not found in observables"
        derivatives = analytical[key]
        # Only the requested atoms, keyed by their index.
        assert sorted(derivatives) == OUTPUT_ATOM_INDICES, (
            f"{key}: expected entries for atoms {OUTPUT_ATOM_INDICES}, got {sorted(derivatives)}"
        )
        for idx in OUTPUT_ATOM_INDICES:
            value, derivative = derivatives[idx]
            # The value reported next to the derivative is the selected atom's own.
            npt.assert_allclose(
                value, np.array(reference[o])[idx], rtol=RTOL, atol=ATOL,
                err_msg=f"{key} value mismatch for {name}, atom {idx}"
            )
            npt.assert_allclose(
                np.array(derivative),
                numerical[idx][o],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"{key} mismatch for {name}, atom {idx}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
