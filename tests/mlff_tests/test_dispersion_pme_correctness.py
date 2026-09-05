"""Scientific regression tests for the dispersion-PME decomposition."""

from ase.units import Bohr, Hartree
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

try:
    # JAX <= 0.5 exposed this context manager from ``experimental``.
    from jax.experimental import enable_x64 as _enable_x64
except ImportError:
    # Newer supported JAX releases moved it to the top-level namespace.
    from jax import enable_x64 as _enable_x64

from so3lr.mlff.nn.observable.dispersion_ref_data import (
    alphas_legacy,
    C6_coef,
    C6_coef_legacy,
)
from so3lr.mlff.nn.observable.observable_sparse import (
    DispersionEnergyKspace,
    DispersionEnergySparse,
    _batched_kspace_dispersion,
    gamma_cubic_fit,
)


class _OnesC6Head(nn.Module):
    """Deterministic C6 head for comparing model generations."""

    def __call__(self, inputs):
        return {
            'c6_ratios': jnp.ones_like(
                inputs['atomic_numbers'], dtype=inputs['positions'].dtype
            )
        }


class _InputC6Head(nn.Module):
    """Expose deterministic predictions without using dataset C6 labels."""

    def __call__(self, inputs):
        return {'c6_ratios': inputs['predicted_c6_ratios']}


def _real_space_energy_fn(*, legacy=True, cutoff=6.0, damping=1.0):
    """Return the PME real-space residual energy for a directed H-H pair."""
    module = DispersionEnergySparse(
        prop_keys=None,
        cutoff_lr=cutoff,
        cutoff_lr_damping=damping,
        hirshfeld_ratios=None,
        c6_ratios=None if legacy else _OnesC6Head(),
        dispersion_energy_scale=1.0,
        do_dispersion_kspace=True,
        legacy=legacy,
        neighborlist_format='sparse',
    )

    def energy(distance):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            dtype=distance.dtype,
        )
        inputs = {
            'positions': positions,
            'node_mask': jnp.array([True, True]),
            'idx_i_lr': jnp.array([0, 1]),
            'idx_j_lr': jnp.array([1, 0]),
            'd_ij_lr': jnp.array([distance, distance]),
            'atomic_numbers': jnp.array([1, 1]),
            'k_smearing_disp': jnp.asarray(2.0, dtype=distance.dtype),
        }
        if legacy:
            inputs['hirshfeld_ratios'] = jnp.ones(2, dtype=distance.dtype)
        else:
            inputs['a0_ratios'] = jnp.ones(2, dtype=distance.dtype)
        return jnp.sum(module.apply({}, inputs)['dispersion_energy'])

    return energy


def test_complete_pme_residual_is_value_and_force_continuous_at_cutoff():
    """C6, C8, and C10 must all vanish smoothly with the neighbor pair."""
    with _enable_x64():
        energy_fn = _real_space_energy_fn()
        force_fn = jax.jit(lambda r: -jax.grad(energy_fn)(r))
        energy_fn = jax.jit(energy_fn)

        cutoff = 6.0
        crossing_step = 0.1
        energy_at = float(energy_fn(jnp.asarray(cutoff)))
        energy_outside = float(energy_fn(jnp.asarray(cutoff + crossing_step)))
        force_at = float(force_fn(jnp.asarray(cutoff)))
        crossing_fd_force = -(
            float(energy_fn(jnp.asarray(cutoff + crossing_step)))
            - float(energy_fn(jnp.asarray(cutoff - crossing_step)))
        ) / (2 * crossing_step)

        # Test the force against a tight central finite difference inside the
        # switching interval as well as across the cutoff itself.
        midpoint = 5.5
        fd_step = 1.0e-4
        midpoint_fd_force = -(
            float(energy_fn(jnp.asarray(midpoint + fd_step)))
            - float(energy_fn(jnp.asarray(midpoint - fd_step)))
        ) / (2 * fd_step)
        midpoint_force = float(force_fn(jnp.asarray(midpoint)))

    assert abs(energy_at) < 1.0e-14
    assert abs(energy_outside) < 1.0e-14
    assert abs(force_at) < 1.0e-13
    assert abs(crossing_fd_force) < 1.0e-6
    npt.assert_allclose(midpoint_force, midpoint_fd_force, rtol=2.0e-6, atol=1.0e-10)


def _expected_legacy_real_space_energy(distance, *, use_pme):
    """Independent H-H expression using only the published legacy tables."""
    alpha = jnp.asarray(alphas_legacy[0], dtype=distance.dtype)
    c6 = jnp.asarray(C6_coef_legacy[0], dtype=distance.dtype)
    gamma = gamma_cubic_fit(alpha)
    radius = distance / jnp.asarray(Bohr, dtype=distance.dtype)
    damping_radius = 2 * 2.54 * alpha ** (1.0 / 7)
    c8 = 5 / gamma * c6
    c10 = 245 / 8 / gamma ** 2 * c6

    c6_full = -c6 / (radius ** 6 + damping_radius ** 6)
    c8_c10 = (
        -c8 / (radius ** 8 + damping_radius ** 8)
        - c10 / (radius ** 10 + damping_radius ** 10)
    )
    if use_pme:
        smearing_bohr = jnp.asarray(2.0 / Bohr, dtype=distance.dtype)
        x = radius ** 2 / (2 * smearing_bohr ** 2)
        c6_long_range = (
            c6 * (1 - jnp.exp(-x) * (1 + x + 0.5 * x ** 2)) / radius ** 6
        )
    else:
        c6_long_range = jnp.asarray(0.0, dtype=distance.dtype)

    return (c6_full + c6_long_range + c8_c10) * jnp.asarray(
        Hartree, dtype=distance.dtype
    )


@pytest.mark.parametrize('use_pme', [False, True])
def test_legacy_real_space_and_pme_subtraction_use_legacy_tables(use_pme):
    """The legacy Casimir-Polder term and PME subtraction must agree."""
    with _enable_x64():
        distance = jnp.asarray(3.0)
        if use_pme:
            energy_fn = _real_space_energy_fn(legacy=True)
        else:
            module = DispersionEnergySparse(
                prop_keys=None,
                cutoff_lr=6.0,
                cutoff_lr_damping=1.0,
                hirshfeld_ratios=None,
                c6_ratios=None,
                dispersion_energy_scale=1.0,
                do_dispersion_kspace=False,
                legacy=True,
                neighborlist_format='sparse',
            )

            def energy_fn(r):
                inputs = {
                    'positions': jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
                    'node_mask': jnp.array([True, True]),
                    'idx_i_lr': jnp.array([0, 1]),
                    'idx_j_lr': jnp.array([1, 0]),
                    'd_ij_lr': jnp.array([r, r]),
                    'atomic_numbers': jnp.array([1, 1]),
                    'hirshfeld_ratios': jnp.ones(2),
                }
                return jnp.sum(module.apply({}, inputs)['dispersion_energy'])

        actual = float(energy_fn(distance))
        expected = float(
            _expected_legacy_real_space_energy(distance, use_pme=use_pme)
        )

    npt.assert_allclose(actual, expected, rtol=2.0e-13, atol=1.0e-14)


def _kspace_energies(*, legacy):
    """Return single-graph and batched PME energies for an H-H cell."""
    module = DispersionEnergyKspace(
        prop_keys=None,
        hirshfeld_ratios=None,
        c6_ratios=None if legacy else _OnesC6Head(),
        do_ewald=False,
        interpolation_nodes=4,
        legacy=legacy,
    )
    inputs = {
        'positions': jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        'k_grid_disp': jnp.ones((8, 8, 8)),
        'k_smearing_disp': jnp.asarray(2.0),
        'cell': jnp.eye(3) * 10.0,
        'node_mask': jnp.array([True, True]),
        'atomic_numbers': jnp.array([1, 1]),
    }
    if legacy:
        inputs['hirshfeld_ratios'] = jnp.ones(2)
    else:
        inputs['c6_ratios'] = jnp.ones(2)

    variables = module.init(jax.random.key(0), inputs)
    single = jnp.sum(module.apply(variables, inputs)['dispersion_energy_kspace'])

    batched_inputs = dict(inputs)
    batched_inputs.update({
        'cell': inputs['cell'][None, ...],
        'batch_segments': jnp.zeros(2, dtype=jnp.int32),
        'graph_mask': jnp.array([True]),
    })
    bound_module = module.bind(variables)
    batched = jnp.sum(
        _batched_kspace_dispersion(
            bound_module, batched_inputs
        )['dispersion_energy_kspace']
    )
    return float(single), float(batched)


def test_legacy_reciprocal_c6_selection_matches_single_and_batched_paths():
    """Both PME paths scale with the selected generation's H reference C6."""
    # Keep this comparison in float32 even if another test module has globally
    # enabled x64; production batching uses int32 segment indices, and the
    # coefficient-selection signal is much larger than float32 roundoff.
    with _enable_x64(False):
        legacy_single, legacy_batched = _kspace_energies(legacy=True)
        v2_single, v2_batched = _kspace_energies(legacy=False)

    expected_ratio = C6_coef_legacy[0] / C6_coef[0]
    npt.assert_allclose(legacy_single, legacy_batched, rtol=2.0e-6, atol=1.0e-9)
    npt.assert_allclose(v2_single, v2_batched, rtol=2.0e-6, atol=1.0e-9)
    npt.assert_allclose(
        legacy_single / v2_single, expected_ratio, rtol=2.0e-6, atol=1.0e-9
    )


def _heteronuclear_kspace_case(*, do_ewald, grid_size):
    """Build one triclinic H-O reference case for either reciprocal solver."""
    module = DispersionEnergyKspace(
        prop_keys=None,
        hirshfeld_ratios=None,
        c6_ratios=_InputC6Head(),
        do_ewald=do_ewald,
        interpolation_nodes=4,
        legacy=False,
    )
    inputs = {
        'positions': jnp.array([
            [0.2, 0.3, 0.4],
            [1.7, 0.8, 0.6],
        ]),
        'k_grid_disp': jnp.ones((grid_size, grid_size, grid_size)),
        'k_smearing_disp': jnp.asarray(2.0),
        'cell': jnp.array([
            [9.0, 0.4, 0.1],
            [0.0, 8.5, 0.3],
            [0.0, 0.0, 8.0],
        ]),
        'node_mask': jnp.array([True, True]),
        'atomic_numbers': jnp.array([1, 8]),
        'predicted_c6_ratios': jnp.array([0.8, 1.3]),
    }
    variables = module.init(jax.random.key(0), inputs)
    return module, variables, inputs


def _total_energy_and_forces(module, variables, inputs):
    """Evaluate one single-graph reciprocal method under JIT and autodiff."""
    def energy_fn(positions):
        return jnp.sum(
            module.apply(
                variables, {**inputs, 'positions': positions}
            )['dispersion_energy_kspace']
        )

    energy = jax.jit(energy_fn)(inputs['positions'])
    forces = jax.jit(
        lambda positions: -jax.grad(energy_fn)(positions)
    )(inputs['positions'])
    return energy, forces


def test_pme_matches_ewald_reference_for_heteronuclear_cell():
    """Converged PME must match an independently evaluated Ewald sum."""
    with _enable_x64(False):
        pme_case = _heteronuclear_kspace_case(do_ewald=False, grid_size=32)
        ewald_case = _heteronuclear_kspace_case(do_ewald=True, grid_size=32)
        pme_energy, pme_forces = _total_energy_and_forces(*pme_case)
        ewald_energy, ewald_forces = _total_energy_and_forces(*ewald_case)

    npt.assert_allclose(pme_energy, ewald_energy, rtol=2.0e-4, atol=1.0e-9)
    relative_force_error = jnp.linalg.norm(
        pme_forces - ewald_forces
    ) / jnp.linalg.norm(ewald_forces)
    assert float(relative_force_error) < 3.0e-3


@pytest.mark.parametrize('do_ewald', [False, True], ids=['pme', 'ewald'])
def test_mixed_periodicity_batch_is_finite_and_matches_single_graph(do_ewald):
    """Zero cells for non-PBC/padding graphs must never reach the solver."""
    with _enable_x64(False):
        module, variables, single_inputs = _heteronuclear_kspace_case(
            do_ewald=do_ewald, grid_size=8
        )
        periodic_positions = single_inputs['positions']
        periodic_cell = single_inputs['cell']
        k_grid = single_inputs['k_grid_disp']

        def single_energy(positions):
            graph_inputs = dict(single_inputs)
            graph_inputs['positions'] = positions
            return jnp.sum(
                module.apply(variables, graph_inputs)['dispersion_energy_kspace']
            )

        single_per_atom = jax.jit(
            lambda positions: module.apply(
                variables, {**single_inputs, 'positions': positions}
            )['dispersion_energy_kspace']
        )(periodic_positions)
        single_forces = jax.jit(
            lambda positions: -jax.grad(single_energy)(positions)
        )(periodic_positions)

        # Graph 0 is periodic H-O, graph 1 is a real non-periodic C atom, and
        # graph 2 is jraph-style padding.  Both skipped graphs intentionally
        # carry singular zero cells, which previously contaminated the result.
        batched_inputs = {
            'positions': jnp.concatenate([
                periodic_positions,
                jnp.array([[0.5, 0.6, 0.7], [0.0, 0.0, 0.0]]),
            ]),
            'k_grid_disp': k_grid,
            'k_smearing_disp': jnp.asarray(2.0),
            'cell': jnp.stack([
                periodic_cell,
                jnp.zeros((3, 3)),
                jnp.zeros((3, 3)),
            ]),
            'node_mask': jnp.array([True, True, True, False]),
            'atomic_numbers': jnp.array([1, 8, 6, 1]),
            'predicted_c6_ratios': jnp.array([0.8, 1.3, 1.1, 0.0]),
            'batch_segments': jnp.array([0, 0, 1, 2], dtype=jnp.int32),
            'graph_mask': jnp.array([True, True, False]),
        }
        bound_module = module.bind(variables)

        def batched_energy(positions):
            graph_inputs = dict(batched_inputs)
            graph_inputs['positions'] = positions
            return jnp.sum(
                _batched_kspace_dispersion(
                    bound_module, graph_inputs
                )['dispersion_energy_kspace']
            )

        batched_per_atom = jax.jit(
            lambda positions: _batched_kspace_dispersion(
                bound_module, {**batched_inputs, 'positions': positions}
            )['dispersion_energy_kspace']
        )(batched_inputs['positions'])
        batched_forces = jax.jit(
            lambda positions: -jax.grad(batched_energy)(positions)
        )(batched_inputs['positions'])

    assert np.all(np.isfinite(np.asarray(batched_per_atom)))
    assert np.all(np.isfinite(np.asarray(batched_forces)))
    npt.assert_allclose(
        batched_per_atom[:2], single_per_atom, rtol=3.0e-6, atol=1.0e-9
    )
    npt.assert_allclose(
        batched_forces[:2], single_forces, rtol=3.0e-5, atol=2.0e-8
    )
    npt.assert_array_equal(np.asarray(batched_per_atom[2:]), np.zeros(2))
    npt.assert_array_equal(np.asarray(batched_forces[2:]), np.zeros((2, 3)))


@pytest.mark.parametrize('do_ewald', [False, True], ids=['pme', 'ewald'])
def test_explicit_vacuum_flags_override_a_nonzero_batch_cell(do_ewald):
    """Keeping a vacuum box must not enable reciprocal-space interactions."""
    with _enable_x64(False):
        module, variables, inputs = _heteronuclear_kspace_case(
            do_ewald=do_ewald, grid_size=8
        )
        inputs.update({
            'cell': inputs['cell'][None, ...],
            'pbc': jnp.zeros((1, 3), dtype=bool),
            'batch_segments': jnp.zeros(2, dtype=jnp.int32),
            'graph_mask': jnp.array([True]),
        })
        bound_module = module.bind(variables)

        def energy(positions):
            return jnp.sum(_batched_kspace_dispersion(
                bound_module, {**inputs, 'positions': positions}
            )['dispersion_energy_kspace'])

        value, gradient = jax.jit(jax.value_and_grad(energy))(
            inputs['positions']
        )
    npt.assert_array_equal(value, 0.0)
    npt.assert_array_equal(gradient, np.zeros((2, 3)))


def test_padded_reciprocal_dispersion_has_finite_c6_gradients():
    """Training differentiates predicted C6 values as well as positions."""
    with _enable_x64(False):
        module, variables, inputs = _heteronuclear_kspace_case(
            do_ewald=False, grid_size=8
        )
        inputs.update({
            'positions': jnp.concatenate([inputs['positions'], jnp.zeros((1, 3))]),
            'cell': jnp.stack([inputs['cell'], jnp.zeros((3, 3))]),
            'atomic_numbers': jnp.array([1, 8, 0]),
            'predicted_c6_ratios': jnp.array([0.8, 1.3, 0.0]),
            'node_mask': jnp.array([True, True, False]),
            'batch_segments': jnp.array([0, 0, 1], dtype=jnp.int32),
            'graph_mask': jnp.array([True, False]),
        })
        bound_module = module.bind(variables)

        def energy(c6):
            return jnp.sum(_batched_kspace_dispersion(
                bound_module, {**inputs, 'predicted_c6_ratios': c6}
            )['dispersion_energy_kspace'])

        gradient = jax.jit(jax.grad(energy))(inputs['predicted_c6_ratios'])
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(gradient[-1]) == 0.0
