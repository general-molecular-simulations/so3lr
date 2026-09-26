"""Scientific regression tests for the dispersion-PME decomposition."""

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
    from jax import enable_x64 as _enable_x64

from so3lr.mlff.nn.observable.dispersion_ref_data import C6_coef, C6_coef_legacy
from so3lr.mlff.nn.observable.observable_sparse import (
    DispersionEnergyKspace,
    DispersionEnergySparse,
    _batched_kspace_dispersion,
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


def _real_space_energy_fn(cutoff=6.0, damping=1.0):
    """Return the (legacy) PME real-space residual energy of an H-H pair."""
    module = DispersionEnergySparse(
        prop_keys=None,
        cutoff_lr=cutoff,
        cutoff_lr_damping=damping,
        hirshfeld_ratios=None,
        dispersion_energy_scale=1.0,
        do_dispersion_kspace=True,
        legacy=True,
        neighborlist_format='sparse',
    )

    def energy(distance):
        inputs = {
            'positions': jnp.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            'node_mask': jnp.array([True, True]),
            'idx_i_lr': jnp.array([0, 1]),
            'idx_j_lr': jnp.array([1, 0]),
            'd_ij_lr': jnp.array([distance, distance]),
            'atomic_numbers': jnp.array([1, 1]),
            'k_smearing_disp': jnp.asarray(2.0, dtype=distance.dtype),
            'hirshfeld_ratios': jnp.ones(2, dtype=distance.dtype),
        }
        return jnp.sum(module.apply({}, inputs)['dispersion_energy'])

    return energy


def test_complete_pme_residual_is_value_and_force_continuous_at_cutoff():
    """C6, C8, and C10 must all vanish smoothly with the neighbor pair."""
    with _enable_x64():
        energy_fn = _real_space_energy_fn(cutoff=6.0)
        force_fn = jax.jit(lambda r: -jax.grad(energy_fn)(r))
        energy_fn = jax.jit(energy_fn)
        e = lambda r: float(energy_fn(jnp.asarray(r)))

        step = 0.1
        crossing_fd_force = -(e(6.0 + step) - e(6.0 - step)) / (2 * step)
        fd_step = 1.0e-4
        midpoint_fd_force = -(e(5.5 + fd_step) - e(5.5 - fd_step)) / (2 * fd_step)

        assert abs(e(6.0)) < 1.0e-14
        assert abs(e(6.0 + step)) < 1.0e-14
        assert abs(float(force_fn(jnp.asarray(6.0)))) < 1.0e-13
        assert abs(crossing_fd_force) < 1.0e-6
        npt.assert_allclose(float(force_fn(jnp.asarray(5.5))), midpoint_fd_force,
                            rtol=2.0e-6, atol=1.0e-10)


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
        'positions': jnp.array([[0.2, 0.3, 0.4], [1.7, 0.8, 0.6]]),
        'k_grid_disp': jnp.ones((grid_size, grid_size, grid_size)),
        'k_smearing_disp': jnp.asarray(2.0),
        'cell': jnp.array([[9.0, 0.4, 0.1], [0.0, 8.5, 0.3], [0.0, 0.0, 8.0]]),
        'node_mask': jnp.array([True, True]),
        'atomic_numbers': jnp.array([1, 8]),
        'predicted_c6_ratios': jnp.array([0.8, 1.3]),
    }
    variables = module.init(jax.random.key(0), inputs)
    return module, variables, inputs


def _single_energy_and_forces(module, variables, inputs):
    """Per-atom energies and forces of the single-graph reciprocal module."""
    def per_atom(positions):
        return module.apply(
            variables, {**inputs, 'positions': positions}
        )['dispersion_energy_kspace']

    energies = jax.jit(per_atom)(inputs['positions'])
    forces = jax.jit(jax.grad(lambda p: -jnp.sum(per_atom(p))))(inputs['positions'])
    return energies, forces


def _batched_energy_and_forces(module, variables, inputs):
    """Per-atom energies and forces of the batched (vmapped) reciprocal path."""
    bound_module = module.bind(variables)

    def per_atom(positions):
        return _batched_kspace_dispersion(
            bound_module, {**inputs, 'positions': positions}
        )['dispersion_energy_kspace']

    energies = jax.jit(per_atom)(inputs['positions'])
    forces = jax.jit(jax.grad(lambda p: -jnp.sum(per_atom(p))))(inputs['positions'])
    return energies, forces


def test_pme_matches_ewald_reference_for_heteronuclear_cell():
    """Converged PME must match an independently evaluated Ewald sum."""
    with _enable_x64(False):
        pme_energy, pme_forces = _single_energy_and_forces(
            *_heteronuclear_kspace_case(do_ewald=False, grid_size=32))
        ewald_energy, ewald_forces = _single_energy_and_forces(
            *_heteronuclear_kspace_case(do_ewald=True, grid_size=32))

    npt.assert_allclose(jnp.sum(pme_energy), jnp.sum(ewald_energy), rtol=2.0e-4, atol=1.0e-9)
    relative_force_error = (
        jnp.linalg.norm(pme_forces - ewald_forces) / jnp.linalg.norm(ewald_forces))
    assert float(relative_force_error) < 3.0e-3


@pytest.mark.parametrize('do_ewald', [False, True], ids=['pme', 'ewald'])
def test_mixed_periodicity_batch_is_finite_and_matches_single_graph(do_ewald):
    """Zero cells for non-PBC/padding graphs must never reach the solver."""
    with _enable_x64(False):
        module, variables, single = _heteronuclear_kspace_case(
            do_ewald=do_ewald, grid_size=8)
        single_energies, single_forces = _single_energy_and_forces(
            module, variables, single)

        # Graph 0 is periodic H-O, graph 1 a real non-periodic C atom, graph 2
        # jraph-style padding. Both skipped graphs carry singular zero cells.
        batched = {
            **single,
            'positions': jnp.concatenate([
                single['positions'], jnp.array([[0.5, 0.6, 0.7], [0.0, 0.0, 0.0]])]),
            'cell': jnp.stack([single['cell'], jnp.zeros((3, 3)), jnp.zeros((3, 3))]),
            'node_mask': jnp.array([True, True, True, False]),
            'atomic_numbers': jnp.array([1, 8, 6, 1]),
            'predicted_c6_ratios': jnp.array([0.8, 1.3, 1.1, 0.0]),
            'batch_segments': jnp.array([0, 0, 1, 2], dtype=jnp.int32),
            'graph_mask': jnp.array([True, True, False]),
        }
        energies, forces = _batched_energy_and_forces(module, variables, batched)

    assert np.all(np.isfinite(np.asarray(energies)))
    assert np.all(np.isfinite(np.asarray(forces)))
    npt.assert_allclose(energies[:2], single_energies, rtol=3.0e-6, atol=1.0e-9)
    npt.assert_allclose(forces[:2], single_forces, rtol=3.0e-5, atol=2.0e-8)
    npt.assert_array_equal(np.asarray(energies[2:]), np.zeros(2))
    npt.assert_array_equal(np.asarray(forces[2:]), np.zeros((2, 3)))


@pytest.mark.parametrize('do_ewald', [False, True], ids=['pme', 'ewald'])
def test_explicit_vacuum_flags_override_a_nonzero_batch_cell(do_ewald):
    """Keeping a vacuum box must not enable reciprocal-space interactions."""
    with _enable_x64(False):
        module, variables, inputs = _heteronuclear_kspace_case(
            do_ewald=do_ewald, grid_size=8)
        inputs.update({
            'cell': inputs['cell'][None, ...],
            'pbc': jnp.zeros((1, 3), dtype=bool),
            'batch_segments': jnp.zeros(2, dtype=jnp.int32),
            'graph_mask': jnp.array([True]),
        })
        energies, forces = _batched_energy_and_forces(module, variables, inputs)

    npt.assert_array_equal(energies, np.zeros(2))
    npt.assert_array_equal(forces, np.zeros((2, 3)))


@pytest.mark.parametrize('do_ewald', [False, True], ids=['pme', 'ewald'])
def test_padded_reciprocal_dispersion_has_finite_c6_gradients(do_ewald):
    """Training differentiates predicted C6 values as well as positions."""
    with _enable_x64(False):
        module, variables, inputs = _heteronuclear_kspace_case(
            do_ewald=do_ewald, grid_size=8)
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
