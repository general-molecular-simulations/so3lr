"""Regression tests for separating C6 targets from physical predictions."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from so3lr.cli.so3lr_eval import calculate_metrics, _model_inputs_for_evaluation
from so3lr.mlff.nn.observable.observable_sparse import DispersionEnergySparse


_PREDICTED_C6_RATIOS = jnp.array([0.75, 1.25])


class _FixedC6Head(nn.Module):
    """Small deterministic stand-in for a trained C6 prediction head."""

    def __call__(self, inputs):
        del inputs
        return {'c6_ratios': _PREDICTED_C6_RATIOS}


def _dispersion_energy_and_forces(c6_labels=None):
    """Evaluate the production dispersion module and differentiate positions."""
    module = DispersionEnergySparse(
        prop_keys=None,
        cutoff_lr=12.0,
        cutoff_lr_damping=2.0,
        hirshfeld_ratios=None,
        c6_ratios=_FixedC6Head(),
        dispersion_energy_scale=1.0,
        do_dispersion_kspace=False,
        legacy=False,
        neighborlist_format='sparse',
    )
    positions = jnp.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]])

    def total_energy(pos):
        pair_displacements = pos[jnp.array([0, 1])] - pos[jnp.array([1, 0])]
        inputs = {
            'positions': pos,
            'node_mask': jnp.array([True, True]),
            'idx_i_lr': jnp.array([0, 1]),
            'idx_j_lr': jnp.array([1, 0]),
            'd_ij_lr': jnp.linalg.norm(pair_displacements, axis=-1),
            'atomic_numbers': jnp.array([1, 8]),
            # In the complete model this is written by the a0 prediction head
            # before the dispersion module runs.
            'a0_ratios': jnp.array([0.9, 1.1]),
        }
        if c6_labels is not None:
            inputs['c6_ratios'] = jnp.asarray(c6_labels)
        return jnp.sum(module.apply({}, inputs)['dispersion_energy'])

    energy, gradient = jax.value_and_grad(total_energy)(positions)
    return np.asarray(energy), np.asarray(-gradient)


def test_energy_and_forces_ignore_present_or_permuted_c6_targets():
    """Dataset C6 labels must not act as inference-time model inputs."""
    energy_without_labels, forces_without_labels = _dispersion_energy_and_forces()
    energy_with_labels, forces_with_labels = _dispersion_energy_and_forces([0.2, 3.0])
    energy_with_permuted_labels, forces_with_permuted_labels = (
        _dispersion_energy_and_forces([3.0, 0.2])
    )

    npt.assert_allclose(energy_with_labels, energy_without_labels, rtol=0, atol=0)
    npt.assert_allclose(
        energy_with_permuted_labels, energy_without_labels, rtol=0, atol=0
    )
    npt.assert_allclose(forces_with_labels, forces_without_labels, rtol=0, atol=0)
    npt.assert_allclose(
        forces_with_permuted_labels, forces_without_labels, rtol=0, atol=0
    )


def test_c6_targets_remain_available_for_evaluation_metrics():
    """Isolation from the forward pass must not discard C6 scoring labels."""
    labels = np.array([0.2, 3.0])
    inputs = {
        'c6_ratios': labels,
        'node_mask': np.array([True, True]),
        'graph_mask': np.array([True]),
    }
    outputs = {'c6_ratios': np.asarray(_PREDICTED_C6_RATIOS)}

    model_inputs = _model_inputs_for_evaluation(inputs)
    metrics = calculate_metrics(outputs, inputs, ['c6_ratios'])

    assert 'c6_ratios' not in model_inputs
    npt.assert_array_equal(inputs['c6_ratios'], labels)
    expected_diff = outputs['c6_ratios'] - labels
    assert metrics['c6_ratios_count'] == 2
    npt.assert_allclose(metrics['c6_ratios_abs_sum'], np.abs(expected_diff).sum())
    npt.assert_allclose(metrics['c6_ratios_sq_sum'], np.square(expected_diff).sum())
