"""The short-range OOD gate and the NaN-label fallback for c6_ratios.

The gate is the one change in the LES-port PR that alters model predictions
(below SR_GATE_ON only), so its safety claims are pinned here: normal chemistry
is untouched (gate exactly 1 at trained distances), clashes fade the learned
heads to constants, padding edges are ignored, and inputs without a short-range
graph disable it. The charge head's fade target is pinned end to end: gating
charges to 0 made charge conservation dump its full correction onto the clashed
atom (~+7 e) and blew up the electrostatics, so the head is faded to its own
zero-environment prediction instead.
"""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from so3lr.mlff.nn.observable.observable_sparse import (
    SR_GATE_OFF,
    SR_GATE_ON,
    _c6_ratios_from_labels_or_head,
    _fade_to_free_atom_ratio,
    short_range_gate,
    switching_fn,
)


def _inputs(d_ij, idx_i, n_atoms, idx_j=None):
    d_ij = jnp.asarray(d_ij)
    idx_i = jnp.asarray(idx_i)
    if idx_j is None:
        # Any partner that is not the sender: these tests only care about d_ij,
        # and idx_i != idx_j is what marks an edge as real.
        idx_j = (idx_i + 1) % max(n_atoms, 2)
    return {
        'd_ij': d_ij,
        'idx_i': idx_i,
        'idx_j': jnp.asarray(idx_j),
        'node_mask': jnp.ones((n_atoms,), dtype=bool),
    }


def test_gate_is_exactly_one_in_the_trained_range():
    """d_ij >= SR_GATE_ON must give gate == 1.0 exactly, not approximately:
    any deviation perturbs every prediction of every normal structure."""
    d = [SR_GATE_ON, 0.96, 1.5, 5.0]  # shortest real bond is O-H ~0.96 A
    gate = short_range_gate(_inputs(d, [0, 0, 1, 1], 2))
    npt.assert_array_equal(np.asarray(gate), np.ones(2))


def test_gate_closes_at_a_clash():
    # 0.13 A is the clash from the motivating blow-up (charge head ~1e8 e).
    gate = short_range_gate(_inputs([0.13, 0.96], [0, 1], 2))
    assert float(gate[0]) == 0.0
    assert float(gate[1]) == 1.0


def test_gate_is_smooth_between_off_and_on():
    d_mid = 0.5 * (SR_GATE_OFF + SR_GATE_ON)
    gate = short_range_gate(_inputs([d_mid], [0], 1))
    assert 0.0 < float(gate[0]) < 1.0


def test_padding_edges_do_not_close_the_gate():
    """Padding edges are marked by idx_i == idx_j; they must not count as
    clashes even though their d_ij is 0."""
    gate = short_range_gate(_inputs([0.96, 0.0, 0.0], [0, 0, 0], 1,
                                    idx_j=[1, 0, 0]))
    npt.assert_array_equal(np.asarray(gate), np.ones(1))


def test_coincident_atoms_close_the_gate():
    """Two atoms at EXACTLY the same position (d_ij == 0) are the worst possible
    out-of-distribution case, so the gate must shut -- it must not be mistaken
    for padding, which is identified by idx_i == idx_j instead."""
    gate = short_range_gate(_inputs([0.0, 0.0], [0, 1], 2, idx_j=[1, 0]))
    npt.assert_array_equal(np.asarray(gate), np.zeros(2))


def test_gate_gradient_is_finite_at_zero_distance():
    import jax

    def total_gate(d):
        return short_range_gate({
            'd_ij': d, 'idx_i': jnp.asarray([0, 1]), 'idx_j': jnp.asarray([1, 0]),
            'node_mask': jnp.ones((2,), dtype=bool),
        }).sum()

    grad = jax.grad(total_gate)(jnp.asarray([0.0, 0.0]))
    assert np.isfinite(np.asarray(grad)).all()


def test_gate_defaults_to_open_without_a_short_range_graph():
    gate = short_range_gate({'node_mask': jnp.ones((3,), dtype=bool)})
    npt.assert_array_equal(np.asarray(gate), np.ones(3))


def test_switching_fn_direction():
    """switching_fn(d, OFF, ON) must be 1 at/below OFF and 0 at/above ON;
    flipping the argument order inverts the gate for every structure."""
    npt.assert_allclose(float(switching_fn(SR_GATE_OFF, SR_GATE_OFF, SR_GATE_ON)), 1.0)
    npt.assert_allclose(float(switching_fn(SR_GATE_ON, SR_GATE_OFF, SR_GATE_ON)), 0.0)


def test_fade_to_free_atom_ratio():
    inputs = _inputs([0.13, 0.96], [0, 1], 2)
    ratios = jnp.asarray([3.7, 0.8])
    faded = _fade_to_free_atom_ratio(ratios, inputs)
    npt.assert_allclose(float(faded[0]), 1.0)   # clash -> free-atom value
    npt.assert_allclose(float(faded[1]), 0.8)   # trained range -> untouched


def test_no_calibrated_charge_constant_remains():
    """The charge fade target is derived from the checkpoint (the head's
    zero-environment prediction), not a hand-calibrated constant: the true value
    differs by several e between model families, so no literal can serve them
    all. Guards against the constant being reintroduced."""
    import so3lr.mlff.nn.observable.observable_sparse as obs

    assert not hasattr(obs, 'SR_GATE_FALLBACK')


class TestC6RatiosLabelFallback:
    """NaN-padded c6_ratios labels must never reach the dispersion forward."""

    @staticmethod
    def _head(inputs):
        return {'c6_ratios': jnp.full((3,), 0.5)}

    def test_no_label_uses_the_head(self):
        out = _c6_ratios_from_labels_or_head(self._head, {})
        npt.assert_allclose(np.asarray(out), 0.5)

    def test_nan_padded_label_falls_back_to_the_head(self):
        out = _c6_ratios_from_labels_or_head(
            self._head, {'c6_ratios': jnp.full((3,), jnp.nan)})
        npt.assert_allclose(np.asarray(out), 0.5)
        assert np.isfinite(np.asarray(out)).all()

    def test_mixed_label_is_elementwise(self):
        label = jnp.asarray([0.9, jnp.nan, 1.1])
        out = _c6_ratios_from_labels_or_head(self._head, {'c6_ratios': label})
        npt.assert_allclose(np.asarray(out), [0.9, 0.5, 1.1])

    def test_finite_labels_pass_through_unchanged(self):
        label = jnp.asarray([0.9, 1.0, 1.1])
        out = _c6_ratios_from_labels_or_head(self._head, {'c6_ratios': label})
        npt.assert_array_equal(np.asarray(out), np.asarray(label))


def test_clash_structure_yields_finite_forces_end_to_end():
    """A steric clash used to give ~1e15 eV/A forces (charge head ~1e8 e ->
    Coulomb ~1e16 eV). With the gate, the ZBL/NLH repulsion carries the
    short-range wall and everything stays finite and sane."""
    from ase import Atoms

    from so3lr import So3lrCalculator

    water = Atoms(
        'OHH',
        positions=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )  # O-H at 0.2 A: far below SR_GATE_ON
    water.calc = So3lrCalculator(model='so3lr-s')
    energy = water.get_potential_energy()
    forces = water.get_forces()
    assert np.isfinite(energy)
    assert np.isfinite(forces).all()
    # Repulsion at 0.2 A is huge but physical; the pre-gate blow-up was ~1e15.
    assert np.abs(forces).max() < 1e6
