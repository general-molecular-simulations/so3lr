"""ensure_adaptive_robust_loss_params: the fine-tuning repair path.

Fine-tuning restores a checkpoint, so fit()'s params-is-None init branch never
runs; this helper attaches or repairs the adaptive robust loss parameters
instead. Lives in its own file (not test_training_utils.py) so stacked PRs
appending there do not collide.
"""
import jax.numpy as jnp
import numpy.testing as npt

from so3lr.mlff.utils import training_utils

def test_ensure_adaptive_robust_loss_params():
    """Fine-tuning restores a checkpoint, so fit()'s params-is-None init branch
    never runs; this helper has to attach/repair the adaptive robust loss
    parameters instead. Three cases: absent -> created; wrong length ->
    re-initialized (positional indexing makes the old values meaningless);
    matching length -> reused unchanged (a continuation run)."""
    checkpoint = {'params': {'observables_0': {'energy_offset': jnp.ones((4,))}}}

    # Absent -> created for exactly the run's targets.
    out = training_utils.ensure_adaptive_robust_loss_params(
        checkpoint, targets=['forces'])
    npt.assert_equal(out['adaptive_robust_loss']['alpha_raw'].shape, (1,))
    npt.assert_equal(out['adaptive_robust_loss']['scale_raw'].shape, (1,))
    assert 'adaptive_robust_loss' not in checkpoint  # input not mutated

    # Length mismatch (pretraining had 5 targets, this run trains 1) -> discard
    # and re-initialize; keeping them would scale `forces` by whatever alpha the
    # pretraining learned for its first target.
    stale = dict(checkpoint)
    stale.update(training_utils.init_adaptive_robust_loss_params(
        targets=['energy', 'forces', 'dipole_vec', 'hirshfeld_ratios', 'c6_ratios']))
    out = training_utils.ensure_adaptive_robust_loss_params(
        stale, targets=['forces'], init_alpha=1.5)
    npt.assert_equal(out['adaptive_robust_loss']['alpha_raw'].shape, (1,))

    # Matching length -> reused as-is (the arrays carry no target names, so a
    # same-length continuation is trusted positionally).
    trained = dict(checkpoint)
    trained.update(training_utils.init_adaptive_robust_loss_params(
        targets=['energy', 'forces'], init_alpha=0.7))
    out = training_utils.ensure_adaptive_robust_loss_params(
        trained, targets=['energy', 'forces'], init_alpha=1.5)
    npt.assert_array_equal(out['adaptive_robust_loss']['alpha_raw'],
                           trained['adaptive_robust_loss']['alpha_raw'])
