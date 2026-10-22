import copy
import logging
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from ase import units

from so3lr.mlff import jraph_utils, training_utils
from so3lr.mlff.data import NpzDataLoaderSparse, transformations
from so3lr.mlff.nn import SO3kratesSparse
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse


TEST_DIR = Path(__file__).parent


# 50 training + 10 validation frames, loaded once. Two ethanol frames fit in one
# batch (see _fit), so 50 training frames make 25 optimizer steps per epoch and
# two epochs make 50 steps.
NUM_FRAMES = 60


@pytest.fixture(scope='module')
def ethanol_frames():
    """60 random ethanol frames, shuffled, as stored (kcal/mol, no shifts).

    Loaded once per module with pick_idx: building all 2000 graphs on every
    call used to dominate this file, and the tests only ever used 60."""
    loader = NpzDataLoaderSparse(input_file=TEST_DIR / 'test_data/ethanol.npz')
    pick_idx = np.random.RandomState(0).permutation(loader.cardinality())[:NUM_FRAMES]
    frames, _ = loader.load(cutoff=5., pick_idx=pick_idx)
    np.random.RandomState(0).shuffle(frames)
    return frames


def _ethanol(frames, num_train=50, num_valid=10):
    """(training, validation) in eV/Angstrom with per-atom energy shifts removed.

    Works on a deep copy: unit_conversion and subtract_atomic_energy_shifts
    rewrite the graphs in place, and the module fixture shares them."""
    all_data = copy.deepcopy(frames)
    energy_unit = units.kcal / units.mol
    energy_mean = transformations.calculate_energy_mean(all_data[:num_train]) * energy_unit
    num_nodes = transformations.calculate_average_number_of_nodes(all_data[:num_train])
    shifts = {a: energy_mean / num_nodes for a in range(119)}

    def prep(frames):
        return list(transformations.subtract_atomic_energy_shifts(
            transformations.unit_conversion(frames, energy_unit=energy_unit, length_unit=units.Angstrom,
                                            dipole_vec_unit=units.Debye),
            atomic_energy_shifts=shifts))

    return prep(all_data[:num_train]), prep(all_data[num_train:num_train + num_valid])


def _so3krates():
    return SO3kratesSparse(
        cutoff=5., num_features=32, num_features_head=8, num_heads=2,
        degrees=[1, 2], message_normalization='avg_num_neighbors', avg_num_neighbors=8.)


def _loss(model, energy=0.001, forces=0.999):
    return training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(model), weights={'energy': energy, 'forces': forces})


def _fit(ckpt_dir, model, loss_fn, data, **kwargs):
    """fit() on (training, validation) with the batch limits these tests share
    (two ethanol frames per batch)."""
    training_data, validation_data = data
    kwargs = {'optimizer': optax.adam(learning_rate=1e-3), 'num_epochs': 2,
              'eval_every_num_steps': 10, **kwargs}
    training_utils.fit(
        model=model, loss_fn=loss_fn, graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
        batch_max_num_edges=72 + 72 + 1, batch_max_num_nodes=9 + 9 + 1,
        batch_max_num_graphs=3, batch_max_num_pairs=100,
        training_data=training_data, validation_data=validation_data,
        ckpt_dir=ckpt_dir, allow_restart=False, use_wandb=False, **kwargs)


def test_fit_with_ema(tmp_path, ethanol_frames):
    """The plain (use_ema=False) path is what every other test here runs.
    Evaluating every 25 steps validates and saves the EMA weights twice."""
    model = _so3krates()
    _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(ethanol_frames),
         eval_every_num_steps=25, use_ema=True, ema_decay=0.99)
    # A checkpoint of the EMA weights should have been written.
    assert any((tmp_path / 'checkpoints').iterdir())


def test_fit_saves_the_final_weights(tmp_path, ethanol_frames):
    """fit() validates and saves the last step even when it is not a multiple of
    eval_every_num_steps: a run shorter than that used to write no checkpoint."""
    model = _so3krates()
    _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(ethanol_frames, 15, 5),
         num_epochs=1, eval_every_num_steps=10_000)
    assert any((tmp_path / 'checkpoints').iterdir())


def test_fit_early_stopping(tmp_path, caplog, ethanol_frames):
    """Early stopping halts training and logs once the validation loss stalls.

    The first evaluation (step 10) always improves on the initial best of inf.
    With an enormous min_delta no later one can, so the evaluations at steps 20
    and 30 exhaust ``patience=2`` and training stops at step 30, inside the 50
    steps the two epochs provide.
    """
    model = _so3krates()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(ethanol_frames),
             early_stopping_patience=2, early_stopping_min_delta=1e9)
    stops = [r.getMessage() for r in caplog.records if 'Early stopping' in r.getMessage()]
    # Exactly once: a loop that logged the stop but kept training would log it again.
    assert len(stops) == 1 and 'Early stopping at step 30' in stops[0], stops


def test_make_optimizer_plateau_knob():
    """``make_optimizer(reduce_lr_on_plateau=True)`` exposes a settable
    ``plateau_lr_scale`` hyperparameter; the default optimizer does not."""
    import optax.tree_utils as otu

    opt_plain = training_utils.make_optimizer(learning_rate=1e-3)
    opt_plateau = training_utils.make_optimizer(
        learning_rate=1e-3, reduce_lr_on_plateau=True)

    params = {'w': jax.numpy.ones((4,))}
    assert otu.tree_get(opt_plain.init(params), 'plateau_lr_scale') is None

    st = opt_plateau.init(params)
    assert otu.tree_get(st, 'plateau_lr_scale') == 1.0
    st = otu.tree_set(st, plateau_lr_scale=0.25)
    assert otu.tree_get(st, 'plateau_lr_scale') == 0.25


def test_fit_reduce_lr_on_plateau(tmp_path, caplog, ethanol_frames):
    """Reduce-LR-on-plateau fires when the validation loss never improves.

    As for early stopping: the step-10 evaluation sets the best loss, and with
    an enormous ``reduce_lr_min_delta`` the evaluations at steps 20 and 30 reach
    ``reduce_lr_patience=2``, so the learning-rate scale is halved at step 30.
    """
    model = _so3krates()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(ethanol_frames),
             # The optimizer must carry the injectable knob for plateau scaling to engage.
             optimizer=training_utils.make_optimizer(learning_rate=1e-3, reduce_lr_on_plateau=True),
             reduce_lr_on_plateau=True, reduce_lr_factor=0.5, reduce_lr_patience=2,
             reduce_lr_min_delta=1e9, reduce_lr_min_scale=1e-3)
    assert any('Reduce-LR-on-plateau' in r.getMessage() and 'scaling LR' in r.getMessage()
               and 'at step 30' in r.getMessage() for r in caplog.records)


def test_fit_two_stage_loss(tmp_path, caplog, ethanol_frames):
    """Two-stage loss reweighting swaps to the stage-two loss at the configured
    epoch and logs the switch. Epochs count from 0, so stage_two_start_epoch=1
    switches at the start of the second (and last) epoch."""
    model = _so3krates()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        # Stage one weights energy heavily; stage two flips to forces.
        _fit(tmp_path / 'checkpoints', model, _loss(model, energy=0.99, forces=0.01),
             _ethanol(ethanol_frames), stage_two_start_epoch=1, stage_two_loss_fn=_loss(model))
    assert any('switching to the stage-two loss at epoch 1' in r.getMessage()
               for r in caplog.records)
