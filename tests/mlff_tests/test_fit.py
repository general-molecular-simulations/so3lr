import logging
from pathlib import Path

import jax
import numpy as np
import optax
from ase import units

from so3lr.mlff import jraph_utils, training_utils
from so3lr.mlff.data import NpzDataLoaderSparse, transformations
from so3lr.mlff.nn import SO3kratesSparse
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse


TEST_DIR = Path(__file__).parent


def _ethanol(num_train=50, num_valid=10):
    """Shuffled ethanol frames in eV/Angstrom with per-atom energy shifts removed:
    (training, validation)."""
    all_data, _ = NpzDataLoaderSparse(input_file=TEST_DIR / 'test_data/ethanol.npz').load(cutoff=5.)
    energy_unit = units.kcal / units.mol
    np.random.RandomState(0).shuffle(all_data)
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
    kwargs = {'optimizer': optax.adam(learning_rate=1e-3), 'num_epochs': 20,
              'eval_every_num_steps': 10, **kwargs}
    training_utils.fit(
        model=model, loss_fn=loss_fn, graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
        batch_max_num_edges=72 + 72 + 1, batch_max_num_nodes=9 + 9 + 1,
        batch_max_num_graphs=3, batch_max_num_pairs=100,
        training_data=training_data, validation_data=validation_data,
        ckpt_dir=ckpt_dir, allow_restart=False, use_wandb=False, **kwargs)


def test_fit_with_ema(tmp_path):
    """The plain (use_ema=False) path is what every other test here runs."""
    model = _so3krates()
    _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(),
         num_epochs=10, eval_every_num_steps=75, use_ema=True, ema_decay=0.99)
    # A checkpoint of the EMA weights should have been written.
    assert any((tmp_path / 'checkpoints').iterdir())


def test_fit_saves_the_final_weights(tmp_path):
    """fit() validates and saves the last step even when it is not a multiple of
    eval_every_num_steps: a run shorter than that used to write no checkpoint."""
    model = _so3krates()
    _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(15, 5),
         num_epochs=1, eval_every_num_steps=10_000)
    assert any((tmp_path / 'checkpoints').iterdir())


def test_fit_early_stopping(tmp_path, caplog):
    """Early stopping halts training and logs once the validation loss stalls.

    With an enormous min_delta the validation loss can never count as an
    improvement, so training must stop after ``patience`` evaluations.
    """
    model = _so3krates()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(),
             early_stopping_patience=2, early_stopping_min_delta=1e9)
    assert any('Early stopping at step' in r.getMessage() for r in caplog.records)


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


def test_fit_reduce_lr_on_plateau(tmp_path, caplog):
    """Reduce-LR-on-plateau fires when the validation loss never improves.

    With an enormous ``reduce_lr_min_delta`` no evaluation counts as an
    improvement, so after ``reduce_lr_patience`` evaluations the learning-rate
    scale must be shrunk (logged once it happens).
    """
    model = _so3krates()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        _fit(tmp_path / 'checkpoints', model, _loss(model), _ethanol(),
             # The optimizer must carry the injectable knob for plateau scaling to engage.
             optimizer=training_utils.make_optimizer(learning_rate=1e-3, reduce_lr_on_plateau=True),
             reduce_lr_on_plateau=True, reduce_lr_factor=0.5, reduce_lr_patience=2,
             reduce_lr_min_delta=1e9, reduce_lr_min_scale=1e-3)
    assert any('Reduce-LR-on-plateau' in r.getMessage() and 'scaling LR' in r.getMessage()
               for r in caplog.records)


def test_fit_two_stage_loss(tmp_path, caplog):
    """Two-stage loss reweighting swaps to the stage-two loss at the configured
    epoch and logs the switch."""
    model = _so3krates()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        # Stage one weights energy heavily; stage two flips to forces.
        _fit(tmp_path / 'checkpoints', model, _loss(model, energy=0.99, forces=0.01), _ethanol(),
             num_epochs=6, stage_two_start_epoch=3, stage_two_loss_fn=_loss(model))
    assert any('switching to the stage-two loss at epoch 3' in r.getMessage()
               for r in caplog.records)
