from ase import units
import jax
import pytest
from so3lr.mlff.nn import SO3kratesSparse
from so3lr.mlff import training_utils
from so3lr.mlff import jraph_utils
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from so3lr.mlff.data import NpzDataLoaderSparse
from so3lr.mlff.data import transformations
import numpy as np
import optax
import pkg_resources


@pytest.mark.parametrize("use_ema", [False, True])
def test_fit(tmp_path, use_ema, disable_jit: bool = False):
    if disable_jit:
        from jax.config import config
        config.update('jax_disable_jit', True)

    filename = 'test_data/ethanol.npz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = NpzDataLoaderSparse(input_file=f)
    all_data, data_stats = loader.load(cutoff=5.)

    num_train = 50
    num_valid = 10

    energy_unit = units.kcal / units.mol
    length_unit = units.Angstrom
    dipole_vec_unit = units.Debye

    numpy_rng = np.random.RandomState(0)
    numpy_rng.shuffle(all_data)

    energy_mean = transformations.calculate_energy_mean(all_data[:num_train]) * energy_unit
    num_nodes = transformations.calculate_average_number_of_nodes(all_data[:num_train])
    energy_shifts = {a: energy_mean / num_nodes for a in range(119)}

    training_data = transformations.subtract_atomic_energy_shifts(
        transformations.unit_conversion(
            all_data[:num_train],
            energy_unit=energy_unit,
            length_unit=length_unit,
            dipole_vec_unit=dipole_vec_unit

        ),
        atomic_energy_shifts=energy_shifts
    )

    validation_data = transformations.subtract_atomic_energy_shifts(
        transformations.unit_conversion(
            all_data[num_train:(num_train + num_valid)],
            energy_unit=energy_unit,
            length_unit=length_unit,
            dipole_vec_unit=dipole_vec_unit
        ),
        atomic_energy_shifts=energy_shifts
    )

    opt = optax.adam(learning_rate=1e-3)

    so3k = SO3kratesSparse(
        cutoff=5.,
        num_features=32,
        num_features_head=8,
        num_heads=2,
        degrees=[1, 2],
        message_normalization='avg_num_neighbors',
        avg_num_neighbors=8.
    )

    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(so3k),
        weights={'energy': 0.001, 'forces': 0.999}
    )

    workdir = tmp_path / 'run_training_sparse'
    workdir.mkdir()

    training_utils.fit(
        model=so3k,
        optimizer=opt,
        loss_fn=loss_fn,
        graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
        num_epochs=10,
        batch_max_num_edges=72+72+1,
        batch_max_num_nodes=9+9+1,
        batch_max_num_graphs=3,
        eval_every_num_steps=75,
        batch_max_num_pairs=100,
        training_data=list(training_data),
        validation_data=list(validation_data),
        ckpt_dir=workdir / 'checkpoints',
        allow_restart=False,
        use_wandb=False,  # Otherwise the GitHub CI fails.
        use_ema=use_ema,
        ema_decay=0.99,
    )

    # A checkpoint should have been written (with EMA weights when use_ema=True).
    assert (workdir / 'checkpoints').exists()
    assert any((workdir / 'checkpoints').iterdir())



def test_fit_early_stopping(tmp_path, caplog):
    """Early stopping halts training and logs once the validation loss stalls.

    With an enormous min_delta the validation loss can never count as an
    improvement, so training must stop after ``patience`` evaluations.
    """
    import logging

    f = pkg_resources.resource_filename(__name__, 'test_data/ethanol.npz')
    loader = NpzDataLoaderSparse(input_file=f)
    all_data, _ = loader.load(cutoff=5.)

    energy_unit = units.kcal / units.mol
    np.random.RandomState(0).shuffle(all_data)
    num_train, num_valid = 50, 10
    energy_mean = transformations.calculate_energy_mean(all_data[:num_train]) * energy_unit
    num_nodes = transformations.calculate_average_number_of_nodes(all_data[:num_train])
    shifts = {a: energy_mean / num_nodes for a in range(119)}

    def prep(sl):
        return transformations.subtract_atomic_energy_shifts(
            transformations.unit_conversion(
                sl, energy_unit=energy_unit, length_unit=units.Angstrom,
                dipole_vec_unit=units.Debye),
            atomic_energy_shifts=shifts)

    training_data = list(prep(all_data[:num_train]))
    validation_data = list(prep(all_data[num_train:num_train + num_valid]))

    so3k = SO3kratesSparse(
        cutoff=5., num_features=32, num_features_head=8, num_heads=2,
        degrees=[1, 2], message_normalization='avg_num_neighbors', avg_num_neighbors=8.)
    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(so3k), weights={'energy': 0.001, 'forces': 0.999})

    workdir = tmp_path / 'run'
    workdir.mkdir()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        training_utils.fit(
            model=so3k, optimizer=optax.adam(learning_rate=1e-3), loss_fn=loss_fn,
            graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
            num_epochs=20, batch_max_num_edges=72 + 72 + 1, batch_max_num_nodes=9 + 9 + 1,
            batch_max_num_graphs=3, eval_every_num_steps=10, batch_max_num_pairs=100,
            training_data=training_data, validation_data=validation_data,
            ckpt_dir=workdir / 'checkpoints', allow_restart=False, use_wandb=False,
            early_stopping_patience=2, early_stopping_min_delta=1e9,
        )

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
    import logging

    f = pkg_resources.resource_filename(__name__, 'test_data/ethanol.npz')
    loader = NpzDataLoaderSparse(input_file=f)
    all_data, _ = loader.load(cutoff=5.)

    energy_unit = units.kcal / units.mol
    np.random.RandomState(0).shuffle(all_data)
    num_train, num_valid = 50, 10
    energy_mean = transformations.calculate_energy_mean(all_data[:num_train]) * energy_unit
    num_nodes = transformations.calculate_average_number_of_nodes(all_data[:num_train])
    shifts = {a: energy_mean / num_nodes for a in range(119)}

    def prep(sl):
        return transformations.subtract_atomic_energy_shifts(
            transformations.unit_conversion(
                sl, energy_unit=energy_unit, length_unit=units.Angstrom,
                dipole_vec_unit=units.Debye),
            atomic_energy_shifts=shifts)

    training_data = list(prep(all_data[:num_train]))
    validation_data = list(prep(all_data[num_train:num_train + num_valid]))

    so3k = SO3kratesSparse(
        cutoff=5., num_features=32, num_features_head=8, num_heads=2,
        degrees=[1, 2], message_normalization='avg_num_neighbors', avg_num_neighbors=8.)
    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(so3k), weights={'energy': 0.001, 'forces': 0.999})

    # Optimizer must carry the injectable knob for plateau scaling to engage.
    opt = training_utils.make_optimizer(learning_rate=1e-3, reduce_lr_on_plateau=True)

    workdir = tmp_path / 'run'
    workdir.mkdir()
    logging.getLogger("SO3LR").propagate = True
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        training_utils.fit(
            model=so3k, optimizer=opt, loss_fn=loss_fn,
            graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
            num_epochs=20, batch_max_num_edges=72 + 72 + 1, batch_max_num_nodes=9 + 9 + 1,
            batch_max_num_graphs=3, eval_every_num_steps=10, batch_max_num_pairs=100,
            training_data=training_data, validation_data=validation_data,
            ckpt_dir=workdir / 'checkpoints', allow_restart=False, use_wandb=False,
            reduce_lr_on_plateau=True, reduce_lr_factor=0.5, reduce_lr_patience=2,
            reduce_lr_min_delta=1e9, reduce_lr_min_scale=1e-3,
        )

    assert any('Reduce-LR-on-plateau' in r.getMessage() and 'scaling LR' in r.getMessage()
               for r in caplog.records)
