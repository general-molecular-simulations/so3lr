from ase import units
import jax
from so3lr.mlff.nn import SO3kratesSparse
from so3lr.mlff import training_utils
from so3lr.mlff import jraph_utils
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from so3lr.mlff.data import NpzDataLoaderSparse
from so3lr.mlff.data import transformations
import numpy as np
import optax
from pathlib import Path
import pkg_resources


def test_fit(disable_jit: bool = False):
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

    workdir = Path('_test_run_training_sparse').expanduser().absolute().resolve()
    workdir.mkdir(exist_ok=True)

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
        use_wandb=False  # Otherwise the GitHub CI fails.
    )


def test_remove_dirs():
    try:
        import shutil
        shutil.rmtree('_test_run_training_sparse')
        shutil.rmtree('wandb')
    except FileNotFoundError:
        pass

