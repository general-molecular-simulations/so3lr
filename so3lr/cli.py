"""Command line interfaces."""
import click
import clu.metrics as clu_metrics
import itertools as it
import jax
import jraph
import numpy as np
import time
import pprint

from ase.io import write
from mlff.utils import jraph_utils
from mlff.utils import evaluation_utils
from mlff.data import AseDataLoaderSparse
from pathlib import Path

from .ascii_string import so3lr_ascii_II
from .jraph_utils import jraph_to_ase_atoms
from .jraph_utils import unbatch_np
from .base_calcuator import make_so3lr


@click.command()
@click.option(
    '--datafile', help='Data file to evaluate the model an. Must be ase.io.read digestible.'
)
@click.option(
    '--batch-size', default=1, help='Number of molecules per batch.'
)
@click.option(
    '--lr-cutoff', default=12., help='Long-range cutoff for SO3LR.'
)
@click.option(
    '--dispersion-energy-lr-cutoff-damping', default=2., help='Damping of long-range start at `--lr-cutoff` - $VALUE'
)
@click.option(
    '--jit-compile/--no-jit-compile', default=True, help='JIT compile the calculation.'
)
@click.option(
    '--save-predictions-to', default=None, help='File path where to save the predictions. Is saved as `extxyz`.'
)
def evaluate_so3lr_on(
        datafile: str,
        batch_size: int,
        lr_cutoff: float,
        dispersion_energy_lr_cutoff_damping: float,
        jit_compile: bool,
        save_predictions_to: str
) -> None:
    click.echo(so3lr_ascii_II)

    # Define the targets.
    targets = (
        'forces', 'dipole_vec', 'hirshfeld_ratios'
    )

    datafile = Path(datafile).resolve().expanduser()

    save_predictions_bool = save_predictions_to is not None

    if save_predictions_bool is True:
        save_predictions_to = Path(save_predictions_to).resolve().expanduser()
        if save_predictions_to.exists():
            raise RuntimeError(
                f'{save_predictions_to} already exists.'
            )

        if save_predictions_to.suffix != '.extxyz':
            raise ValueError(
                f"datafile must have suffix `.extxyz`. "
                f"Received {save_predictions_to.name} with {save_predictions_to.suffix}."
            )

    so3lr_calc = make_so3lr(
        lr_cutoff=lr_cutoff,
        dispersion_energy_lr_cutoff_damping=dispersion_energy_lr_cutoff_damping,
        calculate_forces=True
    )

    loader = AseDataLoaderSparse(
        datafile
    )

    num_data = loader.cardinality()

    data, stats = loader.load(
        cutoff=4.5,
        cutoff_lr=lr_cutoff,
        calculate_neighbors_lr=True if lr_cutoff is not None else False,
        pick_idx=np.arange(num_data)
    )

    # Determine the padding sizes from batch size.
    n_node = stats['max_num_of_nodes'] * batch_size + 1
    n_edge = stats['max_num_of_edges'] * batch_size + 1
    n_graph = batch_size + 1
    n_pairs = stats['max_num_of_nodes'] * (stats['max_num_of_nodes'] - 1) * batch_size + 1

    # Batch the graphs.
    batched_graphs = jraph.dynamically_batch(
        data,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        n_pairs=n_pairs
    )

    if jit_compile is True:
        so3lr_calc = jax.jit(so3lr_calc)

        # Create a dummy batch for jit compile.
        dummy_batched_graphs = jraph.dynamically_batch(
            data[:n_graph],
            n_node=n_node,
            n_edge=n_edge,
            n_graph=n_graph,
            n_pairs=n_pairs
        )

        click.echo('Start JIT compilation.')
        compile_start = time.time()

        _compile_out = jax.block_until_ready(
            so3lr_calc(
                jraph_utils.graph_to_batch_fn(next(dummy_batched_graphs))
            )
        )

        compile_end = time.time()
        compile_time = compile_end - compile_start
        click.echo(f'Compilation completed after {compile_time:.3f} s.\n\n')
    else:
        compile_time = np.nan

    # Create a collections object for the test targets.
    test_collection = clu_metrics.Collection.create(
        **{
            f'{t}_{m}': clu_metrics.Average.from_output(f'{t}_{m}') for (t, m) in
            it.product(targets, ('mae', 'mse'))}
    )

    i = 0
    total_time = 0.
    total_num_structures = 0
    predicted = []

    log_every_t = num_data // 10

    click.echo(f'Start evaluation on {num_data} structures.')
    for graph_batch in batched_graphs:

        if total_num_structures % log_every_t == 0:
            click.echo(f'Completed {total_num_structures} / {num_data} structures.')

        first_iteration_bool = i == 0

        # Transform the batched graph to inputs dict.
        inputs = jraph_utils.graph_to_batch_fn(
            graph_batch
        )
        total_num_structures += inputs['num_of_non_padded_graphs']

        start = time.time()

        output_prediction = jax.block_until_ready(
            so3lr_calc(inputs)
        )

        end = time.time()

        total_time += end - start

        metrics_dict = {}
        # Compute mean absolute error (MAE) and mean squared error (MSE).
        for t in targets:
            msk = assign_mask(t, inputs=inputs)

            metrics_dict[f"{t}_mae"] = evaluation_utils.calculate_mae(
                y_predicted=output_prediction[t], y_true=inputs[t], msk=msk
            )
            metrics_dict[f"{t}_mse"] = evaluation_utils.calculate_mse(
                y_predicted=output_prediction[t], y_true=inputs[t], msk=msk
            )

        if first_iteration_bool is True:
            test_metrics = test_collection.single_from_model_output(**metrics_dict)
        else:
            test_metrics = test_metrics.merge(test_collection.single_from_model_output(**metrics_dict))

        # Only executed if predictions are saved.
        if save_predictions_bool is True:
            graph_batch.nodes['forces_so3lr'] = output_prediction['forces']
            graph_batch.nodes['hirshfeld_ratios_so3lr'] = output_prediction['hirshfeld_ratios']
            graph_batch.globals['energy_so3lr'] = output_prediction['energy']
            graph_batch.globals['dipole_vec_so3lr'] = output_prediction['dipole_vec']

            unbatched_graphs = unbatch_np(graph_batch)
            graph_mask = np.array(inputs['graph_mask'])
            unbatched_graphs = [x for x, cond in zip(unbatched_graphs, graph_mask) if (cond == True).all()]

            predicted += unbatched_graphs

        i += 1

    click.echo(f'Completed {total_num_structures} / {num_data} structures.\n\n')

    # Post evaluation loop.
    test_metrics = test_metrics.compute()

    for t in targets:
        test_metrics[f'{t}_rmse'] = np.sqrt(test_metrics[f'{t}_mse'])

    time_per_batch = total_time / i
    time_per_structure = total_time / total_num_structures

    metrics = dict(
        time_per_batch=time_per_batch,
        time_per_structure=time_per_structure,
        time_evaluation=total_time,
        time_compile=compile_time,
        **test_metrics
    )

    metrics = jax.tree_map(lambda x: float(x), metrics)

    metrics['datafile'] = str(datafile.as_uri())

    click.echo('Collected metrics (units are eV and Angstrom):\n')
    pprint.pprint(metrics)
    click.echo('\n\n')

    if save_predictions_bool is True:
        click.echo(f'Start to save predictions to {str(save_predictions_to.as_uri())}.')

        for n, predicted_graph in enumerate(predicted):
            if n % log_every_t == 0:
                click.echo(f'Completed {n} / {len(predicted)}.')
            atoms = jraph_to_ase_atoms(predicted_graph)
            write(save_predictions_to, images=atoms, append=True)

        click.echo('Save completed.')

    click.echo('Evaluation completed.')



def assign_mask(x, inputs):
    # Assign the mask to the outputs.

    node_mask = inputs['node_mask']
    graph_mask = inputs['graph_mask']

    if x == 'energy':
        msk = graph_mask
    elif x == 'forces':
        msk = node_mask
    elif x == 'stress':
        msk = graph_mask
    elif x == 'dipole_vec':
        msk = graph_mask
    elif x == 'hirshfeld_ratios':
        msk = node_mask
    else:
        raise ValueError(
            f"Evaluate not implemented for target={x}."
        )

    return msk


if __name__ == '__main__':
    evaluate_so3lr_on()