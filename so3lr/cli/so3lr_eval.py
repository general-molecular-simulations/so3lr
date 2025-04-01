"""Evaluation utilities for SO3LR models."""
import jax
import jraph
import numpy as np
import jax.numpy as jnp
import time
import pprint
import click

from ase.io import write
from mlff.utils import jraph_utils
from mlff.utils import evaluation_utils
from mlff.data import AseDataLoaderSparse
from pathlib import Path

from ..jraph_utils import jraph_to_ase_atoms
from ..jraph_utils import unbatch_np
from ..base_calculator import make_so3lr
from .so3lr_md import load_model


def evaluate_so3lr_on(
        datafile,
        batch_size=1,
        lr_cutoff=12.0,
        dispersion_energy_lr_cutoff_damping=2.0,
        jit_compile=True,
        save_predictions_to=None,
        model_path=None,
        precision='float32',
        targets='forces'
):
    """
    Evaluate SO3LR model on a dataset.
    
    Parameters:
    -----------
    datafile : str
        Data file to evaluate the model on. Must be readable by ase.io.read.
    batch_size : int
        Number of molecules per batch.
    lr_cutoff : float
        Long-range cutoff for SO3LR in Å.
    dispersion_energy_lr_cutoff_damping : float
        Damping of long-range start at lr_cutoff - this value.
    jit_compile : bool
        Whether to JIT compile the calculation.
    save_predictions_to : str or None
        File path where to save the predictions (.extxyz format). If None, predictions are not saved.
    model_path : str or None
        Path to the model to evaluate. If None, the default SO3LR model is used.
    precision : str
        Precision to use for the calculation.
    targets : list or None
        Targets to evaluate. If None, all targets are evaluated.
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and timing information.
    """

    total_time_start = time.time()

    # Define the targets.
    targets = targets.split(',')

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
    # Loading the model
    if model_path is None:
        print("Using default SO3LR potential.")
        so3lr_calc = make_so3lr(
            # dtype=precision,
            lr_cutoff=lr_cutoff,
            dispersion_energy_lr_cutoff_damping=dispersion_energy_lr_cutoff_damping,
            calculate_forces=True
        )
    else:
        print(f"Using custom MLFF potential from: {model_path}")
        so3lr_calc = load_model(
            model_path,
            precision= jnp.float64 if precision == 'float64' else jnp.float32,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_lr_cutoff_damping,
            # calculate_forces=True
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
        click.echo(f'Compilation completed after {compile_time:.3f} s.')
    else:
        compile_time = np.nan

    # Create a dictionary to track metrics
    test_metrics = {}
    for t in targets:
        for m in ('mae', 'mse'):
            test_metrics[f'{t}_{m}'] = []

    i = 0
    total_time = 0.
    total_num_structures = 0
    predicted = []

    log_every_t = max(1, num_data // 10)

    click.echo(f'Start evaluation on {num_data} structures.')
    for graph_batch in batched_graphs:

        if total_num_structures % log_every_t == 0:
            click.echo(f'Completed {total_num_structures} / {num_data} structures.')

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

        # Compute mean absolute error (MAE) and mean squared error (MSE).
        for t in targets:
            msk = assign_mask(t, inputs=inputs)

            mae = evaluation_utils.calculate_mae(
                y_predicted=output_prediction[t], y_true=inputs[t], msk=msk
            )
            mse = evaluation_utils.calculate_mse(
                y_predicted=output_prediction[t], y_true=inputs[t], msk=msk
            )
            
            test_metrics[f"{t}_mae"].append(mae)
            test_metrics[f"{t}_mse"].append(mse)

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

    click.echo(f'Completed {total_num_structures} / {num_data} structures.')
    click.echo(f'Time for evaluation = {(total_time):.3f} s.')

    # Compute final metrics
    for key in list(test_metrics.keys()):
        test_metrics[key] = np.mean(test_metrics[key])
    
    # Calculate RMSE from MSE
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

    metrics = jax.tree_util.tree_map(lambda x: float(x), metrics)

    metrics['datafile'] = str(datafile.as_uri())

    click.echo('Collected metrics (units are eV, Angstrom, and seconds):')
    formatted_metrics = {k: f"{v:.3e}" if isinstance(v, float) else v for k, v in metrics.items()}
    click.echo(pprint.pformat(formatted_metrics))

    if save_predictions_bool is True:
        click.echo(f'Start to save predictions to {str(save_predictions_to.as_uri())}.')

        for n, predicted_graph in enumerate(predicted):
            if n % log_every_t == 0:
                click.echo(f'Completed {n} / {len(predicted)}.')
            atoms = jraph_to_ase_atoms(predicted_graph)
            write(save_predictions_to, images=atoms, append=True)

        click.echo('Save completed.')

    total_time_end = time.time()
    click.echo(f'Total time = {(total_time_end - total_time_start):.3f} s.')
    click.echo('Evaluation completed.')
    
    return metrics


def assign_mask(x, inputs):
    """
    Assign the mask to the outputs.
    
    Parameters:
    -----------
    x : str
        Name of the output to mask.
    inputs : dict
        Input dictionary containing masks.
        
    Returns:
    --------
    msk : ndarray
        Mask for the specified output.
    """
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