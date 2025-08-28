"""Evaluation utilities for SO3LR models."""
import jax
import jraph
import numpy as np
import jax.numpy as jnp
import time
import pprint
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm

from ase.io import write
from mlff.utils import jraph_utils, evaluation_utils
from mlff.data import AseDataLoaderSparse
from pathlib import Path
import json
from collections import defaultdict

from ..jraph_utils import jraph_to_ase_atoms, unbatch_np
from ..base_calculator import make_so3lr
from .so3lr_md import load_model, setup_logger

# Get logger
logger = logging.getLogger("SO3LR")


def process_predictions(
    save_to: Optional[str],
    graph_batch: jraph.GraphsTuple,
    inputs: Dict[str, Any],
    output_prediction: Dict[str, Any]
) -> List:
    """
    Process predictions and prepare for saving to file.

    Parameters:
    -----------
    save_to : str or None
        File path where to save the predictions
    graph_batch : jraph.GraphsTuple
        The batch of graphs to process
    inputs : dict
        Input dictionary containing masks and data
    output_prediction : dict
        Model predictions

    Returns:
    --------
    List
        List of processed unbatched graphs
    """
    if save_to is None:
        return []

    # Add predictions to graph nodes and globals
    graph_batch.nodes['forces_so3lr'] = output_prediction['forces']
    graph_batch.nodes['hirshfeld_ratios_so3lr'] = output_prediction['hirshfeld_ratios']
    graph_batch.globals['energy_so3lr'] = output_prediction['energy']
    graph_batch.globals['dipole_vec_so3lr'] = output_prediction['dipole_vec']
    # graph_batch.nodes['c6_ratios_so3lr'] = output_prediction['c6_ratios']

    # Unbatch the graphs and filter out padding
    unbatched_graphs = unbatch_np(graph_batch)
    graph_mask = np.array(inputs['graph_mask'])
    return [x for x, cond in zip(unbatched_graphs, graph_mask) if (cond == True).all()]


def calculate_metrics(
    output_prediction: Dict[str, Any],
    inputs: Dict[str, Any],
    targets: List[str]
) -> Dict[str, float]:
    """
    Calculate MAE and MSE metrics for the specified targets.

    Parameters:
    -----------
    output_prediction : dict
        Model predictions
    inputs : dict
        Input dictionary containing masks and true values
    targets : list
        List of target names to evaluate

    Returns:
    --------
    dict
        Dictionary of calculated metrics
    """
    metrics = {}

    for target in targets:
        mask = assign_mask(target, inputs=inputs)
        y_pred = output_prediction[target]
        y_true = inputs[target]

        diff = jnp.asarray(y_pred) - jnp.asarray(y_true)

        abs_sum = jnp.abs(diff[mask]).sum()
        sq_sum  = (diff[mask] ** 2).sum()
        count   = jnp.asarray(mask).sum()

        # Store as Python scalars to avoid tracer issues later
        metrics[f"{target}_abs_sum"] = float(abs_sum)
        metrics[f"{target}_sq_sum"]  = float(sq_sum)
        metrics[f"{target}_count"]   = int(count)

    return metrics


def save_to_file(
    predicted_graphs: List,
    save_to: str,
    num_data: int
) -> None:
    """
    Save predictions to an extxyz file.

    Parameters:
    -----------
    predicted_graphs : list
        List of predicted graphs to save
    save_to : str
        File path to save predictions to
    num_data : int
        Total number of data points for progress reporting
    """
    logger.info(f'Saving predictions to {str(save_to)}')
    
    # Convert the length to Python int to avoid JAX array issues with tqdm
    total = len(predicted_graphs)
    
    # Use tqdm for progress reporting
    for predicted_graph in tqdm(predicted_graphs, desc="Saving predictions"):
        atoms = jraph_to_ase_atoms(predicted_graph)
        write(save_to, images=atoms, append=True)

    logger.info(f'Successfully saved predictions for {total} structures')


def evaluate_so3lr_on(
        datafile: str,
        batch_size: int = 1,
        lr_cutoff: float = 12.0,
        dispersion_damping: float = 2.0,
        jit_compile: bool = True,
        save_to: Optional[str] = None,
        model_path: Optional[str] = None,
        precision: str = 'float32',
        targets: str = 'forces,dipole_vec,hirshfeld_ratios',
        log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate SO3LR model on a dataset and calculate error metrics.

    This function loads a dataset, processes it in batches, runs the SO3LR model,
    and calculates metrics for the specified targets. It can optionally save the
    predictions to a file in the extxyz format.

    Parameters:
    -----------
    datafile : str
        Data file to evaluate the model on. Must be readable by ase.io.read.
    batch_size : int
        Number of molecules per batch.
    lr_cutoff : float
        Long-range cutoff for SO3LR in Å.
    dispersion_damping : float
        Damping of long-range interactions. Dispersion starts to switch off
        at (lr_cutoff - this value) Å.
    jit_compile : bool
        Whether to JIT compile the calculation.
    save_to : str or None
        File path where to save the predictions (.extxyz format). If None, predictions are not saved.
    model_path : str or None
        Path to the model to evaluate. If None, the default SO3LR model is used.
    precision : str
        Precision to use for the calculation ('float32' or 'float64').
    targets : str
        Comma-separated list of targets to evaluate (forces, energy, dipole_vec,
        hirshfeld_ratios).
    log_file : str or None
        Path to log file. If None, logging only goes to console.

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics (MAE, MSE, RMSE for each target)
        and timing information.
    """
    # Setup logging
    setup_logger(log_file)

    total_time_start = time.time()

    # Parse targets into a list
    target_list = targets.split(',')
    logger.info(f"Evaluating targets: {', '.join(target_list)}")

    # Process input file path
    datafile = Path(datafile).resolve().expanduser()
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")

    # Validate and process output file
    save_predictions_bool = save_to is not None
    if save_predictions_bool:
        save_to = Path(save_to).resolve().expanduser()
        if save_to.exists():
            raise RuntimeError(
                f'Output file already exists: {save_to}')

        if save_to.suffix not in ('.xyz', '.extxyz'):
            raise ValueError(
                f"Output file must have suffix `.xyz` or `.extxyz`. Received: {save_to.suffix}.")

        # Create parent directory if it doesn't exist
        save_to.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the model
    if model_path is None:
        logger.info("Using default SO3LR potential")
        #TODO: add precision handling
        so3lr_calc = make_so3lr(
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping,
            calculate_forces=True
        ) 
        cutoff = 4.5 # Default cutoff for SO3LR
    else:
        logger.info(f"Using custom MLFF potential from: {model_path}")
        so3lr_calc = make_so3lr(
            workdir=model_path,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping,
            calculate_forces=True
        )

        with open(Path(model_path) / "hyperparameters.json", "r") as fp:
            cfg = json.load(fp)
        
        cutoff = cfg['model']['cutoff']
    
    logger.info(f"Using cutoff: {cutoff} Å")

    # Load the data
    logger.info(f"Loading data from {datafile}")
    loader = AseDataLoaderSparse(datafile)
    num_data = loader.cardinality()
    logger.info(f"Found {num_data} structures in dataset")

    # Load the data with neighbor calculation
    try:
        data, stats = loader.load(
            cutoff=cutoff,
            cutoff_lr=lr_cutoff,
            calculate_neighbors_lr=True if lr_cutoff is not None else False,
            pick_idx=np.arange(num_data)
        )
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")

    # Determine padding sizes
    n_node = stats['max_num_of_nodes'] * batch_size + 1
    n_edge = stats['max_num_of_edges'] * batch_size + 1
    n_graph = batch_size + 1
    n_pairs = stats['max_num_of_nodes'] * (stats['max_num_of_nodes'] - 1) * batch_size + 1

    logger.info(f"Batch size: n_node={n_node}, n_edge={n_edge}, n_graph={n_graph}, n_pairs={n_pairs}")

    # Batch the graphs
    batched_graphs = jraph.dynamically_batch(
        data,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        n_pairs=n_pairs
    )

    # JIT compilation if requested
    if jit_compile:
        so3lr_calc = jax.jit(so3lr_calc)

        # Create a dummy batch for compilation
        dummy_batched_graphs = jraph.dynamically_batch(
            data[:n_graph],
            n_node=n_node,
            n_edge=n_edge,
            n_graph=n_graph,
            n_pairs=n_pairs
        )

        logger.info('Starting JIT compilation...')
        compile_start = time.time()

        try:
            _compile_out = jax.block_until_ready(
                so3lr_calc(jraph_utils.graph_to_batch_fn(
                    next(dummy_batched_graphs)))
            )
            compile_end = time.time()
            compile_time = compile_end - compile_start
            logger.info(f'Compilation completed in {compile_time:.3f} seconds')
        except Exception as e:
            raise RuntimeError(f"JIT compilation failed: {str(e)}")
    else:
        compile_time = np.nan

    # running totals for per-atom / per-structure aggregation
    tot_abs = defaultdict(float)  # per target absolute-error sum
    tot_sq  = defaultdict(float)  # per target squared-error sum
    tot_n   = defaultdict(int)    # per target masked count

    i = 0
    total_time = 0.0
    total_num_structures = 0
    predicted = []

    logger.info(f'Starting evaluation on {num_data} structures')
    logger.info('-' * 50)

    try:
        # Create a progress bar that shows structures processed instead of batches
        progress_bar = tqdm(total=num_data, desc="Processing structures", unit="structure")
        
        for graph_batch in batched_graphs:
            # Transform the batched graph to inputs dict
            inputs = jraph_utils.graph_to_batch_fn(graph_batch)
            batch_size = inputs['num_of_non_padded_graphs']
            
            # Update progress bar with actual batch size
            progress_bar.update(int(batch_size))
            total_num_structures += batch_size

            # Run the model
            start = time.time()
            output_prediction = jax.block_until_ready(so3lr_calc(inputs))
            end = time.time()
            total_time += end - start

            # Calculate batch metrics (sums & counts)
            batch_metrics = calculate_metrics(output_prediction, inputs, target_list)

            # Accumulate
            for t in target_list:
                tot_abs[t] += batch_metrics[f"{t}_abs_sum"]
                tot_sq[t]  += batch_metrics[f"{t}_sq_sum"]
                tot_n[t]   += batch_metrics[f"{t}_count"]

            # Process predictions if saving is enabled
            if save_predictions_bool:
                predicted.extend(process_predictions(
                    save_to, graph_batch, inputs, output_prediction
                ))

            i += 1
        
        # Close the progress bar
        progress_bar.close()
        
        logger.info(f'Completed evaluation on {total_num_structures} / {num_data} structures')
        logger.info(f'Pure evaluation time: {total_time:.3f} seconds')
        logger.info('-' * 50)

        # Compute final metrics from totals (global per-atom/per-structure)
        test_metrics = {}
        for t in target_list:
            n = tot_n[t]
            if n <= 0:
                # Avoid division by zero; report NaNs if nothing was counted
                mae = float('nan'); mse = float('nan'); rmse = float('nan')
            else:
                mae = tot_abs[t] / n
                mse = tot_sq[t]  / n
                rmse = float(np.sqrt(mse))
            test_metrics[f'{t}_mae']  = float(mae)
            test_metrics[f'{t}_mse']  = float(mse)
            test_metrics[f'{t}_rmse'] = float(rmse)

        # Compile timing metrics
        time_per_batch = total_time / i if i > 0 else 0
        time_per_structure = total_time / total_num_structures if total_num_structures > 0 else 0

        metrics = {
            'time_per_batch': float(time_per_batch),
            'time_per_structure': float(time_per_structure),
            'time_evaluation': float(total_time),
            'time_compile': float(compile_time),
            **test_metrics
        }

        # Convert numpy types to Python types for serialization
        metrics = jax.tree_util.tree_map(lambda x: float(x), metrics)
        metrics['datafile'] = str(datafile.as_uri())

        # Display results
        logger.info('Evaluation metrics (units are eV, Angstrom, and seconds):')
        formatted_metrics = {k: f"{v:.3e}" if isinstance(v, float) else v for k, v in metrics.items()}
        logger.info(pprint.pformat(formatted_metrics))
        logger.info('-' * 50)

        # Save predictions if requested
        if save_predictions_bool:
            save_to_file(predicted, str(save_to), num_data)

        total_time_end = time.time()
        logger.info(f'Total execution time: {(total_time_end - total_time_start):.3f} seconds')
        logger.info('Evaluation completed successfully')

        return metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def assign_mask(x: str, inputs: Dict[str, Any]) -> np.ndarray:
    """
    Assign the appropriate mask to the outputs based on the target.

    Parameters:
    -----------
    x : str
        Name of the output to mask.
    inputs : dict
        Input dictionary containing masks.

    Returns:
    --------
    np.ndarray
        Mask for the specified output.
    """
    node_mask = inputs['node_mask']
    graph_mask = inputs['graph_mask']

    if x == 'energy':
        return graph_mask
    elif x == 'forces':
        return node_mask
    elif x == 'stress':
        return graph_mask
    elif x == 'dipole_vec':
        return graph_mask
    elif x == 'hirshfeld_ratios':
        return node_mask
    # elif x == 'c6_ratios':
    #     return node_mask
    else:
        raise ValueError(f"Evaluation not implemented for target='{x}'.")