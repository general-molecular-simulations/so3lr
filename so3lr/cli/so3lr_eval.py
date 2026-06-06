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
from so3lr.mlff.utils import jraph_utils, evaluation_utils
from so3lr.mlff.data import AseDataLoaderSparse
from pathlib import Path
import json
from collections import defaultdict

from ..jraph_utils import jraph_to_ase_atoms
from ..base_calculator import make_so3lr
from .so3lr_md import load_model, setup_logger

# Get logger
logger = logging.getLogger("SO3LR")

# Observable quantities that require output_intermediate_quantities from the model.
# All observables are per-atom (node-level) quantities.
OBSERVABLES = {
    'nn_energy', 'partial_charges', 'electrostatic_energy',
    'dispersion_energy', 'zbl_repulsion',
    'nlh_repulsion',  # clear name for the so3lr-s/m/l repulsion (== zbl_repulsion)
    'electrostatic_energy_undamped', 'dispersion_energy_undamped',
    'x',  # final-layer atom embeddings; saved to _x.npz, not extxyz
}

# Map common user-facing alias names to the canonical observable/target name.
# Applied at CLI parse time only (NOT added to OBSERVABLES itself, to avoid
# the same quantity being counted twice in downstream loops). Keep the table
# small and unambiguous; map each alias to a single canonical name.
NORMALIZE_TARGET = {
    # Hirshfeld / a0 volume-ratio label aliases (canonical name is
    # chosen later, per-model, based on legacy_so3lr_bool).
    'hirsh_ratios': 'hirshfeld_ratios',
    # Charges: extxyz files sometimes store this as 'charges'.
    'charges': 'partial_charges',
    # The s/m/l models call their short-range repulsion 'nlh_repulsion';
    # legacy models call it 'zbl_repulsion'. Both names stay in OBSERVABLES
    # so a missing-from-prediction key is still handled gracefully; we do
    # not collapse them here.
}


def process_predictions(
    save_to: Optional[str],
    graph_batch: Tuple[jraph.GraphsTuple, jraph.GraphsTuple],
    inputs: Dict[str, Any],
    output_prediction: Dict[str, Any]
) -> List:
    """
    Process predictions and prepare for saving to file.

    Parameters:
    -----------
    save_to : str or None
        File path where to save the predictions
    graph_batch : tuple of jraph.GraphsTuple
        Tuple of (main_graph, long_range_graph) batches
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

    # Unpack the graph batch tuple
    main_graph, _ = graph_batch

    # Add predictions to graph nodes and globals
    main_graph.nodes['forces_so3lr'] = output_prediction['forces']
    main_graph.globals['energy_so3lr'] = output_prediction['energy']
    main_graph.globals['dipole_vec_so3lr'] = output_prediction['dipole_vec']
    # The polarizability/volume-ratio head is named 'hirshfeld_ratios' for the
    # legacy 'so3lr' model and 'a0_ratios' for the new so3lr-s/m/l models.
    if 'hirshfeld_ratios' in output_prediction:
        main_graph.nodes['hirshfeld_ratios_so3lr'] = output_prediction['hirshfeld_ratios']
    if 'a0_ratios' in output_prediction:
        main_graph.nodes['a0_ratios_so3lr'] = output_prediction['a0_ratios']
    # c6_ratios is only emitted by the so3lr-s/m/l models; the legacy 'so3lr'
    # model has no c6 head, so guard against its absence.
    if 'c6_ratios' in output_prediction:
        main_graph.nodes['c6_ratios_so3lr'] = output_prediction['c6_ratios']

    # Remove non-standard global fields that don't have one row per graph
    # (residue_charge has shape (2,) per graph, residue_segments has shape
    # (num_atoms,) per graph), which causes jraph.unbatch_np to fail when
    # splitting globals evenly by the number of graphs.
    for field in ('residue_charge', 'residue_segments'):
        main_graph.globals.pop(field, None)

    # Add any observable quantities that are present in the predictions.
    # 'x' (atom embeddings) is excluded here — it's saved separately to an NPZ file.
    for obs_key in OBSERVABLES:
        if obs_key in output_prediction and obs_key != 'x':
            main_graph.nodes[f'{obs_key}_so3lr'] = output_prediction[obs_key]

    # Unbatch the graphs and filter out padding using jraph's standard unbatch
    unbatched_graphs = jraph.unbatch_np(main_graph)
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
        # The polarizability/volume-ratio prediction is named 'a0_ratios' (new
        # models) or 'hirshfeld_ratios' (legacy); datasets typically store the
        # label under 'hirshfeld_ratios'. Accept either label key for the ratio.
        label_target = target
        if target in ('a0_ratios', 'hirshfeld_ratios'):
            other = 'hirshfeld_ratios' if target == 'a0_ratios' else 'a0_ratios'
            if (target not in inputs or inputs.get(target) is None) and inputs.get(other) is not None:
                label_target = other

        # Skip targets that have no ground truth in the dataset (e.g., observables)
        if label_target not in inputs or inputs[label_target] is None:
            continue
        if target not in output_prediction:
            continue

        mask = assign_mask(label_target, inputs=inputs)
        y_true = inputs[label_target]
        y_pred = output_prediction[target]

        diff = np.asarray(y_pred) - np.asarray(y_true)

        diff_masked = diff[mask]
        abs_sum = np.abs(diff_masked).sum()
        sq_sum  = (diff_masked ** 2).sum()
        count = diff_masked.size

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
        log_file: Optional[str] = None,
        total_charge: Optional[int] = None,
        num_unpaired_electrons: Optional[int] = None,
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

    # Parse targets into a list (whitespace-tolerant, empty entries dropped).
    # Previously a comma+space ('forces, dipole_vec') silently produced
    # untrimmed ' dipole_vec' which failed the membership check downstream
    # and dropped the target. Validate against the known set and warn with
    # did-you-mean suggestions before any heavy work starts.
    target_list = [t.strip() for t in targets.split(',') if t.strip()]
    # Normalize user-typed alias names to canonical names BEFORE the
    # known-targets membership check, so `--targets charges,hirsh_ratios`
    # works just like `--targets partial_charges,hirshfeld_ratios`.
    _aliased = [(t, NORMALIZE_TARGET[t]) for t in target_list if t in NORMALIZE_TARGET]
    if _aliased:
        for _orig, _canon in _aliased:
            logger.warning(f"Target alias '{_orig}' -> '{_canon}' (using canonical name).")
        target_list = [NORMALIZE_TARGET.get(t, t) for t in target_list]
    _label_targets = {
        'forces', 'energy', 'dipole_vec',
        'hirshfeld_ratios', 'a0_ratios', 'c6_ratios',
        'stress',
    }
    _known_targets = _label_targets | OBSERVABLES
    import difflib as _difflib
    _unknown = [t for t in target_list if t not in _known_targets]
    if _unknown:
        msgs = []
        for t in _unknown:
            sugg = _difflib.get_close_matches(t, sorted(_known_targets), n=3, cutoff=0.6)
            msgs.append(f"  '{t}'" + (f"  (did you mean: {', '.join(sugg)})" if sugg else ''))
        raise ValueError(
            "Unknown --targets entries:\n" + "\n".join(msgs) +
            f"\nKnown labels: {sorted(_label_targets)}; "
            f"known observables: {sorted(OBSERVABLES)}."
        )
    logger.info(f"Evaluating targets: {', '.join(target_list)}")

    # Determine which targets are observables (need output_intermediate_quantities)
    observable_targets = [t for t in target_list if t in OBSERVABLES]
    output_intermediate_quantities = observable_targets if observable_targets else None
    if observable_targets:
        logger.info(f"Observable targets (no ground truth expected): {', '.join(observable_targets)}")

    # Embeddings ('x') are large and saved to a separate compressed NPZ file.
    save_x_bool = 'x' in target_list
    if save_x_bool:
        base_path = save_to if save_to is not None else datafile
        npz_path = Path(base_path).with_name(Path(base_path).stem + '_x.npz')
        all_x = []  # accumulates (n_atoms_i, features) per structure

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

    # Initialize the model. A bundled name (so3lr-s/m/l/so3lr) or a custom path;
    # None falls back to the default model.
    from so3lr.model_registry import resolve_model, available_models
    workdir_path, _, _ = resolve_model(model_path)
    if model_path is None or model_path in available_models():
        logger.info(f"Using bundled SO3LR model: {workdir_path.name}")
    else:
        logger.info(f"Using custom MLFF potential from: {model_path}")

    # Configure JAX process-globals (x64, matmul precision). Without this
    # call, `so3lr eval --precision float64` would silently downcast to
    # float32 because eval bypasses run() (where MD configures these).
    from .so3lr_md import setup_jax_globals
    setup_jax_globals(precision=precision, matmul_precision='highest')

    so3lr_calc = make_so3lr(
        workdir=workdir_path,
        lr_cutoff=lr_cutoff,
        dispersion_energy_cutoff_lr_damping=dispersion_damping,
        calculate_forces=True,
        output_intermediate_quantities=output_intermediate_quantities
    )

    # Read the short-range cutoff from the resolved model's hyperparameters.
    with open(workdir_path / "hyperparameters.json", "r") as fp:
        cfg = json.load(fp)
    cutoff = cfg['model']['cutoff']

    logger.info(f"Using cutoff: {cutoff} Å")

    # The ratio head is named 'hirshfeld_ratios' for the legacy model and
    # 'a0_ratios' for the new so3lr-s/m/l models. Normalize the requested target
    # to the model's output key so metric names (e.g. a0_ratios_rmse) match.
    legacy_model = bool(cfg['model'].get('legacy_so3lr_bool', False))
    ratio_key = 'hirshfeld_ratios' if legacy_model else 'a0_ratios'
    target_list = [ratio_key if t in ('hirshfeld_ratios', 'a0_ratios') else t for t in target_list]

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

    # Override per-structure total_charge / num_unpaired_electrons globals if
    # the user passed explicit CLI flags. Without this, the dataloader silently
    # defaults to neutral singlet (charge=0, multiplicity=1) for inputs whose
    # extxyz info{} lacks `charge=` / `multiplicity=`, so charged or open-shell
    # systems would be evaluated as neutral singlets without warning.
    if total_charge is not None or num_unpaired_electrons is not None:
        _override = {}
        if total_charge is not None:
            _override['total_charge'] = np.array(total_charge, dtype=np.int16).reshape(-1)
            logger.info(f"Overriding total_charge -> {total_charge} for all structures.")
        if num_unpaired_electrons is not None:
            _override['num_unpaired_electrons'] = np.array(num_unpaired_electrons, dtype=np.int16).reshape(-1)
            logger.info(f"Overriding num_unpaired_electrons -> {num_unpaired_electrons} for all structures.")
        _new_data = []
        for _graph, _lr_data in data:
            _new_globals = dict(_graph.globals)
            _new_globals.update(_override)
            _new_data.append((_graph._replace(globals=_new_globals), _lr_data))
        data = _new_data

#   # Check for allowed atom types
#   allowed_atomic_numbers = {1, 6, 7, 8, 9, 15, 16, 17}
#   for i, (graph, _) in enumerate(data):
#       if hasattr(graph, 'nodes') and isinstance(graph.nodes, dict) and 'atomic_numbers' in graph.nodes:
#            atomic_nums = graph.nodes['atomic_numbers']
#       elif isinstance(graph, dict) and 'nodes' in graph and 'atomic_numbers' in graph['nodes']:
#            atomic_nums = graph['nodes']['atomic_numbers']
#       else:
#            # Fallback or skip if structure is unclear, but better to be safe
#            continue
#            
#       current_atomic_numbers = {int(x) for x in atomic_nums}
#       if not current_atomic_numbers.issubset(allowed_atomic_numbers):
#            unsupported = current_atomic_numbers - allowed_atomic_numbers
#            logger.error(f"Structure {i} contains unsupported element(s): {unsupported}. "
#                         f"Supported elements are: H(1), C(6), N(7), O(8), F(9), P(15), S(16), Cl(17).")
#            raise ValueError(f"Unsupported elements in structure {i}: {unsupported}")

    # Determine padding sizes
    n_node = stats['max_num_of_nodes'] * batch_size + 1
    n_edge = stats['max_num_of_edges'] * batch_size + 1
    n_graph = batch_size + 1
    n_pairs = stats['max_num_of_pairs'] * batch_size + 1

    logger.info(f"Batch size: n_node={n_node}, n_edge={n_edge}, n_graph={n_graph}, n_pairs={n_pairs}")

    # Batch the graphs using MLFF's custom batch function that supports n_pairs
    batched_graphs = jraph_utils.dynamically_batch_with_lr(
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
        dummy_batched_graphs = jraph_utils.dynamically_batch_with_lr(
            data[:n_graph],
            n_node=n_node,
            n_edge=n_edge,
            n_graph=n_graph,
            n_pairs=n_pairs
        )

        logger.info('Starting JIT compilation...')
        compile_start = time.time()

        try:
            dummy_batch = next(dummy_batched_graphs)
            _compile_out = jax.block_until_ready(
                so3lr_calc(jraph_utils.graph_to_batch_fn(*dummy_batch))
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
        progress_bar = tqdm(total=num_data, desc="Processing structures", unit="structure",
                            smoothing=0.1, mininterval=0.5)

        for graph_batch in batched_graphs:
            # Transform the batched graph to inputs dict
            # graph_batch is a tuple of (main_graph, long_range_graph)
            inputs = jraph_utils.graph_to_batch_fn(*graph_batch)

            # Run the model
            start = time.time()
            output_prediction = jax.block_until_ready(so3lr_calc(inputs))
            end = time.time()
            total_time += end - start

            # Bulk-transfer outputs and inputs to numpy to avoid repeated syncs
            output_prediction = jax.device_get(output_prediction)
            inputs = jax.device_get(inputs)

            batch_size = int(inputs['num_of_non_padded_graphs'])
            total_num_structures += batch_size
            progress_bar.update(batch_size)

            # Accumulate atom embeddings per structure into all_x
            if save_x_bool and 'x' in output_prediction:
                node_mask = np.array(inputs['node_mask'], dtype=bool)
                x_real = output_prediction['x'][node_mask]  # (total_real_atoms, features)
                n_node_per_graph = graph_batch[0].n_node[:batch_size]
                splits = np.cumsum(n_node_per_graph[:-1]).tolist() if batch_size > 1 else []
                all_x.extend(np.split(x_real, splits))

            # Calculate batch metrics (sums & counts)
            batch_metrics = calculate_metrics(output_prediction, inputs, target_list)

            # Accumulate (only for targets that had ground truth)
            for t in target_list:
                if f"{t}_abs_sum" not in batch_metrics:
                    continue
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

        # Save embeddings to compressed NPZ
        if save_x_bool and all_x:
            x_concat = np.concatenate(all_x, axis=0)  # (total_atoms, features)
            n_atoms = np.array([xi.shape[0] for xi in all_x], dtype=np.int32)
            np.savez_compressed(npz_path, embeddings=x_concat, n_atoms=n_atoms)
            logger.info(f'Embeddings saved to: {npz_path}  '
                        f'(shape: {x_concat.shape}, {len(all_x)} structures)')

        logger.info('-' * 50)

        # Compute final metrics from totals (only for targets with ground truth)
        test_metrics = {}
        for t in target_list:
            if t not in tot_n:
                continue
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
    elif x == 'c6_ratios':
        return node_mask
    elif x in OBSERVABLES:
        return node_mask
    else:
        raise ValueError(f"Evaluation not implemented for target='{x}'.")