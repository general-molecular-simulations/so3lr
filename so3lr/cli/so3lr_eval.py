"""Evaluation utilities for SO3LR models."""
from contextlib import contextmanager
import os
import stat
import tempfile
import jax
import jraph
import numpy as np
import jax.numpy as jnp
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

from ase import Atoms
from ase.io import write
from so3lr.mlff.utils import jraph_utils
from so3lr.mlff.utils.kspace_utils import (
    DEFAULT_KSPACE_DISPERSION_INTERP_NODES, DEFAULT_KSPACE_DISPERSION_SMEARING,
    DEFAULT_KSPACE_DISPERSION_SPACING, DEFAULT_KSPACE_INTERP_NODES,
    DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING, kspace_grid_shape,
    setup_kspace_grid)
from so3lr.mlff.data import AseDataLoaderSparse
from pathlib import Path
from collections import defaultdict

from ..jraph_utils import jraph_to_ase_atoms
from .presentation import get_reporter, progress_counter, reading_progress
from ..base_calculator import make_so3lr
from ..mlff.calculators.potential import cast_float_params
from ..mlff.nn.observable.dispersion_ref_data import (
    check_atomic_numbers_supported,
    supported_atomic_numbers_for_config,
)
from .so3lr_md import run_dtypes, setup_logger, setup_run
from ..pbc import is_fully_periodic

# Get logger
logger = logging.getLogger("SO3LR")

# Observable quantities that require output_intermediate_quantities from the model.
# All observables are per-atom (node-level) quantities.
OBSERVABLES = {
    'nn_energy', 'partial_charges', 'electrostatic_energy',
    'dispersion_energy', 'zbl_repulsion',
    'nlh_repulsion',  # clear name for the so3lr-tauri-s/m/l repulsion (== zbl_repulsion)
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
    graph_batch: Tuple[jraph.GraphsTuple, jraph.GraphsTuple],
    inputs: Dict[str, Any],
    output_prediction: Dict[str, Any]
) -> List:
    """
    Process predictions and prepare for saving to file.

    Parameters:
    -----------
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

    # Unpack the graph batch tuple
    main_graph, _ = graph_batch

    # Add predictions to graph nodes and globals
    main_graph.nodes['forces_so3lr'] = output_prediction['forces']
    main_graph.globals['energy_so3lr'] = output_prediction['energy']
    main_graph.globals['dipole_vec_so3lr'] = output_prediction['dipole_vec']
    # The polarizability/volume-ratio head is named 'hirshfeld_ratios' for the
    # legacy 'so3lr' model and 'a0_ratios' for the new so3lr-tauri-s/m/l models.
    if 'hirshfeld_ratios' in output_prediction:
        main_graph.nodes['hirshfeld_ratios_so3lr'] = output_prediction['hirshfeld_ratios']
    if 'a0_ratios' in output_prediction:
        main_graph.nodes['a0_ratios_so3lr'] = output_prediction['a0_ratios']
    # c6_ratios is only emitted by the so3lr-tauri-s/m/l models; the legacy 'so3lr'
    # model has no c6 head, so guard against its absence.
    if 'c6_ratios' in output_prediction:
        main_graph.nodes['c6_ratios_so3lr'] = output_prediction['c6_ratios']

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


def _check_output_destination(destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not destination.is_file():
        raise RuntimeError(f"Output destination is not a regular file: {destination}")
    if destination.exists() and not overwrite:
        raise RuntimeError(
            f"Output file already exists: {destination}. Refusing to overwrite. "
            "Pass --overwrite or choose a different --save-to path."
        )


@contextmanager
def _staged_output(destination: Path, *, overwrite: bool):
    """Replace one complete file; this is not a multi-file transaction."""
    destination = Path(os.path.realpath(destination))
    _check_output_destination(destination, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.so3lr-eval-', dir=destination.parent) as folder:
        temporary = Path(folder) / destination.name
        yield temporary
        _check_output_destination(destination, overwrite=overwrite)
        if destination.exists():
            temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
        os.replace(temporary, destination)


def save_to_file(
    predicted_graphs: List,
    save_to: str,
    num_data: int,
    *,
    overwrite: bool = True,
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
    get_reporter().phase('Saving', 'Writing predictions')
    
    total = len(predicted_graphs)
    if total == 0:
        raise ValueError("Cannot save an empty prediction collection.")
    progress = progress_counter(total, 'structures written', 'Saving predictions')

    # Label keys that the dataloader fills with NaN when the source extxyz
    # info{} did not actually carry that ground-truth value. The real model
    # outputs live in the matching `_so3lr` keys, so writing NaN labels next
    # to them produces confusing `energy=nan` / `dipole_vec="nan nan nan"`
    # columns in the saved file. Strip a label key only if its stored value
    # is fully non-finite (preserves real labels for evaluations on labelled
    # data).
    _label_keys_to_check = ('energy', 'dipole_vec')

    def _is_all_nan(v):
        try:
            return bool(np.all(~np.isfinite(np.asarray(v, dtype=float))))
        except (TypeError, ValueError):
            return False

    with _staged_output(Path(save_to), overwrite=overwrite) as temporary:
        for index, predicted_graph in enumerate(predicted_graphs):
            progress.update()
            # graphs from the neighbour-list model, Atoms from the flash engine
            atoms = predicted_graph if isinstance(predicted_graph, Atoms) else jraph_to_ase_atoms(predicted_graph)
            for _k in _label_keys_to_check:
                if _k in atoms.info and _is_all_nan(atoms.info[_k]):
                    del atoms.info[_k]
            write(temporary, images=atoms, format='extxyz', append=index > 0)
    progress.close()

    logger.info(f'Successfully saved predictions for {total} structures')


def evaluate_so3lr_on(
        datafile: str,
        batch_size: int = 1,
        lr_cutoff: float = 12.0,
        dispersion_damping: float = 2.0,
        jit_compile: bool = True,
        save_to: Optional[str] = None,
        model_path: Optional[str] = None,
        precision: Optional[str] = None,
        targets: str = 'forces,dipole_vec,hirshfeld_ratios',
        log_file: Optional[str] = None,
        total_charge: Optional[int] = None,
        num_unpaired_electrons: Optional[int] = None,
        overwrite: bool = False,
        kspace_electrostatics: Optional[str] = None,
        kspace_smearing: float = DEFAULT_KSPACE_SMEARING,
        kspace_spacing: float = DEFAULT_KSPACE_SPACING,
        kspace_interp_nodes: int = DEFAULT_KSPACE_INTERP_NODES,
        kspace_dispersion: Optional[str] = None,
        kspace_dispersion_smearing: float = DEFAULT_KSPACE_DISPERSION_SMEARING,
        kspace_dispersion_spacing: float = DEFAULT_KSPACE_DISPERSION_SPACING,
        kspace_dispersion_interp_nodes: int = DEFAULT_KSPACE_DISPERSION_INTERP_NODES,
        charge_embed_total_charge: Optional[float] = None,
        charge_equilibration_total_charge: Optional[float] = None,
        engine: str = 'auto',
        flash_mode: str = 'auto',
        flash_mixed: Optional[bool] = None,
        flash_devices: int = 1,
        flash_kernels: str = 'auto',
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
        The run's precision, on both engines as in MD: 'float32' or 'float64' throughout,
        or 'mixed' (the default), a float32 model around float64 positions; displacements
        are formed in the positions' dtype before the model.
    targets : str
        Comma-separated list of targets to evaluate (forces, energy, dipole_vec,
        hirshfeld_ratios).
    log_file : str or None
        Path to log file. If None, logging only goes to console.
    overwrite : bool
        Replace existing predictions and/or embedding archives only when True.
        Otherwise a pre-existing output raises RuntimeError. Each file is
        replaced separately; the outputs are not a multi-file transaction.
        Matches the --overwrite policy used by perform_md / perform_min.
    engine : str
        'auto': SO3LR-Flash where it computes what is asked (every structure periodic
        and orthorhombic or without a box, the targets energy and/or forces, up to
        so3lr.flash.check.EVAL_AUTO_ENGINES sets of atoms; see so3lr.flash.evaluate),
        else the neighbour-list model; 'flash' or 'standard' forces one. flash_mode and
        flash_devices as for MD; flash_mixed is the deprecated older form of precision.

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics (MAE, MSE, RMSE for each target)
        and timing information.
    """
    # every option, for the flash engine's check (which declines one it does not know)
    flash_settings = dict(locals())

    # Setup logging
    setup_logger(log_file)
    # Configure JAX process-globals (x64, matmul precision) before any array: eval bypasses
    # run(), where MD configures them
    setup_run(flash_settings)

    total_time_start = time.time()
    reporter = get_reporter()

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

    # Process input file path
    datafile = Path(datafile).expanduser().resolve()
    if not datafile.exists():
        raise FileNotFoundError(f"Data file not found: {datafile}")

    # Validate and process output file. Mirrors the --overwrite policy used
    # by perform_md / perform_min so the CLI and direct-Python entry points
    # behave the same way: refuse to clobber by default, opt in explicitly.
    save_predictions_bool = save_to is not None
    if save_predictions_bool:
        save_to = Path(save_to).expanduser().absolute()
        _check_output_destination(save_to, overwrite=overwrite)

        if save_to.suffix not in ('.xyz', '.extxyz'):
            raise ValueError(
                f"Output file must have suffix `.xyz` or `.extxyz`. Received: {save_to.suffix}.")

        # Create parent directory if it doesn't exist
        save_to.parent.mkdir(parents=True, exist_ok=True)

    # Embeddings use the same explicit overwrite policy as predictions. With
    # no --save-to they live beside the input; check before loading the model.
    save_x_bool = 'x' in target_list
    if save_x_bool:
        base_path = save_to if save_to is not None else datafile
        npz_path = base_path.with_name(base_path.stem + '_x.npz')
        _check_output_destination(npz_path, overwrite=overwrite)
        all_x = []

    # Initialize the model. A registered name (see model_registry) or a custom
    # path; None falls back to the default model.
    from so3lr.model_registry import (
        resolve_model, available_models, canonical_models, DEFAULT_MODEL
    )
    workdir_path = resolve_model(model_path).dir
    # Decide bundled vs. custom by comparing the *resolved* directory against
    # the registry's bundled paths, not by short-name membership of the raw
    # input. This way a user who passes an explicit filesystem path that
    # happens to point at a bundled checkpoint directory is correctly logged
    # as 'bundled' (the old name-only check mislabelled this as custom).
    # Iterate canonical names only: they cover every bundled directory, and
    # resolving the floating aliases here would log a redirect line for each.
    _bundled_paths = {resolve_model(n).dir for n in canonical_models()}
    if workdir_path in _bundled_paths:
        # Prefer the user-typed short name when it is a registry key, so a
        # floating alias logs as typed rather than as its target's directory.
        if model_path is not None and model_path in available_models():
            _name = model_path
        elif model_path is None:
            _name = DEFAULT_MODEL
        else:
            _name = workdir_path.name
        logger.info(f"Using bundled SO3LR model: {_name}")
    else:
        logger.info(f"Using custom MLFF potential from: {model_path}")

    # Dimer files report binding energies: one predicted energy per structure is kept for them.
    references = _dimer_pairs(datafile)

    kspace_enabled = (kspace_electrostatics is not None
                      or kspace_dispersion is not None)

    # The engine: SO3LR-Flash where it computes what is asked, else the neighbour-list
    # model below (engine='flash' / 'standard' forces one). The structures are read for
    # it only if the settings, model and device allow it.
    from .so3lr_md import run_on_flash
    from ..flash.check import flash_problem
    flash_settings['targets'] = target_list
    frames = []
    if (engine or 'auto') != 'standard' and flash_problem(flash_settings, [], task='eval') is None:
        # the first structure, then the rest only if flash takes it, and with auto only up
        # to one set of atoms more than it takes (most gas-phase datasets have many: the
        # loader below reads them)
        from ase.io import iread, read
        from ..flash.check import EVAL_AUTO_ENGINES
        frames = [read(str(datafile), index=0)]
        if flash_problem(flash_settings, frames, task='eval') is None:
            frames, kinds = [], set()
            for atoms in iread(str(datafile), index=':'):
                frames.append(atoms)
                kinds.add(atoms.numbers.tobytes())
                if (engine or 'auto') == 'auto' and len(kinds) > EVAL_AUTO_ENGINES:
                    break

    def eval_on_flash(flash_md):
        from ..flash.evaluate import evaluate_flash
        reporter.update_info(plan=f'{len(frames):,} structures')
        reporter.phase('Evaluating', 'SO3LR-Flash')
        # the frames are in memory already; their predictions carry the binding energies
        return evaluate_flash(frames, flash_settings, target_list,
                              keep_predictions=save_predictions_bool or references is not None)
    ran, result = run_on_flash(flash_settings, frames, 'eval', eval_on_flash)
    if ran:
        (tot_abs, tot_sq, tot_n), (t_first, t_rest, n_rest) = result
        logger.info(f'Completed evaluation on {len(frames)} / {len(frames)} structures (flash)')
        # the compile time includes each engine's first call, which cannot be timed apart
        logger.info(f'Engine builds and the calls that compiled: {t_first:.3f} seconds; '
                    f'the other {n_rest} calls: {t_rest:.3f} seconds')
        logger.info('-' * 50)
        return _finish_evaluation(
            target_list=target_list, tot_abs=tot_abs, tot_sq=tot_sq, tot_n=tot_n, total_time=t_rest,
            compile_time=t_first, n_batches=n_rest, n_structures=n_rest, datafile=datafile, save_to=save_to,
            predicted=frames, overwrite=overwrite, total_time_start=total_time_start,
            references=references,
            energies=[float(atoms.info['energy_so3lr']) for atoms in frames] if references else None)
    del frames

    reporter.phase('Loading', 'Loading the force field')
    # the model's and the geometry's dtypes for `precision`, as on flash and in MD
    model_dtype, state_dtype = run_dtypes(flash_settings)
    model_fn = make_so3lr(
        precision=model_dtype,
        workdir=workdir_path,
        kspace_electrostatics=kspace_electrostatics,
        kspace_interp_nodes=kspace_interp_nodes,
        kspace_dispersion=kspace_dispersion,
        kspace_dispersion_interp_nodes=kspace_dispersion_interp_nodes,
        lr_cutoff=lr_cutoff,
        dispersion_energy_cutoff_lr_damping=dispersion_damping,
        calculate_forces=True,
        output_intermediate_quantities=output_intermediate_quantities,
        charge_embed_total_charge=charge_embed_total_charge,
        charge_equilibration_total_charge=charge_equilibration_total_charge
    )

    def so3lr_calc(inputs):
        return model_fn(cast_float_params(inputs, state_dtype))

    # Read the short-range cutoff from the resolved model's hyperparameters.
    from ..model_config import load_model_config
    cfg = load_model_config(workdir_path)
    cutoff = cfg['model']['cutoff']

    logger.info(f"Using cutoff: {cutoff} Å")

    # The ratio head is named 'hirshfeld_ratios' for the legacy model and
    # 'a0_ratios' for the new so3lr-tauri-s/m/l models. Normalize the requested target
    # to the model's output key so metric names (e.g. a0_ratios_rmse) match.
    legacy_model = bool(cfg['model'].get('legacy_so3lr_bool', False))
    supported_z = supported_atomic_numbers_for_config(cfg['model'])
    ratio_key = 'hirshfeld_ratios' if legacy_model else 'a0_ratios'
    _other_ratio = 'a0_ratios' if ratio_key == 'hirshfeld_ratios' else 'hirshfeld_ratios'
    if _other_ratio in target_list:
        logger.info(
            f"Target '{_other_ratio}' renamed to '{ratio_key}' to match the "
            f"{'legacy ' if legacy_model else ''}model's ratio head output key. "
            f"Dataset labels under either key will still be matched."
        )
    target_list = [ratio_key if t in ('hirshfeld_ratios', 'a0_ratios') else t for t in target_list]

    # Load the data
    logger.info(f"Loading data from {datafile}")
    reporter.phase('Loading', 'Reading the dataset and building neighbour lists')
    loader = AseDataLoaderSparse(datafile)
    num_data = loader.cardinality()
    logger.info(f"Found {num_data} structures in dataset")
    reporter.update_info(plan=f'{num_data:,} structures · batch size {batch_size}')

    # Load the data with neighbor calculation
    try:
        data, stats = loader.load(
            cutoff=cutoff,
            cutoff_lr=lr_cutoff,
            calculate_neighbors_lr=True if lr_cutoff is not None else False,
            pick_idx=np.arange(num_data),
            progress=reading_progress(num_data),
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

    # Refuse elements this model was never trained on, before any forward pass.
    # The set is model-dependent: legacy v1 saw only the eight biomolecular
    # elements, the v2 models Z = 1..85. Past that the reference tables are
    # indexed out of bounds inside jit and JAX silently returns a value
    # belonging to some other element rather than raising.
    for i, (graph, _) in enumerate(data):
        nodes = graph['nodes'] if isinstance(graph, dict) else graph.nodes
        atomic_nums = nodes.get('atomic_numbers') if isinstance(nodes, dict) else None
        if atomic_nums is None:
            continue
        check_atomic_numbers_supported(atomic_nums, supported_z,
                                       context=f'{datafile} structure {i}')

    # Determine padding sizes
    n_node = stats['max_num_of_nodes'] * batch_size + 1
    n_edge = stats['max_num_of_edges'] * batch_size + 1
    n_graph = batch_size + 1
    n_pairs = stats['max_num_of_pairs'] * batch_size + 1

    logger.info(f"Batch size: n_node={n_node}, n_edge={n_edge}, n_graph={n_graph}, n_pairs={n_pairs}")

    # --- K-space (Ewald/PME/P3M) long-range ---------------------------------
    # The real-space neighbour list already sums over all images; the
    # reciprocal-space term converges the Coulomb tail, which is what makes
    # `so3lr eval` agree with the ASE calculator on periodic systems.
    #
    # A compiled step needs a fixed mesh shape, and the k-grid arrays carry
    # nothing but that shape (each graph's own cell supplies the geometry), so
    # structures are grouped by mesh shape: one compilation per distinct shape,
    # and structures with different cells but the same mesh share a batch.
    # Outputs are put back into dataset order afterwards.
    if kspace_enabled:
        buckets = {}                      # mesh-shape key -> [dataset index]
        for idx, (_g, _) in enumerate(data):
            cell = (_g.globals or {}).get('cell')
            if cell is None or not np.any(cell):
                raise ValueError(
                    "--kspace-electrostatics/--kspace-dispersion require periodic "
                    "structures, but a structure in the dataset has no cell. "
                    "Drop the k-space flags for non-periodic data."
                )
            pbc = (_g.globals or {}).get('pbc')
            if pbc is None:
                raise ValueError(
                    "K-space evaluation requires explicit PBC metadata for every "
                    "structure; reload the data through the current ASE loader."
                )
            is_fully_periodic(pbc, context=f"K-space structure {idx}")
            cell = np.asarray(cell).reshape(3, 3)
            key = (kspace_grid_shape(kspace_electrostatics, kspace_spacing, cell),
                   kspace_grid_shape(kspace_dispersion, kspace_dispersion_spacing, cell))
            buckets.setdefault(key, []).append(idx)
        logger.info(
            f"K-space enabled: electrostatics={kspace_electrostatics} "
            f"(smearing={kspace_smearing} A, spacing={kspace_spacing} A), "
            f"dispersion={kspace_dispersion}. "
            f"{len(buckets)} distinct mesh shape(s) -> {len(buckets)} compilation(s).")

        groups = []
        for idx_list in buckets.values():
            box = jnp.asarray(data[idx_list[0]][0].globals['cell']).reshape(3, 3)
            k_grid, k_smear = setup_kspace_grid(
                kspace_electrostatics, kspace_smearing, kspace_spacing, box,
                dtype=state_dtype)
            k_grid_d, k_smear_d = setup_kspace_grid(
                kspace_dispersion, kspace_dispersion_smearing,
                kspace_dispersion_spacing, box, dtype=state_dtype)
            groups.append(([data[i] for i in idx_list],
                           {'k_grid': k_grid, 'k_smearing': k_smear,
                            'k_grid_disp': k_grid_d, 'k_smearing_disp': k_smear_d}))
        order = [i for idx_list in buckets.values() for i in idx_list]
    else:
        groups = [(data, None)]
        order = None

    # Batch the graphs using MLFF's custom batch function that supports n_pairs.
    # Without k-space this is a single group with no constants.
    batched_graphs = (
        (batch, kconst)
        for subset, kconst in groups
        for batch in jraph_utils.dynamically_batch_with_lr(
            subset, n_node=n_node, n_edge=n_edge, n_graph=n_graph, n_pairs=n_pairs))

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
        reporter.phase('Compiling', 'Compiling the force field for the first batch')
        compile_start = time.time()

        try:
            dummy_batch = next(dummy_batched_graphs)
            _compile_out = jax.block_until_ready(
                so3lr_calc(jraph_utils.graph_to_batch_fn(
                    *dummy_batch, kspace_constants=groups[0][1]))
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
    energies = []  # per structure, for a dimer file's binding energies

    logger.info(f'Starting evaluation on {num_data} structures')
    logger.info('-' * 50)

    try:
        # Create a progress bar that shows structures processed instead of batches
        reporter.phase('Evaluating')
        progress_bar = progress_counter(num_data, 'structures', 'Processing structures')

        for graph_batch, kspace_constants in batched_graphs:
            # Transform the batched graph to inputs dict
            # graph_batch is a tuple of (main_graph, long_range_graph)
            inputs = jraph_utils.graph_to_batch_fn(
                *graph_batch, kspace_constants=kspace_constants)

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
                predicted.extend(process_predictions(graph_batch, inputs, output_prediction))
            if references is not None:
                real = np.asarray(inputs['graph_mask'], dtype=bool)
                energies.extend(np.asarray(output_prediction['energy']).reshape(-1)[real].tolist())

            i += 1
        
        # Close the progress bar
        progress_bar.close()

        # Grouping reordered the stream; restore dataset order for the
        # per-structure outputs (metrics are order-independent sums).
        if order is not None:
            inverse = np.argsort(order)
            if save_predictions_bool:
                predicted = [predicted[i] for i in inverse]
            if references is not None:
                energies = [energies[i] for i in inverse]
            if save_x_bool:
                all_x = [all_x[i] for i in inverse]
        
        logger.info(f'Completed evaluation on {total_num_structures} / {num_data} structures')
        logger.info(f'Pure evaluation time: {total_time:.3f} seconds')

        # Save embeddings to compressed NPZ
        if save_x_bool and all_x:
            x_concat = np.concatenate(all_x, axis=0)  # (total_atoms, features)
            n_atoms = np.array([xi.shape[0] for xi in all_x], dtype=np.int32)
            with _staged_output(npz_path, overwrite=overwrite) as temporary:
                np.savez_compressed(temporary, embeddings=x_concat, n_atoms=n_atoms)
            logger.info(f'Embeddings saved to: {npz_path}  '
                        f'(shape: {x_concat.shape}, {len(all_x)} structures)')

        logger.info('-' * 50)

        return _finish_evaluation(
            target_list=target_list, tot_abs=tot_abs, tot_sq=tot_sq, tot_n=tot_n, total_time=total_time,
            compile_time=compile_time, n_batches=i, n_structures=total_num_structures, datafile=datafile,
            save_to=save_to, predicted=predicted, overwrite=overwrite, total_time_start=total_time_start,
            references=references, energies=energies)

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def _finish_evaluation(*, target_list, tot_abs, tot_sq, tot_n, total_time, compile_time, n_batches,
                       n_structures, datafile, save_to, predicted, overwrite, total_time_start,
                       references=None, energies=None):
    """The metrics from the per-target sums, the report, and the predictions saved (if
    save_to): the end of ``evaluate_so3lr_on`` for either engine. total_time: seconds of
    the n_batches timed calls, over n_structures structures."""
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
    time_per_batch = total_time / n_batches if n_batches > 0 else 0
    time_per_structure = total_time / n_structures if n_structures > 0 else 0

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

    # Display results: a per-target table (MAE / RMSE / N + unit) is much
    # easier to read than a flat pprint dict where *_mae / *_mse / *_rmse
    # keys interleave with timing fields. The returned `metrics` dict is
    # unchanged so downstream callers keep working.
    _UNIT_MAP = {
        'energy': 'eV',
        'forces': 'eV/Ang',
        'dipole_vec': 'e*Ang',
        'stress': 'eV/Ang^3',
        'hirshfeld_ratios': 'dimensionless',
        'a0_ratios': 'dimensionless',
        'c6_ratios': 'dimensionless',
    }
    # Targets that actually have ground-truth counts (skip observables
    # which have no MAE/RMSE because no label is available).
    # A target whose labels were all missing (NaN-filled) has no meaningful error.
    _scored_targets = [t for t in target_list if t in tot_n and tot_n[t] > 0
                       and np.isfinite(metrics.get(f'{t}_mae', float('nan')))]
    logger.info('Evaluation metrics:')
    if _scored_targets:
        _hdr = f"  {'target':<20}  {'MAE':>12}  {'RMSE':>12}  {'N':>10}  unit"
        logger.info(_hdr)
        logger.info('  ' + '-' * (len(_hdr) - 2))
        for t in _scored_targets:
            mae = metrics.get(f'{t}_mae', float('nan'))
            rmse = metrics.get(f'{t}_rmse', float('nan'))
            n = tot_n[t]
            unit = _UNIT_MAP.get(t, '')
            logger.info(
                f"  {t:<20}  {mae:>12.3e}  {rmse:>12.3e}  {n:>10d}  {unit}"
            )
    else:
        logger.info('  (no reference labels in the dataset: predictions only)')
    reporter = get_reporter()
    if _scored_targets:
        first = _scored_targets[0]
        reporter.outcome(f"{first} MAE {metrics[f'{first}_mae']:.3g} {_UNIT_MAP.get(first, '')}".rstrip())
    if _scored_targets or references is None:  # a dimer file's result is its binding energies
        reporter.results('Errors against the dataset labels', [
            (t, f"MAE {metrics[f'{t}_mae']:.4g} · RMSE {metrics[f'{t}_rmse']:.4g} {_UNIT_MAP.get(t, '')}".rstrip()
                + f" · {tot_n[t]:,} values")
            for t in _scored_targets] or [('Labels', 'none in the dataset · predictions only')])
    logger.info(
        f"  Timing: total={metrics['time_evaluation']:.3f}s, "
        f"compile={metrics['time_compile']:.3f}s, "
        f"per-batch={metrics['time_per_batch']:.3f}s, "
        f"per-structure={metrics['time_per_structure']:.3f}s"
    )
    logger.info('-' * 50)

    _report_binding_energies(references, energies, energy_scored='energy' in _scored_targets)
    reporter.results('Timing', [
        ('Evaluation', f"{metrics['time_evaluation']:.3f} s · {metrics['time_per_structure'] * 1e3:.3g} ms per structure"),
        ('Compilation', f"{metrics['time_compile']:.3f} s")])

    # Save predictions if requested
    if save_to is not None:
        save_to_file(predicted, str(save_to), len(predicted), overwrite=overwrite)

    total_time_end = time.time()
    logger.info(f'Total execution time: {(total_time_end - total_time_start):.3f} seconds')
    logger.info('Evaluation completed successfully')

    return metrics


def _dimer_pairs(datafile):
    """benchmark_Eint per pair for a prepare_dimer_xyz file, else None."""
    from ase.io import iread, read
    from so3lr.binding_energies import BOUND, dimer_references
    try:
        if read(str(datafile), index=0).info.get('structure_type') not in BOUND:
            return None
        return dimer_references(atoms.info for atoms in iread(str(datafile), index=':'))
    except Exception:
        return None


def _report_binding_energies(references, energies, energy_scored=False):
    """For prepare_dimer_xyz files, report binding energies against benchmark_Eint.

    energies: the predicted energy (eV) of each structure in dataset order, at full
    precision (the saved file rounds them).
    The per-frame 'energy' row above compares against interaction-energy labels
    in such files, so it is not a binding-energy error.
    """
    from so3lr.binding_energies import binding_energy_summary
    if references is None or energies is None or len(energies) != 2 * len(references):
        return None
    summary = binding_energy_summary(energies, references)
    logger.info(f"Binding energies (dimer - separated) for {summary['n']} dimers, "
                "against benchmark_Eint in kcal/mol:")
    logger.info(f"  MAE {summary['mae']:.3f} kcal/mol · RMSE {summary['rmse']:.3f} kcal/mol · "
                f"max |error| {summary['max_abs']:.3f} kcal/mol")
    if energy_scored:
        logger.info("  The 'energy' row above compares against interaction-energy labels; "
                    "use the binding energies instead.")
    rows = [('Dimers', f"{summary['n']:,} · against benchmark_Eint"),
            ('MAE', f"{summary['mae']:.3f} kcal/mol"),
            ('RMSE', f"{summary['rmse']:.3f} kcal/mol"),
            ('Max |error|', f"{summary['max_abs']:.3f} kcal/mol")]
    if energy_scored:
        rows.append(('Note', "the energy row compares against interaction-energy labels; "
                             "use these binding energies"))
    reporter = get_reporter()
    reporter.results('Binding energies (dimer − separated)', rows)
    reporter.outcome(f"binding-energy MAE {summary['mae']:.2f} kcal/mol")
    logger.info('-' * 50)
    return summary


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
