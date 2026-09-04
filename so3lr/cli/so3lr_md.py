"""
SO3LR Molecular Dynamics Module

This module provides functions for running molecular dynamics simulations
with the SO3LR Machine Learned Force Field.
It includes functionality for:
- NVT simulations
- NPT simulations
- Geometry optimization
- Trajectory output in hdf5 and extxyz formats
- Restart capabilities

"""
import os
import sys
import time
import json
import logging
import pathlib
from pathlib import Path
from functools import partial
from logging.handlers import RotatingFileHandler
from typing import Dict, Tuple, Union, List, Optional, Any, Callable, MutableMapping, Sequence

import ase
import yaml
import click
import numpy as np
from ase.io import read
import jax
import jax.numpy as jnp
import jax_md
from jax_md import units, partition
from jax_md.space import DisplacementOrMetricFn, raw_transform
from so3lr.mlff.calculators.potential import PotentialSparse
from so3lr.mlff.nn.observable.dispersion_ref_data import (
    check_atomic_numbers_supported,
)
from so3lr.mlff.nn.observable.observable_sparse import (
    compute_avg_dispersion_coefficients,
    dispersion_tail_correction_energy,
)

from so3lr.graph import Graph
from so3lr import So3lrPotential
from so3lr.jaxmd_utils import to_jax_md as _to_jax_md
from so3lr.mlff.utils.kspace_utils import setup_kspace_grid as _setup_kspace_grid

import jaxpme

# Optional PySAGES integration
try:
    import pysages
    from .so3lr_pysages_interface import (
        create_pysages_interface_fns, update_so3lr_after_pysages,
        save_pysages_state, parse_pysages_input, get_pysages_method,
        load_pysages_state
    )
    HAS_PYSAGES = True
except ImportError:
    HAS_PYSAGES = False

import h5py
from collections import namedtuple

from ..md_observables import is_per_atom_observable
from ..md_settings_validation import validate_positive_md_settings


def _model_num_features(model_path) -> Optional[int]:
    """Per-atom feature width of the model, or None if it cannot be read.

    Needed to size the 'x' (embeddings) HDF5 dataset, which -- unlike every
    other observable -- is not one scalar per atom. Read straight from the
    checkpoint's hyperparameters.json: the store is initialized before the
    model itself is loaded.
    """
    try:
        from ..model_registry import resolve_model
        workdir = resolve_model(model_path).dir
        with open(pathlib.Path(workdir) / 'hyperparameters.json') as fp:
            return int(json.load(fp)['model']['num_features'])
    except Exception as e:  # pragma: no cover - informational fallback
        logger.warning(f"Could not determine the model's feature width: {e}")
        return None


def intermediate_quantity_kwargs(observables) -> Dict[str, Any]:
    """Extra kwargs asking the model to emit the requested observables.

    Must fire for ANY requested observable. This used to be gated on
    ``'partial_charges' in observables``, so every other ``--observable`` choice
    was never produced by the model and the run died with ``KeyError`` at the
    end of the first MD cycle.
    """
    if len(observables) > 0:
        return {'output_intermediate_quantities': observables}
    return {}

# HDF5 wrapper to replace mlff.mdx.hdfdict
DataSetEntry = namedtuple('DataSetEntry', ('chunk_length', 'shape', 'dtype'))


class HDF5Store(object):
    """
    Class to append value to a hdf5 file.

    Note on observable naming: HDF5 datasets use the canonical internal
    names emitted by the model ('partial_charges', 'hirshfeld_ratios',
    'c6_ratios', 'dipole_vec', ...), not the ASE-friendly aliases used
    by the extxyz writer ('charges', 'hirsh_ratios'). Specify the
    canonical names in the YAML `observables:` list when
    `output_format: hdf5`.
    """

    def __init__(self, datapath, datasets: Dict[str, DataSetEntry], compression="gzip", mode: str = "w-"):
        self.datapath = datapath
        self.dataset = datasets
        self.i = 0
        self.mode = mode

        with h5py.File(self.datapath, mode=self.mode) as h5f:
            for name, x in self.dataset.items():
                h5f.create_dataset(
                    name,
                    shape=(0,) + x.shape,
                    maxshape=(None,) + x.shape,
                    dtype=x.dtype,
                    compression=compression,
                    chunks=(x.chunk_length,) + x.shape)

    def append(self, x: Dict):
        """Append one or more frames after validating the complete batch.

        The old store appended a single fixed-size *buffer* as one HDF5 row.
        That layout could not represent the final short buffer without either
        dropping frames or broadcasting duplicates.  Dataset shapes are now
        frame-major, and this method grows their leading frame axis by the
        actual number of values supplied.
        """
        if not x:
            return

        unknown = sorted(set(x) - set(self.dataset))
        if unknown:
            raise KeyError(f"Unknown HDF5 dataset(s): {unknown}")

        prepared = {}
        n_frames = None
        for key, values in x.items():
            entry = self.dataset[key]
            array = np.asarray(values, dtype=entry.dtype)
            expected_shape = entry.shape
            if array.ndim != len(expected_shape) + 1 or array.shape[1:] != expected_shape:
                raise ValueError(
                    f"Dataset '{key}' expects a batch shaped "
                    f"(n_frames, {', '.join(map(str, expected_shape))}), "
                    f"received {array.shape}."
                )
            if n_frames is None:
                n_frames = array.shape[0]
            elif array.shape[0] != n_frames:
                raise ValueError(
                    f"All HDF5 arrays must contain the same number of frames; "
                    f"'{key}' has {array.shape[0]}, expected {n_frames}."
                )
            prepared[key] = array

        if not n_frames:
            return

        with h5py.File(self.datapath, mode='a') as h5f:
            # Check the on-disk state before resizing any dataset.  A failed
            # partial write must never leave arrays with mismatched lengths.
            for key in prepared:
                if key not in h5f:
                    raise KeyError(f"HDF5 dataset '{key}' does not exist")
                if h5f[key].shape[0] != self.i:
                    raise RuntimeError(
                        f"HDF5 dataset '{key}' has {h5f[key].shape[0]} frames, "
                        f"but the store expected {self.i}."
                    )

            new_i = self.i + n_frames
            resized = []
            try:
                for key, array in prepared.items():
                    dset = h5f[key]
                    dset.resize((new_i,) + self.dataset[key].shape)
                    resized.append(dset)
                    dset[self.i:new_i] = array
            except Exception:
                for dset in resized:
                    dset.resize((self.i,) + dset.shape[1:])
                h5f.flush()
                raise

            self.i = new_i
            h5f.flush()

# Setup logging
logger = logging.getLogger("SO3LR")

class RestartInNewEnsembleError(Exception):
    def __init__(self, loaded_state, ensemble, dropped_fields=None, reset_only=False):
        """
        Exception raised when the restart file contains a different ensemble
        than the one specified in the simulation, or when the user requested
        a restart-with-reset (`restart_reset: true`). In either case, only
        positions / momenta / masses / box are reused; thermostat and
        barostat state are re-initialised from scratch.

        Args:
            loaded_state: The loaded NpzFile (kept on the exception so the
                caller can re-derive position/mass/momentum/box).
            ensemble: The ensemble the *current* run is targeting.
            dropped_fields: List of restart-file fields that will be
                discarded (thermostat / barostat / reference_box / ...).
                Computed by `load_state`.
            reset_only: True if the user asked for `restart_reset` (in which
                case the ensemble may match but the user explicitly opted in
                to re-initialising integrator state).
        """
        self.ensemble = ensemble
        self.loaded_state = loaded_state
        self.dropped_fields = list(dropped_fields) if dropped_fields else []
        self.reset_only = bool(reset_only)
        saved_ens = str(self.loaded_state.get('ensemble', None)).upper()
        req_ens = str(self.ensemble).upper()
        if reset_only:
            header = (
                f"Restart-with-reset requested (restart_reset=true) for "
                f"ensemble={req_ens}; only positions, momenta, masses and "
                f"box will be re-used from the restart file."
            )
        else:
            header = (
                f"Restart file was saved under ensemble={saved_ens}; loading "
                f"into ensemble={req_ens}. Only positions, momenta, masses "
                f"and box will be re-used from the restart file."
            )
        if self.dropped_fields:
            header += (
                f"\nThe following fields from the .npz will be "
                f"re-initialised: {self.dropped_fields}."
            )
        self.message = header
        super().__init__(self.message)
    def __str__(self):
        return self.message


class RestartSchemaMismatchError(Exception):
    """
    Exception raised when the restart .npz is missing a field that the
    target ensemble requires (e.g. a restart saved by an older so3lr that
    did not yet write `thermostat_tau` is now being loaded by a newer
    build that expects it). Carries the missing field name, the target
    ensemble, and the list of fields actually present in the .npz so the
    user can see at a glance what is wrong with the file.
    """
    def __init__(self, missing_field, ensemble, available_fields, path=None):
        self.missing_field = missing_field
        self.ensemble = ensemble
        self.available_fields = sorted(available_fields)
        self.path = path
        loc = f" in {path}" if path else ""
        self.message = (
            f"Restart file{loc} is missing required field "
            f"'{missing_field}' for ensemble={str(ensemble).upper()}. "
            f"Found fields: {self.available_fields}. "
            f"This usually means the restart was written by an older "
            f"so3lr build whose .npz schema differs from the current one."
        )
        super().__init__(self.message)
    def __str__(self):
        return self.message

def setup_logger(log_file=None, log_level=logging.INFO, console_level=logging.INFO):
    """
    Set up the logger with file and console handlers.

    Parameters:
    -----------
    log_file : str or None
        Path to the log file. If None, logging only goes to console.
    log_level : int
        Logging level for the file handler (default: INFO)
    console_level : int
        Logging level for the console handler (default: INFO)
    
    Raises:
    -------
    RuntimeError
        If unable to set up file logging to the specified log file
    """
    # Get the existing logger (defined at the module level)
    global logger

    # Clear any existing handlers to avoid duplicate output
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the global logger level to the lowest of the two levels
    logger.setLevel(min(log_level, console_level))

    # Create formatter for the file output - includes timestamps and level names
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create formatter for console output - just the message, no prefix
    console_formatter = logging.Formatter('%(message)s')

    # Add the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Add the file handler if a log file is specified
    if log_file:
        try:
            # Make sure parent directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a rotating file handler (10MB max size, keep 5 backups)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Raise an exception instead of falling back to console logging
            raise RuntimeError(f"Could not set up file logging to {log_file}: {e}")

    # Prevent propagation to the root logger to avoid duplicate messages
    logger.propagate = False

    return logger


# Track whether setup_jax_globals has already configured the process so a
# second call from a different entry point doesn't log "Matmul precision: ..."
# twice and so we can tell if a caller's --precision/--matmul-precision is
# being silently ignored because something configured JAX before us.
_JAX_GLOBALS_CONFIGURED: Dict[str, Any] = {}


def setup_jax_globals(precision: str = 'float32', matmul_precision: str = 'highest') -> None:
    """Configure JAX process-globals that every SO3LR entry point depends on.

    Must be called before any JAX op runs. Idempotent — subsequent calls with
    identical arguments are silent; with different arguments they warn and
    keep the first-call values (changing matmul_precision after the first
    JIT'd function may silently re-trace).

    Args:
        precision: 'float32' or 'float64'. Drives jax_enable_x64 (always on,
            since x64 is required for high_precision_sum / mixed-precision
            integrator ops; the precision setting controls array dtype, not
            kernel availability).
        matmul_precision: 'default' / 'high' / 'highest'. 'highest' forces
            full float32 matmuls (no TF32 fallback) which avoids drift on
            GPUs at the cost of throughput.
    """
    if _JAX_GLOBALS_CONFIGURED:
        prev_mp = _JAX_GLOBALS_CONFIGURED.get('matmul_precision')
        prev_p = _JAX_GLOBALS_CONFIGURED.get('precision')
        if prev_mp != matmul_precision or prev_p != precision:
            logger.warning(
                f"setup_jax_globals called again with different values "
                f"(precision: {prev_p} -> {precision}, "
                f"matmul_precision: {prev_mp} -> {matmul_precision}); "
                f"keeping the original. Set both globals in your first call."
            )
        return

    # Always enable x64 so that high_precision_sum and mixed-precision
    # integrator ops work correctly. The 'precision' setting controls
    # which dtype arrays are created in, not whether x64 is available.
    jax.config.update("jax_enable_x64", True)

    # Reduces drift for x32 precision simulations. Must be set before any
    # JIT compilation that touches matmuls.
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    _JAX_GLOBALS_CONFIGURED['precision'] = precision
    _JAX_GLOBALS_CONFIGURED['matmul_precision'] = matmul_precision
    logger.info(
        f"JAX globals set: precision={precision}, "
        f"matmul_precision={matmul_precision}, x64=enabled"
    )


def handle_units(
    unit_system: Callable[[], Dict[str, float]],
    dt: float,
    T: Optional[float] = None,
    P: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """
    Apply unit conversions to timestep, temperature, and pressure.

    The unit_system is a function that returns a dictionary of the units
    with the keys 'time', 'temperature' and 'pressure'.

    Args:
        unit_system: Function that returns a dictionary of units.
        dt: Timestep in picoseconds.
        T: Target temperature in Kelvin. Defaults to None.
        P: Target pressure in bar. Defaults to None.

    Returns:
        Dictionary containing the timestep, temperature and pressure
        in the correct units for JAX MD.
    """
    try:
        unit = unit_system()
        dt *= unit['time']

        if T is not None:
            T *= unit['temperature']
        if P is not None:
            P *= unit['pressure']

        return {
            'dt': dt,
            'T': T,
            'P': P,
        }
    except Exception as e:
        raise ValueError(f"Error converting units: {str(e)}.")


def handle_box(
    shift_displacement: str,
    positions: jnp.ndarray,
    cell: Optional[jnp.ndarray] = None,
    precision: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray, Callable, Callable, bool]:
    """
    Set up box and displacement functions for simulation.

    If the system is periodic, periodic boundary conditions are applied and the
    positions are divided by the box vector. Fractional coordinates are used
    for periodic systems.

    Args:
        shift_displacement: Type of boundary conditions ('free' or 'periodic').
        positions: Positions of the atoms.
        cell: Unit cell vectors for periodic systems. Required if shift_displacement is 'periodic'.

    Raises:
        ValueError: If cell is not provided for periodic boundary conditions or
                   if an unsupported boundary condition is specified.

    Returns:
        Tuple containing:
        - positions: Possibly transformed positions (fractional for periodic)
        - box: Box vector or 0.0 for free boundary conditions
        - displacement: Function to compute displacements
        - shift: Function to shift positions
        - fractional_coordinates: Whether positions are in fractional coordinates
    """
    if shift_displacement == 'periodic':
        if cell is None:
            raise ValueError('Cell must be defined for periodic boundary conditions.')

        box = jnp.array(np.diag(np.array(cell)), dtype=precision)
        if box.shape == (3,3):
            box = jnp.diag(box)
        fractional_coordinates = True

        # Create displacement and shift functions for periodic boundary conditions
        displacement, shift = jax_md.space.periodic_general(
            box=box,
            fractional_coordinates=fractional_coordinates
        )

        # Convert positions to fractional coordinates
        inv_box = 1/box
        positions = raw_transform(inv_box, positions)

    elif shift_displacement == 'periodic_cart':
        if cell is None:
            raise ValueError('Cell must be defined for periodic boundary conditions.')

        box = jnp.array(np.diag(np.array(cell)))
        if box.shape == (3,3):
            box = jnp.diag(box)
        fractional_coordinates = False

        # Create displacement and shift functions for periodic boundary conditions
        # using cartesian coordinates (required for PySAGES)
        displacement, shift = jax_md.space.periodic_general(
            box=box,
            fractional_coordinates=fractional_coordinates
        )

    elif shift_displacement == 'free':
        # Create displacement and shift functions for free boundary conditions
        displacement, shift = jax_md.space.free()
        box = jnp.array([0.0], dtype=precision)
        fractional_coordinates = False

    else:
        raise ValueError(
            f"Unsupported boundary condition: '{shift_displacement}'. "
            "Only 'free', 'periodic', or 'periodic_cart' boundary conditions are supported."
        )
    logger.info(f"Box for SO3LR: {box}")
    return positions, box, displacement, shift, fractional_coordinates


def init_hdf5_store(
    save_to: Union[str, Path],
    batch_size: int,
    num_atoms: int,
    num_box_entries: Union[int, Sequence[int]],
    exist_ok: bool = False,
    observables: Optional[Sequence[str]] = [],
    num_features: Optional[int] = None,
    so3lr_version: Optional[str] = None,
    ensemble: Optional[str] = None,
    md_dt_fs: Optional[float] = None,
    md_steps: Optional[int] = None,
    atomic_numbers: Optional[Any] = None,
    cell_a0: Optional[Any] = None,
    input_file: Optional[str] = None,
    model_path: Optional[str] = None,
    dtype: Any = np.float32,
    fractional_coordinates: Optional[bool] = None,
) -> HDF5Store:
    """
    Initialize an HDF5 storage object for trajectory data.

    Args:
        save_to: Path to save the HDF5 file.
        batch_size: Number of frames per HDF5 chunk / normal write buffer.
        num_atoms: Number of atoms in the system.
        num_box_entries: Shape of a stored box, or the length of a box vector.
        exist_ok: Whether to overwrite the file if it exists.
        observables: Observable names that get their own dataset.
        num_features: Per-atom feature width, needed to size the 'x' dataset.
            Read from the model's hyperparameters by the caller; when None,
            'x' cannot be stored and is reported.
        so3lr_version: SO3LR package version recorded as a root attribute.
        ensemble: Ensemble label (nve/nvt/nvt-langevin/npt) recorded as a root attribute.
        md_dt_fs: MD timestep in femtoseconds recorded as a root attribute.
        md_steps: Number of inner integration steps per write cycle.
        atomic_numbers: 1D iterable of atomic numbers for the initial geometry.
        cell_a0: Initial cell as a (3,3) or (3,) array in Angstrom. Zeros if non-periodic.
        input_file: Original input geometry filename.
        model_path: Resolved SO3LR model name or path.
        dtype: Floating-point storage dtype. This should match the configured
            simulation precision.
        fractional_coordinates: Whether stored positions are fractional.

    Raises:
        RuntimeError: If the file exists and exist_ok is set to False.

    Returns:
        HDF5Store: Initialized HDF5 storage object.
    """
    # Convert to Path object and resolve path
    save_path = Path(save_to).expanduser().resolve()

    # Create parent directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists
    if save_path.exists() and not exist_ok:
        raise RuntimeError(
            f'File exists: {save_path}. '
            f'Set exist_ok=True to override file.'
        )

    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer (received {batch_size!r})."
        )

    storage_dtype = np.dtype(dtype)
    box_shape = (
        (int(num_box_entries),)
        if isinstance(num_box_entries, (int, np.integer))
        else tuple(int(n) for n in num_box_entries)
    )
    if not box_shape or any(n <= 0 for n in box_shape):
        raise ValueError(f"Invalid HDF5 box shape: {box_shape}")

    # Dataset shapes exclude the resizable leading frame dimension.  The
    # configured save buffer controls HDF5 chunking only; it is not part of
    # the public trajectory shape.
    dataset = {
        'positions': DataSetEntry(
            chunk_length=batch_size,
            shape=(num_atoms, 3),
            dtype=storage_dtype
        ),
        'momenta': DataSetEntry(
            chunk_length=batch_size,
            shape=(num_atoms, 3),
            dtype=storage_dtype
        ),
        'box': DataSetEntry(
            chunk_length=batch_size,
            shape=box_shape,
            dtype=storage_dtype
        )
    }

    for o in observables:
        if o == "dipole_vec":
            dataset['dipole_vec'] = DataSetEntry(
                chunk_length=batch_size,
                shape=(1, 3),
                dtype=storage_dtype
            )
        elif o == 'x':
            # Per-atom embeddings: (num_atoms, num_features). Unlike every other
            # observable this is not one scalar per atom, so the dataset can only
            # be sized once the model's feature width is known.
            if num_features is None:
                logger.warning(
                    "Observable 'x' needs the model's feature width to size its "
                    "HDF5 dataset and it could not be determined; use "
                    "--output-format extxyz to record it."
                )
                continue
            dataset['x'] = DataSetEntry(
                chunk_length=batch_size,
                shape=(num_atoms, num_features),
                dtype=storage_dtype
            )
        elif is_per_atom_observable(o) or o in ('hirshfeld_ratios', 'c6_ratios',
                                               'electrostatic_energy_kspace'):
            # Every one-scalar-per-atom observable gets a dataset. This list used
            # to be a fourth hand-maintained copy that omitted five selectable
            # observables (nn_energy, nlh_repulsion, the two *_undamped ones),
            # so requesting them failed at write time with
            # "object '<name>' doesn't exist" even after the model produced them.
            dataset[o] = DataSetEntry(
                chunk_length=batch_size,
                shape=(num_atoms,),
                dtype=storage_dtype
            )
        else:
            logger.warning(
                f"Observable '{o}' has a model-dependent width and cannot be "
                f"stored in HDF5; use --output-format extxyz to record it."
            )

    store = HDF5Store(save_path, datasets=dataset, mode='w')

    # Write self-describing root-group attributes so the .hdf5 file carries
    # provenance on its own (model, ensemble, timestep, system, observables).
    # Only supplied values are written; defaults stay absent. All values are
    # coerced to plain numpy/string types — no nested dicts.
    try:
        with h5py.File(save_path, mode='a') as h5f:
            if so3lr_version is not None:
                h5f.attrs['so3lr_version'] = str(so3lr_version)
            if ensemble is not None:
                h5f.attrs['ensemble'] = str(ensemble)
            if md_dt_fs is not None:
                h5f.attrs['md_dt_fs'] = float(md_dt_fs)
            if md_steps is not None:
                h5f.attrs['md_steps'] = int(md_steps)
            if atomic_numbers is not None:
                h5f.attrs['atomic_numbers'] = np.asarray(
                    atomic_numbers, dtype=np.int32
                )
            if cell_a0 is not None:
                h5f.attrs['cell_a0'] = np.asarray(cell_a0, dtype=storage_dtype)
            if observables is not None:
                h5f.attrs['observables'] = ','.join(str(o) for o in observables)
            if input_file is not None:
                h5f.attrs['input_file'] = str(input_file)
            if model_path is not None:
                h5f.attrs['model_path'] = str(model_path)
            if fractional_coordinates is not None:
                h5f.attrs['fractional_coordinates'] = bool(fractional_coordinates)
            h5f.attrs['trajectory_layout'] = 'frame-major-v1'
            h5f.attrs['box_convention'] = 'ase-cell-vectors-as-rows'
    except Exception as e:
        logger.warning(f"Could not write HDF5 root attributes: {e}")

    return store


def write_to_hdf5(
    hdf5_store: HDF5Store,
    momenta: Optional[List[jnp.ndarray]],
    positions: List[jnp.ndarray],
    boxes: Union[float, jnp.ndarray, List[jnp.ndarray], None],
    obs_dict: Optional[MutableMapping[str, list]] = None,
    indices: Optional[List[int]] = None,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], MutableMapping[str, list]]:
    """
    Write trajectory data to an HDF5 file.

    Writes the accumulated trajectory data to HDF5 storage.

    Args:
        hdf5_store: HDF5 storage object.
        momenta: List of momenta arrays for each frame, or None.
        positions: List of position arrays for each frame.
        boxes: Box vectors for each frame. Can be float, array, list of arrays, or None.
        obs_dict: Dictionary of observables. Defaults to {}.
    Returns:
        Tuple of empty lists for momenta, positions, and boxes.
    """
    try:
        if not positions:
            return (
                momenta if momenta is not None else [],
                positions,
                boxes,
                obs_dict or {},
            )

        if obs_dict is None:
            obs_dict = {}

        n_frames = len(positions)
        if indices is None:
            indices = list(range(positions[0].shape[0]))

        step_data = {
            'positions': np.stack(positions, axis=0)[:, indices],
        }

        if boxes is not None:
            expected_box_shape = hdf5_store.dataset['box'].shape
            box_array = np.asarray(boxes)
            if box_array.ndim == 0:
                box_array = np.full(
                    (n_frames,) + expected_box_shape,
                    box_array.item(),
                )
            elif box_array.shape == expected_box_shape:
                box_array = np.repeat(box_array[None, ...], n_frames, axis=0)
            elif box_array.shape != (n_frames,) + expected_box_shape:
                raise ValueError(
                    f"boxes must have shape {expected_box_shape} or "
                    f"{(n_frames,) + expected_box_shape}; received {box_array.shape}"
                )
            step_data['box'] = box_array

        if momenta is not None:
            if len(momenta) != n_frames:
                raise ValueError(
                    f"momenta contains {len(momenta)} frames; expected {n_frames}"
                )
            step_data['momenta'] = np.stack(momenta, axis=0)[:, indices]

        for o in obs_dict.keys():
            if len(obs_dict[o]) != n_frames:
                raise ValueError(
                    f"Observable '{o}' contains {len(obs_dict[o])} frames; "
                    f"expected {n_frames}"
                )
            # Check if the observable has atom_wise data, this test could be improved
            if obs_dict[o][0].shape == (positions[0].shape[0],):
                step_data[o] = np.stack(obs_dict[o], axis=0)[:, indices]
            else:
                values = np.stack(obs_dict[o], axis=0)
                expected_shape = hdf5_store.dataset[o].shape
                if values.shape[1:] != expected_shape:
                    # Model dipoles are commonly emitted as (3,), while the
                    # HDF5 schema keeps the historical singleton molecular axis.
                    if expected_shape == (1, 3) and values.shape[1:] == (3,):
                        values = values[:, None, :]
                step_data[o] = values
        
        hdf5_store.append(step_data)
        
        obs_dict = {o: [] for o in obs_dict.keys()}

        return [], [], [], obs_dict

    except Exception as e:
        logger.error(f"Failed to write to HDF5 file: {str(e)}")
        sys.exit(1)


def write_to_extxyz(
    output_file: Union[str, Path],
    atoms: ase.Atoms,
    boxes: Union[float, jnp.ndarray, List, None],
    momenta: List,
    positions: List,
    obs_dict: Optional[MutableMapping[str, list]] = None,
    indices: Optional[List[int]] = None,
    fractional_coordinates: bool = True,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], MutableMapping[str, list]]:
    """
    Write trajectory data to an extended XYZ file.

    Creates a copy of the atoms object for each frame and appends it to the
    output file with the corresponding positions, momenta, and box.
    Handles different box formats and applies appropriate scaling to positions.

    Args:
        output_file: Path to the output file.
        atoms: ASE Atoms object template.
        boxes: Box vectors for each frame or single box for all frames.
             Can be float, array, or list of arrays.
        momenta: List of momenta arrays for each frame.
        positions: List of position arrays for each frame.
        obs_dict: Dictionary of observables. Defaults to {}.
        indices: Optional list of atom indices to write.
        fractional_coordinates: If True, positions are in fractional coordinates
            and will be scaled by box vectors. If False, positions are already
            in Cartesian Angstroms and are written directly.

    Returns:
        Tuple of empty lists (momenta, positions, boxes, obs_dict) to clear memory
        on success. On failure, logs the error and calls sys.exit(1) so the user
        sees the failure rather than silently dropping frames for the rest of the run
        (matches the HDF5 writer behavior).
    """
    try:
        if obs_dict is None:
            obs_dict = {}

        # Ensure output_file is a Path object
        output_path = Path(output_file) if isinstance(
            output_file, str) else output_file

        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if we have data to write
        if not positions:
            logger.info('No positions to write to extxyz file')
            return momenta, positions, boxes, obs_dict

        # Convert positions to numpy array
        positions = np.asarray(positions)

        # Create a list of atoms objects for each frame
        atoms_list = []
        for i in range(len(positions)):
            atoms_copy = atoms.copy()

            # Resolve the box for this frame.  Lists are the trajectory form;
            # an ndarray/scalar is a fixed box shared by all frames.
            if isinstance(boxes, list):
                if len(boxes) != len(positions):
                    raise ValueError(
                        f"boxes contains {len(boxes)} frames; expected {len(positions)}"
                    )
                frame_box = boxes[i]
            else:
                frame_box = boxes

            if frame_box is not None and np.any(np.asarray(frame_box) != 0):
                atoms_copy.set_cell(frame_box)
                if fractional_coordinates:
                    # ASE applies the full cell matrix here.  Element-wise
                    # multiplication only happened to work for diagonal boxes
                    # and corrupted coordinates in sheared cells.
                    atoms_copy.set_scaled_positions(positions[i])
                else:
                    atoms_copy.set_positions(positions[i])
            else:
                atoms_copy.set_positions(positions[i])

            # Set momenta if available
            if momenta is not None and i < len(momenta):
                atoms_copy.set_momenta(momenta[i])

            # Canonicalize the extxyz column name so the eval-side reader
            # (see NORMALIZE_TARGET in so3lr/cli/so3lr_eval.py) can locate the
            # data without falling back on its alias table. Keep this map in
            # lock-step with NORMALIZE_TARGET; do NOT collapse the model-
            # specific nlh_repulsion / zbl_repulsion pair.
            _extxyz_obs_alias = {
                'charges': 'partial_charges',
                'hirsh_ratios': 'hirshfeld_ratios',
            }
            for o in obs_dict.keys():
                col = _extxyz_obs_alias.get(o, o)
                if o in ("partial_charges", "charges"):
                    atoms_copy.set_array('partial_charges', np.asarray(obs_dict[o][i]).flatten())

                elif o in ("hirshfeld_ratios", "hirsh_ratios"):
                    atoms_copy.set_array('hirshfeld_ratios', np.asarray(obs_dict[o][i]).flatten())

                elif o == "c6_ratios":
                    atoms_copy.set_array('c6_ratios', np.asarray(obs_dict['c6_ratios'][i]).flatten())

                elif o == "dipole_vec" :
                    atoms_copy.info['dipole_vec'] = np.asarray(obs_dict['dipole_vec'][i]).flatten()

                elif obs_dict[o][0].shape == (positions[0].shape[0],):
                    atoms_copy.set_array(col, np.asarray(obs_dict[o][i]).flatten())
                else:
                    logger.warning(
                        f"Observable '{o}' is not of shape (num_atoms,) and not supported by default for extxyz format."
                    )

            if indices is not None:
                # Filter atoms based on indices
                #atoms_copy = Filter(atoms_copy, indices=indices)
                atoms_copy = atoms_copy[indices]
            atoms_list.append(atoms_copy)

        # Write all atoms to file at once
        file_exists = output_path.exists()
        ase.io.write(output_path, atoms_list,
                     format='extxyz', append=file_exists)

        # Clear lists to free memory
        obs_dict = {o: [] for o in obs_dict.keys()}

        return [], [], [], obs_dict

    except Exception as e:
        logger.error(f"Failed to write to extxyz file: {str(e)}")
        sys.exit(1)


def atoms_to_jnp(
    atoms: ase.Atoms,
    precision: jnp.dtype = jnp.float32
) -> dict:
    """
    Transform the ASE atoms object to a dictionary of jax numpy arrays.

    Args:
        atoms (ase.Atoms): ASE atoms object.
        precision (jnp.dtype, optional): Floating point precision.
                                Defaults to jnp.float32.

    Returns:
        dict: Dictionary containing the positions, species and masses.
    """
    positions = jnp.array(atoms.get_positions(), dtype=precision)
    species = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int32)
    masses = jnp.array(atoms.get_masses(), dtype=precision)
    if not np.all(atoms.get_velocities() == 0):
        momenta = jnp.array(
            atoms.get_velocities() * atoms.get_masses()[:, None],
            dtype=precision,
        )
    elif not np.all(atoms.get_momenta() == 0):
        momenta = jnp.array(atoms.get_momenta(), dtype=precision)
    else:
        momenta = None

    if momenta is not None:
        if momenta.shape[0] > 1:
            momenta = momenta - jnp.mean(momenta, axis=0, keepdims=True)
            logger.info(
                'Loading initial velocities from input file; '
                'subtracted center-of-mass momentum.'
            )
        else:
            logger.info('Loading initial velocities from input file.')

    return {
        'positions': positions,
        'species': species,
        'masses': masses,
        'momenta': momenta
    }


def check_cell(
    cell: np.ndarray,
    lr_cutoff: float,
    kspace_electrostatics: str,
    kspace_smearing: float = None
    ) -> np.ndarray:
    """
    Validate the provided cell matrix for compatibility with JAX-MD requirements.

    Args:
        cell (np.ndarray): The cell matrix to be validated.
        lr_cutoff (float): The long-range cutoff distance.
        kspace_electrostatics (str): Type of k-space electrostatics to use.
        kspace_smearing (float): K-space smearing value.

    Returns:
        np.ndarray: The validated cell matrix if all checks pass, or None if the cell is zero.
    """
    # Sanity check the long-range cutoff up front, before the all-zero cell
    # early return. Otherwise non-periodic systems would silently accept a
    # too-small lr_cutoff because this check used to live below the return.
    if lr_cutoff < 10 and (kspace_electrostatics == None or kspace_smearing == None):
        raise ValueError(f'Long-range cutoff below 10 Å is not supported yet. Provided cutoff: {lr_cutoff} Å.')

    if np.all(cell == 0):
        return None
    
    # Create a mask with True only on the diagonal
    mask = np.eye(cell.shape[0], dtype=bool)
    if np.any(cell[~mask] != 0):
        raise ValueError(f'JAX-MD currently supports only orthogonal cells. Provided cell: {cell}') 

    # if np.any(np.diag(cell) < lr_cutoff * 2):
    #     raise ValueError(f'Each dimension of the cell {np.diag(cell)} must be at least twice the long-range cutoff distance [{lr_cutoff} Å].')

    return cell


def cell_from_atoms_boundary_conditions(
    atoms: ase.Atoms,
    lr_cutoff: float,
    kspace_electrostatics: str,
    kspace_smearing: float = None,
) -> Optional[np.ndarray]:
    """Return a periodic cell only when all three ASE PBC axes are enabled.

    The current JAX-MD neighbor-list and reciprocal-space plumbing carries a
    box but not a per-axis PBC mask, so mixed boundary conditions cannot be
    represented faithfully.  A nonzero ASE cell with ``pbc=False`` is merely a
    bounding cell and must not silently make a gas-phase system periodic.
    """
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if np.any(pbc) and not np.all(pbc):
        raise ValueError(
            f"Mixed-axis periodic boundary conditions {pbc.tolist()} are not "
            f"supported by SO3LR MD. Use either pbc=[True, True, True] for "
            f"three-dimensional periodicity or pbc=[False, False, False] for "
            f"a nonperiodic system."
        )

    if not np.any(pbc):
        # Preserve check_cell's cutoff validation without interpreting the
        # (possibly non-orthogonal) bounding cell as a periodic simulation box.
        return check_cell(
            np.zeros((3, 3)), lr_cutoff, kspace_electrostatics,
            kspace_smearing,
        )

    return check_cell(
        np.asarray(atoms.get_cell()), lr_cutoff, kspace_electrostatics,
        kspace_smearing,
    )

def _is_legacy_model(model_dir) -> bool:
    """Return True if the resolved model is the legacy v1 SO3LR (legacy_so3lr_bool)."""
    try:
        with open(pathlib.Path(model_dir) / 'hyperparameters.json', 'r') as fp:
            cfg = json.load(fp)
        return bool(cfg.get('model', {}).get('legacy_so3lr_bool', False))
    except Exception:
        return False


def load_model(
    model_path: str,
    precision: jnp.dtype,
    lr_cutoff: float = 12.0,
    dispersion_damping: float = 2.,
    kspace_electrostatics: str = None,
    kspace_interp_nodes: int = 4,
    kspace_dispersion: str = None,
    kspace_dispersion_interp_nodes: int = 4,
    charge_embed_total_charge: float = None,
    charge_equilibration_total_charge: float = None,
    **kwargs
) -> PotentialSparse:
    """
    Load a trained MLFF model from a checkpoint directory.

    Args:
        model_path (str): Path to the model checkpoint directory.
        precision (jnp.dtype): Precision to use for the calculation.
        lr_cutoff (float, optional): Long-range cutoff for SO3LR in Å. Defaults to 12.0.
        dispersion_damping (float, optional): Cutoff for dispersion
                                energy damping in Å. Defaults to 2.0.
        kspace_dispersion (str, optional): Method for dispersion PME ('ewald', 'pme', or None).
        kspace_dispersion_interp_nodes (int, optional): Interpolation nodes for dispersion PME.
        charge_embed_total_charge (float, optional): Override the total charge seen by
                                the charge embedding. None (default) uses the system's
                                total charge.
        charge_equilibration_total_charge (float, optional): Override the total charge
                                that charge equilibration conserves to. None (default)
                                uses the system's total charge. Independent of the
                                embedding override.

    Returns:
        PotentialSparse: The loaded MLFF model.
    """
    logger.info(f'Loading model from {model_path}')

    # Resolve a registered model name (see model_registry) or a filesystem path.
    # Raises FileNotFoundError if it is neither a known name nor an existing path.
    from so3lr.model_registry import resolve_model
    spec = resolve_model(model_path)
    model_dir, from_file = spec.dir, spec.from_file
    params_pkl = model_dir / 'params.pkl'

    # The legacy v1 model used real-space cutoff electrostatics; k-space
    # (pme/ewald/p3m) is untested for it. Warn but allow.
    if kspace_electrostatics is not None and _is_legacy_model(model_dir):
        logger.warning(
            f"k-space electrostatics ('{kspace_electrostatics}') is untested for the legacy "
            f"v1 SO3LR model, which was trained with real-space cutoff electrostatics. "
            f"Results may be unreliable; consider a real-space long-range cutoff instead.")

    long_range_kw = dict(
        cutoff_lr=lr_cutoff,
        dispersion_energy_cutoff_lr_damping=dispersion_damping,
        neighborlist_format_lr='ordered_sparse',
        coulomb_kspace_method=kspace_electrostatics,
        coulomb_kspace_interp_nodes=kspace_interp_nodes,
    )

    if kspace_dispersion is not None:
        do_ewald = (kspace_dispersion == 'ewald')
        long_range_kw['dispersion_kspace_do_ewald'] = do_ewald
        long_range_kw['dispersion_kspace_interp_nodes'] = kspace_dispersion_interp_nodes

    def _create_potential(from_file_flag: bool) -> PotentialSparse:
        return PotentialSparse.create_from_workdir(
            workdir=model_dir,
            from_file=from_file_flag,
            long_range_kwargs=long_range_kw,
            dtype=precision,
            charge_embed_total_charge=charge_embed_total_charge,
            charge_equilibration_total_charge=charge_equilibration_total_charge,
            **kwargs
        )

    try:
        potential = _create_potential(from_file)
    except FileNotFoundError:
        if (not from_file) and params_pkl.exists():
            logger.warning(
                'Checkpoint loading failed for %s; falling back to params.pkl.',
                model_dir,
            )
            potential = _create_potential(True)
        else:
            raise

    return potential


def to_jax_md_custom(
        potential,  # the mlff potential
        species: jnp.ndarray,
        displacement_or_metric: DisplacementOrMetricFn,
        box,  # box if it exists, check jax_md documentation for conventions
        dr_threshold: float = 0.,  # currently dr_threshold > 0 is experimental
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        minimum_cell_size_multiplier: float = 1.0,
        disable_cell_list: bool = False,
        fractional_coordinates: bool = True,
        total_charge: float = 0.,
        precision: jnp.dtype = jnp.float32,
        num_unpaired_electrons: float = 0.,
        **neighbor_kwargs
):
    """Create neighbor functions and a jax-md energy function for the MD engine.

    Thin wrapper around the shared :func:`so3lr.jaxmd_utils.to_jax_md` helper, so
    the CLI and the public helper stay in sync (previously these were two
    independent copies). Uses ``capacity_multiplier=None`` so each per-list
    ``buffer_size_multiplier`` is used directly as the neighbor-list capacity
    multiplier (the historical CLI behaviour), and applies a single
    ``minimum_cell_size_multiplier`` to both the short- and long-range lists.

    Args:
        potential: The MLFF potential object.
        species: Array of atomic species.
        displacement_or_metric: Function to compute displacements.
        box: Box definition for periodic systems.
        dr_threshold: Threshold for neighborhood cutoff.
        buffer_size_multiplier_sr: Capacity multiplier for short-range neighbor list.
        buffer_size_multiplier_lr: Capacity multiplier for long-range neighbor list.
        minimum_cell_size_multiplier: Multiplier for cell size.
        disable_cell_list: Whether to disable cell lists.
        fractional_coordinates: Whether to use fractional coordinates.
        total_charge: Total charge of the system.
        precision: Precision to use for the auxiliary graph arrays.
        num_unpaired_electrons: Number of unpaired electrons (spin).
        **neighbor_kwargs: Additional keyword arguments for neighbor functions.

    Returns:
        Tuple of neighbor function, long-range neighbor function, and energy function.
    """
    return _to_jax_md(
        potential=potential,
        displacement_or_metric=displacement_or_metric,
        box_size=box,
        species=species,
        dr_threshold=dr_threshold,
        capacity_multiplier=None,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        minimum_cell_size_multiplier_sr=minimum_cell_size_multiplier,
        minimum_cell_size_multiplier_lr=minimum_cell_size_multiplier,
        disable_cell_list=disable_cell_list,
        fractional_coordinates=fractional_coordinates,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
        precision=precision,
        **neighbor_kwargs,
    )


def process_model(
    potential: PotentialSparse,
    species: jnp.ndarray,
    displacement: DisplacementOrMetricFn,
    box: jnp.ndarray,
    total_charge: float = 0.,
    num_unpaired_electrons: float = 0.,
    buffer_size_multiplier_lr: float = 1.25,
    buffer_size_multiplier_sr: float = 1.25,
    minimum_cell_size_multiplier: float = 1.0,
    precision: jnp.dtype = jnp.float32,
    fractional_coordinates: bool = False,
    has_aux: bool = False,
) -> Tuple[callable, callable, callable, Optional[callable]]:
    """
    Process the model and create the neighbor functions and energy function.

    Args:
        potential (PotentialSparse): Force field model.
        species (jnp.ndarray): Atomic species.
        displacement (DisplacementOrMetricFn): Displacement function.
        box (jnp.ndarray): Box of the system.
        total_charge (float, optional): Total charge of the system. Defaults to 0..
        buffer_size_multiplier_lr (float, optional): Buffer for long range
                                            neighborlist. Defaults to 1.25.
        buffer_size_multiplier_sr (float, optional): Buffer for short range
                                            neighborlist. Defaults to 1.25.
        minimum_cell_size_multiplier (float, optional): Multiplier for cell size.
                                            Defaults to 1.0.
        precision (jnp.dtype, optional): Floating point precision.
                                            Defaults to jnp.float32.
        fractional_coordinates (bool, optional): Whether to use fractional
                    coordinates or not (needs to be True for periodic systems).
                                            Defaults to False.

    Returns:
        Tuple[callable, callable, callable]: Neighbor functions and energy function.
    """

    # Refuse elements this model was never trained on, before any jit tracing.
    # Both perform_md and perform_min funnel through here, and the set is
    # model-dependent: legacy v1 saw only the eight biomolecular elements, the
    # v2 models Z = 1..85. Past that the reference tables are indexed out of
    # bounds inside jit, so JAX returns a value belonging to some other element
    # instead of raising.
    check_atomic_numbers_supported(
        species, getattr(potential, 'supported_atomic_numbers', None))

    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = to_jax_md_custom(
        potential=potential,
        species=species,
        displacement_or_metric=displacement,
        box=box,
        disable_cell_list=False if fractional_coordinates else True,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        minimum_cell_size_multiplier=minimum_cell_size_multiplier,
    )

    return neighbor_fn, neighbor_fn_lr, energy_or_obs_fn


def box_to_diag(box):
    """Extract diagonal from box matrix, or return as-is if already 1D."""
    if box is not None and hasattr(box, 'ndim') and box.ndim == 2:
        return jnp.diag(box)
    return box


def box_for_trajectory(box):
    """Convert a JAX-MD box to the row-vector convention used by ASE/HDF5.

    JAX-MD represents a general cell as the affine transform ``x = box @ s``
    for column-vector fractional coordinates.  ASE stores lattice vectors as
    rows, so the corresponding cell is ``box.T``.  Vectors and scalars have no
    convention ambiguity and are returned unchanged.
    """
    if box is not None and hasattr(box, 'ndim') and box.ndim == 2:
        return jnp.swapaxes(box, -1, -2)
    return box


def format_pressure_tensor(state, npt_coupling, pressure_unit):
    """Format instantaneous pressure tensor for logging.

    Returns (header_str, value_str) appropriate for the coupling mode:
    - isotropic: 1 number (scalar pressure = trace/3)
    - semi-isotropic: 3 numbers (Pxx, Pyy, Pzz)
    - anisotropic: 9 numbers (full tensor, row-major)
    """
    if not isinstance(state, jax_md.simulate.NPTNoseHooverFlexState):
        return '', ''

    P_tensor = jax_md.simulate.pressure_tensor_flex(state)
    P_tensor = P_tensor / pressure_unit

    coupling = jax_md.simulate._normalize_coupling(npt_coupling)
    if coupling == 'isotropic':
        p_scalar = float(jnp.trace(P_tensor) / 3)
        return 'P [bar]', f'{p_scalar:.1f}'
    elif coupling == 'semi_isotropic':
        pxx, pyy, pzz = float(P_tensor[0, 0]), float(P_tensor[1, 1]), float(P_tensor[2, 2])
        return 'P [bar] (xx,yy,zz)', f'({pxx:.1f},{pyy:.1f},{pzz:.1f})'
    else:  # anisotropic
        vals = [float(P_tensor[i, j]) for i in range(3) for j in range(3)]
        header = 'P [bar] (xx,xy,xz,yx,yy,yz,zx,zy,zz)'
        val_str = '(' + ','.join(f'{v:.1f}' for v in vals) + ')'
        return header, val_str


def log_nbrs_info(nbrs, label: str = 'SR', box=None):
    """Log neighbor list allocation diagnostics."""
    N = nbrs.reference_position.shape[0]
    fmt = nbrs.format.name if hasattr(nbrs.format, 'name') else str(nbrs.format)
    max_occ = nbrs.max_occupancy
    if nbrs.format in (partition.NeighborListFormat.Sparse,
                        partition.NeighborListFormat.OrderedSparse):
        actual_occ = int(jnp.sum(nbrs.idx[0] < N))
    else:
        actual_occ = int(jnp.sum(nbrs.idx < N))
    util = actual_occ / max_occ * 100 if max_occ > 0 else 0
    mem_mb = nbrs.idx[:, :max_occ].size * nbrs.idx.dtype.itemsize / (1024**2)
    logger.info(f'  {label} ({fmt}): occupancy={actual_occ}, allocated={max_occ} '
                f'(util: {util:.1f}%, memory: {mem_mb:.0f} MB)')
    if nbrs.cell_list_capacity is not None:
        cell_size_ang = nbrs.cell_size
        if box is not None and cell_size_ang is not None:
            box_diag = jnp.diag(box) if box.ndim == 2 else box
            cell_size_ang = float(nbrs.cell_size * jnp.min(box_diag))
        logger.info(f'  {label} cell list: capacity={nbrs.cell_list_capacity}, '
                    f'cell_size={cell_size_ang:.2f} A')


# Per-overflow cell-list headroom, added to cell_capacity on each reallocation.
OVERFLOW_EXTRA_SR = 2
OVERFLOW_EXTRA_LR = 8
MAX_OVERFLOW_RETRIES = 4


def check_overflow(
    neighbor_fn: callable,
    neighbor_fn_lr: callable,
    nbrs: jax_md.partition.NeighborList,
    nbrs_lr: jax_md.partition.NeighborList,
    state,
    box: jnp.ndarray,
    sr_extra: int = 0,
    lr_extra: int = 0,
) -> Tuple[jax_md.partition.NeighborList, jax_md.partition.NeighborList, bool, bool]:
    """

    Check if the neighbor lists have overflowed and reallocate them if needed.

    Args:
        neighbor_fn (callable): Function that creates the short range neighbor list.
        neighbor_fn_lr (callable): Function that creates the long range neighbor list.
        nbrs (jax_md.partition.NeighborList): Short-range neighbor list.
        nbrs_lr (jax_md.partition.NeighborList): Long-range neighbor list.
        state (jax_md.simulate.State): State of the simulation.
        box (jnp.ndarray): Box of the system.

    Returns:
        Tuple[jax_md.partition.NeighborList, jax_md.partition.NeighborList, bool]:
        Updated neighbor lists and a boolean indicating if the lists overflowed.
    """

    sr_overflow = False
    lr_overflow = False
    box_diag = box_to_diag(box)
    if nbrs.did_buffer_overflow:
        sr_overflow = True
        logger.info(f'Neighbor list overflowed, reallocating (SR extra_capacity={sr_extra}).')
        nbrs = neighbor_fn.allocate(state.position, box=box_diag, extra_capacity=sr_extra)
        log_nbrs_info(nbrs, 'SR', box=box)
    if nbrs_lr is not None:
        if nbrs_lr.did_buffer_overflow:
            lr_overflow = True
            logger.info(f'Long-range neighbor list overflowed, reallocating (LR extra_capacity={lr_extra}).')
            nbrs_lr = neighbor_fn_lr.allocate(state.position, box=box_diag, extra_capacity=lr_extra)
            log_nbrs_info(nbrs_lr, 'LR', box=box)
    return nbrs, nbrs_lr, sr_overflow, lr_overflow


def compute_quantities(
    energy_fn: callable,
    state,
    nbrs: jax_md.partition.NeighborList,
    nbrs_lr: jax_md.partition.NeighborList,
    box: jnp.ndarray,
    k_grid: jnp.ndarray,
    unit: dict,
    ensemble: str,
    T: float,
    P: float = None,
    k_grid_disp: jnp.ndarray = None,
) -> Tuple[float, float, float, float, float]:
    """
    Compute the kinetic energy, potential energy, Hamiltonian, temperature
    and pressure of the system.
    *** Warning:Pressure calculation is too memory intensive for large systems 
        so it just returns 0 pressure at the moment ***

    Args:
        energy_fn (callable): Function that calculates the energy.
        state (jax_md.simulate.State): State of the simulattion.
        nbrs (jax_md.partition.NeighborList): Short-range neighbor list.
        nbrs_lr (jax_md.partition.NeighborList): Long-range neighbor list.
        box (jnp.ndarray): Box of the system.
        unit (dict): Dictionary containing the units.
        T (float, optional): Target temperature. Defaults to None.
        P (float, optional): Target pressure. Defaults to None.

    Returns:
        Tuple[float, float, float, float, float]: Kinetic energy, potential 
                             energy, Hamiltonian, temperature and pressure.
    """

    KE = jax_md.quantity.kinetic_energy(
        momentum=state.momentum,
        mass=state.mass
    )
    H = None

    if nbrs_lr is not None:
        PE = energy_fn(
            state.position,
            neighbor=nbrs.idx,
            neighbor_lr=nbrs_lr.idx,
            box=box,
            k_grid=k_grid,
            k_grid_disp=k_grid_disp
        )

        if ensemble == 'nvt':
            H = jax_md.simulate.nvt_nose_hoover_invariant(
                energy_fn,
                state,
                kT=T,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                box=box,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            ) / unit['energy']
        elif ensemble == 'nvt-langevin':
            H = KE + PE
        elif ensemble == 'npt':
            if isinstance(state, jax_md.simulate.NPTNoseHooverFlexState):
                H = jax_md.simulate.npt_nose_hoover_flex_invariant(
                    energy_fn,
                    state,
                    pressure=P,
                    kT=T,
                    neighbor=nbrs.idx,
                    neighbor_lr=nbrs_lr.idx,
                    k_grid=k_grid,
                    k_grid_disp=k_grid_disp
                )
            else:
                H = jax_md.simulate.npt_nose_hoover_invariant(
                    energy_fn,
                    state,
                    pressure=P,
                    kT=T,
                    neighbor=nbrs.idx,
                    neighbor_lr=nbrs_lr.idx,
                    k_grid=k_grid,
                    k_grid_disp=k_grid_disp
                )
        elif ensemble == 'nve':
            # NVE: the conserved quantity is the total energy E = KE + PE.
            H = KE + PE
        else:
            H = None

    else:
        PE = energy_fn(
            state.position,
            neighbor=nbrs.idx,
            box=box,
            k_grid=k_grid,
            k_grid_disp=k_grid_disp
        )
        if ensemble == 'nvt':
            H = jax_md.simulate.nvt_nose_hoover_invariant(
                energy_fn,
                state,
                kT=T,
                neighbor=nbrs.idx,
                box=box,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            ) / unit['energy']
        elif ensemble == 'nvt-langevin':
            H = KE + PE
        elif ensemble == 'npt':
            H = jax_md.simulate.npt_nose_hoover_invariant(
                energy_fn,
                state,
                pressure=P,
                kT=T,
                neighbor=nbrs.idx,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )
        elif ensemble == 'nve':
            # NVE: the conserved quantity is the total energy E = KE + PE.
            H = KE + PE
        else:
            H = None

    current_T = jax_md.quantity.temperature(
        momentum=state.momentum,
        mass=state.mass
    ) / unit['temperature']

    # Too memory intensive to calculate pressure for large systems
    if P is not None:
        if nbrs_lr is not None:
            # current_P = jax_md.quantity.pressure(
            #    energy_fn,
            #    state.position,
            #    box=box,
            #    neighbor=nbrs.idx,
            #    neighbor_lr=nbrs_lr.idx,
            #    kinetic_energy=KE,
            #    k_grid=k_grid
            # )
            current_P = 0.0
        else:
            # current_P = jax_md.quantity.pressure(
            #    energy_fn,
            #    state.position,
            #    box=box,
            #    neighbor=nbrs.idx,
            #    kinetic_energy=KE,
            #    k_grid=k_grid
            # )
            current_P = 0.0
    else:
        current_P = None

    return KE, PE, H, current_T, current_P

def create_nve_fn(
    energy_fn: callable,
    shift: callable,
    dt: float,
    T: float,
    box: jnp.ndarray,
    lr: bool,
)-> Tuple[callable, callable]:
    """
    Create the NVE init/apply functions.

    Args:
        energy_fn (callable): Function that calculates the energy.
        shift (callable): Function that shifts the positions.
        dt (float): Time step.
        box (jnp.ndarray): Box of the system.
        lr (bool): whether to use long-range interactions.

    Returns:
        Tuple[callable, callable]: Init and apply functions.
    """

    init_fn, apply_fn = jax_md.simulate.nve(
        energy_fn,
        shift,
        dt=dt,
        kT=T
    )

    init_fn = jax.jit(init_fn)
    apply_fn = jax.jit(apply_fn)

    step_md_fn = create_md_fn('nve',lr,apply_fn)

    return init_fn, step_md_fn

def create_nhc_fn(
    energy_fn: callable,
    shift: callable,
    dt: float,
    T: float,
    box: jnp.ndarray,
    nhc_kwargs: dict,
    lr: bool
) -> Tuple[callable, callable]:
    """
    Create the NHC thermostat functions.

    Args:
        energy_fn (callable): Function that calculates the energy.
        shift (callable): Function that shifts the positions.
        dt (float): Time step.
        T (float): Target temperature.
        box (jnp.ndarray): Box of the system.
        nhc_kwargs (dict): Settings for the NHC thermostat.
        lr (bool): whether to use long-range interactions.

    Returns:
        Tuple[callable, callable]: Init and apply functions.
    """

    init_fn, apply_fn = jax_md.simulate.nvt_nose_hoover(
        energy_fn,
        shift,
        dt=dt,
        kT=T,
        **nhc_kwargs
    )
    init_fn = jax.jit(init_fn)
    apply_fn = jax.jit(apply_fn)

    step_md_fn = create_md_fn('nvt', lr, apply_fn, T)

    return init_fn, step_md_fn

def create_langevin_fn(
    energy_fn: callable,
    shift: callable,
    dt: float,
    T: float,
    box: jnp.ndarray,
    langevin_kwargs: dict,
    lr: bool
) -> Tuple[callable, callable]:
    """
    Create the Langevin Thermostat functions.

    Args:
        energy_fn (callable): Function that calculates the energy.
        shift (callable): Function that shifts the positions.
        dt (float): Time step.
        T (float): Target temperature.
        box (jnp.ndarray): Box of the system.
        nhc_kwargs (dict): Settings for the NHC thermostat.
        lr (bool): whether to use long-range interactions.

    Returns:
        Tuple[callable, callable]: Init and apply functions.
    """

    init_fn, apply_fn = jax_md.simulate.nvt_langevin(
        energy_fn,
        shift,
        dt=dt,
        kT=T,
        **langevin_kwargs
    )
    init_fn = jax.jit(init_fn)
    apply_fn = jax.jit(apply_fn)

    step_md_fn = create_md_fn('nvt', lr, apply_fn, T)

    return init_fn, step_md_fn

def create_npt_nhc_fn(
    energy_fn,
    shift,
    dt,
    T,
    P,
    nhc_kwargs,
    barostat_kwargs,
    lr,
    npt_coupling='isotropic'
):
    if npt_coupling == 'isotropic':
        init_fn, apply_fn = jax_md.simulate.npt_nose_hoover(
            energy_fn,
            shift,
            dt=dt,
            kT=T,
            pressure=P,
            barostat_kwargs=barostat_kwargs,
            thermostat_kwargs=nhc_kwargs
        )
        init_fn = jax.jit(init_fn)
        apply_fn = jax.jit(apply_fn)
        step_md_fn = create_md_fn("npt", lr, apply_fn, T, P)
    else:
        # Use full float32 matmul precision to avoid TF32 drift on GPUs
        # jax.config.update("jax_default_matmul_precision", "highest")
        # logger.info("Flex NPT: matmul precision set to 'highest' (recommended)")
        init_fn, apply_fn = jax_md.simulate.npt_nose_hoover_flex(
            energy_fn,
            shift,
            dt=dt,
            kT=T,
            pressure=P,
            coupling=npt_coupling,
            barostat_kwargs=barostat_kwargs,
            thermostat_kwargs=nhc_kwargs
        )
        init_fn = jax.jit(init_fn)
        apply_fn = jax.jit(apply_fn)
        step_md_fn = create_md_fn("npt_flex", lr, apply_fn, T, P)

    return init_fn, step_md_fn

def create_nve_step_fn(
    lr: bool,
    apply_fn: callable,
)-> callable:
    """
    Create the NVE step function.

    Args:
        lr (bool): Whether to use long-range interactions.
        apply_fn (callable): Function that applies the MD step.

    Returns:
        callable: NVT step function.
    """

    if lr:
        @jax.jit
        def step_nve_fn_lr(
            i: int,
            state):
            state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = state
            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                box=box,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )
            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )
            nbrs_lr = nbrs_lr.update(
                state.position,
                neighbor=nbrs_lr.idx,
                box=box
            )
            return state, nbrs, nbrs_lr, box, k_grid, k_grid_disp
        return step_nve_fn_lr

    else:
        @jax.jit
        def step_nve_fn(
            i: int,
            state):
            state, nbrs, box, k_grid, k_grid_disp = state
            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                box=box,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )
            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )
            return state, nbrs, box, k_grid, k_grid_disp
        return step_nve_fn

def create_nvt_step_fn(
    lr: bool,
    apply_fn: callable,
    T: float,
) -> callable:
    """
    Create the NVT step function.

    Args:
        lr (bool): Whether to use long-range interactions.
        apply_fn (callable): Function that applies the MD step.
        T (float): Target temperature.

    Returns:
        callable: NVT step function.
    """

    if lr:
        @jax.jit
        def step_nvt_fn_lr(i: int, state):
            state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = state
            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                kT=T,
                box=box,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )
            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )
            nbrs_lr = nbrs_lr.update(
                state.position,
                neighbor=nbrs_lr.idx,
                box=box
            )
            return state, nbrs, nbrs_lr, box, k_grid, k_grid_disp
        return step_nvt_fn_lr
    else:
        @jax.jit
        def step_nvt_fn(i: int, state):
            state, nbrs, box, k_grid, k_grid_disp = state
            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                kT=T,
                box=box,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )
            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )
            return state, nbrs, box, k_grid, k_grid_disp
        return step_nvt_fn


def create_npt_step_fn(
    lr: bool,
    apply_fn: callable,
    T: float,
    P: float
) -> callable:
    """
    Create the NPT step function.

    Args:
        lr (bool): Whether to use long-range interactions.
        apply_fn (callable): Function that applies the MD step.
        T (float): Target temperature.
        P (float): Target pressure.

    Returns:
        callable: NPT step function.
    """

    if lr:
        @jax.jit
        def step_npt_fn(i, state):
            state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = state

            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                kT=T,
                pressure=P,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )

            box = jax_md.simulate.npt_box(state)

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )

            nbrs_lr = nbrs_lr.update(
                state.position,
                neighbor=nbrs_lr.idx,
                box=box
            )

            return state, nbrs, nbrs_lr, box, k_grid, k_grid_disp
        return step_npt_fn
    else:
        @jax.jit
        def step_npt_fn(i, state):
            state, nbrs, box, k_grid, k_grid_disp = state

            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                kT=T,
                pressure=P,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )

            box = jax_md.simulate.npt_box(state)

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )

            return state, nbrs, box, k_grid, k_grid_disp
        return step_npt_fn


def create_npt_flex_step_fn(
    lr: bool,
    apply_fn: callable,
    T: float,
    P: float
) -> callable:
    """
    Create the NPT step function for flexible cell (anisotropic/semi-isotropic).

    Args:
        lr (bool): Whether to use long-range interactions.
        apply_fn (callable): Function that applies the MD step.
        T (float): Target temperature.
        P (float): Target pressure.

    Returns:
        callable: NPT flex step function.
    """

    if lr:
        @jax.jit
        def step_npt_flex_fn(i, state):
            state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = state

            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                kT=T,
                pressure=P,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )

            box = jax_md.simulate.npt_box_flex(state)
            box_diag = box_to_diag(box)

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box_diag
            )

            nbrs_lr = nbrs_lr.update(
                state.position,
                neighbor=nbrs_lr.idx,
                box=box_diag
            )

            return state, nbrs, nbrs_lr, box, k_grid, k_grid_disp
        return step_npt_flex_fn
    else:
        @jax.jit
        def step_npt_flex_fn(i, state):
            state, nbrs, box, k_grid, k_grid_disp = state

            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                kT=T,
                pressure=P,
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )

            box = jax_md.simulate.npt_box_flex(state)

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box_to_diag(box)
            )

            return state, nbrs, box, k_grid, k_grid_disp
        return step_npt_flex_fn


def create_md_fn(
    ensemble: str,
    lr: bool,
    apply_fn: callable,
    T: float = None,
    P: float = None
) -> callable:
    """
    Create the MD step function for the given ensemble.

    Args:
        ensemble (str): String specifying the ensemble.
        lr (bool): Whether to use long-range interactions.
        apply_fn (callable): Function that applies the MD step.
        T (float, optional): Target temperature. Defaults to None.
        P (float, optional): Target pressure. Defaults to None.

    Raises:
        NotImplementedError: If ensemble is not NVT or NPT.

    Returns:
        callable: MD step function.
    """
    ensemble = ensemble.lower()
    if ensemble == 'nve':
        return create_nve_step_fn(lr, apply_fn)
    elif ensemble == 'nvt':
        return create_nvt_step_fn(lr, apply_fn, T)
    elif ensemble == 'npt':
        return create_npt_step_fn(lr, apply_fn, T, P)
    elif ensemble == 'npt_flex':
        return create_npt_flex_step_fn(lr, apply_fn, T, P)
    else:
        raise NotImplementedError(
            f'Ensemble "{ensemble}" is not supported. Only NVE, NVT and NPT ensembles are currently implemented.')

def create_obs_fn(
    energy_or_obs_fn: callable,
    observables: Sequence[str],
    lr: bool,
)-> Tuple[callable, MutableMapping[str, list]]:
    """
    Creates an observation function for monitoring simulation observables.

    Args:
        energy_or_obs_fn (callable): A function that computes the energy or observables of the system.
        observables (Sequence[str]): A list of observable names to monitor during the simulation.
        lr (bool): A flag indicating whether to use a low-rank approximation.

    Returns:
        Tuple[callable, MutableMapping[str, list]]: A tuple containing:
            - A callable observation function.
            - A mutable dictionary mapping observable names to their respective lists of recorded values.
    """

    obs_dict = {o: [] for o in observables}

    if lr:
        @jax.jit
        def obs_fn(state):
            state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = state
            return  energy_or_obs_fn(
                        state.position,
                        neighbor=nbrs.idx,
                        neighbor_lr=nbrs_lr.idx,
                        box=box,
                        k_grid=k_grid,
                        k_grid_disp=k_grid_disp,
                        has_aux=True
                    )
    else:
        @jax.jit
        def obs_fn(state):
            # The caller always passes the 6-element carry, long-range or not.
            # Unpacking only five names here raised "too many values to unpack"
            # at the end of the first MD cycle for any short-range-only model.
            # nbrs_lr is unused on this branch.
            state, nbrs, _nbrs_lr, box, k_grid, k_grid_disp = state
            return  energy_or_obs_fn(
                        state.position,
                        neighbor=nbrs.idx,
                        box=box,
                        k_grid=k_grid,
                        k_grid_disp=k_grid_disp,
                        has_aux=True
                    )

    return obs_fn, obs_dict

def setup_kspace_grid(
    kspace_electrostatics: str,
    kspace_smearing: float,
    kspace_spacing: float,
    box: jnp.ndarray,
    k_grid: jnp.ndarray = None
    )-> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Setup the k-space grid for Ewald summation.
    Delegates to so3lr.mlff.utils.kspace_utils.setup_kspace_grid.
    """
    return _setup_kspace_grid(
        kspace_electrostatics=kspace_electrostatics,
        kspace_smearing=kspace_smearing,
        kspace_spacing=kspace_spacing,
        box=box,
        k_grid=k_grid,
    )

def check_kspace_grid(
    k_grid: jnp.array,
    kspace_electrostatics: str,
    kspace_smearing: float,
    kspace_spacing: float,
    box: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Check and setup the k-space grid for Ewald summation.

    Args:
        kspace_electrostatics (str): Type of k-space electrostatics (None or 'ewald' or 'pme').
        kspace_smearing (float): Smearing value for the k-space grid.
        kspace_spacing (float): Spacing for the k-space grid.
        box (jnp.ndarray): Box of the system.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: k-space grid and smearing values.
    """

    if kspace_electrostatics is None:
        return None
    else:
        k_spacing = jnp.array([kspace_spacing], dtype=jnp.float32)
        k_smearing = jnp.array([kspace_smearing], dtype=jnp.float32)
        if kspace_electrostatics == 'ewald':
            if k_grid.shape != jaxpme.kspace.get_kgrid_ewald_shape(box,lr_wavelength=k_spacing):
                k_grid = setup_kspace_grid(kspace_electrostatics, kspace_smearing, kspace_spacing, box)[0]
                logger.info(f'k-grid shape: {k_grid.shape}')
                return k_grid
        elif kspace_electrostatics == 'pme' or kspace_electrostatics == 'p3m':
            if k_grid.shape != jaxpme.kspace.get_kgrid_mesh_shape(box,mesh_spacing=k_spacing):
                k_grid = setup_kspace_grid(kspace_electrostatics, kspace_smearing, kspace_spacing, box)[0]
                logger.info(f'k-grid shape: {k_grid.shape}')
                return k_grid
        else:
            raise ValueError(
                f'Invalid kspace_electrostatics value: {kspace_electrostatics}. '
                'Expected None, "ewald", "pme", or "p3m".'
            )


def validate_and_log_input(initial_geometry, input_file_path,
                           total_charge=None, num_unpaired_electrons=None,
                           verbose=True):
    """Validate a parsed structure and (optionally) log a human-readable summary.

    Raises a clear error if positions or atomic numbers are missing. When
    ``verbose`` is True, logs what was read (counts, elements, periodicity,
    charge/spin) plus a short example of the first atoms, and warns about large
    net charge/spin in periodic cells. Pass ``verbose=False`` to validate only
    (avoids duplicate logging when a later stage logs the full summary).
    """
    n_atoms = len(initial_geometry)
    if n_atoms == 0:
        raise ValueError(f"Input geometry file {input_file_path} contains no atoms.")

    atomic_numbers = initial_geometry.get_atomic_numbers()
    if atomic_numbers is None or len(atomic_numbers) == 0:
        raise ValueError(f"Input geometry file {input_file_path} has no atomic numbers (atom types).")

    positions = initial_geometry.get_positions()
    if positions is None or len(positions) == 0:
        raise ValueError(f"Input geometry file {input_file_path} has no atomic positions.")

    if not verbose:
        return

    symbols = initial_geometry.get_chemical_symbols()
    elements = sorted(set(symbols))
    is_periodic = bool(np.any(initial_geometry.get_pbc()))

    logger.info(f"Read {n_atoms} atoms ({initial_geometry.get_chemical_formula()}) from {input_file_path}")
    logger.info(f"  Elements:     {', '.join(elements)}")
    if is_periodic:
        diag = np.diag(np.array(initial_geometry.get_cell()))
        logger.info(f"  Periodic:     yes  (cell lengths along x,y,z = {diag[0]:.3f}, {diag[1]:.3f}, {diag[2]:.3f} Å)")
    else:
        logger.info(f"  Periodic:     no")
    if total_charge is not None:
        logger.info(f"  Total charge: {total_charge}    Spin (unpaired e-): {num_unpaired_electrons}")

    # Short example of what was parsed, so users can confirm the input.
    logger.info("  Example (first atoms read: symbol  x  y  z [Å]):")
    for i in range(min(3, n_atoms)):
        x, y, z = positions[i]
        logger.info(f"    {symbols[i]:>2}  {x:11.5f} {y:11.5f} {z:11.5f}")

    # Warn about poorly-behaved electrostatics/spin in periodic cells.
    if is_periodic and total_charge is not None and abs(total_charge) > 2:
        logger.warning(
            f"Net charge {total_charge:+} e in a periodic cell — long-range electrostatics "
            f"assume a neutral cell; consider neutralizing the system (counter-ions or a "
            f"uniform background charge).")
    if is_periodic and num_unpaired_electrons is not None and abs(num_unpaired_electrons) > 2:
        logger.warning(
            f"Large spin ({num_unpaired_electrons} unpaired electrons) in a periodic cell — "
            f"verify this is intended.")


def load_atom_masks(
    mask_path: Optional[Union[str, Path]],
    n_atoms: int,
    mobile_atom_indices: Optional[Sequence[int]] = None,
    fixed_atom_indices: Optional[Sequence[int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load mobile/fixed atom indices and return ``(mobile, fixed)`` arrays."""
    mobile = None
    fixed = None

    if mask_path:
        data = np.load(str(mask_path))
        if 'mobile_atom_indices' in data:
            mobile = np.asarray(data['mobile_atom_indices'], dtype=int)
        if 'fixed_atom_indices' in data:
            fixed = np.asarray(data['fixed_atom_indices'], dtype=int)

    if mobile_atom_indices is not None:
        mobile = np.asarray(mobile_atom_indices, dtype=int)
    if fixed_atom_indices is not None:
        fixed = np.asarray(fixed_atom_indices, dtype=int)

    all_indices = np.arange(n_atoms, dtype=int)
    if mobile is None and fixed is None:
        return None, None
    if mobile is None:
        fixed_set = set(np.asarray(fixed, dtype=int).tolist())
        mobile = np.asarray([i for i in all_indices if i not in fixed_set], dtype=int)
    if fixed is None:
        mobile_set = set(np.asarray(mobile, dtype=int).tolist())
        fixed = np.asarray([i for i in all_indices if i not in mobile_set], dtype=int)

    if np.any(mobile < 0) or np.any(mobile >= n_atoms):
        raise ValueError('mobile_atom_indices contains out-of-range indices')
    if np.any(fixed < 0) or np.any(fixed >= n_atoms):
        raise ValueError('fixed_atom_indices contains out-of-range indices')
    return mobile, fixed


def make_mobile_mask(mobile_indices: Optional[np.ndarray], n_atoms: int, dtype) -> Optional[jnp.ndarray]:
    if mobile_indices is None:
        return None
    mask = np.zeros((n_atoms, 1), dtype=np.float32)
    mask[np.asarray(mobile_indices, dtype=int), 0] = 1.0
    return jnp.asarray(mask, dtype=dtype)


def positions_to_cartesian(position, box, fractional_coordinates: bool):
    if fractional_coordinates and box is not None:
        return raw_transform(box, position)
    return position


def cartesian_to_positions(cartesian_position, box, fractional_coordinates: bool):
    if not fractional_coordinates or box is None:
        return cartesian_position
    if jnp.isscalar(box) or box.size == 1:
        return cartesian_position / box
    if box.ndim == 1:
        return cartesian_position / box
    if box.ndim == 2:
        return jnp.linalg.solve(box, cartesian_position.T).T
    raise ValueError(f'Box must be scalar, vector, or matrix. Found {box}.')


def apply_atom_constraints_to_state(
    state,
    mobile_mask: Optional[jnp.ndarray],
    reference_position,
    box=None,
    fractional_coordinates: bool = False,
    reference_is_cartesian: bool = False,
):
    if mobile_mask is None:
        return state
    if reference_is_cartesian:
        reference_position = cartesian_to_positions(reference_position, box, fractional_coordinates)
    zero_momentum = jnp.zeros_like(state.momentum)
    zero_force = jnp.zeros_like(state.force)
    return state.set(
        position=jnp.where(mobile_mask == 1, state.position, reference_position),
        momentum=jnp.where(mobile_mask == 1, state.momentum, zero_momentum),
        force=jnp.where(mobile_mask == 1, state.force, zero_force),
    )


def constrain_md_carry(
    carry,
    lr: bool,
    mobile_mask: Optional[jnp.ndarray],
    reference_position,
    fractional_coordinates: bool,
    reference_is_cartesian: bool,
):
    if mobile_mask is None:
        return carry
    if lr:
        state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = carry
        state = apply_atom_constraints_to_state(
            state,
            mobile_mask,
            reference_position,
            box=box,
            fractional_coordinates=fractional_coordinates,
            reference_is_cartesian=reference_is_cartesian,
        )
        box_for_nbrs = box_to_diag(box) if hasattr(box, 'ndim') and box.ndim == 2 else box
        nbrs = nbrs.update(state.position, neighbor=nbrs.idx, box=box_for_nbrs)
        nbrs_lr = nbrs_lr.update(state.position, neighbor=nbrs_lr.idx, box=box_for_nbrs)
        return state, nbrs, nbrs_lr, box, k_grid, k_grid_disp
    state, nbrs, box, k_grid, k_grid_disp = carry
    state = apply_atom_constraints_to_state(
        state,
        mobile_mask,
        reference_position,
        box=box,
        fractional_coordinates=fractional_coordinates,
        reference_is_cartesian=reference_is_cartesian,
    )
    box_for_nbrs = box_to_diag(box) if hasattr(box, 'ndim') and box.ndim == 2 else box
    nbrs = nbrs.update(state.position, neighbor=nbrs.idx, box=box_for_nbrs)
    return state, nbrs, box, k_grid, k_grid_disp


def wrap_step_with_constraints(
    step_md_fn,
    lr: bool,
    mobile_mask: Optional[jnp.ndarray],
    reference_position,
    fractional_coordinates: bool,
    reference_is_cartesian: bool,
):
    if mobile_mask is None:
        return step_md_fn

    @jax.jit
    def constrained_step(i, carry):
        return constrain_md_carry(
            step_md_fn(i, carry),
            lr,
            mobile_mask,
            reference_position,
            fractional_coordinates,
            reference_is_cartesian,
        )

    return constrained_step


def output_position_array(position, box, fractional_coordinates: bool, output_cartesian_positions: bool):
    if fractional_coordinates and output_cartesian_positions and box is not None:
        return positions_to_cartesian(position, box, fractional_coordinates)
    return position


def save_restart_checkpoints(
    state,
    box,
    current_cycle: int,
    *,
    ensemble: str,
    fractional_coordinates: bool,
    restart_save_path: Optional[str] = None,
    pysages_result=None,
    restart_pysages_save_path: Optional[str] = None,
) -> None:
    """Persist SO3LR and optional PySAGES state for one accepted cycle."""
    if restart_save_path is not None:
        save_state(
            state,
            box,
            current_cycle,
            restart_save_path,
            ensemble=ensemble,
            fractional_coordinates=fractional_coordinates,
        )
    if restart_pysages_save_path is not None:
        save_pysages_state(pysages_result, restart_pysages_save_path)


def perform_md(
    all_settings: Dict,
    opt_structure: Optional[jnp.ndarray] = None,
    restart: Optional[bool] = False,
    log_input: bool = True,
) -> None:
    """
    Perform molecular dynamics simulation with the given settings.

    Args:
        all_settings (Dict): Settings for the MD simulation.
        opt_structure (Optional[jnp.ndarray], optional): Optimized structure, if 
                                    already available. Defaults to None.
    """
    # Setup logger
    log_file = all_settings.get('log_file')
    setup_logger(log_file)

    # Extract settings with defaults
    input_file = all_settings.get('input_file')
    output_file = all_settings.get('output_file')
    restart_save_path = all_settings.get('restart_save_path')
    restart_load_path = all_settings.get('restart_load_path')
    overwrite = bool(all_settings.get('overwrite', False))

    # Refuse to clobber existing output unless the user opted in via
    # --overwrite. The previous behavior was format-dependent: HDF5 silently
    # truncated, extxyz silently appended. With this guard the policy is
    # uniform across formats and across restarts.
    if (output_file and Path(output_file).exists() and not overwrite
            and not restart_load_path
            and not all_settings.get('_relax_wrote_output')):
        raise RuntimeError(
            f"Output file already exists: {output_file}. Refusing to overwrite. "
            f"Pass --overwrite to overwrite, choose a different --output, or "
            f"pass --restart-load to resume from a checkpoint."
        )

    # Model parameters
    model_path = all_settings.get('model_path')
    precision = all_settings.get('precision')
    lr_cutoff = all_settings.get('lr_cutoff')
    dispersion_damping = all_settings.get('dispersion_damping')
    buffer_size_multiplier_sr = all_settings.get('buffer_size_multiplier_sr')
    buffer_size_multiplier_lr = all_settings.get('buffer_size_multiplier_lr')
    minimum_cell_size_multiplier = all_settings.get('minimum_cell_size_multiplier', 1.0)
    total_charge = all_settings.get('total_charge')
    num_unpaired_electrons = all_settings.get('num_unpaired_electrons', 0)

    # Ewald electrostatics settings
    kspace_electrostatics = all_settings.get('kspace_electrostatics',None)
    kspace_smearing = all_settings.get('kspace_smearing', 4.0)
    kspace_spacing = all_settings.get('kspace_spacing', 2.0)
    kspace_interp_nodes = all_settings.get('kspace_interp_nodes', 4)
    charge_embed_total_charge = all_settings.get('charge_embed_total_charge', None)
    charge_equilibration_total_charge = all_settings.get('charge_equilibration_total_charge', None)
    kspace_npt_cycles = all_settings.get('kspace_npt_cycles', 100)

    # Dispersion PME settings
    kspace_dispersion = all_settings.get('kspace_dispersion', None)
    kspace_dispersion_smearing = all_settings.get('kspace_dispersion_smearing', 2.0)
    kspace_dispersion_spacing = all_settings.get('kspace_dispersion_spacing', 3.0)
    kspace_dispersion_interp_nodes = all_settings.get('kspace_dispersion_interp_nodes', 4)

    # MD parameters
    # md_dt is in PICOSECONDS here (the CLI YAML loader converted fs -> ps
    # in so3lr_cli.parse_settings; see md_settings.yaml for the fs convention).
    # It stays in ps until handle_units() below converts it to jax-md natural
    # units for the integrator.
    md_dt = all_settings.get('md_dt')
    md_T = all_settings.get('md_T')
    md_P = all_settings.get('md_P')
    md_cycles = all_settings.get('md_cycles')
    md_steps = all_settings.get('md_steps')
    ensemble = all_settings.get('ensemble')

    # NVE has no thermostat, so 'md_T' only seeds the initial velocity sample.
    # If the user specified 'init_T' (documented in md_settings.yaml under the
    # NVE section), honour it. The CLI subcommand path already sets both to the
    # same --temperature value; this path matters for YAML-only NVE runs.
    if ensemble == 'nve':
        _init_T = all_settings.get('init_T')
        if _init_T is not None:
            md_T = _init_T

    # Nose-Hoover thermostat parameters
    nhc_chain_length = all_settings.get('nhc_chain_length')
    nhc_steps = all_settings.get('nhc_steps')
    nhc_thermo = all_settings.get('nhc_thermo')
    nhc_baro = all_settings.get('nhc_baro')
    nhc_sy_steps = all_settings.get('nhc_sy_steps')

    # NPT coupling mode: 'isotropic', 'semi-isotropic', or 'anisotropic'
    npt_coupling = all_settings.get('npt_coupling', 'isotropic')

    # Langevin thermostat parameters
    langevin_removecmmotion = all_settings.get('langevin_removecmmotion')
    langevin_thermo = all_settings.get('langevin_thermo')

    # langevin_thermo is the dimensionless timescale used to build the friction
    # gamma = 1 / (md_dt * langevin_thermo). A zero (or negative) value silently
    # produces ZeroDivisionError / inf gamma far downstream; catch it here.
    if ensemble == 'nvt-langevin' and (langevin_thermo is None or langevin_thermo <= 0):
        raise ValueError(
            f"--langevin-thermo must be > 0 (received {langevin_thermo}). "
            f"It is a dimensionless timescale; the friction is gamma = 1 / (md_dt * langevin_thermo)."
        )

    # Format control
    output_format = all_settings.get('output_format')
    save_buffer = all_settings.get('save_buffer')
    seed = all_settings.get('seed')
    relax_before_run = all_settings.get('relax_before_run')
    observables = all_settings.get('observables', [])
    output_atom_indices = all_settings.get('output_atom_indices', None)
    dispersion_tail_correction = all_settings.get('dispersion_tail_correction', False)

    # Atom constraints / enhanced output (optional)
    output_cartesian_positions = all_settings.get('output_cartesian_positions', False)
    atom_mask_path = all_settings.get('atom_mask_path')
    mobile_atom_indices_setting = all_settings.get('mobile_atom_indices')
    fixed_atom_indices_setting = all_settings.get('fixed_atom_indices')

    # Normalise the constraint mode so spellings like 'Cartesian_anchor' or
    # 'cartesian-anchor' do not silently fall back to fractional anchoring
    # (which under NPT lets the "frozen" atoms physically drift when the box
    # rescales). Recognised values: 'cartesian_anchor', 'fractional'.
    _allowed_constraint_modes = {'cartesian_anchor', 'fractional'}
    fixed_atom_constraint_raw = all_settings.get('fixed_atom_constraint', 'cartesian_anchor')
    fixed_atom_constraint = str(fixed_atom_constraint_raw).strip().lower().replace('-', '_')
    if fixed_atom_constraint not in _allowed_constraint_modes:
        raise ValueError(
            f"Unknown fixed_atom_constraint={fixed_atom_constraint_raw!r}. "
            f"Allowed values: {sorted(_allowed_constraint_modes)}. "
            f"Note: 'cartesian_anchor' (default, NPT-safe) and 'fractional' "
            f"are the only supported modes."
        )

    # Enhanced sampling (PySAGES)
    do_enhanced_sampling = all_settings.get('do_enhanced_sampling', False)
    pysages_input_path = all_settings.get('pysages_input', './pysages_input.yaml')
    restart_pysages_save_path = all_settings.get('restart_pysages_save_path')
    restart_pysages_load_path = all_settings.get('restart_pysages_load_path')
    enforce_cartesian_coordinates = all_settings.get('enforce_cartesian_coordinates', False)

    # This entry point is also a public Python API and may bypass Click/YAML
    # parsing, so enforce the same count/frequency constraints here before any
    # modulo, division, model loading, or output-file mutation occurs.
    validate_positive_md_settings({
        'md_cycles': md_cycles,
        'md_steps': md_steps,
        'save_buffer': save_buffer,
        'kspace_npt_cycles': kspace_npt_cycles,
    })

    if do_enhanced_sampling and not HAS_PYSAGES:
        raise ImportError(
            "PySAGES is required for enhanced sampling but is not installed. "
            "Install it with: pip install so3lr[pysages]"
        )

    # Methods that only add a fixed bias potential (no histogram accumulation
    # or adaptive forces) work correctly with any ensemble.
    PYSAGES_ENSEMBLE_UNRESTRICTED_METHODS = {'harmonic', 'unbiased'}

    if do_enhanced_sampling:
        pysages_set_dict = parse_pysages_input(pysages_input_path)
        pysages_method_name = pysages_set_dict['method'].lower()
        if (pysages_method_name not in PYSAGES_ENSEMBLE_UNRESTRICTED_METHODS
                and ensemble not in ('nvt',)):
            raise ValueError(
                f"Enhanced sampling method '{pysages_method_name}' currently only "
                f"supports the NVT ensemble (Nose-Hoover thermostat), but "
                f"'{ensemble}' was selected. Methods that support all ensembles: "
                f"{', '.join(sorted(PYSAGES_ENSEMBLE_UNRESTRICTED_METHODS))}."
            )

    if do_enhanced_sampling and restart_pysages_load_path is not None:
        restart_pysages = True
    else:
        restart_pysages = False

    # Handling of restart
    if restart_save_path is not None:
        create_restart = True
    else:
        create_restart = False

    # Geometry, cell and charge
    input_file_path = all_settings.get('input_file')
    if input_file_path is None:
        raise ValueError('Initial geometry file path must be provided in settings.')
    try:
        initial_geometry = read(input_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find initial geometry file: {input_file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to read geometry file {input_file_path}: {e}")

    validate_and_log_input(initial_geometry, input_file_path,
                           total_charge=total_charge,
                           num_unpaired_electrons=num_unpaired_electrons,
                           verbose=log_input)
    if opt_structure is not None:
        initial_geometry.set_positions(opt_structure)

    cell = cell_from_atoms_boundary_conditions(
        initial_geometry,
        lr_cutoff,
        kspace_electrostatics,
        kspace_smearing,
    )

    if md_P is not None and cell is None:
        raise ValueError(
            'Cell must be defined for NPT simulations. The input geometry does not contain cell information.')

    # K-space electrostatics (Ewald/PME/P3M) is only meaningful for periodic
    # systems. Without a cell the model still runs but the long-range term is
    # nonsense — refuse so we do not silently produce wrong physics.
    if kspace_electrostatics is not None and cell is None:
        raise ValueError(
            f"kspace_electrostatics={kspace_electrostatics!r} requires a periodic "
            f"cell, but the input geometry has none. Either supply a cell, or "
            f"omit kspace_electrostatics."
        )

    if cell is not None:
        # PySAGES requires Cartesian positions, BUT for NPT we must keep JAX-MD internal
        # coordinates fractional (unit cube). The PySAGES interface performs the
        # Cartesian<->fractional conversion each step.
        if do_enhanced_sampling and ensemble == 'npt':
            if enforce_cartesian_coordinates:
                logger.info(
                    "NPT+PySAGES: overriding enforce_cartesian_coordinates=False "
                    "(JAX-MD NPT requires fractional coordinates internally)."
                )
            enforce_cartesian_coordinates = False
        else:
            # For non-NPT ensembles, PySAGES truly requires internal Cartesian coords.
            if do_enhanced_sampling and not enforce_cartesian_coordinates:
                logger.warning(
                    "PySAGES requires cartesian coordinates for periodic systems. "
                    "Setting enforce_cartesian_coordinates=True."
                )
                enforce_cartesian_coordinates = True

        shift_displacement = 'periodic_cart' if enforce_cartesian_coordinates else 'periodic'
        initial_geometry.wrap()
    else:
        shift_displacement = 'free'

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)

    dtype = jnp.float64 if precision == 'float64' else jnp.float32
    (position, box, displacement, shift, fractional_coordinates) = handle_box(
        shift_displacement, initial_geometry_dict['positions'], cell, precision=dtype)

    mobile_atom_indices, fixed_atom_indices = load_atom_masks(
        atom_mask_path,
        n_atoms=position.shape[0],
        mobile_atom_indices=mobile_atom_indices_setting,
        fixed_atom_indices=fixed_atom_indices_setting,
    )
    mobile_mask = make_mobile_mask(mobile_atom_indices, position.shape[0], dtype)
    if fixed_atom_indices is not None and len(fixed_atom_indices) > 0:
        logger.info(
            f'Fixed atoms: {len(fixed_atom_indices)}; mobile atoms: {len(mobile_atom_indices)}; '
            f'constraint={fixed_atom_constraint}'
        )

    save_buffer = min(md_cycles, save_buffer)
    hdf5_store = None

    current_cycle = 0
    # Loading restart
    if restart:
        logger.info(f'Loading state from {restart_load_path}.')
        try:
            state, box, current_cycle = load_state(
                path_to_load=restart_load_path,
                ensemble=ensemble,
                restart_reset=all_settings.get('restart_reset', False),
                fractional_coordinates=fractional_coordinates,
                precision=dtype
            )
            position = state.position
        except RestartInNewEnsembleError as e:
            logger.warning(e)
            position = e.loaded_state['position']
            initial_geometry_dict['masses'] = e.loaded_state['mass']
            initial_geometry_dict['momenta'] = e.loaded_state['momentum']
            box = e.loaded_state['box']
            if np.all(box == 0) or box is None:
                box = jnp.array([0.0], dtype=jnp.float64 if precision == 'float64' else jnp.float32)
            saved_fractional = bool(e.loaded_state.get('fractional_coordinates', True))
            if saved_fractional and not fractional_coordinates:
                logger.info('Converting restart positions from fractional to cartesian coordinates.')
                position = positions_to_cartesian(position, jnp.array(box), True)
            elif not saved_fractional and fractional_coordinates:
                logger.info('Converting restart positions from cartesian to fractional coordinates.')
                position = cartesian_to_positions(position, jnp.array(box), True)
            restart = False
        except RestartSchemaMismatchError as e:
            # Surface the full schema-mismatch message (missing field, ensemble,
            # available fields) instead of letting it get wrapped into an
            # opaque RuntimeError below.
            raise RuntimeError(str(e)) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load restart state from {restart_load_path}: {str(e)}")

        if restart_pysages:
            restart_data_pysages = load_pysages_state(restart_pysages_load_path)
            # Match the dtypes of the pickled PySAGES result to the run precision,
            # exactly as load_state() does for the .npz state above. The PySAGES
            # snapshot is pickled at whatever dtype the *originating* run carried
            # (a fresh run keeps e.g. the Nose-Hoover chain `tau` at float32 even
            # under x64, because jax-md hardcodes `tau = f32(tau)` in its
            # integrator factories). On restart that pickled snapshot seeds the
            # fori_loop carry's snapshot slot, while the jax-md state comes from
            # load_state() normalized to `dtype`. If the two disagree, the loop
            # carry input (snapshot float32) != output (re-snapshot of the
            # float64 state) and jax.lax.scan rejects it:
            #   loop_carry[...].chain_data['tau']: float32 vs float64
            #
            # The PySAGES Result is a plain class (not a registered pytree), so
            # we normalize its pytree-valued `snapshots` attribute (a list of
            # registered NamedTuples) rather than the Result itself -- a
            # tree_map over the Result would treat it as a single opaque leaf
            # and do nothing.
            #
            # IMPORTANT: only `snapshots` is normalized, NOT `states`. The
            # method `states` (e.g. ABFState: Fsum/Wp/force/bias) are the
            # sampler's internal accumulators, which are *not* part of the
            # integrator carry that `_match_float_dtypes` pins to the run
            # precision. PySAGES always computes them at its float64 working
            # precision (the method view promotes the snapshot to float64 via
            # `_as_default_float`), and a fresh run carries them at float64
            # regardless of the integrator dtype. Casting them down to float32
            # here both (a) silently degrades the accumulated mean force to
            # float32 for the continued run, and (b) breaks ABF's out-of-bounds
            # `cond` in `estimate_force`: the `average_force` branch derives its
            # dtype from `Fsum` (now float32) while the `restraints_force`
            # branch stays at the float64 of the restraint bounds baked into
            # `method`, so `lax.cond` rejects the mismatched branches with
            #   "true_fun output float64[1] vs false_fun output float32[1]".
            # Leaving `states` at their pickled (float64) dtype matches exactly
            # what a fresh run produces.
            def _to_run_precision(x):
                if hasattr(x, 'dtype') and jnp.issubdtype(
                    jnp.asarray(x).dtype, jnp.floating
                ):
                    return jnp.asarray(x).astype(dtype)
                return x

            if restart_data_pysages.snapshots is not None:
                restart_data_pysages.snapshots = jax.tree_util.tree_map(
                    _to_run_precision, restart_data_pysages.snapshots
                )

    patched_potential_kwargs = intermediate_quantity_kwargs(observables)

    # Loading the model. A bundled name (so3lr-s/m/l/so3lr) or a custom path; None
    # falls back to the default model. load_model resolves names and paths alike.
    from so3lr.model_registry import DEFAULT_MODEL, available_models
    selected_model = model_path if model_path is not None else DEFAULT_MODEL
    if selected_model in available_models():
        logger.info(f"Using bundled SO3LR model: {selected_model}")
    else:
        logger.info(f"Using custom MLFF potential from: {selected_model}")
    potential = load_model(
        selected_model,
        precision=jnp.float64 if precision == 'float64' else jnp.float32,
        lr_cutoff=lr_cutoff,
        dispersion_damping=dispersion_damping,
        kspace_electrostatics=kspace_electrostatics,
        kspace_interp_nodes=kspace_interp_nodes,
        kspace_dispersion=kspace_dispersion,
        kspace_dispersion_interp_nodes=kspace_dispersion_interp_nodes,
        charge_embed_total_charge=charge_embed_total_charge,
        charge_equilibration_total_charge=charge_equilibration_total_charge,
        **patched_potential_kwargs
    )

    # Setting up the model
    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = process_model(
        potential=potential,
        species=initial_geometry_dict['species'],
        displacement=displacement,
        box=box,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        minimum_cell_size_multiplier=minimum_cell_size_multiplier,
        has_aux=len(observables)>0,
    )

    k_grid, k_smearing = setup_kspace_grid(
        kspace_electrostatics=kspace_electrostatics,
        kspace_spacing=kspace_spacing,
        kspace_smearing=kspace_smearing,
        box=box
    )

    # Setup dispersion k-space grid (reuse setup_kspace_grid with dispersion params)
    k_grid_disp, k_smearing_disp = setup_kspace_grid(
        kspace_electrostatics=kspace_dispersion,
        kspace_spacing=kspace_dispersion_spacing,
        kspace_smearing=kspace_dispersion_smearing,
        box=box
    )

    energy_or_obs_fn = partial(energy_or_obs_fn, k_smearing=k_smearing, k_smearing_disp=k_smearing_disp)
    energy_fn = jax.jit(partial(energy_or_obs_fn, has_aux=False))

    # Allocating the neighbor lists
    if neighbor_fn_lr is not None:
        lr = True
    else:
        lr = False

    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box) if lr else None

    logger.info('Neighbor list allocated:')
    log_nbrs_info(nbrs, 'SR', box=box)
    if nbrs_lr is not None:
        log_nbrs_info(nbrs_lr, 'LR', box=box)

    # --- Dispersion tail correction for NPT ---
    if dispersion_tail_correction and ensemble == 'npt':
        logger.info('Computing dispersion tail correction coefficients...')

        # Run one forward pass to get Hirshfeld and C6 ratios
        if lr:
            _energy, _aux = energy_or_obs_fn(
                position, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx,
                box=box, k_grid=k_grid, k_grid_disp=k_grid_disp, has_aux=True
            )
        else:
            _energy, _aux = energy_or_obs_fn(
                position, neighbor=nbrs.idx,
                box=box, k_grid=k_grid, k_grid_disp=k_grid_disp, has_aux=True
            )

        # New models expose the ratio as 'a0_ratios'; the legacy model uses
        # 'hirshfeld_ratios'. Accept either so an explicit opt-in works for both.
        _hirshfeld_ratios = _aux.get('a0_ratios', _aux.get('hirshfeld_ratios'))
        if _hirshfeld_ratios is None or 'c6_ratios' not in _aux:
            raise KeyError(
                "dispersion_tail_correction requires per-atom 'a0_ratios'/"
                "'hirshfeld_ratios' and 'c6_ratios' in the model output; the "
                "selected model does not provide them (the legacy model has no "
                "C6 head). Disable dispersion_tail_correction for this model."
            )
        _c6_ratios = _aux['c6_ratios']
        _species = initial_geometry_dict['species']

        # Log per-element diagnostics
        import numpy as _np
        from so3lr.mlff.nn.observable.dispersion_ref_data import alphas as _alphas_ref, C6_coef as _C6_ref
        for _z in _np.unique(_np.asarray(_species)):
            _mask = _np.asarray(_species) == _z
            _hr = _np.asarray(_hirshfeld_ratios)[_mask]
            _cr = _np.asarray(_c6_ratios)[_mask]
            _c6_free = _C6_ref[_z - 1]
            logger.info(
                f'  Z={_z}: count={_mask.sum()}, '
                f'C6_free={_c6_free:.2f}, '
                f'hirshfeld_ratio={_np.mean(_hr):.4f} +/- {_np.std(_hr):.4f}, '
                f'c6_ratio={_np.mean(_cr):.4f} +/- {_np.std(_cr):.4f}, '
                f'effective C6_i={_c6_free * _np.mean(_cr):.2f}'
            )

        C6_avg, C8_avg, C10_avg = compute_avg_dispersion_coefficients(
            _species, _hirshfeld_ratios, _c6_ratios
        )
        logger.info(f'Tail correction coefficients (a.u.): C6_avg={C6_avg:.2f}, C8_avg={C8_avg:.2f}, C10_avg={C10_avg:.2f}')

        # When dispersion PME is active, C6 is handled exactly in k-space — zero out C6 tail correction
        if kspace_dispersion is not None:
            logger.info('Dispersion PME active: setting C6_avg=0 in tail correction (C6 handled by PME).')
            C6_avg = 0.0

        _N = len(_species)
        _r_on = lr_cutoff - dispersion_damping
        _r_off = lr_cutoff

        original_energy_fn = energy_fn
        def energy_fn_with_tail(R, **kwargs):
            E = original_energy_fn(R, **kwargs)
            box_kw = kwargs.get('box', None)
            if box_kw is not None:
                pert = kwargs.get('perturbation', None)
                if pert is not None:
                    box_kw = box_kw * pert
                # dispersion_tail_correction_energy expects 1D box
                if hasattr(box_kw, 'ndim') and box_kw.ndim == 2:
                    box_kw = jnp.diag(box_kw)
                E_tail = dispersion_tail_correction_energy(
                    box_kw, _N, C6_avg, C8_avg, C10_avg, _r_on, _r_off
                )
                E = E + E_tail
            return E
        energy_fn = jax.jit(energy_fn_with_tail)
        logger.info('Dispersion tail correction enabled for NPT.')

    # Apply units
    # Unit boundary: handle_units takes (dt in ps, T in K, P in bar) and returns
    # the same quantities in jax-md natural units. After this block md_dt is
    # NO LONGER in ps — every downstream use (nhc_tau, gamma, step_fn ctors,
    # pysages interface) treats md_dt as a jax-md natural-units timestep.
    # Anything past this point that wants ps must convert back.
    unit_dict = handle_units(
        units.metal_unit_system,
        md_dt,
        md_T,
        md_P,
    )

    md_dt_ps = float(md_dt)  # capture ps value before it becomes jax-md natural units (used by the drift watchdog below)
    md_dt = dtype(unit_dict['dt'])  # ps -> jax-md natural units
    md_T = dtype(unit_dict['T']) if unit_dict['T'] is not None else None
    md_P = dtype(unit_dict['P']) if unit_dict['P'] is not None else None

    # set nhc_tau to 0 for NVE ensemble only
    if ensemble == 'nve':
        nhc_tau = 0
        langevin_thermo = 0
    elif ensemble == 'nvt-langevin':
        nhc_tau = 0
    else:
        nhc_tau = md_dt * nhc_thermo
        langevin_thermo = 0

    nhc_kwargs = {
        'chain_length': nhc_chain_length,
        'chain_steps': nhc_steps,
        'sy_steps': nhc_sy_steps,
        'tau': nhc_tau
    }


    if ensemble == 'npt':
        nhc_barostat_kwargs = nhc_kwargs.copy()
        nhc_barostat_kwargs['tau'] = md_dt * nhc_baro

    if ensemble == 'nvt-langevin':
        langevin_kwargs = {
            'gamma': 1.0 / (md_dt * langevin_thermo),
            'center_velocity': langevin_removecmmotion
        }

    # Setting up the thermostat and/or barostat jax functions

    if ensemble == 'nve':
        init_fn, step_md_fn = create_nve_fn(
            energy_fn,
            shift,
            md_dt,
            md_T,
            box,
            lr,
        )
    elif ensemble == 'nvt':
        init_fn, step_md_fn = create_nhc_fn(
            energy_fn,
            shift,
            md_dt,
            md_T,
            box,
            nhc_kwargs,
            lr
        )
    elif ensemble == 'nvt-langevin':
        init_fn, step_md_fn = create_langevin_fn(
            energy_fn,
            shift,
            md_dt,
            md_T,
            box,
            langevin_kwargs,
            lr
        )
    else:
        logger.info(f"NPT coupling mode:         {npt_coupling}")
        init_fn, step_md_fn = create_npt_nhc_fn(
            energy_fn,
            shift,
            md_dt,
            md_T,
            md_P,
            nhc_kwargs,
            nhc_barostat_kwargs,
            lr,
            npt_coupling=npt_coupling
        )

    rng_key = jax.random.PRNGKey(seed)

    if not restart:
        if lr:
            state = init_fn(
                rng_key,
                position,
                box=box,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                kT=md_T,
                mass=initial_geometry_dict['masses'],
                momenta=initial_geometry_dict.get('momenta'),
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )
        else:
            state = init_fn(
                rng_key,
                position,
                box=box,
                neighbor=nbrs.idx,
                kT=md_T,
                mass=initial_geometry_dict['masses'],
                momenta=initial_geometry_dict.get('momenta'),
                k_grid=k_grid,
                k_grid_disp=k_grid_disp
            )

    # Setup additional observables
    obs_dict={}
    if len(observables)>0:
        obs_fn, obs_dict = create_obs_fn(
            energy_or_obs_fn,
            observables,
            lr
        )


    # Running the MD
    logger.info('Starting MD simulation...')
    pressure_unit = units.metal_unit_system()['pressure']
    is_flex_npt = isinstance(state, jax_md.simulate.NPTNoseHooverFlexState)
    if ensemble == 'npt':
        # Header tabs MUST mirror the data-line tab count (see ~line 2712 and
        # ~line 2947). Data uses a single \t between every column; previously
        # the header padded with \t\t\t around the Box column, which made the
        # column titles slide three tabs right of their values.
        if is_flex_npt:
            p_header, _ = format_pressure_tensor(state, npt_coupling, pressure_unit)
            logger.info(f'Step\tE [eV]\tKE\tPE\tH\tTemp [K]\t{p_header}\tBox [A]\ttime/step [s]')
        else:
            logger.info('Step\tE [eV]\tKE\tPE\tH\tTemp [K]\tBox [A]\ttime/step [s]')
    else:
        logger.info('Step\tE [eV]\tKE\tPE\tH\tTemp [K]\ttime/step [s]')

    # Calculate some quantities for printing
    KE, PE, H, current_T, _ = compute_quantities(
        energy_fn,
        state,
        nbrs,
        nbrs_lr,
        box,
        k_grid,
        units.metal_unit_system(),
        ensemble,
        md_T,
        md_P,
        k_grid_disp=k_grid_disp
    )
    if ensemble == 'npt':
        p_str = ''
        if is_flex_npt:
            _, p_str = format_pressure_tensor(state, npt_coupling, pressure_unit)
            p_str = f'\t{p_str}'
        box_disp = box_to_diag(box)
        if len(box_disp) == 1:
            logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{(H or 0.0):.3f}\t{current_T:.1f}{p_str}\t{box_disp[0]:.3f}\t{0.0:.2e}')
        else:
            logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{(H or 0.0):.3f}\t{current_T:.1f}{p_str}\t'+
                        f'({box_disp[0]:.3f},{box_disp[1]:.3f},{box_disp[2]:.3f})\t{0.0:.2e}')
    else:
        logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{(H or 0.0):.3f}\t{current_T:.1f}\t{0.0:.2e}')

    # --- Conserved-quantity drift watchdog --------------------------------
    # For NVE the conserved quantity is the total energy E = KE+PE; for the
    # Nose-Hoover NVT/NPT thermostats it is the NH invariant. compute_quantities()
    # returns the appropriate one in `H`. We track its drift as a RATE
    # normalised per atom and per ps (meV/atom/ps) -- a size- and length-
    # independent measure, unlike a relative % of the (huge) absolute energy.
    #
    # Calibrated on 1 ns NVE water (SO3LR-M, 12288 atoms): well-behaved
    # dt=0.5 fs runs drift <=0.011 meV/atom/ps (float32 == float64 here; the
    # ~0.011 is a model/PME floor, not a precision effect); dt=1.0 fs runs reach
    # 0.025-0.058. The default 0.1 threshold sits above all of these, so it only
    # fires when the integration is clearly unstable (wrong timestep, blow-up).
    #
    # The reference is taken AFTER an equilibration warmup (drift_warmup_ps),
    # because (a) the first ps are a thermal transient where some drift is
    # normal and (b) a rate measured over <~1 ps is swamped by per-step noise
    # (~0.05 eV here). The same warmup is reused as the minimum baseline before
    # the first evaluation, so the earliest a warning can fire is ~2*warmup ps.
    # The warning fires at most once. NVT-Langevin is stochastic (KE+PE not
    # conserved) and is excluded.
    energy_drift_threshold = float(all_settings.get('energy_drift_threshold', 0.1))  # meV/atom/ps
    drift_warmup_ps = float(all_settings.get('drift_warmup_ps', 5.0))
    drift_ref_H = None      # conserved quantity at the reference point (eV)
    drift_ref_t = None      # simulated time at the reference point (ps)
    drift_warned = False

    momenta, positions, boxes = [], [], []
    cycle_md = 0
    # NB: current_cycle is initialised to 0 above (line ~2243) and overwritten
    # by load_state() when a restart .npz is loaded. We must NOT reset it
    # again here, otherwise the log step counter restarts at zero after a
    # restart (the run looks like a fresh run in the log even though state
    # is correctly loaded).
    total_time_for_steps = 0
    first_loop = True
    last_checkpoint_cycle = None

    # PySAGES setup (pysages_set_dict was already parsed during validation)
    if do_enhanced_sampling:
        method = get_pysages_method(
            pysages_set_dict,
            nbrs=nbrs,
            species=initial_geometry_dict['species'],
            box=box
        )
        if restart_pysages:
            logger.info('Restarting from previous PySAGES run.')
            pysages_result = restart_data_pysages
        else:
            pysages_result = None

    # For npt_flex, the step function returns a [3,3] box matrix via npt_box_flex.
    # Ensure the initial box matches this shape so fori_loop carry types are consistent.
    if npt_coupling in ('semi-isotropic', 'anisotropic') and box.ndim == 1:
        box = jnp.diag(box)

    # Initialize trajectory storage only after restart loading and flexible-cell
    # normalization.  Those paths can change a vector box into a full matrix;
    # creating the HDF5 schema earlier permanently chose the wrong shape.
    if output_format == 'hdf5':
        if output_atom_indices is None:
            num_atoms = position.shape[0]
        else:
            num_atoms = len(output_atom_indices)

        trajectory_box = np.asarray(box_for_trajectory(box), dtype=np.dtype(dtype))
        if not np.any(trajectory_box):
            _cell_a0 = np.zeros((3, 3), dtype=np.dtype(dtype))
        elif trajectory_box.ndim == 1 and trajectory_box.shape == (3,):
            _cell_a0 = np.diag(trajectory_box)
        else:
            _cell_a0 = trajectory_box

        try:
            _atomic_numbers_attr = np.asarray(
                initial_geometry.get_atomic_numbers(), dtype=np.int32
            )
            if output_atom_indices is not None:
                _atomic_numbers_attr = _atomic_numbers_attr[output_atom_indices]
        except Exception:
            _atomic_numbers_attr = None

        # md_dt_ps remains the pre-unit-conversion value captured above.
        _md_dt_fs = md_dt_ps * 1000.0
        from .. import __version__ as _so3lr_version
        _num_features = _model_num_features(model_path) if 'x' in observables else None
        hdf5_store = init_hdf5_store(
            save_to=output_file,
            batch_size=save_buffer,
            num_atoms=num_atoms,
            num_box_entries=tuple(trajectory_box.shape),
            exist_ok=True,
            observables=observables,
            num_features=_num_features,
            so3lr_version=_so3lr_version,
            ensemble=ensemble,
            md_dt_fs=_md_dt_fs,
            md_steps=md_steps,
            atomic_numbers=_atomic_numbers_attr,
            cell_a0=_cell_a0,
            input_file=input_file,
            model_path=model_path,
            dtype=np.dtype(dtype),
            fractional_coordinates=(
                fractional_coordinates and not output_cartesian_positions
            ),
        )

    sr_overflow_extra = 0
    lr_overflow_extra = 0
    overflow_retries = 0

    # Apply atom constraints (frozen/mobile atoms) to the initial state and wrap the
    # integrator step so fixed atoms stay pinned each step. No-op when mobile_mask is None.
    if mobile_mask is not None:
        reference_is_cartesian = fixed_atom_constraint == 'cartesian_anchor'
        reference_position = (
            positions_to_cartesian(position, box, fractional_coordinates)
            if reference_is_cartesian else position
        )
        state = apply_atom_constraints_to_state(
            state,
            mobile_mask,
            reference_position,
            box=box,
            fractional_coordinates=fractional_coordinates,
            reference_is_cartesian=reference_is_cartesian,
        )
        step_md_fn = wrap_step_with_constraints(
            step_md_fn,
            lr,
            mobile_mask,
            reference_position,
            fractional_coordinates,
            reference_is_cartesian,
        )

    while cycle_md < md_cycles:
        old_time = time.time()
        if do_enhanced_sampling:
            generate_context = create_pysages_interface_fns(
                lr, state, box, step_md_fn, md_dt, nbrs,
                ensemble=ensemble, nbrs_lr=nbrs_lr,
                k_grid=k_grid, k_grid_disp=k_grid_disp
            )

            if pysages_result:
                candidate_result = jax.block_until_ready(
                    pysages.run(pysages_result, generate_context, md_steps,
                                context_args={})
                )
            else:
                candidate_result = jax.block_until_ready(
                    pysages.run(method, generate_context, md_steps)
                )

            new_state, nbrs, nbrs_lr, new_box, k_grid, k_grid_disp = jax.block_until_ready(
                update_so3lr_after_pysages(
                    candidate_result, lr, init_fn, rng_key, md_T, md_dt,
                    nbrs, ensemble=ensemble, nbrs_lr=nbrs_lr,
                    k_grid=k_grid, k_grid_disp=k_grid_disp
                )
            )

        elif lr:
            new_state, nbrs, nbrs_lr, new_box, k_grid, k_grid_disp = jax.block_until_ready(
                jax.lax.fori_loop(
                    0,
                    md_steps,
                    step_md_fn,
                    (state, nbrs, nbrs_lr, box, k_grid, k_grid_disp)
                )
            )
        else:
            new_state, nbrs, new_box, k_grid, k_grid_disp = jax.block_until_ready(
                jax.lax.fori_loop(
                    0,
                    md_steps,
                    step_md_fn,
                    (state, nbrs, box, k_grid, k_grid_disp)
                )
            )

        time_per_step = (time.time() - old_time) / md_steps

        # Do not count first `allocation` loop for time_per_step calculation
        if not first_loop:
            total_time_for_steps += time_per_step
        else:
            first_loop = False

        # Checking if the neighbor lists overflowed
        nbrs, nbrs_lr, sr_overflow, lr_overflow = check_overflow(
            neighbor_fn,
            neighbor_fn_lr,
            nbrs,
            nbrs_lr,
            new_state,
            new_box,
            sr_extra=sr_overflow_extra,
            lr_extra=lr_overflow_extra,
        )
        overflow = sr_overflow or lr_overflow

        if overflow:
            # Add extra capacity for consecutive overflows
            if sr_overflow:
                sr_overflow_extra += OVERFLOW_EXTRA_SR
            if lr_overflow:
                lr_overflow_extra += OVERFLOW_EXTRA_LR
            overflow_retries += 1
            if overflow_retries > MAX_OVERFLOW_RETRIES:
                raise RuntimeError(
                    f"Neighbor-list overflow did not stabilize after "
                    f"{MAX_OVERFLOW_RETRIES} reallocations "
                    f"(SR extra_capacity={sr_overflow_extra}, "
                    f"LR extra_capacity={lr_overflow_extra}). "
                    f"Increase buffer_size_multiplier_sr and/or _lr."
                    f"Double-check your starting/final structure."
                )

        if not overflow:
            # Cycle succeeded: reset escalation state for the next cycle.
            sr_overflow_extra = 0
            lr_overflow_extra = 0
            overflow_retries = 0
            if do_enhanced_sampling:
                pysages_result = candidate_result

            cycle_md += 1
            current_cycle += 1

            state = new_state
            box = new_box

            if (ensemble == 'npt') and (kspace_npt_cycles is not None) and (current_cycle % kspace_npt_cycles == 0):
                box_for_kgrid = box_to_diag(box)
                k_grid,_ = setup_kspace_grid(
                    kspace_electrostatics,
                    kspace_smearing,
                    kspace_spacing,
                    box_for_kgrid,
                    k_grid=k_grid
                )
                if kspace_dispersion is not None:
                    k_grid_disp,_ = setup_kspace_grid(
                        kspace_dispersion,
                        kspace_dispersion_smearing,
                        kspace_dispersion_spacing,
                        box_for_kgrid,
                        k_grid=k_grid_disp
                    )

            # Calculate some quantities for printing
            KE, PE, H, current_T, _ = compute_quantities(
                energy_fn,
                state,
                nbrs,
                nbrs_lr,
                box,
                k_grid,
                units.metal_unit_system(),
                ensemble,
                md_T,
                md_P,
                k_grid_disp=k_grid_disp
            )

            # Abort if the dynamics diverged (NaN/Inf). Cheap host-side check on
            # the per-cycle scalars that were already synced for logging.
            if not np.all(np.isfinite([float(KE), float(PE), float(current_T)])):
                logger.error(
                    f"Non-finite value encountered at step {current_cycle*md_steps} "
                    f"(KE={KE}, PE={PE}, T={current_T}). The simulation has diverged; "
                    f"aborting. Check the timestep, temperature, and input structure.")
                raise RuntimeError(
                    "MD simulation diverged: NaN/Inf detected in energy or temperature.")

            # Watchdog: drift RATE of the conserved quantity in meV/atom/ps.
            # The reference is set once, after drift_warmup_ps of equilibration;
            # the warning then fires at most once. Skipped for stochastic
            # Langevin, where KE+PE is not conserved.
            if (not drift_warned and ensemble != 'nvt-langevin'
                    and H is not None and np.isfinite(float(H)) and float(H) != 0.0):
                t_ps = current_cycle * md_steps * md_dt_ps
                if drift_ref_H is None:
                    # Still warming up: set the reference once past the warmup.
                    if t_ps >= drift_warmup_ps:
                        drift_ref_H = float(H)
                        drift_ref_t = t_ps
                else:
                    window_ps = t_ps - drift_ref_t
                    # Require at least one warmup of baseline so the rate is not
                    # dominated by per-step noise.
                    if window_ps >= drift_warmup_ps:
                        n_atoms_md = state.position.shape[0]
                        drift_rate = (float(H) - drift_ref_H) / n_atoms_md / window_ps * 1000.0
                        if abs(drift_rate) > energy_drift_threshold:
                            logger.warning(
                                f"Conserved-quantity drift {drift_rate:+.3f} meV/atom/ps exceeds "
                                f"{energy_drift_threshold:.3f} meV/atom/ps at step "
                                f"{current_cycle*md_steps} (measured over {window_ps:.1f} ps after a "
                                f"{drift_warmup_ps:.1f} ps warmup). This usually means the integration "
                                f"is unstable; consider a smaller timestep or float64 precision. "
                                f"(Set energy_drift_threshold [meV/atom/ps] to tune, or drift_warmup_ps "
                                f"to change the equilibration window; some drift early in the run is normal.)")
                            drift_warned = True

            if ensemble == 'npt':
                p_str = ''
                if is_flex_npt:
                    _, p_val = format_pressure_tensor(state, npt_coupling, pressure_unit)
                    p_str = f'\t{p_val}'
                box_disp = box_to_diag(box)
                if len(box_disp) == 1:
                    logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{(H or 0.0):.3f}\t{current_T:.1f}{p_str}\t{box_disp[0]:.3f}\t{time_per_step:.2e}')
                else:
                    logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{(H or 0.0):.3f}\t{current_T:.1f}{p_str}\t'+
                                f'({box_disp[0]:.3f},{box_disp[1]:.3f},{box_disp[2]:.3f})\t{time_per_step:.2e}')
            else:
                logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{(H or 0.0):.3f}\t{current_T:.1f}\t{time_per_step:.2e}')

            out_pos = output_position_array(
                state.position, box, fractional_coordinates, output_cartesian_positions
            )
            positions.append(np.array(out_pos))
            momenta.append(np.array(state.momentum))
            if box is not None:
                boxes.append(np.array(box_for_trajectory(box)))

            # Calculate observables
            if len(observables)>0:
                obs_data = obs_fn((state, nbrs, nbrs_lr, box, k_grid, k_grid_disp))
                for key in obs_dict.keys():
                    obs_dict[key].append(obs_data[1][key])

            if len(positions) >= save_buffer:
                # Checkpoints track accepted simulation state, not successful
                # trajectory I/O. Save them first so a disk/write failure does
                # not erase the latest restartable state.
                if create_restart or (
                    do_enhanced_sampling and restart_pysages_save_path is not None
                ):
                    save_restart_checkpoints(
                        state,
                        box,
                        current_cycle,
                        ensemble=ensemble,
                        fractional_coordinates=fractional_coordinates,
                        restart_save_path=restart_save_path if create_restart else None,
                        pysages_result=pysages_result if do_enhanced_sampling else None,
                        restart_pysages_save_path=(
                            restart_pysages_save_path if do_enhanced_sampling else None
                        ),
                    )
                    last_checkpoint_cycle = current_cycle

                # Saving the output
                if output_format == 'hdf5':
                    positions, momenta, boxes, obs_dict = write_to_hdf5(
                        hdf5_store,
                        momenta,
                        positions,
                        boxes,
                        obs_dict,
                        output_atom_indices
                    )
                elif output_format == 'extxyz':
                    positions, momenta, boxes, obs_dict = write_to_extxyz(
                        output_file,
                        initial_geometry,  # Template atoms
                        boxes,
                        momenta,
                        positions,
                        obs_dict,
                        output_atom_indices,
                        fractional_coordinates=(
                            fractional_coordinates and not output_cartesian_positions
                        )
                    )

    # Always persist the accepted final state, even when the number of cycles
    # is not a multiple of save_buffer.  This is intentionally independent of
    # trajectory flushing: if the tail write fails, users can still restart
    # from the last completed MD cycle.
    if (
        last_checkpoint_cycle != current_cycle
        and (
            create_restart
            or (do_enhanced_sampling and restart_pysages_save_path is not None)
        )
    ):
        save_restart_checkpoints(
            state,
            box,
            current_cycle,
            ensemble=ensemble,
            fractional_coordinates=fractional_coordinates,
            restart_save_path=restart_save_path if create_restart else None,
            pysages_result=pysages_result if do_enhanced_sampling else None,
            restart_pysages_save_path=(
                restart_pysages_save_path if do_enhanced_sampling else None
            ),
        )

    # A frame-major HDF5 layout and extxyz both support arbitrary batch sizes,
    # so flush the final short buffer instead of dropping or padding it.
    if positions:
        if output_format == 'hdf5':
            positions, momenta, boxes, obs_dict = write_to_hdf5(
                hdf5_store,
                momenta,
                positions,
                boxes,
                obs_dict,
                output_atom_indices
            )
        elif output_format == 'extxyz':
            positions, momenta, boxes, obs_dict = write_to_extxyz(
                output_file,
                initial_geometry,
                boxes,
                momenta,
                positions,
                obs_dict,
                output_atom_indices,
                fractional_coordinates=(
                    fractional_coordinates and not output_cartesian_positions
                )
            )

    logger.info('Results saved to: ' + output_file)
    # The first cycle is excluded from timing (it includes JIT compilation /
    # neighbor-list allocation), so at least two cycles are needed for an average.
    timed_cycles = cycle_md - 1
    if timed_cycles > 0:
        average_time_per_step = total_time_for_steps / timed_cycles
        logger.info(f'Average time per step: {average_time_per_step:.2e} seconds', )
        logger.info(f'Average time per atom per step: {average_time_per_step / len(initial_geometry):.2e} seconds')
    else:
        average_time_per_step = None
        logger.info('Average time per step: not available (needs at least 2 MD cycles; '
                    'the first cycle is excluded as warm-up).')

    if output_format == 'extxyz':
        logger.info('Consider saving the trajectory in HDF5 format, which has no overhead for long runs (--output traj_name.hdf5).')

    try:
        backend = jax.default_backend()
    except RuntimeError:
        backend = "cpu"

    # The 3.25e-6 s/atom baseline was measured on an A100 GPU, so the warning is
    # only meaningful on A100 hardware. On other GPUs (slower cards, ROCm/MI-series,
    # consumer GPUs) the comparison is misleading, so gate it on the actual device kind.
    try:
        device_kind = jax.devices()[0].device_kind
    except Exception:
        device_kind = ""
    is_a100 = "A100" in (device_kind or "")

    if average_time_per_step is not None and backend in ["gpu", "cuda", "rocm"] and is_a100:
        if average_time_per_step > 1.2 * 3.25e-6 * len(initial_geometry):
            logger.warning('Ideally, the average time per step should be close to {:.2e} seconds (measured on an A100 GPU)'.format(
                3.25e-6 * len(initial_geometry)))
            logger.warning('Consider decreasing the buffer sizes if the system has equilibrated (--buffer-sr 1.15, --buffer-lr 1.1)')
            logger.warning('and/or increasing the number of steps in each jax.lax.fori_loop (--steps 1000)')
            if output_format == 'extxyz':
                logger.warning('and/or saving the trajectory in HDF5 format (--output traj_name.hdf5)')


def create_min_fn(
    lr: bool,
    min_apply: callable,
    box=None
):
    if lr:
        @jax.jit
        def step_min_fn_lr(i, state):

            state, nbrs, nbrs_lr = state

            state = min_apply(
                state,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                box=box
            )

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx
            )

            nbrs_lr = nbrs_lr.update(
                state.position,
                neighbor=nbrs_lr.idx
            )

            return state, nbrs, nbrs_lr
        return step_min_fn_lr
    else:
        @jax.jit
        def step_min_fn(i, state):

            state, nbrs = state

            state = min_apply(
                state,
                neighbor=nbrs.idx,
                box=box
            )

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx
            )

            return state, nbrs
        return step_min_fn


def create_fire_fn(
    energy_fn: callable,
    shift: callable,
    min_start_dt: float,
    min_max_dt: float,
    min_n_min: int,
    lr: bool,
    box: jnp.ndarray,
) -> Tuple[callable, callable]:
    """
    Create the FIRE minimization functions.
    Returns the init and apply functions.

    Args:
        energy_fn (callable): Function that calculates the energy.
        shift (callable): Function that shifts the positions.
        min_start_dt (float): Starting time step for the minimization.
        min_max_dt (float): Maximum time step for the minimization.
        min_n_min (int): Number of minimization steps.
        lr (bool): Whether to use long-range interactions.
        box (jnp.ndarray): Box of the system.

    Returns:
        Tuple[callable, callable]: Init and apply functions.
    """
    fire_init, fire_apply = jax_md.minimize.fire_descent(
        energy_fn,
        shift,
        dt_start=min_start_dt,
        dt_max=min_max_dt,
        n_min=min_n_min
    )

    fire_init = jax.jit(fire_init)
    fire_apply = jax.jit(fire_apply)

    step_min_fn = create_min_fn(lr, fire_apply, box)

    return fire_init, step_min_fn


def perform_min(
    all_settings: Dict,
    log_input: bool = True,
) -> Union[jnp.ndarray, List[jnp.ndarray]]:
    """
    Perform energy minimization (geometry optimization) with the given settings.

    Args:
        all_settings (Dict): Settings for the minimization.

    Returns:
        Union[jnp.ndarray, List[jnp.ndarray]]: Optimized structure or trajectory.
    """
    # Configure JAX process-globals (x64, matmul precision). Idempotent if
    # already called by run(); needed here for `so3lr opt`, which previously
    # silently ignored --precision float64 because it never went through run().
    setup_jax_globals(
        precision=all_settings.get('precision', 'float32'),
        matmul_precision=all_settings.get('matmul_precision', 'highest'),
    )

    # Extract settings with defaults
    input_file = all_settings.get('input_file')
    output_file = all_settings.get('output_file')
    output_format = all_settings.get('output_format')
    overwrite = bool(all_settings.get('overwrite', False))

    # Refuse to clobber existing output unless --overwrite was passed.
    if output_file and Path(output_file).exists() and not overwrite:
        raise RuntimeError(
            f"Output file already exists: {output_file}. Refusing to overwrite. "
            f"Pass --overwrite to overwrite, or choose a different --output."
        )

    # Model parameters
    model_path = all_settings.get('model_path')
    precision = all_settings.get('precision')
    lr_cutoff = all_settings.get('lr_cutoff')
    dispersion_damping = all_settings.get('dispersion_damping')
    buffer_size_multiplier_sr = all_settings.get('buffer_size_multiplier_sr')
    buffer_size_multiplier_lr = all_settings.get('buffer_size_multiplier_lr')
    minimum_cell_size_multiplier = all_settings.get('minimum_cell_size_multiplier', 1.0)
    total_charge = all_settings.get('total_charge')
    num_unpaired_electrons = all_settings.get('num_unpaired_electrons', 0)

    # Ewald electrostatics settings
    kspace_electrostatics = all_settings.get('kspace_electrostatics',None)
    kspace_smearing = all_settings.get('kspace_smearing', 4.0)
    kspace_spacing = all_settings.get('kspace_spacing', 2.0)
    kspace_interp_nodes = all_settings.get('kspace_interp_nodes', 4)
    charge_embed_total_charge = all_settings.get('charge_embed_total_charge', None)
    charge_equilibration_total_charge = all_settings.get('charge_equilibration_total_charge', None)

    # Settings for the minimization
    min_cycles = all_settings.get('min_cycles')
    min_steps = all_settings.get('min_steps')
    min_start_dt = all_settings.get('min_start_dt')
    min_max_dt = all_settings.get('min_max_dt')
    min_n_min = all_settings.get('min_n_min')
    force_convergence = all_settings.get('force_convergence')  # eV/A, can be None

    # Atom-constraint settings. perform_min does NOT enforce these (FIRE-with-
    # constraints is not yet implemented), but the previous behavior was to
    # silently ignore them — so `relax_before_run: true` with frozen atoms
    # would drift the "frozen" atoms during pre-MD relaxation, putting the
    # MD constraint logic at the wrong reference position. Refuse here with
    # a clear message so the user can either drop the constraints or disable
    # the pre-MD relaxation. (When perform_md calls perform_min, the same
    # check fires before any work is done.)
    _mob = all_settings.get('mobile_atom_indices')
    _fix = all_settings.get('fixed_atom_indices')
    _msk = all_settings.get('atom_mask_path')
    if any(v is not None for v in (_mob, _fix, _msk)):
        raise NotImplementedError(
            "Atom constraints (fixed_atom_indices / mobile_atom_indices / "
            "atom_mask_path) are not supported during geometry optimization "
            "(`so3lr opt` or `relax_before_run: true`): the FIRE minimizer "
            "would silently drift the 'frozen' atoms before MD step 0. "
            "Either remove the constraint settings, or set "
            "`relax_before_run: false` and pre-relax in a separate step."
        )

    # Geometry, cell and charge
    input_file_path = all_settings.get('input_file')
    if input_file_path is None:
        raise ValueError('Initial geometry file path must be provided')
    try:
        initial_geometry = read(input_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find initial geometry file: {input_file_path}")

    validate_and_log_input(initial_geometry, input_file_path,
                           total_charge=total_charge,
                           num_unpaired_electrons=num_unpaired_electrons,
                           verbose=log_input)
    cell = initial_geometry.get_cell()
    cell = check_cell(cell, lr_cutoff, kspace_electrostatics, kspace_smearing)

    if cell is not None:
        shift_displacement = 'periodic'
        initial_geometry.wrap()
    else:
        shift_displacement = 'free'

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)

    dtype = jnp.float64 if precision == 'float64' else jnp.float32
    (position, box, displacement, shift, fractional_coordinates) = handle_box(
        shift_displacement, initial_geometry_dict['positions'], cell, precision=dtype)

    # Loading the model. A bundled name (so3lr-s/m/l/so3lr) or a custom path; None
    # falls back to the default model. load_model resolves names and paths alike.
    from so3lr.model_registry import DEFAULT_MODEL, available_models
    selected_model = model_path if model_path is not None else DEFAULT_MODEL
    if selected_model in available_models():
        logger.info(f"Using bundled SO3LR model: {selected_model}")
    else:
        logger.info(f"Using custom MLFF potential from: {selected_model}")
    potential = load_model(
        selected_model,
        precision=jnp.float64 if precision == 'float64' else jnp.float32,
        lr_cutoff=lr_cutoff,
        dispersion_damping=dispersion_damping,
        kspace_electrostatics=kspace_electrostatics,
        kspace_interp_nodes=kspace_interp_nodes,
        charge_embed_total_charge=charge_embed_total_charge,
        charge_equilibration_total_charge=charge_equilibration_total_charge,
    )

    # Setting up the model
    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = process_model(
        potential=potential,
        species=initial_geometry_dict['species'],
        displacement=displacement,
        box=box,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        minimum_cell_size_multiplier=minimum_cell_size_multiplier,
    )

    k_grid, k_smearing = setup_kspace_grid(
        kspace_electrostatics=kspace_electrostatics,
        kspace_spacing=kspace_spacing,
        kspace_smearing=kspace_smearing,
        box=box
    )

    energy_fn = jax.jit(partial(energy_or_obs_fn, has_aux=False, k_grid=k_grid, k_smearing=k_smearing))
    force_fn = jax.jit(jax_md.quantity.force(energy_fn))

    # Allocating the neighbor lists
    if neighbor_fn_lr is not None:
        lr = True
    else:
        lr = False

    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box) if lr else None

    # Setting up the minimization functions
    min_init_fn, min_step_fn = create_fire_fn(
        energy_fn=energy_fn,
        shift=shift,
        min_start_dt=min_start_dt,
        min_max_dt=min_max_dt,
        min_n_min=min_n_min,
        lr=lr,
        box=box
    )

    if lr:
        min_state = min_init_fn(
            position,
            box=box,
            neighbor=nbrs.idx,
            neighbor_lr=nbrs_lr.idx
        )
    else:
        min_state = min_init_fn(
            position,
            box=box,
            neighbor=nbrs.idx
        )

    logger.info('Starting minimization...')
    logger.info('Step\tE [eV]\tF_max [eV/Å]')

    # Initial energy and force
    E = energy_fn(
        position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx if lr else None,
        box=box
    )
    F = force_fn(
        position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx if lr else None,
        box=box
    )
    f_max = np.abs(F).max()
    logger.info('{}\t {:.3f}\t {:.3f}'.format(0, E, f_max))

    # If tracking trajectory, store all positions
    minimization_trajectory = []
    minimization_trajectory.append(np.array(position))

    # `min_cycles` is a hard ceiling on outer FIRE cycles. With
    # `force_convergence` set, the loop exits early once f_max drops below
    # the threshold; otherwise it runs the full `min_cycles` cycles and
    # warns the user that the requested convergence was never reached.
    total_iter = 0
    i = 1
    while total_iter < min_cycles:
        if lr:
            min_state, nbrs, nbrs_lr = jax.lax.fori_loop(
                0,
                min_steps,
                min_step_fn,
                (min_state, nbrs, nbrs_lr)
            )

            E = energy_fn(
                min_state.position,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                box=box
            )

            F = force_fn(
                min_state.position,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                box=box
            )
        else:
            min_state, nbrs = jax.lax.fori_loop(
                0,
                min_steps,
                min_step_fn,
                (min_state, nbrs)
            )

            E = energy_fn(
                min_state.position,
                neighbor=nbrs.idx,
                box=box
            )

            F = force_fn(
                min_state.position,
                neighbor=nbrs.idx,
                box=box
            )

        f_max = np.abs(F).max()
        logger.info('{}\t {:.3f}\t {:.3f}'.format(i*min_steps, E, f_max))

        # Save the current positions to the trajectory
        minimization_trajectory.append(np.array(min_state.position))

        # Check if force convergence criterion is met
        if force_convergence is not None and f_max <= force_convergence:
            logger.info(
                f"Convergence criterion met at step {i*min_steps}: Fmax = {f_max:.3f} eV/Å")
            break

        total_iter += 1
        i += 1
    else:
        # Loop exited because the hard min_cycles ceiling was hit without
        # meeting the requested force_convergence threshold (if any).
        if force_convergence is not None and f_max > force_convergence:
            logger.warning(
                f"FIRE minimization hit the min_cycles ceiling "
                f"({min_cycles} cycles * {min_steps} steps = "
                f"{min_cycles*min_steps} total steps) without reaching "
                f"force_convergence={force_convergence:.3f} eV/Å "
                f"(last Fmax = {f_max:.3f} eV/Å). Saving best-so-far "
                f"positions. Increase --min-cycles or relax --force-conv "
                f"if tighter convergence is required.")

    # Save optimization result and return optimized geometry
    positions = minimization_trajectory
    if output_format == 'extxyz':
        write_to_extxyz(output_file, initial_geometry, boxes=box, momenta=None, positions=positions)
    elif output_format == 'hdf5':
        logger.warning(f"Output format is 'hdf5', changing output file extension to 'xyz' for minimization.") #TODO: Saving in xyz for simplicity for now, add hdf5 support
        output_file = output_file.rsplit('.', 1)[0] + '_opt.xyz'
        write_to_extxyz(output_file, initial_geometry, boxes=box, momenta=None, positions=positions)
        # write_to_hdf5(hdf5_store, momenta=None, positions=positions, boxes=box)
    
    logger.info(f"Optimization trajectory with {len(minimization_trajectory)} frames saved to {output_file}")

    if box is None or np.any(box == 0):
        return min_state.position
    else:
        return min_state.position * box


def save_state(
    state,
    box: jnp.array,
    cycle: int,
    path_to_save: str,
    ensemble: str = 'nvt',
    fractional_coordinates: bool = True
) -> None:
    """
    Save the state of the simulation to a .npz file.

    Args:
        state: Current state of the simulation.
        box (jnp.array): Cell vectors of the simulation.
        cycle (int): Current MD cycle number.
        path_to_save (str): Path to save the .npz file.
        ensemble (str, optional): Specifies which statistical ensemble
                                    is being run. Defaults to 'nvt'.
        fractional_coordinates (bool, optional): Whether positions are stored
            in fractional coordinates. Defaults to True.
    """

    state_dict = dict(
        position=state.position,
        momentum=state.momentum,
        force=state.force,
        mass=state.mass,
        # Additional necessary information
        box=box,
        step=cycle,
        fractional_coordinates=fractional_coordinates
    )

    if ensemble.lower() == 'nvt':
        state_dict.update(
            thermostat_position=state.chain.position,
            thermostat_momentum=state.chain.momentum,
            thermostat_mass=state.chain.mass,
            thermostat_tau=state.chain.tau,
            thermostat_kinetic_energy=state.chain.kinetic_energy,
            thermostat_degrees_of_freedom=state.chain.degrees_of_freedom,
        )

    if ensemble.lower() == 'npt':
        is_flex = isinstance(state, jax_md.simulate.NPTNoseHooverFlexState)
        state_dict.update(
            thermostat_position=state.thermostat.position,
            thermostat_momentum=state.thermostat.momentum,
            thermostat_mass=state.thermostat.mass,
            thermostat_tau=state.thermostat.tau,
            thermostat_kinetic_energy=state.thermostat.kinetic_energy,
            thermostat_degrees_of_freedom=state.thermostat.degrees_of_freedom,
            barostat_position=state.barostat.position,
            barostat_momentum=state.barostat.momentum,
            barostat_mass=state.barostat.mass,
            barostat_tau=state.barostat.tau,
            barostat_kinetic_energy=state.barostat.kinetic_energy,
            barostat_degrees_of_freedom=state.barostat.degrees_of_freedom,
            reference_box=state.reference_box,
            box_position=state.box_position,
            box_momentum=state.box_momentum,
            box_mass=state.box_mass,
            npt_flex=is_flex,
        )
        if is_flex:
            state_dict['stress_tensor'] = state.stress_tensor
        else:
            state_dict['dUdV'] = state.dUdV

    if ensemble.lower() == 'nvt-langevin':
        # The BAOAB Langevin state (NVTLangevinState) carries only an RNG key
        # beyond the core position/momentum/force/mass fields. Persist it so the
        # stochastic trajectory continues deterministically across a restart.
        state_dict.update(rng=state.rng)
    state_dict.update(ensemble=ensemble)
    np.savez(path_to_save, **state_dict)


def load_state(
    path_to_load: str,
    ensemble: str = 'nvt',
    restart_reset: bool = False,
    fractional_coordinates: bool = True,
    precision: jnp.dtype = jnp.float32,
) -> Tuple:
    """

    Loads the state of the simulation from a .npz file and returns the state,
    the box and the cycle number. If the coordinate system of the restart file
    differs from the current run, positions and momenta are converted.

    Args:
        path_to_load (str): Path to the .npz file containing the state of the
                            simulation.
        ensemble (str, optional): Specifies which ensemble is loaded.
                                    Defaults to 'nvt'.
        restart_reset (bool, optional): Whether to reset the restart.
                                    Defaults to False.
        fractional_coordinates (bool, optional): Whether the current run uses
            fractional coordinates. Used to detect and convert coordinate
            mismatches with the restart file. Defaults to True.

    Returns:
        Tuple: Tuple containing the state, the box and the cycle number.
    """

    loaded_state = np.load(path_to_load, allow_pickle=True)

    # Helper: enumerate which fields in the .npz are about to be dropped if
    # we cannot/won't reconstruct the integrator state (ensemble mismatch or
    # restart_reset). We treat everything *except* the core position/momentum/
    # mass/force/box/step/ensemble/fractional_coordinates as droppable.
    _avail_fields = (
        sorted(loaded_state.files)
        if hasattr(loaded_state, 'files')
        else sorted(loaded_state.keys())
    )
    _core_fields = {
        'position', 'momentum', 'mass', 'force', 'box', 'step',
        'ensemble', 'fractional_coordinates',
    }
    _dropped = [f for f in _avail_fields if f not in _core_fields]

    if restart_reset:
        raise RestartInNewEnsembleError(
            loaded_state, ensemble,
            dropped_fields=_dropped, reset_only=True,
        )
    if ensemble.lower() != loaded_state.get('ensemble', None):
        # Log the mismatch BEFORE raising so the caller's warning surfaces the
        # full picture (saved ensemble, requested ensemble, dropped fields).
        logger.info(
            f"Restart was saved under ensemble="
            f"{str(loaded_state.get('ensemble', None)).upper()}; loading into "
            f"ensemble={str(ensemble).upper()}; the following fields will be "
            f"re-initialised: {_dropped}."
        )
        raise RestartInNewEnsembleError(
            loaded_state, ensemble, dropped_fields=_dropped,
        )

    if ensemble.lower() == 'nve':
        try:
            state = jax_md.simulate.NVEState(
                position=jnp.array(loaded_state['position']),
                momentum=jnp.array(loaded_state['momentum']),
                force=jnp.array(loaded_state['force']),
                mass=jnp.array(loaded_state['mass'])
            )
        except KeyError as _ke:
            raise RestartSchemaMismatchError(
                missing_field=str(_ke).strip("'\""),
                ensemble=ensemble,
                available_fields=_avail_fields,
                path=path_to_load,
            ) from _ke

    elif ensemble.lower() == 'nvt':
        try:
            state = jax_md.simulate.NVTNoseHooverState(
                position=jnp.array(loaded_state['position']),
                momentum=jnp.array(loaded_state['momentum']),
                force=jnp.array(loaded_state['force']),
                mass=jnp.array(loaded_state['mass']),
                chain=jax_md.simulate.NoseHooverChain(
                    position=loaded_state['thermostat_position'],
                    momentum=loaded_state['thermostat_momentum'],
                    mass=loaded_state['thermostat_mass'],
                    tau=loaded_state['thermostat_tau'],
                    kinetic_energy=loaded_state['thermostat_kinetic_energy'],
                    degrees_of_freedom=loaded_state['thermostat_degrees_of_freedom']
                )
            )
        except KeyError as _ke:
            raise RestartSchemaMismatchError(
                missing_field=str(_ke).strip("'\""),
                ensemble=ensemble,
                available_fields=_avail_fields,
                path=path_to_load,
            ) from _ke

    elif ensemble.lower() == 'npt':
        is_flex = bool(loaded_state.get('npt_flex', False))
        try:
            nhc_barostat = jax_md.simulate.NoseHooverChain(
                position=loaded_state['barostat_position'],
                momentum=loaded_state['barostat_momentum'],
                mass=loaded_state['barostat_mass'],
                tau=loaded_state['barostat_tau'],
                kinetic_energy=loaded_state['barostat_kinetic_energy'],
                degrees_of_freedom=loaded_state['barostat_degrees_of_freedom']
            )
            nhc_thermostat = jax_md.simulate.NoseHooverChain(
                position=loaded_state['thermostat_position'],
                momentum=loaded_state['thermostat_momentum'],
                mass=loaded_state['thermostat_mass'],
                tau=loaded_state['thermostat_tau'],
                kinetic_energy=loaded_state['thermostat_kinetic_energy'],
                degrees_of_freedom=loaded_state['thermostat_degrees_of_freedom']
            )
        except KeyError as _ke:
            raise RestartSchemaMismatchError(
                missing_field=str(_ke).strip("'\""),
                ensemble=ensemble,
                available_fields=_avail_fields,
                path=path_to_load,
            ) from _ke
        try:
            if is_flex:
                mass = loaded_state['mass']
                if mass.ndim == 1:
                    mass = mass.reshape(-1, 1)
                state = jax_md.simulate.NPTNoseHooverFlexState(
                    position=loaded_state['position'],
                    velocity=loaded_state['momentum'] / mass,
                    force=loaded_state['force'],
                    mass=mass,
                    reference_box=loaded_state['reference_box'],
                    box_position=loaded_state['box_position'],
                    box_velocity=loaded_state['box_momentum'] / loaded_state['box_mass'],
                    box_mass=loaded_state['box_mass'],
                    barostat=nhc_barostat,
                    thermostat=nhc_thermostat,
                    stress_tensor=loaded_state.get('stress_tensor', None),
                )
            else:
                state = jax_md.simulate.NPTNoseHooverState(
                    position=loaded_state['position'],
                    momentum=loaded_state['momentum'],
                    force=loaded_state['force'],
                    mass=loaded_state['mass'],
                    reference_box=loaded_state['reference_box'],
                    box_position=loaded_state['box_position'],
                    box_momentum=loaded_state['box_momentum'],
                    box_mass=loaded_state['box_mass'],
                    barostat=nhc_barostat,
                    thermostat=nhc_thermostat,
                    dUdV=loaded_state.get('dUdV', np.float32(0.0)),
                )
        except KeyError as _ke:
            raise RestartSchemaMismatchError(
                missing_field=str(_ke).strip("'\""),
                ensemble=ensemble,
                available_fields=_avail_fields,
                path=path_to_load,
            ) from _ke

    elif ensemble.lower() == 'nvt-langevin':
        # The Langevin integrator state is fully described by the core fields
        # plus the RNG key (see save_state). A restart file written before
        # Langevin restarts were supported lacks 'rng' and surfaces as a clear
        # schema mismatch rather than an opaque crash.
        try:
            state = jax_md.simulate.NVTLangevinState(
                position=jnp.array(loaded_state['position']),
                momentum=jnp.array(loaded_state['momentum']),
                force=jnp.array(loaded_state['force']),
                mass=jnp.array(loaded_state['mass']),
                rng=jnp.array(loaded_state['rng']),
            )
        except KeyError as _ke:
            raise RestartSchemaMismatchError(
                missing_field=str(_ke).strip("'\""),
                ensemble=ensemble,
                available_fields=_avail_fields,
                path=path_to_load,
            ) from _ke
    else:
        raise NotImplementedError(
            'Only NVE, NVT, NVT-Langevin and NPT ensembles are supported'
        )

    try:
        box = loaded_state['box']
        cycle = loaded_state['step']
    except KeyError as _ke:
        raise RestartSchemaMismatchError(
            missing_field=str(_ke).strip("'\""),
            ensemble=ensemble,
            available_fields=_avail_fields,
            path=path_to_load,
        ) from _ke
    if np.all(box == 0) or box is None:
        box = jnp.array([0.0], dtype=precision)

    # Detect coordinate system mismatch and convert if needed.
    # Old restart files without the flag are assumed to use fractional
    # coordinates (the historical default for periodic systems).
    saved_fractional = bool(loaded_state.get('fractional_coordinates', True))

    if saved_fractional and not fractional_coordinates:
        # Restart was saved in fractional coords, current run uses cartesian.
        # In jax_md with periodic_general, momenta and forces are stored
        # identically regardless of coordinate system — only positions differ.
        logger.info('Converting restart positions from fractional to cartesian coordinates.')
        state = state.set(
            position=positions_to_cartesian(state.position, jnp.array(box), True)
        )
    elif not saved_fractional and fractional_coordinates:
        # Restart was saved in cartesian coords, current run uses fractional.
        logger.info('Converting restart positions from cartesian to fractional coordinates.')
        state = state.set(
            position=cartesian_to_positions(state.position, jnp.array(box), True)
        )

    # Normalize the reconstructed state/box to match the dtypes (and weak typing)
    # of a fresh init_fn carry, so the restart fori_loop type-checks under x64.
    # `load_state` otherwise returns raw numpy arrays straight from the npz:
    #   * `position` stays numpy -> numpy gather raises IndexError on the sentinel
    #     neighbor index in the pre-loop forward passes.
    #   * `dUdV`/`stress_tensor` come back *strongly* typed float64, whereas a
    #     fresh run carries `dUdV` as a *weakly* typed float64 scalar. A strong
    #     float64 dUdV promotes the float32 box -> float64 inside the integrator,
    #     so the loop carry input (float32) != output (float64) and jax.lax.scan
    #     rejects it. Rebuilding dUdV weak (a Python float for the scalar case)
    #     restores the fresh-run behaviour; everything else -> run precision jnp.
    import dataclasses
    _weak_fields = {'dUdV', 'stress_tensor'}

    def _to_precision(x):
        a = jnp.asarray(x)
        return a.astype(precision) if jnp.issubdtype(a.dtype, jnp.floating) else a

    _new_vals = {}
    for _f in dataclasses.fields(state):
        _v = getattr(state, _f.name)
        if _v is None:
            _new_vals[_f.name] = None
        elif _f.name in _weak_fields:
            _a = jnp.asarray(_v)
            _new_vals[_f.name] = float(_a) if _a.ndim == 0 else _a.astype(precision)
        else:
            _new_vals[_f.name] = jax.tree_util.tree_map(_to_precision, _v)
    state = dataclasses.replace(state, **_new_vals)

    box = jnp.asarray(box, precision)

    # Report exactly what was restored from the .npz so the log is self-explanatory.
    _n_atoms = int(np.asarray(state.position).shape[0])
    _avail = sorted(loaded_state.files) if hasattr(loaded_state, 'files') else sorted(loaded_state.keys())
    logger.info(f"Restored state from .npz: ensemble={ensemble.lower()}, "
                f"n_atoms={_n_atoms}, step/cycle={int(np.asarray(cycle))}, "
                f"saved_fractional={saved_fractional}, precision={np.dtype(precision).name}")
    logger.info(f"  position shape: {tuple(np.asarray(state.position).shape)}; "
                f"box: {np.asarray(box).tolist()}")
    if ensemble.lower() == 'npt':
        logger.info(f"  npt_flex={bool(loaded_state.get('npt_flex', False))}; "
                    f"box_position shape: {tuple(np.asarray(state.box_position).shape)}")
    logger.info(f"  fields present in .npz: {_avail}")

    return state, box, cycle


def run(
    settings: dict
) -> None:
    """
    Run the MD simulation using the provided settings. Relaxation before the
    MD run is optional. Sets the precision of the simulation.

    Default precision is float32. If float64 is desired, set the precision
    key in the settings dictionary to 'float64'.

    Default behavior is to not relax the system before the MD run. To relax
    the system before the MD run, set the relax_before_run key in the settings
    dictionary to True.

    Args:
        settings (dict): Dictionary containing the settings for the MD 
                            simulation and relaxation.
    """

    relax_before_run = settings.get('relax_before_run', True)

    setup_jax_globals(
        precision=settings.get('precision', 'float32'),
        matmul_precision=settings.get('matmul_precision', 'highest'),
    )
    matmul_precision = _JAX_GLOBALS_CONFIGURED.get('matmul_precision', 'highest')

    # Check for allowed atom types
    input_file_path = settings.get('input_file')
    if input_file_path is not None:
        try:
            initial_geometry = read(input_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find initial geometry file: {input_file_path}")

        # Log the full input summary once here; the delegated perform_min /
        # perform_md calls below validate only (log_input=False).
        validate_and_log_input(initial_geometry, input_file_path,
                               total_charge=settings.get('total_charge'),
                               num_unpaired_electrons=settings.get('num_unpaired_electrons'))

        # allowed_atomic_numbers = {1, 6, 7, 8, 9, 15, 16, 17}
        # current_atomic_numbers = {int(x) for x in initial_geometry.get_atomic_numbers()}
        # if not current_atomic_numbers.issubset(allowed_atomic_numbers):
        #     unsupported = current_atomic_numbers - allowed_atomic_numbers
        #     logger.error(f"Input geometry contains unsupported element(s): {unsupported}. "
        #                  f"Supported elements are: H(1), C(6), N(7), O(8), F(9), P(15), S(16), Cl(17).")
        #     raise ValueError(f"Input geometry contains unsupported element(s): {unsupported}")

    # Load restart file if needed
    restart_load_path = settings.get('restart_load_path')
    if restart_load_path is not None:
        if os.path.exists(restart_load_path):
            restart = True
        else:
            logger.error(f"Restart load path does not exist.")
            sys.exit(1)
    else:
        restart = False

    # Relax the geometry
    if relax_before_run:
        if restart:
            logger.info('Restarting MD, skipping relaxation.')
            opt_structure = None
        else:
            opt_structure = perform_min(settings, log_input=False)
            # perform_min just wrote the optimization trajectory to output_file
            # (extxyz). Mark it so the MD overwrite guard below does not refuse
            # the file THIS run's own relax phase created. Stale output left by a
            # previous run is still caught by perform_min's own guard above.
            settings['_relax_wrote_output'] = True
            logger.info("=" * 60)
    else:
        opt_structure = None

    # Perform MD with relaxed structure
    perform_md(settings, opt_structure, restart, log_input=False)
