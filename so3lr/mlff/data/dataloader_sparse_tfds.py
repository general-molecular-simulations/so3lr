import hashlib
import importlib.metadata
import json
import jraph
import logging
import numpy as np
import queue
import sys
import traceback
import wandb
from ase import Atoms
from ase.neighborlist import neighbor_list
from collections.abc import Mapping
from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr
from so3lr.mlff.utils.param_utils import MAX_NUM_THEORY_LEVELS

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from multiprocessing import Manager, get_context
from pathlib import Path
from typing import Optional

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None
    logging.warning(
        "For using QCMLDataLoader please install tensorflow."
    )
try:
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    tfds = None
    logging.warning(
        "For using QCMLDataLoader please install tensorflow_datasets."
    )


logger = logging.getLogger("SO3LR")

_TFDS_STREAM_FINGERPRINT_VERSION = 1
_TFDS_EPOCH_STREAM_VERSION = 1
_NEIGHBOUR_BUCKET_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def _canonical_fingerprint(payload):
    """Return a stable SHA-256 digest for JSON-compatible loader metadata."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _installed_version(distribution, fallback='unavailable'):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _json_compatible(value):
    """Normalize ConfigDict/NumPy/Path values for canonical JSON hashing."""
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f'Cannot fingerprint configuration value of type {type(value).__name__}.'
    )


class _ClosePropagatingSkipIterator:
    """Lazily skip a prefix and forward ``close`` to the source iterator."""

    def __init__(self, iterable, count):
        self._iterator = iter(iterable)
        self._remaining = count
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            while self._remaining:
                next(self._iterator)
                self._remaining -= 1
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        close = getattr(self._iterator, 'close', None)
        if close is not None:
            close()

    def __del__(self):
        # Best-effort cleanup if consumer code exits its loop by exception.
        try:
            self.close()
        except Exception:
            pass


def convert_dataset_units_tf(element, energy_unit, length_unit, dipole_vec_unit):
    """Convert one TFDS record to SO3LR's canonical eV/Angstrom units.

    Conversion happens before filtering and neighbor construction.  In
    particular, positions and the periodic cell must be scaled together: cell
    offsets are integer lattice translations and would otherwise be combined
    with positions expressed in a different unit.
    """
    converted = dict(element)

    def _scale(key, factor):
        if key in converted:
            value = converted[key]
            converted[key] = value * tf.cast(factor, value.dtype)

    _scale('energy', energy_unit)
    _scale('forces', energy_unit / length_unit)
    _scale('stress', energy_unit / length_unit ** 3)
    _scale('positions', length_unit)
    _scale('cell', length_unit)
    _scale('dipole_vec', dipole_vec_unit)
    return converted


def force_is_valid_tf(forces, max_force):
    """Return whether all force labels are finite and within the threshold."""
    forces = tf.convert_to_tensor(forces)
    finite = tf.reduce_all(tf.math.is_finite(forces))
    max_abs_force = tf.cond(
        tf.size(forces) > 0,
        lambda: tf.reduce_max(tf.abs(forces)),
        lambda: tf.zeros((), dtype=forces.dtype),
    )
    return tf.logical_and(
        finite,
        max_abs_force <= tf.cast(max_force, forces.dtype),
    )


def minimum_distance_np(positions, cell, pbc):
    """Return the closest distinct-atom distance using the requested PBC axes.

    ASE's minimum-image implementation supports triclinic cells and mixed PBC.
    Periodic images of the same atom are deliberately excluded: the data filter
    describes an interatomic, rather than a shortest lattice-vector, distance.
    """
    positions = np.asarray(positions)
    cell = np.asarray(cell)
    pbc = np.asarray(pbc, dtype=bool)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        return np.nan
    if positions.shape[0] < 2:
        return np.inf
    if not np.all(np.isfinite(positions)):
        return np.nan
    if np.any(pbc) and (cell.shape != (3, 3) or not np.all(np.isfinite(cell))):
        return np.nan

    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)
    distances = atoms.get_all_distances(mic=bool(np.any(pbc)))
    distinct_pairs = distances[np.triu_indices(positions.shape[0], k=1)]
    return float(np.min(distinct_pairs))


def minimum_distance_is_valid_tf(positions, cell, pbc, min_distance):
    """TensorFlow wrapper for the PBC-aware minimum-distance criterion."""
    valid = tf.py_function(
        func=lambda pos, c, periodic: np.asarray(
            minimum_distance_is_valid_np(
                pos, c, periodic, min_distance
            ),
            dtype=np.bool_,
        ),
        inp=[positions, cell, pbc],
        Tout=tf.bool,
    )
    valid.set_shape([])
    return valid


def minimum_distance_is_valid_np(positions, cell, pbc, min_distance):
    """Check for close contacts without allocating an all-pairs matrix.

    The filter only needs to know whether a violating pair exists.  A spatial
    hash handles free-space records using only direct dependencies and exits at
    the first close pair; ASE's neighbor list handles triclinic and mixed-axis
    periodic cells.  Periodic images of the same atom are excluded, matching
    :func:`minimum_distance_np`.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    min_distance = float(min_distance)

    if positions.ndim != 2 or positions.shape[1:] != (3,):
        return False
    if pbc.shape != (3,):
        return False
    if not np.all(np.isfinite(positions)) or not np.isfinite(min_distance):
        return False
    if min_distance < 0:
        return False
    if positions.shape[0] < 2 or min_distance == 0:
        return True
    if np.any(pbc) and (cell.shape != (3, 3) or not np.all(np.isfinite(cell))):
        return False

    if not np.any(pbc):
        # Cubes have side ``min_distance``.  Any violating pair must therefore
        # be in the same cube or one of its 26 neighbours.  Dense buckets
        # terminate on their first close pair rather than accumulating every
        # matching pair as query_pairs/all-pairs approaches do.
        scaled = positions / min_distance
        if not np.all(np.isfinite(scaled)):
            return False
        buckets = {}
        cutoff_squared = min_distance * min_distance
        for position, bucket_coords in zip(positions, np.floor(scaled)):
            bucket = tuple(int(coord) for coord in bucket_coords)
            for offset in _NEIGHBOUR_BUCKET_OFFSETS:
                neighbouring_bucket = tuple(
                    coord + delta for coord, delta in zip(bucket, offset)
                )
                for other in buckets.get(neighbouring_bucket, ()):
                    difference = position - other
                    if float(np.dot(difference, difference)) < cutoff_squared:
                        return False
            buckets.setdefault(bucket, []).append(position)
        return True

    # The documented predicate is distance >= threshold, so ask ASE only for
    # neighbours just below the boundary rather than rejecting an exact tie.
    violating_cutoff = np.nextafter(min_distance, -np.inf)
    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)
    i, j, distances = neighbor_list(
        'ijd', atoms, violating_cutoff, self_interaction=False
    )
    return not bool(np.any((i != j) & (distances < min_distance)))


def epoch_size_from_dataset_sizes(sizes, weights, mode):
    """Compute the finite number of raw records attempted in an epoch."""
    if len(sizes) != len(weights):
        raise ValueError('sizes and weights must have the same length.')
    if any(int(size) < 0 for size in sizes):
        raise ValueError('dataset sizes must be non-negative.')
    if mode == 'train':
        return sum(
            int(size) for size, weight in zip(sizes, weights) if weight > 0
        )
    return sum(int(size) for size in sizes)


def combine_epoch_datasets(datasets, weights, mode, epoch_size, seed):
    """Build a deterministic, finite dataset for one logical epoch.

    A training epoch makes ``epoch_size`` weighted draws with replacement.  In
    the usual case where weights are dataset sizes, each source is visited once
    in expectation.  Validation never repeats sources and therefore visits
    every selected record exactly once; weights affect training only.
    """
    if not datasets:
        raise ValueError('At least one TFDS dataset is required.')
    if epoch_size < 0:
        raise ValueError(f'epoch_size must be non-negative, got {epoch_size}.')

    if mode == 'train':
        repeated = [dataset.repeat() for dataset in datasets]
        combined = tf.data.Dataset.sample_from_datasets(
            repeated,
            weights=weights,
            seed=seed,
            stop_on_empty_dataset=False,
        )
        return combined.take(epoch_size)

    return tf.data.Dataset.sample_from_datasets(
        datasets,
        seed=seed,
        stop_on_empty_dataset=False,
    )


def compute_edges_np(positions, atomic_numbers, cell, pbc, cutoff):
    """Compute edges using ASE neighbor_list, handling both PBC and non-PBC.

    Args:
        positions: Atom positions as numpy array (N, 3).
        atomic_numbers: Atomic numbers as numpy array (N,).
        cell: Unit cell as numpy array (3, 3). Ignored when pbc is all False.
        pbc: Periodic boundary conditions as numpy array (3,) of bool.
        cutoff: Distance cutoff for edge creation.

    Returns:
        receivers: Center atom indices (int32).
        senders: Neighbor atom indices (int32).
        cell_offsets: Cell offset vectors (N_edges, 3) int32, empty (0, 3) if non-PBC.
    """
    mol = Atoms(positions=positions, numbers=atomic_numbers, cell=cell, pbc=pbc)
    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=False)
        return (
            np.array(i, dtype=np.int32),
            np.array(j, dtype=np.int32),
            np.array(S, dtype=np.int32),
        )
    else:
        # Non-PBC: use a fast vectorized all-pairs distance test (matches the old
        # loader's TF-native compute_edges_tf). ase.neighbor_list's binning can
        # stall/explode for some gas-phase structures and was the cause of the
        # dataloader-worker hang.
        pos = np.asarray(positions, dtype=np.float64)
        n = pos.shape[0]
        d = pos[None, :, :] - pos[:, None, :]
        dist = np.sqrt(np.einsum('ijk,ijk->ij', d, d))
        keep = (dist < cutoff) & ~np.eye(n, dtype=bool)
        i, j = np.where(keep)
        return (
            i.astype(np.int32),
            j.astype(np.int32),
            np.zeros((i.shape[0], 3), dtype=np.int32),  # dummy offsets for uniform output shape
        )

def standardize_dataset_fields(dataset, expected_fields=None):
    """Standardize dataset fields to ensure all datasets have the same structure.

    Args:
        dataset: TensorFlow dataset to standardize
        expected_fields: Dictionary of expected field names and their default specs

    Returns:
        Dataset with standardized fields
    """
    if expected_fields is None:
        # Define the complete expected structure based on your data
        expected_fields = {
            'atomic_numbers': {'dtype': tf.int32, 'shape': (None,)},
            'charge': {'dtype': tf.int32, 'shape': (1,)},
            'dipole_vec': {'dtype': tf.float32, 'shape': (3,)},
            'energy': {'dtype': tf.float32, 'shape': (None,)},
            'forces': {'dtype': tf.float32, 'shape': (None, 3)},
            'hirshfeld_ratios': {'dtype': tf.float32, 'shape': (None,)},
            'c6_ratios': {'dtype': tf.float32, 'shape': (None,)},
            'multiplicity': {'dtype': tf.int32, 'shape': (1,)},
            'positions': {'dtype': tf.float32, 'shape': (None, 3)},
            'stress': {'dtype': tf.float32, 'shape': (6,)},
            'theory_level': {'dtype': tf.int32, 'shape': (1,)},
            'cell': {'dtype': tf.float32, 'shape': (3, 3)},
            'pbc': {'dtype': tf.bool, 'shape': (3,)},
        }

    def add_missing_fields(sample):
        """Add missing fields with appropriate default values."""
        standardized_sample = {}

        # Copy existing fields
        for key, value in sample.items():
            standardized_sample[key] = value

        # Add missing cell (zeros = no cell)
        if 'cell' not in sample:
            standardized_sample['cell'] = tf.zeros((3, 3), dtype=tf.float32)
        # Add missing pbc: infer from cell if present, otherwise default to non-periodic
        if 'pbc' not in sample:
            cell = standardized_sample.get('cell', sample.get('cell'))
            has_cell = tf.reduce_any(tf.not_equal(cell, 0.0))
            standardized_sample['pbc'] = tf.where(
                has_cell,
                tf.constant([True, True, True], dtype=tf.bool),
                tf.constant([False, False, False], dtype=tf.bool),
            )

        # Add missing hirshfeld_ratios field only
        if 'hirshfeld_ratios' not in sample:
            # For hirshfeld_ratios, create NaN values with same length as atomic_numbers
            num_atoms = tf.shape(sample['atomic_numbers'])[0]
            hirshfeld_ratios_spec = expected_fields['hirshfeld_ratios']
            dtype = hirshfeld_ratios_spec['dtype']
            default_value = tf.fill((num_atoms,), tf.constant(float('nan'), dtype=dtype))
            standardized_sample['hirshfeld_ratios'] = default_value
        if 'c6_ratios' not in sample:
            # For c6_ratios, create NaN values with same length as atomic_numbers
            num_atoms = tf.shape(sample['atomic_numbers'])[0]
            c6_ratios_spec = expected_fields['c6_ratios']
            dtype = c6_ratios_spec['dtype']
            default_value = tf.fill((num_atoms,), tf.constant(float('nan'), dtype=dtype))
            standardized_sample['c6_ratios'] = default_value


        return standardized_sample

    return dataset.map(add_missing_fields, num_parallel_calls=tf.data.AUTOTUNE)

def create_graph_tuple_tf(
        element,
        cutoff: float,
        calculate_neighbors_lr: bool = False,
        cutoff_lr: Optional[float] = None,
        max_num_theory_levels: int = MAX_NUM_THEORY_LEVELS, #get from the config
) -> jraph.GraphsTuple:

    """Takes a data element and wraps relevant components in a GraphsTuple."""
    nodes_dict = dict()
    globals_dict = dict()

    atomic_numbers = element['atomic_numbers']
    positions = element['positions']

    nodes_dict['positions'] = positions
    nodes_dict['atomic_numbers'] = atomic_numbers
    num_atoms = tf.shape(atomic_numbers)[0]

    properties = element.keys()
    if 'forces' in properties:
        nodes_dict['forces'] = element['forces']
    if 'theory_level' in properties:
        theory_level = tf.reshape(element['theory_level'], (1,))

        # theory level 0 - PBE0+MBD-NL, 1 - PBE+MBD, 4 - OMol level
        # use real levels for training
        # theory_level0 = theory_level #real levels
        # theory_level0 = tf.where(tf.equal(theory_level0, 1), 0, theory_level0) #merge PBE0+MBD and PBE0+MBD-NL heads, for the final head only
        # use final fine-tuned head for inference
        theory_level0 = tf.reshape(5, (1,))

        globals_dict['theory_level'] = theory_level0
        theory_mask = tf.one_hot(theory_level0, depth=max_num_theory_levels)  # (1, num_theory_levels)
        globals_dict['theory_mask'] = theory_mask

    if 'hirshfeld_ratios' in properties and theory_level[0] == 0:  # Only include Hirshfeld ratios for theory level 0.
        nodes_dict['hirshfeld_ratios'] = element['hirshfeld_ratios']
    else:
        # Hack to deal with symbolic tensors, since this depends on the number of atoms in the molecule.
        hirshfeld_ratios = tf.reshape(
            tf.zeros_like(atomic_numbers, dtype=tf.float32) * tf.constant([np.nan], dtype=tf.float32),
            (-1, )
        )
        nodes_dict['hirshfeld_ratios'] = hirshfeld_ratios
    if 'c6_ratios' in properties and theory_level[0] == 0:  # Only include C6 ratios for theory level 0.
        nodes_dict['c6_ratios'] = element['c6_ratios']
    else:
        # Hack to deal with symbolic tensors, since this depends on the number of atoms in the molecule.
        c6_ratios = tf.reshape(
            tf.zeros_like(atomic_numbers, dtype=tf.float32) * tf.constant([np.nan], dtype=tf.float32),
            (-1, )
        )
        nodes_dict['c6_ratios'] = c6_ratios

    if 'energy' in properties and theory_level[0] != 4: #do not use omol energies
    #if 'energy' in properties:
        en = tf.reshape(element['energy'], (1,))
        globals_dict['energy'] = en
        #if en < 100 and theory_level[0] != 4:
    else:
        energy = np.empty((1,))
        energy[:] = np.nan
        globals_dict['energy'] = tf.convert_to_tensor(energy, dtype=tf.float32)
    if 'multiplicity' in properties:
        globals_dict['num_unpaired_electrons'] = tf.reshape(element['multiplicity'], (1,)) - 1
    if 'charge' in properties:
        globals_dict['total_charge'] = tf.reshape(element['charge'], (1,))

    if 'stress' in properties:
        globals_dict['stress'] = tf.reshape(element['stress'], (1, 6))
    else:
        stress = np.empty((1, 6))
        stress[:] = np.nan
        globals_dict['stress'] = tf.convert_to_tensor(stress, dtype=tf.float32)
    if 'dipole_vec' in properties: # and num_atoms < 100:
        globals_dict['dipole_vec'] = tf.reshape(element['dipole_vec'], (1, 3))
    else:
        dipole_vec = np.empty((1, 3))
        dipole_vec[:] = np.nan
        globals_dict['dipole_vec'] = tf.convert_to_tensor(dipole_vec, dtype=tf.float32)

    # Read cell and pbc from the element
    cell = element['cell']
    pbc = element['pbc']

    # Use tf.py_function to call numpy-based ASE neighbor_list
    centers, others, cell_offsets = tf.py_function(
        func=lambda pos, Z, c, p: compute_edges_np(
            pos.numpy(), Z.numpy(), c.numpy(), p.numpy(), cutoff
        ),
        inp=[positions, atomic_numbers, cell, pbc],
        Tout=[tf.int32, tf.int32, tf.int32],
    )
    centers.set_shape([None])
    others.set_shape([None])
    cell_offsets.set_shape([None, 3])

    # Always include cell_offset and cell so the output structure is uniform
    # across all elements (required by tf.data.Dataset.map).
    # Non-PBC entries will have zeros; _numpy_tuple_iterator strips them later.
    edges_dict = {'cell_offset': cell_offsets}
    globals_dict['cell'] = tf.reshape(cell, (1, 3, 3))

    if calculate_neighbors_lr is True:
        if cutoff_lr is None:
            raise ValueError(
                f'cutoff_lr must be specified for {calculate_neighbors_lr=}. Received {cutoff_lr=}.'
            )
        centers_lr, others_lr, cell_offsets_lr = tf.py_function(
            func=lambda pos, Z, c, p: compute_edges_np(
                pos.numpy(), Z.numpy(), c.numpy(), p.numpy(), cutoff_lr
            ),
            inp=[positions, atomic_numbers, cell, pbc],
            Tout=[tf.int32, tf.int32, tf.int32],
        )
        centers_lr.set_shape([None])
        others_lr.set_shape([None])
        cell_offsets_lr.set_shape([None, 3])
    else:
        centers_lr = tf.constant([], dtype=tf.int32)
        others_lr = tf.constant([], dtype=tf.int32)
        cell_offsets_lr = tf.zeros((0, 3), dtype=tf.int32)

    num_edges_lr = tf.shape(centers_lr)[0]
    num_nodes = tf.shape(atomic_numbers)[0]
    num_edges = tf.shape(centers)[0]

    graph = jraph.GraphsTuple(
        n_node=tf.reshape(num_nodes, (1,)),
        n_edge=tf.reshape(num_edges, (1,)),
        # Central nodes (idx_i) receive information from the neighboring nodes (idx_j).
        receivers=centers,
        senders=others,
        nodes=nodes_dict,
        globals=globals_dict,
        edges=edges_dict,
    )
    # Return long-range data as a plain dict of TF tensors so it's compatible with
    # tf.data.Dataset.map(). Converted to LongRangeNeighborList when consuming the
    # numpy iterator.
    lr_dict = {
        'idx_i_lr': centers_lr,
        'idx_j_lr': others_lr,
        'cell_offsets_lr': cell_offsets_lr,
        'n_pairs': tf.reshape(num_edges_lr, (1,)),
    }
    return graph, lr_dict

@dataclass
class WorkerConfig:
    """Configuration for worker processes.

    Contains all parameters needed by worker processes to load and process data.
    """
    split: str
    cutoff: float
    batch_max_num_nodes: int
    batch_max_num_edges: int
    batch_max_num_graphs: int
    batch_max_num_pairs: int
    shuffle_seed: int
    n_workers: int
    worker_idx: int
    max_force_filter: Optional[float]
    min_distance_filter: Optional[float]
    calculate_neighbors_lr: bool
    cutoff_lr: Optional[float]
    train_seed: int
    energy_unit: float
    length_unit: float
    dipole_vec_unit: float
    epoch_size: int
    cancellation_event: object
    mode: str
    input_folders: list
    dataset_weights: list
    num_train: list
    num_valid: list

class StopToken:
    """Token to signal that a worker has finished processing."""
    pass


@dataclass
class WorkerError:
    """Serializable worker failure sent back to the parent process."""

    worker_idx: int
    message: str
    traceback: str


class QCMLDataLoaderSparseParallel:
    """Data loader for TFDS datasets that loads data in parallel."""

    STOP_TOKEN = StopToken()

    def __init__(self, config, input_folders, dataset_weights, length_unit,
                 energy_unit, dipole_vec_unit=1.0):
        """Initialize the data loader with configuration parameters.

        Args:
            config: The configuration object containing all necessary parameters.
            input_folders: List of paths to the input data folders.
            dataset_weights: List of weights for each dataset.
            length_unit: Unit for length measurements.
            energy_unit: Unit for energy measurements.
            dipole_vec_unit: Unit for dipole-vector measurements.
        """
        self.config = config
        self.input_folders = input_folders
        if dataset_weights is None:
            dataset_weights = [1.0] * len(input_folders)
        if len(dataset_weights) != len(input_folders):
            raise ValueError(
                'dataset_weights must contain one entry per input folder.'
            )
        if (not all(np.isfinite(weight) and weight >= 0 for weight in dataset_weights)
                or not any(weight > 0 for weight in dataset_weights)):
            raise ValueError('dataset_weights must be finite, non-negative, and not all zero.')
        self.dataset_weights = [float(weight) for weight in dataset_weights]
        self.length_unit = length_unit
        self.energy_unit = energy_unit
        self.dipole_vec_unit = dipole_vec_unit

        # Set parameters from config
        self.calculate_neighbors_lr = config.data.neighbors_lr_bool
        # TFDS records are converted before neighbors and filters, so these
        # user-facing values remain in canonical Angstrom/eV units.
        self.cutoff = config.model.cutoff
        self.cutoff_lr = config.data.neighbors_lr_cutoff if self.calculate_neighbors_lr else None
        data_filter = getattr(config.data, 'filter', None)
        self.max_force_filter = (
            getattr(data_filter, 'max_force', None) if data_filter is not None else None
        )
        self.min_distance_filter = (
            getattr(data_filter, 'min_distance', None) if data_filter is not None else None
        )
        if self.max_force_filter is not None and self.max_force_filter < 0:
            raise ValueError('data.filter.max_force must be non-negative.')
        if self.min_distance_filter is not None and self.min_distance_filter < 0:
            raise ValueError('data.filter.min_distance must be non-negative.')
        self.train_seed = config.training.training_seed
        self.batch_max_num_nodes = config.training.batch_max_num_nodes
        self.batch_max_num_edges = config.training.batch_max_num_edges
        self.batch_max_num_graphs = config.training.batch_max_num_graphs
        self.batch_max_num_pairs = config.training.batch_max_num_pairs if config.training.batch_max_num_pairs is not None else 0

        # num_train and num_valid can be used to limit the number of examples taken from TFDS splits
        if hasattr(config.data, 'datasets'):
            self.num_train = [d.get('num_train', None) for d in config.data.datasets]
            self.num_valid = [d.get('num_valid', None) for d in config.data.datasets]
        else:
            self.num_train = [getattr(config.training, 'num_train', None)]
            self.num_valid = [getattr(config.training, 'num_valid', None)]

        # Log information about train/valid limits
        train_total = sum(x for x in self.num_train if x is not None)
        valid_total = sum(x for x in self.num_valid if x is not None)
        if train_total > 0 or valid_total > 0:
            print(f"Using TFDS splits with limits: num_train={train_total if train_total > 0 else 'unlimited'}, "
                  f"num_valid={valid_total if valid_total > 0 else 'unlimited'}")
        else:
            print("Using full TFDS splits without limits")

        try:
            self.n_proc = int(config.training.batch_n_proc)
            print(f"Using {self.n_proc} processes for parallel data loading")
        except (AttributeError, TypeError, ValueError):
            self.n_proc = 8
            print("Warning: No number of processes specified. Defaulting to 8.")
        if self.n_proc < 1:
            raise ValueError('training.batch_n_proc must be at least 1.')

        # multithread stuff # important
        ctx = get_context("spawn")
        self.manager = ctx.Manager()
        self.executor = ProcessPoolExecutor(max_workers=self.n_proc, mp_context=ctx)

        # Cache for cardinality
        self._cardinality = None
        self._dataset_sizes_cache = {}
        self._epoch_counters = {'train': 0, 'validation': 0}

    @staticmethod
    def _safe_put(output_queue, item, cancellation_event=None):
        """Put an item without leaving a worker blocked after consumer exit."""
        while cancellation_event is None or not cancellation_event.is_set():
            try:
                output_queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _dataset_sizes(self, mode):
        """Return exact selected-record counts for an epoch's source splits."""
        if mode in self._dataset_sizes_cache:
            return self._dataset_sizes_cache[mode]

        split_name = 'train' if mode == 'train' else 'validation'
        limits = self.num_train if mode == 'train' else self.num_valid
        sizes = []
        for i, folder in enumerate(self.input_folders):
            builder = tfds.builder_from_directory(folder)
            try:
                size = int(builder.info.splits[split_name].num_examples)
            except (KeyError, TypeError, AttributeError):
                dataset = builder.as_dataset(split=split_name)
                size = int(tf.data.experimental.cardinality(dataset).numpy())
                if size < 0:
                    raise ValueError(
                        f'Cannot determine cardinality of TFDS split {split_name!r} '
                        f'in {folder!r}; a finite epoch cannot be defined.'
                    )
            limit = limits[i] if i < len(limits) else None
            if limit is not None:
                if int(limit) < 0:
                    raise ValueError(f'{split_name} limit must be non-negative, got {limit}.')
                size = min(size, int(limit))
            sizes.append(size)

        self._dataset_sizes_cache[mode] = sizes
        return sizes

    def _epoch_size(self, mode):
        return epoch_size_from_dataset_sizes(
            self._dataset_sizes(mode), self.dataset_weights, mode
        )

    def cardinality(self) -> int:
        """Calculate total number of examples across all datasets, respecting limits.

        Returns:
            Total number of examples considering num_train and num_valid limits.
        """
        if self._cardinality is not None:
            return self._cardinality

        total_examples = 0
        for i, folder in enumerate(self.input_folders):
            builder = tfds.builder_from_directory(folder)

            # Count examples across relevant splits, respecting limits
            for split_name in ['train', 'validation', 'test']:
                try:
                    # Construct split string with limits if specified
                    if split_name == 'train' and i < len(self.num_train) and self.num_train[i] is not None:
                        split_str = f'train[:{self.num_train[i]}]'
                    elif split_name == 'validation' and i < len(self.num_valid) and self.num_valid[i] is not None:
                        split_str = f'validation[:{self.num_valid[i]}]'
                    else:
                        split_str = split_name

                    dataset = builder.as_dataset(split=split_str)
                    count = tf.data.experimental.cardinality(dataset).numpy()
                    total_examples += count
                except Exception:
                    # Skip if split doesn't exist
                    pass

        self._cardinality = total_examples
        return total_examples

    def dataset_info(self):
        """Return per-dataset training sizes and normalized weights for epoch tracking.

        Returns:
            dict with keys:
                'names': list of dataset folder basenames
                'sizes': list of per-dataset training set sizes
                'weights': normalized sampling weights (sum to 1.0)
        """
        from pathlib import Path
        names = [Path(f).name for f in self.input_folders]
        sizes = self._dataset_sizes('train')

        # Normalize weights
        raw_weights = self.dataset_weights if self.dataset_weights is not None else [1.0] * len(self.input_folders)
        total_weight = sum(raw_weights)
        normalized_weights = [w / total_weight for w in raw_weights]

        return {
            'names': names,
            'sizes': sizes,
            'weights': normalized_weights,
        }

    def _dataset_identity_metadata(self):
        """Describe the TFDS builders without reading every record payload.

        Builder metadata files are hashed, and shard names/sizes are recorded.
        This catches ordinary dataset/version/schema replacement while keeping
        restart startup independent of total dataset size.
        """
        identities = []
        for folder in self.input_folders:
            folder_path = Path(folder).expanduser().resolve()
            builder = tfds.builder_from_directory(folder_path)
            info = builder.info
            splits = {}
            for name, split_info in sorted(info.splits.items()):
                splits[str(name)] = {
                    'num_examples': int(split_info.num_examples),
                    'shard_lengths': [
                        int(length)
                        for length in getattr(split_info, 'shard_lengths', ())
                    ],
                }

            metadata_hashes = {}
            for metadata_path in sorted(folder_path.glob('*.json')):
                if not metadata_path.is_file():
                    continue
                hasher = hashlib.sha256()
                with metadata_path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        hasher.update(chunk)
                metadata_hashes[metadata_path.name] = hasher.hexdigest()

            shards = []
            for shard_path in sorted(folder_path.glob('*tfrecord*')):
                if shard_path.is_file():
                    shards.append({
                        'name': shard_path.name,
                        'size_bytes': int(shard_path.stat().st_size),
                    })

            config_name = getattr(info, 'config_name', None)
            identities.append({
                'path': str(folder_path),
                'name': str(getattr(info, 'name', folder_path.parent.name)),
                'config_name': (
                    None if config_name is None else str(config_name)
                ),
                'version': str(getattr(info, 'version', 'unknown')),
                'full_name': str(getattr(info, 'full_name', 'unknown')),
                'splits': splits,
                'metadata_sha256': metadata_hashes,
                'record_shards': shards,
            })
        return identities

    def restart_fingerprint_payload(self):
        """Return the deterministic loader/config identity used for restart.

        The payload intentionally excludes ``num_epochs`` so a completed run
        can be extended.  It includes every setting that determines record
        selection, filtering, graph construction, batching, and worker order.
        """
        trajectory_config = None
        if getattr(self, 'config', None) is not None:
            full_config = _json_compatible(self.config)
            training_config = dict(full_config.get('training', {}))
            # These settings affect run length/reporting, not the mathematical
            # next update.  In particular, excluding num_epochs permits a
            # checkpointed run to be extended.
            for key in (
                    'allow_restart', 'num_epochs', 'use_wandb',
                    'wandb_init_args', 'log_gradient_values'):
                training_config.pop(key, None)
            trajectory_config = {
                'data': full_config.get('data'),
                'model': full_config.get('model'),
                'optimizer': full_config.get('optimizer'),
                'training': training_config,
                'neighborlist_format_lr': full_config.get(
                    'neighborlist_format_lr'
                ),
            }

        return {
            'fingerprint_schema_version': _TFDS_STREAM_FINGERPRINT_VERSION,
            'epoch_stream_version': _TFDS_EPOCH_STREAM_VERSION,
            'loader': f'{type(self).__module__}.{type(self).__qualname__}',
            'datasets': self._dataset_identity_metadata(),
            'selected_train_sizes': [int(x) for x in self._dataset_sizes('train')],
            'selected_validation_sizes': [
                int(x) for x in self._dataset_sizes('validation')
            ],
            'dataset_weights': [float(x) for x in self.dataset_weights],
            'num_train': [None if x is None else int(x) for x in self.num_train],
            'num_valid': [None if x is None else int(x) for x in self.num_valid],
            'batch_budgets': {
                'nodes': int(self.batch_max_num_nodes),
                'edges': int(self.batch_max_num_edges),
                'graphs': int(self.batch_max_num_graphs),
                'pairs': int(self.batch_max_num_pairs),
            },
            'neighbors': {
                'cutoff': float(self.cutoff),
                'calculate_lr': bool(self.calculate_neighbors_lr),
                'cutoff_lr': (
                    None if self.cutoff_lr is None else float(self.cutoff_lr)
                ),
            },
            'filters': {
                'max_force': (
                    None if self.max_force_filter is None
                    else float(self.max_force_filter)
                ),
                'min_distance': (
                    None if self.min_distance_filter is None
                    else float(self.min_distance_filter)
                ),
            },
            'unit_factors': {
                'energy': float(self.energy_unit),
                'length': float(self.length_unit),
                'dipole_vec': float(self.dipole_vec_unit),
            },
            'training_seed': int(self.train_seed),
            'n_proc': int(self.n_proc),
            'graph_schema': {
                'max_num_theory_levels': int(MAX_NUM_THEORY_LEVELS),
                'shuffle_buffer_size': 10_000,
            },
            'trajectory_config': trajectory_config,
            'runtime': {
                'python': '.'.join(map(str, sys.version_info[:3])),
                'numpy': np.__version__,
                'tensorflow': _installed_version(
                    'tensorflow', getattr(tf, '__version__', 'unavailable')
                ),
                'tensorflow_datasets': _installed_version(
                    'tensorflow-datasets',
                    getattr(tfds, '__version__', 'unavailable'),
                ),
                'ase': _installed_version('ase'),
                'jraph': _installed_version('jraph'),
            },
        }

    def restart_fingerprint(self):
        """SHA-256 identity for deterministic TFDS trajectory reconstruction."""
        return _canonical_fingerprint(self.restart_fingerprint_payload())

    @staticmethod
    def _preprocess(dataset,
                    batch_max_num_nodes,
                    batch_max_num_edges,
                    batch_max_num_graphs,
                    batch_max_num_pairs,
                    cutoff,
                    calculate_neighbors_lr,
                    cutoff_lr,
                    max_force_filter,
                    min_distance_filter,
                    energy_unit,
                    length_unit,
                    dipole_vec_unit,
                    train_seed,
                    shuffle=True):
        """Preprocess the dataset by creating graph tuples and batching.

        Args:
            dataset: TensorFlow dataset to preprocess
            batch_max_num_nodes: Maximum number of nodes in a batch
            batch_max_num_edges: Maximum number of edges in a batch
            batch_max_num_graphs: Maximum number of graphs in a batch
            batch_max_num_pairs: Maximum number of pairs in a batch
            cutoff: Distance cutoff for edge creation
            calculate_neighbors_lr: Whether to calculate long-range neighbors
            cutoff_lr: Distance cutoff for long-range neighbors
            max_force_filter: Maximum force value for filtering
            min_distance_filter: Minimum distinct-atom separation in Angstrom
            energy_unit: Conversion factor from raw energies to eV
            length_unit: Conversion factor from raw lengths to Angstrom
            dipole_vec_unit: Conversion factor for raw dipole-vector labels
            train_seed: Random seed for shuffling
            shuffle: Whether to shuffle this finite epoch before batching.

        Returns:
            Batched dataset of graph tuples
        """

        # Normalize all geometry and labels before applying canonical-unit
        # filters or constructing neighbor lists.
        dataset = dataset.map(
            lambda element: convert_dataset_units_tf(
                element, energy_unit, length_unit, dipole_vec_unit
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if max_force_filter is not None and 'forces' in dataset.element_spec:
            dataset = dataset.filter(
                lambda element: force_is_valid_tf(
                    element['forces'], max_force_filter
                )
            )

        if min_distance_filter is not None:
            dataset = dataset.filter(
                lambda element: minimum_distance_is_valid_tf(
                    element['positions'], element['cell'], element['pbc'],
                    min_distance_filter,
                )
            )

        if shuffle:
            # A new seed is supplied for every training epoch, so this dataset
            # itself should not silently reshuffle if iterated more than once.
            dataset = dataset.shuffle(
                buffer_size=10_000,
                reshuffle_each_iteration=False,
                seed=train_seed,
            )

        # Create graph tuples only after rejecting invalid structures.
        dataset = dataset.map(
            lambda element: create_graph_tuple_tf(
                element,
                cutoff=cutoff,
                calculate_neighbors_lr=calculate_neighbors_lr,
                cutoff_lr=cutoff_lr,
                max_num_theory_levels=MAX_NUM_THEORY_LEVELS
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Prefetch for better performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Create batches using the new wrapper that handles both budgets
        def _numpy_tuple_iterator(ds):
            """Convert TF dataset of (graph, lr_dict) tuples to numpy iterator."""
            for graph, lr_dict in ds.as_numpy_iterator():
                # Check if this structure has PBC (non-zero cell)
                cell = graph.globals.get('cell')
                has_pbc = cell is not None and np.any(cell != 0.0)

                # Strip PBC fields for non-periodic systems to match ASE/NPZ behavior
                if not has_pbc:
                    # Remove cell from globals
                    globals_new = {k: v for k, v in graph.globals.items() if k != 'cell'}
                    # Remove cell_offset from edges
                    edges_new = {k: v for k, v in graph.edges.items() if k != 'cell_offset'}
                    graph = graph._replace(globals=globals_new, edges=edges_new)

                # Determine cell_offsets_lr
                offsets = np.asarray(lr_dict['cell_offsets_lr'])
                cell_offsets_lr = offsets if has_pbc and offsets.shape[0] > 0 else None

                lr_data = {
                    'idx_i_lr': np.asarray(lr_dict['idx_i_lr']),
                    'idx_j_lr': np.asarray(lr_dict['idx_j_lr']),
                    'cell_offsets_lr': cell_offsets_lr,
                    'n_pairs': np.asarray(lr_dict['n_pairs']),
                }
                yield graph, lr_data

        return dynamically_batch_with_lr(
            _numpy_tuple_iterator(dataset),
            n_node=batch_max_num_nodes,
            n_edge=batch_max_num_edges,
            n_graph=batch_max_num_graphs,
            n_pairs=batch_max_num_pairs,
        )

    @staticmethod
    def _worker(config, output_queue):
        """Worker function that loads data for the given indices and puts it into the queue.

        Args:
            config: WorkerConfig with all parameters
            output_queue: Queue to put processed batches into
        """
        try:
            # We need a deterministic shuffle seed, s.t. workers shuffle files in the same way
            read_config = tfds.ReadConfig(
                shuffle_seed=config.shuffle_seed,
            )

            # Load all datasets
            datasets = []
            for i, folder in enumerate(config.input_folders):
                builder = tfds.builder_from_directory(folder)
                # Use proper TFDS splits created by xyz_to_tfds_6.py with optional limits
                if config.mode == 'train':
                    if i < len(config.num_train) and config.num_train[i] is not None:
                        split = f'train[:{config.num_train[i]}]'
                    else:
                        split = 'train'
                else:  # validation
                    if i < len(config.num_valid) and config.num_valid[i] is not None:
                        split = f'validation[:{config.num_valid[i]}]'
                    else:
                        split = 'validation'

                dataset = builder.as_dataset(
                    split=split,
                    shuffle_files=config.mode == 'train',
                    read_config=read_config,
                )

                # Standardize dataset fields to ensure compatibility
                dataset = standardize_dataset_fields(dataset)

                datasets.append(dataset)

            # Training sources repeat only inside this bounded mixer.  The
            # resulting iterator always terminates at the configured epoch size.
            dataset = combine_epoch_datasets(
                datasets,
                weights=config.dataset_weights,
                mode=config.mode,
                epoch_size=config.epoch_size,
                seed=config.shuffle_seed,
            )

            # Shard the combined dataset
            dataset = dataset.shard(num_shards=config.n_workers, index=config.worker_idx)

            for batch in QCMLDataLoaderSparseParallel._preprocess(dataset,
                                                batch_max_num_nodes=config.batch_max_num_nodes,
                                                batch_max_num_edges=config.batch_max_num_edges,
                                                batch_max_num_graphs=config.batch_max_num_graphs,
                                                batch_max_num_pairs=config.batch_max_num_pairs,
                                                cutoff=config.cutoff,
                                                calculate_neighbors_lr=config.calculate_neighbors_lr,
                                                cutoff_lr=config.cutoff_lr,
                                                max_force_filter=config.max_force_filter,
                                                min_distance_filter=config.min_distance_filter,
                                                energy_unit=config.energy_unit,
                                                length_unit=config.length_unit,
                                                dipole_vec_unit=config.dipole_vec_unit,
                                                train_seed=config.train_seed,
                                                shuffle=config.mode == 'train',
                                                ):
                if not QCMLDataLoaderSparseParallel._safe_put(
                    output_queue, batch, config.cancellation_event
                ):
                    break
        except Exception as e:
            QCMLDataLoaderSparseParallel._safe_put(
                output_queue,
                WorkerError(
                    worker_idx=config.worker_idx,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
                config.cancellation_event,
            )
        finally:
            # Finished processing data, always put a stop token
            QCMLDataLoaderSparseParallel._safe_put(
                output_queue,
                QCMLDataLoaderSparseParallel.STOP_TOKEN,
                config.cancellation_event,
            )

    def _generator(self, split, mode, epoch_seed):
        """Generator reads data from the queue.

        Args:
            split: Dataset split to use
            mode: 'train' or 'validation'

        Yields:
            Batches of data
        """
        # One queue per worker lets the consumer drain batches round-robin.
        # This makes batch order independent of process scheduling while still
        # allowing every worker to prepare two batches ahead.
        output_queues = [self.manager.Queue(maxsize=2) for _ in range(self.n_proc)]
        cancellation_event = self.manager.Event()
        epoch_size = self._epoch_size(mode)

        futures = {}
        for i in range(self.n_proc):
            config = WorkerConfig(
                worker_idx=i,
                n_workers=self.n_proc,
                split=split,
                shuffle_seed=epoch_seed,
                batch_max_num_nodes=self.batch_max_num_nodes,
                batch_max_num_edges=self.batch_max_num_edges,
                batch_max_num_graphs=self.batch_max_num_graphs,
                batch_max_num_pairs=self.batch_max_num_pairs,
                cutoff=self.cutoff,
                calculate_neighbors_lr=self.calculate_neighbors_lr,
                cutoff_lr=self.cutoff_lr,
                max_force_filter=self.max_force_filter,
                min_distance_filter=self.min_distance_filter,
                train_seed=epoch_seed,
                energy_unit=self.energy_unit,
                length_unit=self.length_unit,
                dipole_vec_unit=self.dipole_vec_unit,
                epoch_size=epoch_size,
                cancellation_event=cancellation_event,
                input_folders=self.input_folders,
                dataset_weights=self.dataset_weights,
                num_train=self.num_train,
                num_valid=self.num_valid,
                mode=mode
            )
            futures[i] = self.executor.submit(
                QCMLDataLoaderSparseParallel._worker, config, output_queues[i]
            )

        active_workers = list(range(self.n_proc))
        ctr = 0
        try:
            while active_workers:
                next_active_workers = []
                for worker_idx in active_workers:
                    while True:
                        try:
                            batch = output_queues[worker_idx].get(timeout=3)
                            break
                        except queue.Empty:
                            future = futures[worker_idx]
                            if future.done():
                                try:
                                    worker_result = future.result()
                                except Exception as e:
                                    raise RuntimeError(
                                        f'TFDS worker {worker_idx} terminated '
                                        f'without reporting its error: {e}'
                                    ) from e
                                raise RuntimeError(
                                    f'TFDS worker {worker_idx} exited without a '
                                    f'completion token (result={worker_result!r}).'
                                )
                            continue
                        except Exception as e:
                            raise RuntimeError(
                                f'Could not read from TFDS worker {worker_idx}: {e}'
                            ) from e

                    ctr += 1
                    if ctr % 1000 == 0 and wandb.run is not None:
                        size = sum(output_queue.qsize() for output_queue in output_queues)
                        wandb.log({"queue_size_"+mode: size})

                    if isinstance(batch, WorkerError):
                        raise RuntimeError(
                            f'TFDS worker {batch.worker_idx} failed: {batch.message}\n'
                            f'{batch.traceback}'
                        )
                    if not isinstance(batch, StopToken):
                        next_active_workers.append(worker_idx)
                        yield batch
                active_workers = next_active_workers
        finally:
            # Early stopping or a consumer exception closes this generator.
            # Wake blocked workers so the shared executor remains usable.
            cancellation_event.set()

    def next_epoch(self, split, mode, parallel: bool = True,
                   advance_epoch: bool = True, epoch_index: Optional[int] = None,
                   skip_batches: int = 0):
        """Loads the data for ONE epoch.

        Args:
            split: Dataset split to use
            mode: 'train' or 'validation'
            parallel: Use the multiprocessing pipeline. Pass ``False`` for a
                lightweight consumer that reads only a few batches, such as a
                metadata probe. Parallel generators cancel their workers when
                closed before the finite epoch is exhausted.
            advance_epoch: Increment the deterministic seed counter. Internal
                metadata probes pass ``False`` so they do not alter the first
                real training epoch.
            epoch_index: Explicit logical training epoch. Checkpoint restart
                uses this to rebuild the same seeded stream before skipping
                batches already represented by the checkpoint.
            skip_batches: Number of deterministic batches to discard from the
                start of the selected epoch.

        Returns:
            Generator yielding batches of data
        """
        if isinstance(skip_batches, bool) or int(skip_batches) != skip_batches or skip_batches < 0:
            raise ValueError(f'skip_batches must be a non-negative integer, got {skip_batches!r}.')
        skip_batches = int(skip_batches)

        if mode not in self._epoch_counters:
            self._epoch_counters[mode] = 0
        # Validation must be a stable measurement of unchanged parameters.
        # Reuse a fixed seed/order rather than advancing the training RNG state
        # at every evaluation.
        if mode == 'validation':
            selected_epoch_index = 0
        else:
            if epoch_index is not None and (
                    isinstance(epoch_index, bool)
                    or int(epoch_index) != epoch_index):
                raise ValueError(
                    f'epoch_index must be a non-negative integer, got {epoch_index!r}.'
                )
            selected_epoch_index = (
                self._epoch_counters[mode]
                if epoch_index is None else int(epoch_index)
            )
            if selected_epoch_index < 0:
                raise ValueError(
                    f'epoch_index must be non-negative, got {selected_epoch_index}.'
                )
            if advance_epoch:
                self._epoch_counters[mode] = max(
                    self._epoch_counters[mode], selected_epoch_index + 1
                )
        epoch_seed = (int(self.train_seed) + selected_epoch_index) % (2**31 - 1)

        if mode == 'train' and parallel: # use multiprocessing for training batches
            iterator = self._generator(split, mode, epoch_seed)
        else:
            iterator = self._generator_sync(split, mode, epoch_seed)
        return (
            _ClosePropagatingSkipIterator(iterator, skip_batches)
            if skip_batches else iterator
        )

    def _generator_sync(self, split, mode, epoch_seed):
        """Synchronous data generator for single-process operation.

        Args:
            split: Dataset split to use

        Yields:
            Batches of data
        """
        # Load all datasets
        datasets = []
        for i, folder in enumerate(self.input_folders):
            builder = tfds.builder_from_directory(folder)
            # Use proper TFDS splits created by xyz_to_tfds_6.py with optional limits
            if mode == 'train':
                if i < len(self.num_train) and self.num_train[i] is not None:
                    split = f'train[:{self.num_train[i]}]'
                else:
                    split = 'train'
            else:  # validation
                if i < len(self.num_valid) and self.num_valid[i] is not None:
                    split = f'validation[:{self.num_valid[i]}]'
                else:
                    split = 'validation'

            read_config = tfds.ReadConfig(shuffle_seed=epoch_seed)
            dataset = builder.as_dataset(
                split=split,
                shuffle_files=mode == 'train',
                read_config=read_config,
            )

            # Standardize dataset fields to ensure compatibility
            dataset = standardize_dataset_fields(dataset)

            datasets.append(dataset)

        dataset = combine_epoch_datasets(
            datasets,
            weights=self.dataset_weights,
            mode=mode,
            epoch_size=self._epoch_size(mode),
            seed=epoch_seed,
        )

        for batch in self._preprocess(dataset,
                                   batch_max_num_nodes=self.batch_max_num_nodes,
                                   batch_max_num_edges=self.batch_max_num_edges,
                                   batch_max_num_graphs=self.batch_max_num_graphs,
                                   batch_max_num_pairs=self.batch_max_num_pairs,
                                   cutoff=self.cutoff,
                                   calculate_neighbors_lr=self.calculate_neighbors_lr,
                                   cutoff_lr=self.cutoff_lr,
                                   max_force_filter=self.max_force_filter,
                                   min_distance_filter=self.min_distance_filter,
                                   energy_unit=self.energy_unit,
                                   length_unit=self.length_unit,
                                   dipole_vec_unit=self.dipole_vec_unit,
                                   train_seed=epoch_seed,
                                   shuffle=mode == 'train',
                                  ):
            yield batch
