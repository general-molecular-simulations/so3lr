import jraph
import logging
import numpy as np
from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr

from dataclasses import dataclass
from functools import partial, partialmethod
from typing import Optional
import queue
import wandb
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, get_context

try:
    import tensorflow as tf
except ModuleNotFoundError:
    logging.warning(
        "For using QCMLDataLoader please install tensorflow."
    )
try:
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    logging.warning(
        "For using QCMLDataLoader please install tensorflow_datasets."
    )


logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


def compute_edges_tf(
    positions,
    cutoff: float
):
    """Compute edges between atoms based on distance cutoff.
    
    Args:
        positions: Atom positions tensor
        cutoff: Distance cutoff for edge creation
        
    Returns:
        centers: Indices of center atoms
        others: Indices of neighbor atoms
    """
    num_atoms = tf.shape(positions)[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = tf.norm(displacements, axis=-1)
    mask = ~tf.eye(num_atoms, dtype=tf.bool)  # Get rid of self-connections.
    keep_edges = tf.where((distances < cutoff) & mask)
    centers = tf.cast(keep_edges[:, 0], dtype=tf.int32)  # center indices
    others = tf.cast(keep_edges[:, 1], dtype=tf.int32)  # neighbor indices
    return centers, others

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
        }

    def add_missing_fields(sample):
        """Add missing fields with appropriate default values."""
        standardized_sample = {}

        # Copy existing fields
        for key, value in sample.items():
            standardized_sample[key] = value

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
        max_num_theory_levels: int = 16, #get from the config
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

        theory_level0 = theory_level #real levels
        theory_level0 = tf.where(tf.equal(theory_level0, 1), 0, theory_level0) #merge PBE0+MBD and PBE0+MBD-NL heads, for the final head only
        theory_level0 = tf.reshape(5, (1,)) #for final head level = 5 or 6

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

    if 'energy' in properties and theory_level[0] != 4:
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

    centers, others = compute_edges_tf(
        positions=positions,
        cutoff=cutoff
    )

    if calculate_neighbors_lr is True:
        if cutoff_lr is None:
            raise ValueError(
                f'cutoff_lr must be specified for {calculate_neighbors_lr=}. Received {cutoff_lr=}.'
            )
        centers_lr, others_lr = compute_edges_tf(
            positions=positions,
            cutoff=cutoff_lr
        )
    else:
        centers_lr = tf.constant([], dtype=tf.int64)
        others_lr = tf.constant([], dtype=tf.int64)

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
        edges=dict(),  # Don't set to None, since otherwise tf.data.Dataset.to_numpy_generator() does not work due to
        # call to None.numpy().
    )
    # Return long-range data as a plain dict of TF tensors so it's compatible with
    # tf.data.Dataset.map(). Converted to LongRangeNeighborList when consuming the
    # numpy iterator.
    lr_dict = {
        'idx_i_lr': centers_lr,
        'idx_j_lr': others_lr,
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
    max_force_filter: float
    calculate_neighbors_lr: bool
    cutoff_lr: float
    train_seed: int
    mode: str
    input_folders: list
    dataset_weights: list
    num_train: list
    num_valid: list

class StopToken:
    """Token to signal that a worker has finished processing."""
    pass

class QCMLDataLoaderSparseParallel:
    """Data loader for TFDS datasets that loads data in parallel."""

    STOP_TOKEN = StopToken()

    def __init__(self, config, input_folders, dataset_weights, length_unit, energy_unit):
        """Initialize the data loader with configuration parameters.

        Args:
            config: The configuration object containing all necessary parameters.
            input_folders: List of paths to the input data folders.
            dataset_weights: List of weights for each dataset.
            length_unit: Unit for length measurements.
            energy_unit: Unit for energy measurements.
        """
        self.config = config
        self.input_folders = input_folders
        self.dataset_weights = dataset_weights
        self.length_unit = length_unit
        self.energy_unit = energy_unit

        # Set parameters from config
        self.calculate_neighbors_lr = config.data.neighbors_lr_bool
        self.cutoff = config.model.cutoff / self.length_unit
        self.cutoff_lr = config.data.neighbors_lr_cutoff / self.length_unit if self.calculate_neighbors_lr else None
        self.max_force_filter = config.data.filter.max_force / self.energy_unit * self.length_unit if hasattr(config.data.filter, 'max_force') else 1.e6
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

        #TODO: add checks for num_train and num_valid
        try:
            self.n_proc = int(config.training.batch_n_proc)
            print(f"Using {self.n_proc} processes for parallel data loading")
        except:
            self.n_proc = 8
            print("Warning: No number of processes specified. Defaulting to 8.")

        # multithread stuff # important
        ctx = get_context("spawn")
        self.manager = ctx.Manager()
        self.executor = ProcessPoolExecutor(max_workers=self.n_proc, mp_context=ctx)

        # Cache for cardinality
        self._cardinality = None

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
                    train_seed):
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
            train_seed: Random seed for shuffling

        Returns:
            Batched dataset of graph tuples
        """

        # First create graph tuples (returns (graph, long_range) tuples)
        dataset = dataset.map(
            lambda element: create_graph_tuple_tf(
                element,
                cutoff=cutoff,
                calculate_neighbors_lr=calculate_neighbors_lr,
                cutoff_lr=cutoff_lr,
                max_num_theory_levels=16
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Apply force filter (access graph from the tuple)
        dataset = dataset.filter(
            lambda graph, lr_dict: tf.math.less(tf.math.reduce_max(graph.nodes['forces']), tf.constant(max_force_filter))
        )

        # Shuffle
        dataset = dataset.shuffle(
            buffer_size=10_000,
            reshuffle_each_iteration=True,
            seed=train_seed
        )

        # Prefetch for better performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        #TODO: need to add data.transformations.unit_conversion_graph before filtering

        # Create batches using the new wrapper that handles both budgets
        def _numpy_tuple_iterator(ds):
            """Convert TF dataset of (graph, lr_dict) tuples to numpy iterator."""
            for graph, lr_dict in ds.as_numpy_iterator():
                lr_np = jraph.GraphsTuple(
                    nodes=None,
                    edges=None,
                    senders=np.asarray(lr_dict['idx_j_lr']),
                    receivers=np.asarray(lr_dict['idx_i_lr']),
                    n_node=np.asarray(graph.n_node),
                    n_edge=np.asarray(lr_dict['n_pairs']),
                    globals=None,
                )
                yield graph, lr_np

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

                dataset = builder.as_dataset(split=split, shuffle_files=True, read_config=read_config)

                # Standardize dataset fields to ensure compatibility
                dataset = standardize_dataset_fields(dataset)

                if config.mode == 'train':
                    dataset = dataset.repeat() # to avoid exhausting the smaller dataset, makes one epoch infinite

                datasets.append(dataset)

            # Combine datasets with weighted sampling
            dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=config.dataset_weights)

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
                                                train_seed=config.train_seed
                                                ):
                output_queue.put(batch)
        except Exception as e:
            print(f"[!] Error in worker {config.worker_idx}: {e}")
        finally:
            # Finished processing data, always put a stop token
            QCMLDataLoaderSparseParallel._safe_put(output_queue, QCMLDataLoaderSparseParallel.STOP_TOKEN)

    def _generator(self, split, mode):
        """Generator reads data from the queue.

        Args:
            split: Dataset split to use
            mode: 'train' or 'validation'

        Yields:
            Batches of data
        """
        shuffle_seed = np.random.randint(0, 2**31 - 1) # different shuffle seed for each epoch
        output_queue = self.manager.Queue(maxsize=2) # cache maximum of 2 batches

        for i in range(self.n_proc):
            config = WorkerConfig(
                worker_idx=i,
                n_workers=self.n_proc,
                split=split,
                shuffle_seed=shuffle_seed,
                batch_max_num_nodes=self.batch_max_num_nodes,
                batch_max_num_edges=self.batch_max_num_edges,
                batch_max_num_graphs=self.batch_max_num_graphs,
                batch_max_num_pairs=self.batch_max_num_pairs,
                cutoff=self.cutoff,
                calculate_neighbors_lr=self.calculate_neighbors_lr,
                cutoff_lr=self.cutoff_lr,
                max_force_filter=self.max_force_filter,
                train_seed=self.train_seed,
                input_folders=self.input_folders,
                dataset_weights=self.dataset_weights,
                num_train=self.num_train,
                num_valid=self.num_valid,
                mode=mode
            )
            self.executor.submit(QCMLDataLoaderSparseParallel._worker, config, output_queue)

        n_stop_token = 0
        ctr = 0
        while True:
            ctr += 1
            try:
                batch = output_queue.get(timeout=3)
            except queue.Empty:
                # Timeout expired, try again
                continue
            except Exception as e:
                print(f"[!] Unexpected error retrieving item from queue: {e}")
                break

            if ctr % 1000 == 0:
                size = output_queue.qsize()
                if wandb.run is not None:
                    wandb.log({"queue_size_"+mode: size})

            if isinstance(batch, StopToken):
                n_stop_token += 1

                # wait for all processes to finish
                if n_stop_token == self.n_proc:
                    break
            else:
                yield batch

    def next_epoch(self, split, mode):
        """Loads the data for ONE epoch.

        Args:
            split: Dataset split to use

        Returns:
            Generator yielding batches of data
        """
        if mode == 'train': # use multiporcessing for training batch
            return self._generator(split, mode)
        else:
            return self._generator_sync(split, mode)

    def _generator_sync(self, split, mode):
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

            dataset = builder.as_dataset(split=split, shuffle_files=True)

            # Standardize dataset fields to ensure compatibility
            dataset = standardize_dataset_fields(dataset)

            datasets.append(dataset)

        # Combine datasets with weighted sampling
        dataset = tf.data.Dataset.sample_from_datasets(datasets) #, weights=self.dataset_weights)

        for batch in self._preprocess(dataset,
                                   batch_max_num_nodes=self.batch_max_num_nodes,
                                   batch_max_num_edges=self.batch_max_num_edges,
                                   batch_max_num_graphs=self.batch_max_num_graphs,
                                   batch_max_num_pairs=self.batch_max_num_pairs,
                                   cutoff=self.cutoff,
                                   calculate_neighbors_lr=self.calculate_neighbors_lr,
                                   cutoff_lr=self.cutoff_lr,
                                   max_force_filter=self.max_force_filter,
                                   train_seed=self.train_seed
                                  ):
            yield batch