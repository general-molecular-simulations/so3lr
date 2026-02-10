import h5py
import jax.numpy as jnp
from dataclasses import dataclass
import jraph
from tqdm import tqdm
import numpy as np
import logging

from functools import partial, partialmethod

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


def compute_senders_and_receivers(
        positions, cutoff: float
):
    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = jnp.linalg.norm(displacements, axis=-1)
    mask = ~jnp.eye(num_atoms, dtype=jnp.bool_) # get rid of self interactions
    keep_edges = jnp.where((distances < cutoff) & mask)
    senders = keep_edges[0].astype(jnp.int32)
    receivers = keep_edges[1].astype(jnp.int32)
    return senders, receivers


def compute_senders_and_receivers_np(
    positions, cutoff: float
):
    """Computes an edge list from atom positions and a fixed cutoff radius."""
    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(displacements, axis=-1)
    mask = ~np.eye(num_atoms, dtype=np.bool_) # get rid of self interactions
    keep_edges = np.where((distances < cutoff) & mask)
    senders = keep_edges[0].astype(np.int32)
    receivers = keep_edges[1].astype(np.int32)
    return senders, receivers


@dataclass
class SpiceDataLoaderSparse:
    input_file: str

    def cardinality(self):
        data = h5py.File(self.input_file)
        n = 0
        for i in data:
            n += len(data[i]['conformations'])
        return n

    def load(self, cutoff: float, pick_idx: np.ndarray = None):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        raise NotImplementedError(
            'Loading SPICE dataset is not supported at the moment.'
        )

        data = h5py.File(self.input_file)

        max_num_of_nodes = 0
        max_num_of_edges = 0
        logging.mlff(
            f"Load data from {self.input_file} and calculate neighbors within cutoff={cutoff} Ang ..."
        )
        loaded_data = []
        i = 0
        for k in tqdm(data):
            conformations = data[k]['conformations']
            atomic_numbers = data[k]['atomic_numbers']
            forces = data[k]['dft_total_gradient']
            energy = data[k]['dft_total_energy']
            for n in range(len(conformations)):
                if keep(i):
                    senders, receivers = compute_senders_and_receivers_np(conformations[n], cutoff=cutoff)

                    g = jraph.GraphsTuple(
                        n_node=np.array([len(atomic_numbers)]),
                        n_edge=np.array([len(receivers)]),
                        globals=dict(energy=np.array(energy[n]).reshape(-1)),
                        nodes=dict(
                            atomic_numbers=np.array(atomic_numbers).astype(np.int16),
                            positions=np.array(conformations[n]),
                            forces=np.array(forces[n])
                        ),
                        edges=dict(cell=None, cell_offsets=None),
                        receivers=np.array(senders),  # opposite convention in mlff
                        senders=np.array(receivers)
                    )
                    lr = jraph.GraphsTuple(
                        nodes=None, edges=None,
                        senders=np.array([], dtype=np.int64),
                        receivers=np.array([], dtype=np.int64),
                        n_node=np.array([len(atomic_numbers)]),
                        n_edge=np.array([0]),
                        globals=None,
                    )
                    loaded_data += [(g, lr)]
                    num_nodes = len(g.nodes['atomic_numbers'])
                    num_edges = len(g.receivers)
                    max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                    max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges
                else:
                    pass
                i += 1

        if pick_idx is not None:
            if max(pick_idx) >= i:
                raise RuntimeError(
                    f'`max(pick_idx) = {max(pick_idx)} >= cardinality = {i} of the dataset`.'
                )

        logging.mlff("... done!")

        return loaded_data, {'max_num_of_nodes': max_num_of_nodes, 'max_num_of_edges': max_num_of_edges}
