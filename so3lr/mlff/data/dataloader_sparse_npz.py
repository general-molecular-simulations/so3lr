from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
import jraph
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import logging

from functools import partial, partialmethod

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


@dataclass
class NpzDataLoaderSparse:
    input_file: Optional[str] = None
    input_folder: Optional[str] = None

    def make_file_list(self):
        if self.input_folder is not None:
            if self.input_file is not None:
                raise ValueError(
                    f"Only input_folder or input_file can be specified. "
                    f"Received {self.input_folder=} and {self.input_file=}."
                )
            input_folder = Path(self.input_folder).expanduser().resolve()

            file_list = [
                Path(x.path).expanduser().resolve() for x in os.scandir(input_folder) if Path(x.path).suffix == '.npz'
            ]
        else:
            if self.input_file is None:
                raise ValueError(
                    f"Either input_folder or input_file must be set. "
                    f"Received {self.input_folder=} and {self.input_file=}."
                )
            input_file = Path(self.input_file).expanduser().resolve()
            file_list = [input_file]

        return file_list

    def cardinality(self):
        file_list = self.make_file_list()
        total_atoms = 0
        for file in file_list:
            total_atoms += len(np.load(file)['positions'])
        return total_atoms

    def load(
            self,
            cutoff: float,
            calculate_neighbors_lr: bool = False,
            cutoff_lr: Optional[float] = None,
            pick_idx: np.ndarray = None
    ):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        np_data = np.load(self.input_file)

        keys = list(np_data.keys())
        data_iterator = zip(*(np_data[k] for k in keys))

        loaded_data = []
        max_num_of_nodes = 0
        max_num_of_edges = 0

        logging.mlff(
            f"Load data from {self.input_file}."
        )
        logging.mlff(
            f"Calculate short range neighbors within cutoff={cutoff} Ang."
        )
        if calculate_neighbors_lr:
            logging.mlff(
                f"Calculate long-range neighbors within cutoff_lr={cutoff_lr} Ang."
            )
        i = 0
        for tup in tqdm(data_iterator):
            if keep(i):
                entry = {k: v for (k, v) in zip(keys, tup)}
                result = entry_to_jraph(
                    entry,
                    cutoff=cutoff,
                    calculate_neighbors_lr=calculate_neighbors_lr,
                    cutoff_lr=cutoff_lr
                )
                graph, _ = result
                num_nodes = len(graph.nodes['atomic_numbers'])
                num_edges = len(graph.receivers)
                max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges
                loaded_data.append(result)
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


def entry_to_jraph(
        entry,
        cutoff: float,
        self_interaction: bool = False,
        calculate_neighbors_lr: bool = False,
        cutoff_lr: Optional[float] = None,

) -> jraph.GraphsTuple:
    """Convert an entry to a jraph.GraphTuple object.

    Args:
        entry (Dict): Entry.
        cutoff (float): Cutoff radius for neighbor interactions.
        self_interaction (bool): Include self-interaction in neighbor list.
        calculate_neighbors_lr (bool): Calculate long-range neighborhood.
        cutoff_lr (float): Cutoff for the long-range neighborhood.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object.
    """

    atomic_numbers = np.array(entry['atomic_numbers'], dtype=np.int64)
    positions = entry['positions']

    total_charge = entry.get('total_charge')
    multiplicity = entry.get('multiplicity')

    forces = entry.get('forces')
    energy = entry.get('energy')
    stress = entry.get('stress')

    hirshfeld_ratios = entry.get('hirsh_ratios')
    dipole = entry.get('dipole_vec')

    cell = entry.get('cell')
    pbc = entry.get('pbc')

    num_atoms = len(atomic_numbers)

    # Forces are NaN when not present.
    if forces is None:
        forces = np.empty((num_atoms, 3))
        forces[:] = np.nan

    # Energy is NaN when not present.
    if energy is None:
        # Energy from ASE is only a scalar.
        energy = np.nan

    max_num_theory_levels = 16
    theory_level = np.reshape(1, (1,))
    theory_mask = np.eye(max_num_theory_levels)[theory_level] # (1, num_theory_levels)

    # Dipoles are NaN when not present.
    if dipole is None:
        dipole = np.empty((3,))
        dipole[:] = np.nan

    # Stress is NaN when not present.
    if stress is None:
        stress = np.empty((6,))
        stress[:] = np.nan

    # Total charges are assumed to be zero when not specified.
    if total_charge is None:
        total_charge = np.array(0, np.int16).reshape(-1)
    else:
        total_charge = np.array(total_charge, dtype=np.int16).reshape(-1)

    # Multiplicity is assumed to be one when not specified.
    if multiplicity is None:
        multiplicity = np.array(1, dtype=np.int16).reshape(-1)
    else:
        multiplicity = np.array(multiplicity, dtype=np.int16).reshape(-1)

    # Hirshfeld ratios are set to NaN when not present.
    if hirshfeld_ratios is None:
        hirshfeld_ratios = np.empty((num_atoms,))
        hirshfeld_ratios[:] = np.nan
    else:
        hirshfeld_ratios = np.array(hirshfeld_ratios).reshape(num_atoms, )

    mol = Atoms(positions=positions, numbers=atomic_numbers, cell=cell, pbc=pbc)
    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=self_interaction)
        edge_features = {
            "cell": np.repeat(np.array(cell)[None], repeats=len(S), axis=0),
            "cell_offset": np.array(S)
        }
    else:
        i, j = neighbor_list('ij', mol, cutoff, self_interaction=self_interaction)
        edge_features = dict()  # No edge features.

    senders = np.array(j)
    receivers = np.array(i)

    if calculate_neighbors_lr:
        if cutoff_lr is None:
            raise ValueError(
                f'cutoff_lr must be specified for {calculate_neighbors_lr=}. Received {cutoff_lr=}.'
            )
        if mol.get_pbc().any():
            idx_i_lr, idx_j_lr, S_lr = neighbor_list('ijS', mol, cutoff_lr, self_interaction=self_interaction)
            idx_i_lr = np.array(idx_i_lr)
            idx_j_lr = np.array(idx_j_lr)
            cell_offsets_lr = np.array(S_lr)
        else:
            idx_i_lr, idx_j_lr = neighbor_list('ij', mol, cutoff_lr, self_interaction=self_interaction)
            idx_i_lr = np.array(idx_i_lr)
            idx_j_lr = np.array(idx_j_lr)
            cell_offsets_lr = None
    else:
        # No long-range indices are calculated.
        idx_i_lr = np.array([], dtype=int)
        idx_j_lr = np.array([], dtype=int)
        cell_offsets_lr = None

    n_node = np.array([num_atoms])

    n_edge = np.array([len(senders)])

    # Evaluates to zero if long-range edges are not calculated. When batching the graphs, this ensures that
    # n_pairs can not lead to overflow when no long-range indices are present.
    n_pairs = np.array([len(idx_i_lr)])

    # Add axis one for vector quantities such that batching yields e.g. (num_graphs, 6) for stress and (num_graphs, 3)
    # for dipole_vec. Otherwise batching via jraph yields (num_graphs*6, ) and (num_graphs*3, ). This is ultimately
    # convention that has to be taken into account when calculating the loss. Scalars like energy are represented
    # as (num_graphs, ).
    global_context = {
        "energy": energy.reshape(-1),
        "theory_level": theory_level.reshape(-1),
        "theory_mask": theory_mask, #.reshape(1, max_num_theory_levels),
        "stress": stress.reshape(1, 6),
        "dipole_vec": dipole.reshape(1, 3),
        "total_charge": total_charge.reshape(-1),
        "num_unpaired_electrons": multiplicity.reshape(-1) - 1,
    }

    # Edges follow a similar convention where e.g. for positions and forces one has (num_nodes, 3) and for scalars
    # like hirshfeld volumes (num_nodes, ).
    node_features = {
        "positions": positions.reshape(num_atoms, 3),
        "atomic_numbers": atomic_numbers.reshape(num_atoms),
        "forces": forces.reshape(num_atoms, 3),
        "hirshfeld_ratios": hirshfeld_ratios.reshape(num_atoms)
    }

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context,
    )
    # Build LR edge features - needs both cell and cell_offsets for PBC support
    if cell_offsets_lr is not None:
        lr_edge_features = {
            "cell": np.repeat(np.array(cell)[None], repeats=len(cell_offsets_lr), axis=0),
            "cell_offsets": cell_offsets_lr
        }
    else:
        lr_edge_features = None

    long_range = jraph.GraphsTuple(
        nodes=None,
        edges=lr_edge_features,
        senders=idx_j_lr,
        receivers=idx_i_lr,
        n_node=n_node,
        n_edge=n_pairs,
        globals=None,
    )
    return graph, long_range
