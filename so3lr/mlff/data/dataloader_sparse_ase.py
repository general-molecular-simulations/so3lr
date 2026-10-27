from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import iread
from ase.io.formats import ioformats
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import numpy as np
import jraph
import os
from pathlib import Path

from ..utils.param_utils import MAX_NUM_THEORY_LEVELS

import logging

logger = logging.getLogger("SO3LR")


def compute_senders_and_receivers_np(
    positions, cutoff: float
):
    """Computes an edge list from atom positions and a fixed cutoff radius."""
    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(displacements, axis=-1)
    distances_min = distances[distances > 0].min()
    mask = ~np.eye(num_atoms, dtype=np.bool_)  # get rid of self interactions
    keep_edges = np.where((distances < cutoff) & mask)
    senders = keep_edges[0].astype(np.int32)
    receivers = keep_edges[1].astype(np.int32)
    return senders, receivers, distances_min


# Calculate residue information for dimer calculations
def selection_indices(selection):
    """0-based atom indices of a 1-based selection like '1-4' or '5' (empty if blank)."""
    start, _, end = str(selection).partition('-')
    if not start:
        return np.array([], dtype=int)
    return np.arange(int(start) - 1, int(end or start))
    

@dataclass
class AseDataLoaderSparse:
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
            file_list = []
            for entry in os.scandir(input_folder):
                if Path(entry.path).suffix[1:] in set(ioformats.keys()):
                    file_list.append(Path(entry.path).expanduser().resolve())
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
            atoms = ['' for _ in iread(file)]
            total_atoms += len(atoms)
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
            # Build the lookup once. `idx in pick_idx` on an ndarray is a full
            # vectorised scan per structure, i.e. O(N*M) over the dataset.
            pick_idx_set = set(np.asarray(pick_idx).tolist())

            def keep(idx: int):
                return idx in pick_idx_set

        loaded_data = []
        max_num_of_nodes = 0
        max_num_of_edges = 0
        max_num_of_pairs = 0

        file_list = self.make_file_list()

        logger.info(f"Loading data from file_list: {file_list} ...")

        i = 0
        for file_path in file_list:
            logger.info(
                f"Load data from {file_path}."
            )
            logger.info(
                f"Calculate short range neighbors within cutoff={cutoff} Ang."
            )
            if calculate_neighbors_lr:
                logger.info(
                    f"Calculate long-range neighbors within cutoff_lr={cutoff_lr} Ang."
                )
            for a in tqdm(iread(file_path), mininterval=60, maxinterval=600):
                if keep(i):
                    result = ASE_to_jraph(
                        a,
                        cutoff=cutoff,
                        calculate_neighbors_lr=calculate_neighbors_lr,
                        cutoff_lr=cutoff_lr
                    )

                    loaded_data.append(result)

                    if result is not None:
                        graph, lr_data = result
                        num_nodes = len(graph.nodes['atomic_numbers'])
                        num_edges = len(graph.receivers)
                        num_pairs = int(lr_data['n_pairs'][0])
                        max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                        max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges
                        max_num_of_pairs = max_num_of_pairs if num_pairs <= max_num_of_pairs else num_pairs
                else:
                    pass
                i += 1

        if pick_idx is not None:
            if max(pick_idx) >= i:
                raise RuntimeError(
                    f'`max(pick_idx) = {max(pick_idx)} >= cardinality = {i} of the dataset`.'
                )
        logger.info("... done!")
        logger.info(
            f"Data stats: max_num_of_nodes={max_num_of_nodes}, "
            f"max_num_of_edges={max_num_of_edges}, "
            f"max_num_of_pairs={max_num_of_pairs}"
        )

        return loaded_data, {'max_num_of_nodes': max_num_of_nodes, 'max_num_of_edges': max_num_of_edges, 'max_num_of_pairs': max_num_of_pairs}


def charge_and_multiplicity(mol: Atoms):
    """The total charge and spin multiplicity in ``mol.info``, None where absent.

    Spin multiplicity (2S+1): OMol25/OPoly26 extxyz store it under `spin=`, other
    datasets use `multiplicity=`. Prefer the canonical `multiplicity`, falling back
    to `spin` so open-shell structures are not silently read as closed-shell
    singlets (num_unpaired_electrons = multiplicity - 1).
    """
    multiplicity = mol.info.get('multiplicity')
    if multiplicity is None:
        multiplicity = mol.info.get('spin')
    return mol.info.get('charge'), multiplicity


def ASE_to_jraph(
        mol: Atoms,
        cutoff: float,
        self_interaction: bool = False,
        calculate_neighbors_lr: bool = True,
        cutoff_lr: float = Optional[None]
):
    """Convert an ASE Atoms object to a jraph.GraphTuple object.

    Args:
        mol (Atoms): ASE Atoms object.
        cutoff (float): Cutoff radius for neighbor interactions.
        self_interaction (bool): Include self-interaction in neighbor list.
        calculate_neighbors_lr (bool): Calculate long-range neighborhood.
        cutoff_lr (float): Cutoff for the long-range neighborhood.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object if filter != True else None.
    """

    atomic_numbers = np.array(mol.get_atomic_numbers(), dtype=np.int64)
    positions = np.array(mol.get_positions())

    num_atoms = len(atomic_numbers)

    if mol.calc is not None:
        try:
            energy = np.array(mol.get_potential_energy()).reshape(-1)
        except PropertyNotImplementedError:
            energy = None
        try:
            forces = np.array(mol.get_forces())
        except PropertyNotImplementedError:
            forces = None
        try:
            stress = np.array(mol.get_stress())
        except PropertyNotImplementedError:
            stress = None
        try:
            dipole = np.array(mol.get_dipole_moment())
        except PropertyNotImplementedError:
            dipole = None
    else:
        energy = None
        forces = None
        stress = None
        dipole = None

    # Forces are NaN when not present.
    if forces is None:
        forces = np.empty((num_atoms, 3))
        forces[:] = np.nan

    # Energy is NaN when not present.
    if energy is None:
        energy = np.array([np.nan])

    max_num_theory_levels = MAX_NUM_THEORY_LEVELS
    theory_level = np.reshape(5, (1,))
    theory_mask = np.eye(max_num_theory_levels)[theory_level] # (1, num_theory_levels)

    # Dipoles are NaN when not present.
    if dipole is None:
        dipole = np.empty((3, ))
        dipole[:] = np.nan

    # Stress is NaN when not present.
    if stress is None:
        stress = np.empty((6, ))
        stress[:] = np.nan

    # Read additional properties from .info and .arrays in Atoms object.
    total_charge, multiplicity = charge_and_multiplicity(mol)
    # The polarizability/volume-ratio label has been stored under several keys
    # over time. Accept any of them and pick the first that exists, so users
    # don't silently get NaN labels when their extxyz uses the canonical
    # 'hirshfeld_ratios' name or the new-model 'a0_ratios' name. Aliases
    # checked, in order of preference:
    #   'hirshfeld_ratios' (canonical) -> 'hirsh_ratios' (legacy short form)
    #   -> 'a0_ratios' (so3lr-tauri-s/m/l output name).
    hirshfeld_ratios = None
    for _ratio_key in ('hirshfeld_ratios', 'hirsh_ratios', 'a0_ratios'):
        if _ratio_key in mol.arrays:
            hirshfeld_ratios = mol.arrays.get(_ratio_key)
            break

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
        hirshfeld_ratios = np.empty((num_atoms, ))
        hirshfeld_ratios[:] = np.nan
    else:
        hirshfeld_ratios = np.array(hirshfeld_ratios).reshape(num_atoms, )

    # C6 ratios (MBD-NL); set to NaN when not present.
    c6_ratios = mol.arrays.get('c6_ratios')
    if c6_ratios is None:
        c6_ratios = np.empty((num_atoms, ))
        c6_ratios[:] = np.nan
    else:
        c6_ratios = np.array(c6_ratios).reshape(num_atoms, )

    # Initialize monomer-specific information for dimer binding energy calculations,
    # as described in the Methods section of SO3LR paper.
    residue_charge = None
    residue_segments = None
    # Read dimer-specific properties for dimer calculations
    structure_type = mol.info.get('structure_type', None)
    charge_a = mol.info.get('charge_a', None)
    charge_b = mol.info.get('charge_b', None)
    selection_a = mol.info.get('selection_a', None)
    selection_b = mol.info.get('selection_b', None)
    # Only create residue info if all dimer properties are present
    if all(x is not None for x in [charge_a, charge_b, selection_a, selection_b]):
        idx_a, idx_b = selection_indices(selection_a), selection_indices(selection_b)
        # Validate that the selections split all atoms into two non-empty monomers
        both = np.sort(np.concatenate([idx_a, idx_b]))
        if len(idx_a) and len(idx_b) and np.array_equal(both, np.arange(num_atoms)):
            residue_segments = np.zeros(num_atoms, np.int32)
            if structure_type == 'dimer_translated':
                # Separated monomers: each keeps its own charge.
                residue_segments[idx_b] = 1
                residue_charge = np.array([[int(charge_a), int(charge_b)]])
            else:
                # The dimer itself (prepare_dimer_xyz writes both into one file): one residue with
                # the total charge, so both kinds batch together with the same fields.
                residue_charge = np.array([[int(total_charge[0]), 0]])

    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=self_interaction)
        cell = np.array(mol.get_cell())
        edge_features = {
            "cell_offset": np.array(S)
        }
        senders = np.array(j)
        receivers = np.array(i)
    else:
        cell = None
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
        idx_i_lr = np.array([], dtype=np.int32)
        idx_j_lr = np.array([], dtype=np.int32)
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
        "theory_mask": theory_mask, #.reshape(1, max_num_theory_levels), TODO: double check if I need to add .reshape...
        "stress": stress.reshape(1, 6),
        "dipole_vec": dipole.reshape(1, 3),
        "total_charge": total_charge.reshape(-1),
        "num_unpaired_electrons": multiplicity.reshape(-1) - 1,
        # ASE already used these axis-specific flags to construct the neighbor
        # offsets. Preserve them so reciprocal-space callers can reject mixed
        # PBC rather than silently promoting it to three-dimensional PBC.
        "pbc": np.asarray(mol.get_pbc(), dtype=bool).reshape(1, 3),
    }

    if cell is not None:
        global_context["cell"] = np.array(cell).reshape(1, 3, 3)

    if residue_charge is not None:
        global_context["residue_charge"] = residue_charge

    # Edges follow a similar convention where e.g. for positions and forces one has (num_nodes, 3) and for scalars
    # like hirshfeld volumes (num_nodes, ).
    node_features = {
            "positions": positions.reshape(num_atoms, 3),
            "atomic_numbers": atomic_numbers.reshape(num_atoms),
            "forces": forces.reshape(num_atoms, 3),
            "hirshfeld_ratios": hirshfeld_ratios.reshape(num_atoms),
            "c6_ratios": c6_ratios.reshape(num_atoms)
            }
    if residue_segments is not None:
        node_features["residue_segments"] = residue_segments

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context,
    )
    lr_data = {
        'idx_i_lr': idx_i_lr,
        'idx_j_lr': idx_j_lr,
        'cell_offsets_lr': cell_offsets_lr,
        'n_pairs': n_pairs,
    }
    return graph, lr_data
