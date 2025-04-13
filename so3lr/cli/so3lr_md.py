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
import logging
import pathlib
from pathlib import Path
from functools import partial
from logging.handlers import RotatingFileHandler
from typing import Dict, Tuple, Union, List, Optional, Any, Callable

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
from mlff.mdx.potential import MLFFPotentialSparse
from mlff.mdx.hdfdict import DataSetEntry, HDF5Store

from so3lr.graph import Graph
from so3lr import So3lrPotential

# Setup logging
logger = logging.getLogger("SO3LR")


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
        P: Target pressure in atmospheres. Defaults to None.

    Returns:
        Dictionary containing the timestep, temperature and pressure
        in the correct units for JAX MD.
    """
    try:
        unit = unit_system()
        dt *= unit['time']

        T *= unit['temperature']

        if P is not None:
            P *= unit['pressure']

        return {
            'dt': dt,
            'T': T,
            'P': P
        }
    except Exception as e:
        raise ValueError(f"Error converting units: {str(e)}.")


def handle_box(
    shift_displacement: str,
    positions: jnp.ndarray,
    cell: Optional[jnp.ndarray] = None
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

        box = jnp.array(np.diag(np.array(cell)))
        fractional_coordinates = True

        # Create displacement and shift functions for periodic boundary conditions
        displacement, shift = jax_md.space.periodic_general(
            box=box,
            fractional_coordinates=fractional_coordinates
        )

        # Convert positions to fractional coordinates
        inv_box = 1/box
        positions = raw_transform(inv_box, positions)

    elif shift_displacement == 'free':
        # Create displacement and shift functions for free boundary conditions
        displacement, shift = jax_md.space.free()
        box = jnp.array(0.0)
        fractional_coordinates = False

    else:
        raise ValueError(
            f"Unsupported boundary condition: '{shift_displacement}'. "
            "Only 'free' or 'periodic' boundary conditions are supported."
        )

    return positions, box, displacement, shift, fractional_coordinates


def init_hdf5_store(
    save_to: Union[str, Path],
    batch_size: int,
    num_atoms: int,
    num_box_entries: int,
    exist_ok: bool = False
) -> HDF5Store:
    """
    Initialize an HDF5 storage object for trajectory data.

    Args:
        save_to: Path to save the HDF5 file.
        batch_size: Batch size for the storage.
        num_atoms: Number of atoms in the system.
        num_box_entries: Number of entries in the box vector (typically 3).
        exist_ok: Whether to overwrite the file if it exists.

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

    # Define dataset structure
    dataset = {
        'positions': DataSetEntry(
            chunk_length=1,
            shape=(batch_size, num_atoms, 3),
            dtype=np.float32
        ),
        'momenta': DataSetEntry(
            chunk_length=1,
            shape=(batch_size, num_atoms, 3),
            dtype=np.float32
        ),
        'box': DataSetEntry(
            chunk_length=1,
            shape=(batch_size, num_box_entries),
            dtype=np.float32
        )
    }

    return HDF5Store(save_path, datasets=dataset, mode='w')


def write_to_hdf5(
    hdf5_store: HDF5Store,
    momenta: Optional[List[jnp.ndarray]],
    positions: List[jnp.ndarray],
    boxes: Union[float, jnp.ndarray, List[jnp.ndarray], None],
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Write trajectory data to an HDF5 file.

    Writes the accumulated trajectory data to HDF5 storage.

    Args:
        hdf5_store: HDF5 storage object.
        momenta: List of momenta arrays for each frame, or None.
        positions: List of position arrays for each frame.
        boxes: Box vectors for each frame. Can be float, array, list of arrays, or None.

    Returns:
        Tuple of empty lists for momenta, positions, and boxes.
    """
    try:
        step_data = {
            'positions': jnp.stack(positions, axis=0),
        }

        if boxes[0] != 0:
            step_data['box'] = jnp.stack(boxes, axis=0)
    
        if momenta is not None:
            step_data['momenta'] = jnp.stack(momenta, axis=0)
        
        step_data = jax.tree.map(
            lambda u: np.asarray(u), step_data
            )
        
        hdf5_store.append(step_data)
        
        return [], [], []

    except Exception as e:
        logger.error(f"Failed to write to HDF5 file: {str(e)}")
        sys.exit(1)


def write_to_extxyz(
    output_file: Union[str, Path],
    atoms: ase.Atoms,
    boxes: Union[float, jnp.ndarray, List, None],
    momenta: List,
    positions: List
) -> Tuple[List, List, List]:
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

    Returns:
        Tuple of empty lists (momenta, positions, boxes) to clear memory on success.
        On failure, returns the original lists.
    """
    try:
        # Ensure output_file is a Path object
        output_path = Path(output_file) if isinstance(
            output_file, str) else output_file

        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if we have data to write
        if not positions:
            logger.info('No positions to write to extxyz file')
            return momenta, positions, boxes

        # Convert positions to numpy array
        positions = np.asarray(positions)

        # Create a list of atoms objects for each frame
        atoms_list = []
        for i in range(len(positions)):
            atoms_copy = atoms.copy()

            # TODO: replace with raw_transform(box, positions)
            # Handle different box types and set positions accordingly
            if isinstance(boxes, float):
                if boxes == 0:
                    atoms_copy.set_positions(positions[i])
                else:
                    atoms_copy.set_cell(boxes)
                    atoms_copy.set_positions(positions[i] * boxes)
            elif isinstance(boxes, (jnp.ndarray, np.ndarray)):
                if np.any(boxes == 0):
                    atoms_copy.set_positions(positions[i])
                else:
                    atoms_copy.set_cell(boxes)
                    atoms_copy.set_positions(positions[i] * boxes)
            elif isinstance(boxes, list):
                if np.any(boxes[0] == 0):
                    atoms_copy.set_positions(positions[i])
                else:
                    atoms_copy.set_cell(boxes[i])
                    atoms_copy.set_positions(positions[i] * boxes[i])
            else:
                atoms_copy.set_positions(positions[i])

            # Set momenta if available
            if momenta and i < len(momenta):
                atoms_copy.set_momenta(momenta[i])

            atoms_list.append(atoms_copy)

        # Write all atoms to file at once
        file_exists = output_path.exists()
        ase.io.write(output_path, atoms_list,
                     format='extxyz', append=file_exists)

        # Clear lists to free memory
        return [], [], []

    except Exception as e:
        logger.warning(f"Failed to write to extxyz file: {str(e)}")
        # Return the original lists in case of failure
        return [], [], []


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

    return {
        'positions': positions,
        'species': species,
        'masses': masses
    }

def check_cell(cell: np.ndarray, lr_cutoff: float) -> np.ndarray:
    """
    Validate the provided cell matrix for compatibility with JAX-MD requirements.

    Args:
        cell (np.ndarray): The cell matrix to be validated.
        lr_cutoff (float): The long-range cutoff distance.

    Returns:
        np.ndarray: The validated cell matrix if all checks pass, or None if the cell is zero.
    """
    if np.all(cell == 0):
        return None
    
    # Create a mask with True only on the diagonal
    mask = np.eye(cell.shape[0], dtype=bool)
    if np.any(cell[~mask] != 0):
        raise ValueError(f'JAX-MD currently supports only orthogonal cells. Provided cell: {cell}') 

    if np.any(np.diag(cell) < lr_cutoff * 2):
        raise ValueError(f'Each dimension of the cell {np.diag(cell)} must be at least twice the long-range cutoff distance [{lr_cutoff} Å].')
        
    if lr_cutoff < 10:
        raise ValueError(f'Long-range cutoff below 10 Å is not supported yet. Provided cutoff: {lr_cutoff} Å.')

    return cell

def load_model(
    model_path: str,
    precision: jnp.dtype,
    lr_cutoff: float = 12.0,
    dispersion_damping: float = 2.
) -> MLFFPotentialSparse:
    """
    Load a trained MLFF model from a checkpoint directory.

    Args:
        model_path (str): Path to the model checkpoint directory.
        precision (jnp.dtype): Precision to use for the calculation.
        lr_cutoff (float, optional): Long-range cutoff for SO3LR in Å. Defaults to 12.0.
        dispersion_damping (float, optional): Cutoff for dispersion
                                energy damping in Å. Defaults to 2.0.

    Returns:
        MLFFPotentialSparse: The loaded MLFF model.
    """
    logger.info(f'Loading model from {model_path}')

    # Verify model path exists
    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Load from ckpt directory path
    potential = MLFFPotentialSparse.create_from_ckpt_dir(
        model_dir,
        from_file=True,
        long_range_kwargs=dict(
            cutoff_lr=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping,
            neighborlist_format_lr='ordered_sparse'
        ),
        dtype=precision
    )

    return potential


def neighbor_list_featurizer_custom(displacement_fn, species, total_charge=0., precision=jnp.float32):
    """
    Create a function that builds a graph from positions and neighbors.

    Args:
        displacement_fn: Function to compute displacements.
        species: Array of atomic species.
        total_charge: Total charge of the system.
        precision: Precision to use for calculations.

    Returns:
        Function that creates a graph from positions and neighbors.
    """
    def featurize(R, neighbor, neighbor_lr, **kwargs):
        idx_i = neighbor[0]  # shape: P
        idx_j = neighbor[1]  # shape: P
        idx_i_lr = neighbor_lr[0]  # shape: P
        idx_j_lr = neighbor_lr[1]  # shape: P

        Ra = R[idx_i]
        Rb = R[idx_j]
        Ra_lr = R[idx_i_lr]
        Rb_lr = R[idx_j_lr]

        box = kwargs.get('box', None)

        d = jax.vmap(partial(displacement_fn, **kwargs))
        dR = d(Ra, Rb)
        dR_lr = d(Ra_lr, Rb_lr)

        graph = Graph(
            positions=None,
            nodes=species,
            edges=dR,
            centers=idx_i,
            others=idx_j,
            mask=None,
            total_charge=jnp.array([total_charge], dtype=precision),
            num_unpaired_electrons=jnp.array([0.]),
            edges_lr=dR_lr,
            idx_i_lr=idx_i_lr,
            idx_j_lr=idx_j_lr,
            cell=box  # Will be None for free boundary conditions
        )

        return graph

    return featurize


def to_jax_md_custom(
        potential,  # the mlff potential
        species: jnp.ndarray,
        displacement_or_metric: DisplacementOrMetricFn,
        box,  # box if it exists, check jax_md documentation for conventions
        dr_threshold: float = 0.,  # currently dr_threshold > 0 is experimental
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        minimum_cell_size_multiplier_sr: float = 1.0,
        minimum_cell_size_multiplier_lr: float = 1.0,
        disable_cell_list: bool = False,
        fractional_coordinates: bool = True,
        total_charge: float = 0.,
        precision: jnp.dtype = jnp.float32,
        **neighbor_kwargs
):
    """
    Create neighbor functions and energy function for JAX MD.

    Args:
        potential: The MLFF potential object.
        species: Array of atomic species.
        displacement_or_metric: Function to compute displacements.
        box: Box definition for periodic systems.
        dr_threshold: Threshold for neighborhood cutoff.
        capacity_multiplier: Capacity multiplier for neighbor lists.
        buffer_size_multiplier_sr: Buffer size multiplier for short-range interactions.
        buffer_size_multiplier_lr: Buffer size multiplier for long-range interactions.
        minimum_cell_size_multiplier_sr: Minimum cell size multiplier for short-range.
        minimum_cell_size_multiplier_lr: Minimum cell size multiplier for long-range.
        disable_cell_list: Whether to disable cell lists.
        fractional_coordinates: Whether to use fractional coordinates.
        total_charge: Total charge of the system.
        precision: Precision to use for calculations.
        **neighbor_kwargs: Additional keyword arguments for neighbor functions.

    Returns:
        Tuple of neighbor function, long-range neighbor function, and energy function.
    """
    # Create the neighbor_fn
    neighbor_fn = partition.neighbor_list(
        displacement_or_metric,
        box,
        potential.cutoff,  # load the cutoff of the model from the MLFFPotential
        dr_threshold,
        capacity_multiplier,
        buffer_size_multiplier_sr,  # as buffer_size_multiplier
        minimum_cell_size_multiplier_sr,
        fractional_coordinates=fractional_coordinates,
        # only sparse is supported in mlff
        format=partition.NeighborListFormat(1),
        disable_cell_list=disable_cell_list,
        **neighbor_kwargs)

    # Create the neighbor_fn for long-range cutoff
    neighbor_fn_lr = partition.neighbor_list(
        displacement_or_metric,
        box,
        potential.long_range_cutoff,
        dr_threshold,
        capacity_multiplier,
        buffer_size_multiplier_lr,  # as buffer_size_multiplier
        minimum_cell_size_multiplier_lr,
        fractional_coordinates=fractional_coordinates,
        # long-range modules can handle OrderedSparse.
        format=partition.NeighborListFormat(2),
        disable_cell_list=disable_cell_list,
        **neighbor_kwargs)

    featurizer = neighbor_list_featurizer_custom(
        displacement_or_metric,
        species,
        total_charge,
        precision
    )

    # Create an energy_fn that is compatible with jax_md
    def energy_fn(
            R,
            neighbor,
            neighbor_lr,
            **energy_fn_kwargs
    ):
        graph = featurizer(R, neighbor, neighbor_lr, **energy_fn_kwargs)
        return potential(graph).sum()

    return neighbor_fn, neighbor_fn_lr, energy_fn


def process_model(
    potential: MLFFPotentialSparse,
    species: jnp.ndarray,
    displacement: DisplacementOrMetricFn,
    box: jnp.ndarray,
    total_charge: float = 0.,
    buffer_size_multiplier_lr: float = 1.25,
    buffer_size_multiplier_sr: float = 1.25,
    precision: jnp.dtype = jnp.float32,
    fractional_coordinates: bool = False
) -> Tuple[callable, callable, callable]:
    """
    Process the model and create the neighbor functions and energy function.

    Args:
        potential (MLFFPotentialSparse): Force field model.
        species (jnp.ndarray): Atomic species.
        displacement (DisplacementOrMetricFn): Displacement function.
        box (jnp.ndarray): Box of the system.
        total_charge (float, optional): Total charge of the system. Defaults to 0..
        buffer_size_multiplier_lr (float, optional): Buffer for short range
                                            neighborlist. Defaults to 1.25.
        buffer_size_multiplier_sr (float, optional): Buffer for long range
                                            neighborlist. Defaults to 1.25.
        precision (jnp.dtype, optional): Floating point precision.
                                            Defaults to jnp.float32.
        fractional_coordinates (bool, optional): Whether to use fractional 
                    coordinates or not (needs to be True for periodic systems). 
                                            Defaults to False.

    Returns:
        Tuple[callable, callable, callable]: Neighbor functions and energy function.
    """

    neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md_custom(
        potential=potential,
        species=species,
        displacement_or_metric=displacement,
        box=box,
        disable_cell_list=False if fractional_coordinates else True,
        total_charge=total_charge,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
    )
    energy_fn = jax.jit(energy_fn)

    return neighbor_fn, neighbor_fn_lr, energy_fn


def check_overflow(
    neighbor_fn: callable,
    neighbor_fn_lr: callable,
    nbrs: jax_md.partition.NeighborList,
    nbrs_lr: jax_md.partition.NeighborList,
    state,
    box: jnp.ndarray,
) -> Tuple[jax_md.partition.NeighborList, jax_md.partition.NeighborList, bool]:
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

    overflown = False
    if nbrs.did_buffer_overflow:
        overflown = True
        logger.info('Neighbor list overflowed, reallocating.')
        nbrs = neighbor_fn.allocate(
            state.position,
            box=box
        )
    if nbrs_lr is not None:
        if nbrs_lr.did_buffer_overflow:
            overflown = True
            logger.info('Long-range neighbor list overflowed, reallocating.')
            nbrs_lr = neighbor_fn_lr.allocate(
                state.position,
                box=box
            )
    return nbrs, nbrs_lr, overflown


def compute_quantities(
    energy_fn: callable,
    state,
    nbrs: jax_md.partition.NeighborList,
    nbrs_lr: jax_md.partition.NeighborList,
    box: jnp.ndarray,
    unit: dict,
    T: float,
    P: float = None
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
            box=box
        )

        if P is None:
            H = jax_md.simulate.nvt_nose_hoover_invariant(
                energy_fn,
                state,
                kT=T,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                box=box
            ) / unit['energy']
        else:
            H = jax_md.simulate.npt_nose_hoover_invariant(
                energy_fn,
                state,
                pressure=P,
                kT=T,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx
            )

    else:
        PE = energy_fn(
            state.position,
            neighbor=nbrs.idx,
            box=box
        )
        if P is None:
            H = jax_md.simulate.nvt_nose_hoover_invariant(
                energy_fn,
                state,
                kT=T,
                neighbor=nbrs.idx,
                box=box
            ) / unit['energy']
        else:
            H = jax_md.simulate.npt_nose_hoover_invariant(
                energy_fn,
                state,
                pressure=P,
                kT=T,
                neighbor=nbrs.idx
            )

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
            #    kinetic_energy=KE
            # )
            current_P = 0.0
        else:
            # current_P = jax_md.quantity.pressure(
            #    energy_fn,
            #    state.position,
            #    box=box,
            #    neighbor=nbrs.idx,
            #    kinetic_energy=KE
            # )
            current_P = 0.0
    else:
        current_P = None

    return KE, PE, H, current_T, current_P


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
        box=box,
        thermostat_kwargs=nhc_kwargs
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
    lr
):
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

    return init_fn, step_md_fn


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
            state, nbrs, nbrs_lr, box = state
            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                kT=T,
                box=box
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
            return state, nbrs, nbrs_lr, box
        return step_nvt_fn_lr
    else:
        @jax.jit
        def step_nvt_fn(i: int, state):
            state, nbrs, box, = state
            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                kT=T,
                box=box
            )
            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )
            return state, nbrs, box
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
            state, nbrs, nbrs_lr, box, = state

            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                kT=T,
                pressure=P
            )

            box = jax_md.simulate.npt_box(state)

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )

            nbrs_lr = nbrs_lr.update(
                state.position,
                neighbor_lr=nbrs_lr.idx,
                box=box
            )

            return state, nbrs, nbrs_lr, box
        return step_npt_fn
    else:
        @jax.jit
        def step_npt_fn(i, state):
            state, nbrs, box, = state

            state = apply_fn(
                state,
                neighbor=nbrs.idx,
                kT=T,
                pressure=P
            )

            box = jax_md.simulate.npt_box(state)

            nbrs = nbrs.update(
                state.position,
                neighbor=nbrs.idx,
                box=box
            )

            return state, nbrs, box
        return step_npt_fn


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
    if ensemble == 'nvt':
        return create_nvt_step_fn(lr, apply_fn, T)
    elif ensemble == 'npt':
        return create_npt_step_fn(lr, apply_fn, T, P)
    else:
        raise NotImplementedError(
            f'Ensemble "{ensemble}" is not supported. Only NVT and NPT ensembles are currently implemented.')


def perform_md(
    all_settings: Dict,
    opt_structure: Optional[jnp.ndarray] = None,
    restart: Optional[bool] = False,
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

    # Model parameters
    model_path = all_settings.get('model_path')
    precision = all_settings.get('precision')
    lr_cutoff = all_settings.get('lr_cutoff')
    dispersion_damping = all_settings.get('dispersion_damping')
    buffer_size_multiplier_sr = all_settings.get('buffer_size_multiplier_sr')
    buffer_size_multiplier_lr = all_settings.get('buffer_size_multiplier_lr')
    total_charge = all_settings.get('total_charge')

    # MD parameters
    md_dt = all_settings.get('md_dt')
    md_T = all_settings.get('md_T')
    md_P = all_settings.get('md_P')
    md_cycles = all_settings.get('md_cycles')
    md_steps = all_settings.get('md_steps')

    # Thermostat parameters
    nhc_chain_length = all_settings.get('nhc_chain_length')
    nhc_steps = all_settings.get('nhc_steps')
    nhc_thermo = all_settings.get('nhc_thermo')
    nhc_baro = all_settings.get('nhc_baro')
    nhc_sy_steps = all_settings.get('nhc_sy_steps')

    # Format control
    output_format = all_settings.get('output_format')
    save_buffer = all_settings.get('save_buffer')
    seed = all_settings.get('seed')
    relax_before_run = all_settings.get('relax_before_run')

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

    if opt_structure is not None:
        initial_geometry.set_positions(opt_structure)

    cell = initial_geometry.get_cell()
    cell = check_cell(cell, lr_cutoff)        

    if md_P is not None and cell is None:
        raise ValueError(
            'Cell must be defined for NPT simulations. The input geometry does not contain cell information.')

    if cell is not None:
        shift_displacement = 'periodic'
        initial_geometry.wrap()
    else:
        shift_displacement = 'free'

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)

    (position, box, displacement, shift, fractional_coordinates) = handle_box(
        shift_displacement, initial_geometry_dict['positions'], cell)

    save_buffer = min(md_cycles, save_buffer)
    # Initialize HDF5 storage
    hdf5_store = None
    if output_format == 'hdf5':
        hdf5_store = init_hdf5_store(
            save_to=output_file,
            batch_size=save_buffer,
            num_atoms=position.shape[0],
            num_box_entries=1,
            exist_ok=True
        )

    current_cycle = 0
    # Loading restart
    if restart:
        logger.info(f'Loading state from {restart_load_path}.')
        try:
            state, box, current_cycle = load_state(
                path_to_load=restart_load_path,
                ensemble='nvt' if md_P is None else 'npt'
            )
            position = state.position
        except Exception as e:
            raise RuntimeError(f"Failed to load restart state from {restart_load_path}: {str(e)}")
        

    # Loading the model
    if model_path is None:
        # Load default SO3LR
        logger.info("Using default SO3LR potential.")
        potential = So3lrPotential(
            dtype=jnp.float64 if precision == 'float64' else jnp.float32,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping
        )
    else:
        # Verify model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Custom model path not found: {model_path}")
        logger.info(f"Using custom MLFF potential from: {model_path}")
        potential = load_model(
            model_path,
            precision=jnp.float64 if precision == 'float64' else jnp.float32,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping
        )

    # Setting up the model
    neighbor_fn, neighbor_fn_lr, energy_fn = process_model(
        potential=potential,
        species=initial_geometry_dict['species'],
        displacement=displacement,
        box=box,
        total_charge=total_charge,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr
    )

    # Allocating the neighbor lists
    if neighbor_fn_lr is not None:
        lr = True
    else:
        lr = False

    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box) if lr else None

    # Apply units
    unit_dict = handle_units(
        units.metal_unit_system,
        md_dt,
        md_T,
        md_P,
    )

    md_dt = unit_dict['dt']
    md_T = unit_dict['T']
    md_P = unit_dict['P']

    # Setting up the thermostat and/or barostat
    nhc_tau = md_dt * nhc_thermo
    nhc_kwargs = {
        'chain_length': nhc_chain_length,
        'chain_steps': nhc_steps,
        'sy_steps': nhc_sy_steps,
        'tau': nhc_tau
    }

    if md_P is not None:
        nhc_barostat_kwargs = nhc_kwargs.copy()
        nhc_barostat_kwargs['tau'] = md_dt * nhc_baro

    rng_key = jax.random.PRNGKey(seed)

    if md_P is None:
        init_fn, step_md_fn = create_nhc_fn(
            energy_fn,
            shift,
            md_dt,
            md_T,
            box,
            nhc_kwargs,
            lr
        )
    else:
        init_fn, step_md_fn = create_npt_nhc_fn(
            energy_fn,
            shift,
            md_dt,
            md_T,
            md_P,
            nhc_kwargs,
            nhc_barostat_kwargs,
            lr
        )

    if not restart:
        if lr:
            state = init_fn(
                rng_key,
                position,
                box=box,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx,
                kT=md_T,
                mass=initial_geometry_dict['masses']
            )
        else:
            state = init_fn(
                rng_key,
                position,
                box=box,
                neighbor=nbrs.idx,
                kT=md_T,
                mass=initial_geometry_dict['masses']
            )

    # Running the MD
    logger.info('Starting MD simulation...')
    logger.info('Step\tE [eV]\tKE\tPE\tH\tTemp [K]\ttime/step [s]')

    # Calculate some quantities for printing
    KE, PE, H, current_T, _ = compute_quantities(
        energy_fn,
        state,
        nbrs,
        nbrs_lr,
        box,
        units.metal_unit_system(),
        md_T,
        md_P
    )

    logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{H:.3f}\t{current_T:.1f}\t{0.0:.2e}')

    momenta, positions, boxes = [], [], []
    cycle_md = 0
    current_cycle = 0
    total_time_for_steps = 0
    first_loop = True

    while cycle_md < md_cycles:
        old_time = time.time()
        if lr:
            new_state, nbrs, nbrs_lr, new_box = jax.block_until_ready(
                jax.lax.fori_loop(
                    0,
                    md_steps,
                    step_md_fn,
                    (state, nbrs, nbrs_lr, box)
                )
            )
        else:
            new_state, nbrs, new_box = jax.block_until_ready(
                jax.lax.fori_loop(
                    0,
                    md_steps,
                    step_md_fn,
                    (state, nbrs, box)
                )
            )

        time_per_step = (time.time() - old_time) / md_steps

        # Do not count first `allocation` loop for time_per_step calculation
        if not first_loop:
            total_time_for_steps += time_per_step
        else:
            first_loop = False

        # Checking if the neighbor lists overflowed
        nbrs, nbrs_lr, overflow = check_overflow(
            neighbor_fn,
            neighbor_fn_lr,
            nbrs,
            nbrs_lr,
            new_state,
            new_box
        )
        if not overflow:
            cycle_md += 1
            current_cycle += 1

            state = new_state
            box = new_box

            # Calculate some quantities for printing
            KE, PE, H, current_T, _ = compute_quantities(
                energy_fn,
                state,
                nbrs,
                nbrs_lr,
                box,
                units.metal_unit_system(),
                md_T,
                md_P
            )

            logger.info(f'{current_cycle*md_steps}\t{KE+PE:.3f}\t{KE:.3f}\t{PE:.3f}\t{H:.3f}\t{current_T:.1f}\t{time_per_step:.2e}')

            positions.append(np.array(state.position))
            momenta.append(np.array(state.momentum))
            if box is not None:
                boxes.append(np.array(box))

            if ((len(positions) % save_buffer == 0 and len(positions) > 0) or (len(positions) == md_cycles)):
                # Saving the output
                if output_format == 'hdf5':
                    positions, momenta, boxes = write_to_hdf5(
                        hdf5_store,
                        momenta,
                        positions,
                        boxes,
                    )
                elif output_format == 'extxyz':
                    positions, momenta, boxes = write_to_extxyz(
                        output_file,
                        initial_geometry,  # Template atoms
                        boxes,
                        momenta,
                        positions,
                    )
                # Saving the state for restart
                if create_restart:
                    save_state(
                        state,
                        box,
                        current_cycle,
                        restart_save_path,
                        ensemble='nvt' if md_P is None else 'npt'
                    )

    logger.info('Results saved to: ' + output_file)
    average_time_per_step = total_time_for_steps / (cycle_md - 1)
    logger.info(f'Average time per step: {average_time_per_step:.2e} seconds')

    if output_format == 'extxyz':
        logger.info('Consider saving the trajectory in HDF5 format, which has no overhead for long runs (--output traj_name.hdf5).')

    if jax.default_backend() in ["gpu", "cuda", "rocm"]:
        if average_time_per_step > 1.2 * 3.25e-6 * len(initial_geometry):
            logger.warn('Ideally, the average time per step should be close to {:.2e} seconds (measured on an A100 GPU)'.format(
                3.25e-6 * len(initial_geometry)))
            logger.warn('Consider decreasing the buffer sizes if the system has equilibrated (--buffer-sr 1.15, --buffer-lr 1.1)')
            logger.warn('and/or increasing the number of steps in each jax.lax.fori_loop (--steps 1000)')
            if output_format == 'extxyz':
                logger.warn('and/or saving the trajectory in HDF5 format (--output traj_name.hdf5)')


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
                neighbor_lr=nbrs_lr.idx
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
) -> Union[jnp.ndarray, List[jnp.ndarray]]:
    """
    Perform energy minimization (geometry optimization) with the given settings.

    Args:
        all_settings (Dict): Settings for the minimization.

    Returns:
        Union[jnp.ndarray, List[jnp.ndarray]]: Optimized structure or trajectory.
    """
    # Extract settings with defaults
    input_file = all_settings.get('input_file')
    output_file = all_settings.get('output_file')
    output_format = all_settings.get('output_format')

    # Model parameters
    model_path = all_settings.get('model_path')
    precision = all_settings.get('precision')
    lr_cutoff = all_settings.get('lr_cutoff')
    dispersion_damping = all_settings.get('dispersion_damping')
    buffer_size_multiplier_sr = all_settings.get('buffer_size_multiplier_sr')
    buffer_size_multiplier_lr = all_settings.get('buffer_size_multiplier_lr')
    total_charge = all_settings.get('total_charge')
    
    # Settings for the minimization
    min_cycles = all_settings.get('min_cycles')
    min_steps = all_settings.get('min_steps')
    min_start_dt = all_settings.get('min_start_dt')
    min_max_dt = all_settings.get('min_max_dt')
    min_n_min = all_settings.get('min_n_min')
    force_convergence = all_settings.get('force_convergence')  # eV/A, can be None

    # Geometry, cell and charge
    input_file_path = all_settings.get('input_file')
    if input_file_path is None:
        raise ValueError('Initial geometry file path must be provided')
    try:
        initial_geometry = read(input_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find initial geometry file: {input_file_path}")
    cell = initial_geometry.get_cell()
    cell = check_cell(cell, lr_cutoff)        

    if cell is not None:
        shift_displacement = 'periodic'
        initial_geometry.wrap()
    else:
        shift_displacement = 'free'

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)

    (position, box, displacement, shift, fractional_coordinates) = handle_box(
        shift_displacement, initial_geometry_dict['positions'], cell)

    # Loading the model
    if model_path is None:
        # Load default SO3LR
        logger.info("Using default SO3LR potential.")
        potential = So3lrPotential(
            dtype=jnp.float64 if precision == 'float64' else jnp.float32,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping
        )
    else:
        # Load custom MLFF
        logger.info(f"Using custom MLFF potential from: {model_path}")
        potential = load_model(
            model_path,
            precision=jnp.float64 if precision == 'float64' else jnp.float32,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping
        )

    # Setting up the model
    neighbor_fn, neighbor_fn_lr, energy_fn = process_model(
        potential=potential,
        species=initial_geometry_dict['species'],
        displacement=displacement,
        box=box,
        total_charge=total_charge,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr
    )
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

    num_cycles = 0
    i = 1
    while num_cycles < min_cycles:
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
        if force_convergence is not None:
            if f_max > force_convergence:
                num_cycles -= 1
            else:
                logger.info(
                    f"Convergence criterion met at step {i*min_steps}: Fmax = {f_max:.3f} eV/Å")
                break

        num_cycles += 1
        i += 1

    # Save optimization result and return optimized geometry
    positions = minimization_trajectory
    if output_format == 'extxyz':
        write_to_extxyz(output_file, initial_geometry, boxes=box, momenta=None, positions=positions)
    elif output_format == 'hdf5':
        logger.warn(f"Output format is 'hdf5', changing output file extension to 'xyz' for minimization.") #TODO: Saving in xyz for simplicity for now, add hdf5 support
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
    ensemble: str = 'nvt'
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
    """

    state_dict = dict(
        position=state.position,
        momentum=state.momentum,
        force=state.force,
        mass=state.mass,
        # Additional necessary information
        box=box,
        step=cycle
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
        )
    np.savez(path_to_save, **state_dict)


def load_state(
    path_to_load: str,
    ensemble: str = 'nvt'
) -> Tuple:
    """

    Loads the state of the simulation from a .npz file and returns the state,
    the box and the cycle number.

    Args:
        path_to_load (str): Path to the .npz file containing the state of the
                            simulation.
        ensemble (str, optional): Specifies which ensemble is loaded. 
                                    Defaults to 'nvt'.

    Returns:
        Tuple: Tuple containing the state, the box and the cycle number.
    """

    loaded_state = np.load(path_to_load, allow_pickle=True)

    if ensemble.lower() == 'nvt':
        state = jax_md.simulate.NVTNoseHooverState(
            position=loaded_state['position'],
            momentum=loaded_state['momentum'],
            force=loaded_state['force'],
            mass=loaded_state['mass'],
            chain=jax_md.simulate.NoseHooverChain(
                position=loaded_state['thermostat_position'],
                momentum=loaded_state['thermostat_momentum'],
                mass=loaded_state['thermostat_mass'],
                tau=loaded_state['thermostat_tau'],
                kinetic_energy=loaded_state['thermostat_kinetic_energy'],
                degrees_of_freedom=loaded_state['thermostat_degrees_of_freedom']
            )
        )

    elif ensemble.lower() == 'npt':
        state = jax_md.simulate.NPTNoseHooverState(
            position=loaded_state['position'],
            momentum=loaded_state['momentum'],
            force=loaded_state['force'],
            mass=loaded_state['mass'],
            reference_box=loaded_state['reference_box'],
            box_position=loaded_state['box_position'],
            box_momentum=loaded_state['box_momentum'],
            box_mass=loaded_state['box_mass'],
            barostat=jax_md.simulate.NoseHooverChain(
                position=loaded_state['barostat_position'],
                momentum=loaded_state['barostat_momentum'],
                mass=loaded_state['barostat_mass'],
                tau=loaded_state['barostat_tau'],
                kinetic_energy=loaded_state['barostat_kinetic_energy'],
                degrees_of_freedom=loaded_state['barostat_degrees_of_freedom']
            ),
            thermostat=jax_md.simulate.NoseHooverChain(
                position=loaded_state['thermostat_position'],
                momentum=loaded_state['thermostat_momentum'],
                mass=loaded_state['thermostat_mass'],
                tau=loaded_state['thermostat_tau'],
                kinetic_energy=loaded_state['thermostat_kinetic_energy'],
                degrees_of_freedom=loaded_state['thermostat_degrees_of_freedom']
            )
        )
    else:
        raise NotImplementedError('Only NVT and NPT ensembles are supported')

    box = None if loaded_state['box'].item() is None else loaded_state['box']
    cycle = loaded_state['step']

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

    # Set the precision
    if settings.get('precision', "float32").lower() == 'float64':
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

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
            opt_structure = perform_min(settings)
            logger.info("=" * 60)
    else:
        opt_structure = None

    # Perform MD with relaxed structure
    perform_md(settings, opt_structure, restart)
