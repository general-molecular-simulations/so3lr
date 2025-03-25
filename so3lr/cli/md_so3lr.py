import jax
import jax.numpy as jnp
import jax_md
import jax_md.quantity
import numpy as np
from ase.io import read
from jax_md import units
from typing import Dict, Tuple
from mlff.mdx.potential import MLFFPotentialSparse
import ase
from functools import partial
import time
import os
import yaml
from jax_md import partition
from jax_md.space import Box, DisplacementOrMetricFn, raw_transform
from so3lr.graph import Graph
from mlff.mdx.hdfdict import DataSetEntry, HDF5Store
import pathlib
import logging
from so3lr import So3lrPotential
import click

HELP_STRING = """
# MD settings and default values.\n
\n
# === General settings ===\n
initial_geometry: "path/to/geometry.xyz"         # (str) Path to the initial geometry file\n
model_path: "path/to/model/"                     # (str) Path to the model directory\n
use_so3lr: True                                  # (bool) Whether to use SO3LR \n
precision: "float32"                             # (str) Numerical precision, e.g., 'float32' or 'float64'\n

# === Cutoffs and Buffers ===\n
lr_cutoff: 12.0                                  # (float) Long-range cutoff distance in Å\n
dispersion_energy_cutoff_lr_damping: 2.0         # (float) Damping factor for long-range dispersion interactions\n
buffer_size_multiplier_sr: 1.25                  # (float) Buffer size multiplier for short-range interactions\n
buffer_size_multiplier_lr: 1.25                  # (float) Buffer size multiplier for long-range interactions\n
hdf5_buffer_size: 5                              # (int) Number of frames to buffer before writing to the HDF5 file\n

# === File Paths ===\n
trajectory_hdf5_file: "trajectory.hdf5"          # (str) Output file for trajectory data in HDF5 format\n
restart_save_path: null                          # (str) Optional path to save restart data from a previous run\n
restart_load_path: null                          # (str) Optional path to load restart data from a previous run\n

# === Minimization settings ===\n
min_cycles: 10                                   # (int) Number of minimization cycles to perform\n
min_steps: 10                                    # (int) Number of steps per minimization cycle\n
min_start_dt: 0.05                               # (float) Initial timestep for minimization\n
min_max_dt: 0.1                                  # (float) Maximum timestep for minimization\n
min_n_min: 2                                     # (int) Number of minimizers to average during minimization\n

# === Molecular Dynamics (MD) settings ===\n
relax_before_run: False                          # (bool) Whether to perform a relaxation before the MD run\n
md_dt: 0.0005                                    # (float) MD timestep (in picoseconds)\n
md_T: 300.0                                      # (float) Simulation temperature (in Kelvin)\n
md_cycles: 100                                   # (int) Number of MD cycles to run\n
md_steps: 100                                    # (int) Number of steps per MD cycle\n

# === Nose-Hoover Chain (NVT) settings ===\n
nhc_chain_length: 3                              # (int) Length of the Nose-Hoover thermostat chain\n
nhc_steps: 2                                     # (int) Number of integration steps per MD step\n
nhc_thermo: 100                                  # (float) Thermostat timescale (in femtoseconds)\n
nhc_tau: null                                    # (float) Thermostat coupling constant (default: md_dt * nhc_thermo)\n

# === Nose-Hoover Chain (NPT) settings ===\n
md_P: null                                       # (float) Target pressure for the system (in atm). Toggles NPT run.\n
nhc_baro: 1000                                   # (float) Barostat timescale\n
nhc_sy_steps: 3                                  # (int) Number of Suzuki-Yoshida integration steps\n
nhc_npt_tau: null                                # (float) Barostat coupling constant (default: md_dt * nhc_baro)\n

# === Miscellaneous settings ===\n
total_charge: 0                                  # (int) Total charge of the system\n
seed: 0                                          # (int) Random seed for MD.\n


"""

def handle_units(
    unit_system: callable,
    dt: float,
    T: float = None,
    P: float = None
)-> dict:
    """
    Applies the units to the timestep, temperature and pressure.
    The unit_system is a function that returns a dictionary of the units
    with the keys 'time', 'temperature' and 'pressure'.

    Args:
        unit_system (callable): Function that returns a dictionary of units.
        dt (float): Timestep.
        T (float, optional): Target temperature. Defaults to None.
        P (float, optional): Target pressure. Defaults to None.

    Returns:
        dict: Dictionary containing the timestep, temperature and pressure
                in the correct units.
    """
    
    unit = unit_system()
    dt *= unit['time']
    if T is not None:
        T *= unit['temperature']
    if P is not None:
        P *= unit['pressure']
    
    return {
        'dt': dt,
        'T': T,
        'P': P
    }  

def handle_box(
    shift_displacement: str,
    positions: jnp.ndarray,
    cell: jnp.ndarray = None
)-> Tuple[jnp.ndarray, jnp.ndarray, callable, callable, bool]:
    """
    Handle the box and the displacement functions. 
    
    If the system is periodic, periodic boundary conditions are applied and the
    positions are divided by the box vector. Also fractional coordinates are
    then applied.

    Args:
        shift_displacement (str): Type of boundary conditions.
        positions (jnp.ndarray): Positions of the atoms.
        cell (jnp.ndarray, optional): Cell of the system. Defaults to None.

    Raises:
        NotImplementedError: If the shift_displacement is not 'free' or 'periodic'.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, callable, callable, bool]: 
            Positions, box, displacement function, shift function and
            whether fractional coordinates are used.
    """
    
    if shift_displacement == 'periodic':
        assert cell is not None, 'Cell must be defined for periodic boundary conditions'
        box = jnp.array(np.diag(np.array(cell)))
        fractional_coordinates = True
        displacement, shift = jax_md.space.periodic_general(
            box=box,
            fractional_coordinates=fractional_coordinates
            )
        # Convert positions to fractional coordinates
        inv_box = 1/box
        positions = raw_transform(inv_box, positions)
    
    elif shift_displacement == 'free':
        displacement, shift = jax_md.space.free()
        box = None
        fractional_coordinates = False
        
    else:
        raise NotImplementedError(
            "Only 'free' or 'perodic' boundary conditions are supported"
        )

    return positions, box, displacement, shift, fractional_coordinates

def init_hdf5_store(
    save_to: str, 
    batch_size: int, 
    num_atoms: int,
    num_box_entries: int,
    exist_ok: bool = False
)-> HDF5Store:
    """
    Initialize the HDF5 storage object.

    Args:
        save_to (str): Path to save the HDF5 file.
        batch_size (int): Batch size for the storage.
        num_atoms (int): Number of atoms in the system.
        num_box_entries (int): Number of entries in the box vector.
        exist_ok (bool, optional): Whether to overwrite the file if it exists.
                                    Defaults to False.

    Raises:
        RuntimeError: If the file exists and exist_ok is set to False.

    Returns:
        HDF5Store: _description_
    """
    parent_dir = pathlib.Path(save_to).expanduser().resolve().parent
    parent_dir.mkdir(exist_ok=True)
    
    _save_to = pathlib.Path(save_to)
    if _save_to.exists():
        if exist_ok is False:
            raise RuntimeError(
                f'File exists save_to={_save_to}. '
                f'Set exists_ok=True to override file.'
            )
    
    dataset = {
        'positions': DataSetEntry(
            chunk_length=1, 
            shape=(batch_size, num_atoms, 3), 
            dtype=np.float32
        ),
        'velocities': DataSetEntry(
            chunk_length=1, 
            shape=(batch_size, num_atoms, 3), 
            dtype=np.float32
        ),
        'box': DataSetEntry(
            chunk_length=1, 
            shape=(batch_size, 3), 
            dtype=np.float32
        )
    }

    return HDF5Store(_save_to, datasets=dataset, mode='w')

def write_to_hdf5(
    hdf5_store: HDF5Store,
    velocities: list,
    positions: list,
    boxs: list,
)-> Tuple[list, list, list]:
    """
    Write the positions, velocities and box to the hdf5 file.
    Returns empty lists for the positions, velocities and boxs in order
    to store the next batch of data.
    
    Args:
        hdf5_store (HDF5Store): HDF5 storage object.
        velocities (list): List of velocities.
        positions (list): List of positions.
        boxs (list): List of box vectors.

    Returns:
        Tuple[list, list, list]: Empty lists for the positions, velocities and boxs.
    """
    
    step_data = {
        'positions': jnp.stack(positions, axis=0),
        'velocities': jnp.stack(velocities, axis=0),
        'box': jnp.stack(boxs, axis=0) if len(boxs) > 0 else None
    }
    
    step_data = jax.tree.map(
        lambda u: np.asarray(u), step_data
        )
    
    hdf5_store.append(step_data)
    
    return [], [], []

class Featurizer:
    """
    Featurizer class that creates the graph for the given model. 
    When called returns a function that creates the graph for the 
    short-range or long-range interactions.
    """
    def __init__(
        self,
        displacement_fn: callable,
        species: jnp.ndarray,
        total_charge: float = 0.,
        precision: jnp.dtype = jnp.float32,
        lr_bool: bool = False
    )-> None:
    
        self.displacement_fn = displacement_fn
        self.species = species
        self.lr_bool = lr_bool
        self.total_charge = total_charge
        self.precision = precision
        
    def featurize_lr(
        self,
        R: jnp.ndarray,
        neighbor: Tuple[jnp.ndarray, jnp.ndarray],
        neighbor_lr: Tuple[jnp.ndarray, jnp.ndarray],
        **kwargs
    )-> Graph:
        """
        Create the graph for the long-range interactions.

        Args:
            R (jnp.ndarray): Positions of the atoms.
            neighbor (Tuple[jnp.ndarray, jnp.ndarray]): Senders and receivers
                                                of the short-range interactions.
            neighbor_lr (Tuple[jnp.ndarray, jnp.ndarray]): Senders and receivers
                                                of the long-range interactions.
            **kwargs: Additional keyword arguments.

        Returns:
            Graph: Graph object containing all information for the MLFF.
        """
        
        idx_i = neighbor[0]  # shape: P
        idx_j = neighbor[1]  # shape: P
        idx_i_lr = neighbor_lr[0]  # shape: P
        idx_j_lr = neighbor_lr[1]  # shape: P

        Ra = R[idx_i]
        Rb = R[idx_j]
        Ra_lr = R[idx_i_lr]
        Rb_lr = R[idx_j_lr]

        if 'box' in kwargs:
            box = kwargs.get('box')

        # Edges
        d = jax.vmap(partial(self.displacement_fn, **kwargs))
        dR = d(Ra, Rb)
        dR_lr = d(Ra_lr, Rb_lr)

        graph = Graph(
            positions=None,
            nodes=self.species,
            edges=dR,
            centers=idx_i,
            others=idx_j,
            mask=None,
            total_charge=jnp.array([self.total_charge], dtype=self.precision),
            num_unpaired_electrons=jnp.array([0.]),
            edges_lr=dR_lr,
            idx_i_lr=idx_i_lr,
            idx_j_lr=idx_j_lr,
            cell=box  # will raise an error if box not in kwargs.
        )
        return graph
    
    def featurize(
        self,
        R: jnp.ndarray,
        neighbor: Tuple[jnp.ndarray, jnp.ndarray],
        **kwargs
    )-> Graph:
        """
        Create the graph for the short-range interactions.

        Args:
            R (jnp.ndarray): Positions of the atoms.
            neighbor (Tuple[jnp.ndarray, jnp.ndarray]): Senders and receivers
                                                of the short-range interactions.

        Returns:
            Graph: Graph object containing all information for the MLFF.
        """
        idx_i = neighbor[0]
        idx_j = neighbor[1]
        
        Ra = R[idx_i]
        Rb = R[idx_j]
        
        if 'box' in kwargs:
            box = kwargs.get('box')
        
        d = jax.vmap(partial(self.displacement_fn, **kwargs))
        dR = d(Ra, Rb)
        
        graph = Graph(
            positions=None,
            nodes=self.species,
            edges=dR,
            centers=idx_i,
            others=idx_j,
            mask=None,
            total_charge=jnp.array([self.total_charge], dtype=self.precision),
            num_unpaired_electrons=jnp.array([0.]),
            edges_lr=None,
            idx_i_lr=None,
            idx_j_lr=None,
            cell=box
        )
        
        return graph
    
    def __call__(
        self
    ):
        if self.lr_bool:
            return self.featurize_lr
        else:
            return self.featurize

def to_jax_md_custom(
        potential: MLFFPotentialSparse,  
        displacement_or_metric: DisplacementOrMetricFn,
        box: Box,  
        species: jnp.ndarray = None, 
        total_charge: float = 0., 
        precision: jnp.dtype = jnp.float32,
        dr_threshold: float = 0., 
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        minimum_cell_size_multiplier_sr: float = 1.0,
        minimum_cell_size_multiplier_lr: float = 1.0,
        disable_cell_list: bool = False,
        fractional_coordinates: bool = True,
        **neighbor_kwargs
)-> Tuple[callable, callable, callable]:
    """
    Create the neighbor functions and energy function for the given potential
    and displacement function.

    Args:
        potential (MLFFPotentialSparse): Force field model.
        displacement_or_metric (DisplacementOrMetricFn): Displacement function.
        box (Box): Box of the system.
        species (jnp.ndarray, optional): Array of atomic species. Defaults to None.
        total_charge (float, optional): Charge of the system. Defaults to 0..
        precision (jnp.dtype, optional): Floating point precision. Defaults to jnp.float32.
        dr_threshold (float, optional): . Defaults to 0..
        capacity_multiplier (float, optional): Memory multiplier for neighborlist.
                                            Defaults to 1.25.
        buffer_size_multiplier_sr (float, optional): Buffer for short-range
                                            neighborlist. Defaults to 1.25.
        buffer_size_multiplier_lr (float, optional): Buffer for long-range
                                            neighborlist. Defaults to 1.25.
        minimum_cell_size_multiplier_sr (float, optional): Memory mutliplier for
                                            the short-range cell. Defaults to 1.0.
        minimum_cell_size_multiplier_lr (float, optional): Memory mutliplier for
                                            the long-range cell. Defaults to 1.0.
        disable_cell_list (bool, optional): If set to `True` then the neighbor
                    list is constructed using only distances. Defaults to False.
        fractional_coordinates (bool, optional): Whether to use fractional 
                                                    coordinates. Defaults to True.

    Returns:
        Tuple[callable, callable, callable]: Neighbor functions and energy function.
    """
    # create the neighbor_fn
    neighbor_fn = partition.neighbor_list(
        displacement_or_metric,
        box,
        potential.cutoff,  
        dr_threshold,
        capacity_multiplier,
        buffer_size_multiplier_sr,  
        minimum_cell_size_multiplier_sr,
        fractional_coordinates=fractional_coordinates,
        format=partition.NeighborListFormat(1),  # only sparse is supported in mlff
        disable_cell_list=disable_cell_list,
        **neighbor_kwargs)

    if potential.long_range_cutoff is not None:
    # create the neighbor_fn for long-range cutoff
        neighbor_fn_lr = partition.neighbor_list(
            displacement_or_metric,
            box,
            potential.long_range_cutoff,
            dr_threshold,
            capacity_multiplier,
            buffer_size_multiplier_lr,  # as buffer_size_multiplier
            minimum_cell_size_multiplier_lr,
            fractional_coordinates=fractional_coordinates,
            format=partition.NeighborListFormat(2),  # long-range modules can handle OrderedSparse.
            disable_cell_list=disable_cell_list,
            **neighbor_kwargs)

        # function that constructs the graph
        featurizer = Featurizer(
            displacement_fn=displacement_or_metric,
            species=species,
            lr_bool=True,
            total_charge=total_charge,
            precision=precision
            )()

        # create an energy_fn that is compatible with jax_md
        def energy_fn(
                R,
                neighbor,
                neighbor_lr,
                **energy_fn_kwargs
        ):
            graph = featurizer(R, neighbor, neighbor_lr, **energy_fn_kwargs)
            return potential(graph).sum()

        return neighbor_fn, neighbor_fn_lr, energy_fn
    
    else:
        featurizer = Featurizer(
            displacement_fn=displacement_or_metric,
            species=species,
            lr_bool=False,
            total_charge=total_charge,
            precision=precision
        )()

        # create an energy_fn that is compatible with jax_md
        def energy_fn(
                R,
                neighbor,
                **energy_fn_kwargs
        ):
            graph = featurizer(R, neighbor, **energy_fn_kwargs)
            return potential(graph).sum()

        return neighbor_fn, None, energy_fn

def atoms_to_jnp(
    atoms: ase.Atoms,
    precision: jnp.dtype = jnp.float32
)-> dict:
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

def load_model(
    model_path: str,
    precision: jnp.dtype,
    lr_cutoff: float = 12.0,
    dispersion_energy_cutoff_lr_damping: float = 2.
)-> MLFFPotentialSparse:
    """
    Load the model from the given path and return the potential.

    Args:
        model_path (str): Path to the model.
        precision (jnp.dtype): Floating point precision.
        lr_cutoff (float, optional): Long-range cutoff radius. Defaults to 12.0.
        dispersion_energy_cutoff_lr_damping (float, optional): Cutoff for dispersion
                                        potential damping function. Defaults to 2..

    Returns:
        MLFFPotentialSparse: Force field model.
    """
    long_range_kwargs = {
        'cutoff_lr': lr_cutoff,
        'dispersion_energy_cutoff_lr_damping': dispersion_energy_cutoff_lr_damping,
        'neighborlist_format_lr': 'ordered_sparse'
    }

    potential = MLFFPotentialSparse.create_from_workdir(
        workdir=model_path,
        dtype=jnp.float32 if precision == 'float32' else jnp.float64,
        long_range_kwargs=long_range_kwargs
    )
    return potential

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
)-> Tuple[callable, callable, callable]:
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
    logger: logging.Logger = None
    )-> Tuple[jax_md.partition.NeighborList, jax_md.partition.NeighborList, bool]:
    """
    
    Check if the neighbor lists have overflowed and reallocate them if needed.

    Args:
        neighbor_fn (callable): Function that creates the short range neighbor list.
        neighbor_fn_lr (callable): Function that creates the long range neighbor list.
        nbrs (jax_md.partition.NeighborList): Short-range neighbor list.
        nbrs_lr (jax_md.partition.NeighborList): Long-range neighbor list.
        state (jax_md.simulate.State): State of the simulation.
        box (jnp.ndarray): Box of the system.
        logger (logging.Logger, optional): Logger object. Defaults to None.

    Returns:
        Tuple[jax_md.partition.NeighborList, jax_md.partition.NeighborList, bool]:
        Updated neighbor lists and a boolean indicating if the lists overflowed.
    """
    
    overflown = False
    if nbrs.did_buffer_overflow:
        overflown = True
        if logger:
            logger.debug('Neighbor list overflowed, reallocating.')
        nbrs = neighbor_fn.allocate(
            state.position,
            box=box
        )
    if nbrs_lr is not None:
        if nbrs_lr.did_buffer_overflow:
            overflown = True
            if logger:
                logger.debug('Long-range neighbor list overflowed, reallocating.')
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
    T: float = None,
    P:float = None
)-> Tuple[float, float, float, float, float]:
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
        
        if T is not None and P is None:
            H = jax_md.simulate.nvt_nose_hoover_invariant(
                energy_fn, 
                state, 
                kT=T,
                neighbor=nbrs.idx,
                neighbor_lr=nbrs_lr.idx, 
                box=box
            ) / unit['energy']
            
        if T is not None and P is not None:
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
        if T is not None and P is None:
            H = jax_md.simulate.nvt_nose_hoover_invariant(
                energy_fn, 
                state, 
                kT=T,
                neighbor=nbrs.idx,
                box=box
            ) / unit['energy']
        if T is not None and P is not None:
            H = jax_md.simulate.npt_nose_hoover_invariant(
                energy_fn,
                state,
                pressure=P,
                kT=T,
                neighbor=nbrs.idx
            )

    
    current_T = jax_md.quantity.temperature(
        momentum=state.momentum,
        mass = state.mass
    ) / unit['temperature']
    
    # Too memory intensive to calculate pressure for large systems
    if P is not None:
        if nbrs_lr is not None:
            #current_P = jax_md.quantity.pressure(
            #    energy_fn,
            #    state.position,
            #    box=box,
            #    neighbor=nbrs.idx,
            #    neighbor_lr=nbrs_lr.idx,
            #    kinetic_energy=KE
            #)
            current_P = 0.0
        else:
            #current_P = jax_md.quantity.pressure(
            #    energy_fn,
            #    state.position,
            #    box=box,
            #    neighbor=nbrs.idx,
            #    kinetic_energy=KE
            #)
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
)-> Tuple[callable, callable]:
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

    step_md_fn = create_md_fn('nvt',lr,apply_fn, T)
    
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

    step_md_fn = create_md_fn("npt", lr,apply_fn, T, P)
    
    return init_fn, step_md_fn

def create_nvt_step_fn(
    lr: bool,
    apply_fn: callable,
    T: float,
)-> callable:
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
        def step_nvt_fn_lr(
            i: int,
            state):
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
        def step_nvt_fn(
            i: int,
            state):
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
)-> callable:
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
)-> callable:
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
        raise NotImplementedError('Only NVT and NPT ensembles are supported')
    
def perform_md(
    all_settings: Dict,
    opt_structure: jnp.ndarray = None,
    logger: logging.Logger = None,
)-> None:
    """
    Performs MD simulation using the settings provided in all_settings.
    If opt_structure is provided, the simulation will start from this 
    geometry e.g. an optimized structure. This is only an array with
    the coordinates of the atoms. Thus, it still expects a geometry file
    to read from in order to get the cell and species information.

    Args:
        all_settings (Dict): Dictionary containing the settings for the
                            MD simulation.
        opt_structure (jnp.ndarray, optional): Structure to start the MD
                                                from. If not provided starts 
                                                from geometry that is read from
                                                the path given in the settings.
                                                Defaults to None.
        logger (logging.Logger, optional): Logger object. Defaults to None.

    Raises:
        ValueError: If model_path is not defined and use_so3lr is set to False.
    """

    # Model path, settings and precision
    model_path = all_settings.get('model_path')
    use_so3lr = all_settings.get('use_so3lr', True)
    
    if model_path is None and not use_so3lr:
        raise ValueError(
            'Model path must be defined or use_so3lr must be set to True'
        )
    
    if use_so3lr and model_path is not None:
        logger.warning('Model path is ignored when use_so3lr is set to True')
        
    precision = all_settings.get('precision', 'float32')
    precision = jnp.float32 if precision == 'float32' else jnp.float64
    lr_cutoff = all_settings.get('lr_cutoff', 12.0)
    dispersion_energy_cutoff = all_settings.get('dispersion_energy_cutoff_lr_damping', 2.0)
    
    # Seed for the thermostat
    seed = all_settings.get('seed', 0)
    
    # Buffer for the neighbor lists
    buffer_size_multiplier_lr = all_settings.get('buffer_size_multiplier_lr', 1.25)
    buffer_size_multiplier_sr = all_settings.get('buffer_size_multiplier_sr', 1.25)

    # Settings for saving the trajectory to a hdf5 file
    hdf5_buffer_size = all_settings.get('hdf5_buffer_size', 5)
    trajectory_hdf5_file = all_settings.get('trajectory_hdf5_file', 'trajectory.hdf5')
    
    # Handling of restart
    restart_save_path = all_settings.get('restart_save_path')
    if restart_save_path is not None:
        create_restart = True
    else:
        create_restart = False
        
    restart_load_path = all_settings.get('restart_load_path')
    if restart_load_path is not None:
        if os.path.exists(restart_load_path):
            restart = True
        else:
            if logger:
                logger.warning('Restart file does not exist, starting from scratch')
            restart = False
    else:
        restart = False  
    
    # Settings for the MD simulation
    md_dt = all_settings.get('md_dt', 0.0005)
    md_T = all_settings.get('md_T', 300)
    md_cycles = all_settings.get('md_cycles',100)
    md_steps = all_settings.get('md_steps',100)
    
    # Thermostat settings
    nhc_chain_length = all_settings.get('nhc_chain_length', 3)
    nhc_steps = all_settings.get('nhc_steps', 2)
    nhc_tau = all_settings.get('nhc_tau')
    nhc_thermo = all_settings.get('nhc_thermo', 100)
    
    # Barostat settings
    md_P = all_settings.get('md_P')
    nhc_baro = all_settings.get('nhc_baro', 1000)
    nhc_sy_steps = all_settings.get('nhc_sy_steps', 3)
    nhc_npt_tau = all_settings.get('nhc_npt_tau')
    
    # Geometry, cell and charge
    total_charge = all_settings.get('total_charge', 0.)
    initial_geometry = all_settings.get('initial_geometry')
    if initial_geometry is None:
        raise ValueError('Initial geometry must be defined')
    initial_geometry = read(initial_geometry)

    cell = initial_geometry.get_cell()
    if np.all(cell == 0):
        cell = None
    
    if cell is not None:
        shift_displacement = 'periodic'
        initial_geometry.wrap()
    else:
        shift_displacement = 'free'

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)

    (
    position,
    box,
    displacement,
    shift,
    fractional_coordinates
    ) = handle_box(
        shift_displacement,
        initial_geometry_dict['positions'],
        cell = cell
    )

    if opt_structure is not None:
        position = jnp.array(opt_structure, dtype=precision)

    current_cycle = 0
    # Loading restart 
    if restart:
        if logger:
            logger.info(f'Loading state from {restart_load_path}.')
        state, box, current_cycle = load_state(
            path_to_load=restart_load_path,
            ensemble='nvt' if md_P is None else 'npt'
        )
        position = state.position
    
    # Initialize the hdf5 storage
    hdf5_store = init_hdf5_store(
        save_to=trajectory_hdf5_file,
        batch_size=hdf5_buffer_size,
        num_atoms=position.shape[0],
        num_box_entries=1,
        exist_ok=True
    )

    # Loading the model
    if use_so3lr:
        potential = So3lrPotential(
            dtype=precision,
            lr_cutoff=lr_cutoff,
            dispersion_energy_lr_cutoff_damping=dispersion_energy_cutoff
        )
    else:
        potential = load_model(
            model_path,
            precision= jnp.float32 if precision == 'float32' else jnp.float64,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff
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
        
    nbrs = neighbor_fn.allocate(
        position,
        box=box
    )

    if lr:
        nbrs_lr = neighbor_fn_lr.allocate(
            position,
            box=box
        )
    else:
        nbrs_lr = None
    
    # Apply units
    unit_dict = handle_units(
        units.metal_unit_system,
        md_dt,
        md_T,
        md_P
    )
    
    md_dt = unit_dict['dt']
    md_T = unit_dict['T']
    md_P = unit_dict['P']
    
    # Setting up the thermostat and/or barostat
    if nhc_tau is None:
        nhc_tau = md_dt * nhc_thermo
    if nhc_npt_tau is None:
        nhc_npt_tau = md_dt * nhc_baro
        
    nhc_kwargs = {
        'chain_length': nhc_chain_length,
        'chain_steps': nhc_steps,
        'sy_steps': nhc_sy_steps,
        'tau': nhc_tau   
    }
    
    if md_P is not None:
        nhc_barostat_kwargs = nhc_kwargs.copy()
        nhc_barostat_kwargs['tau'] = nhc_npt_tau
        
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
    if logger:
        logger.info('Starting MD simulation')
        if md_T is not None:
            logger.info('Step\tKE\tPE\tTotal Energy\tTemperature\tH\ttime/steps')
        else:
            logger.info('Step\tKE\tPE\tTotal Energy\ttime/steps')
            
        logger.info('----------------------------------------')

    velocities, positions, boxs = [], [], []
    total_time = time.time()
    while current_cycle < md_cycles:
        old_time = time.time()
        
        if lr:
            new_state, nbrs, nbrs_lr, new_box = jax.lax.fori_loop(
                0,
                md_steps,
                step_md_fn,
                (state, nbrs, nbrs_lr, box)
            )
        else:
            new_state, nbrs, new_box = jax.lax.fori_loop(
                0,
                md_steps,
                step_md_fn,
                (state, nbrs, box)
            )
        
        new_time = time.time()
        
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
            if logger:
                if current_T is not None:
                    logger.info(f'{current_cycle*md_steps}\t{KE:.2f}\t{PE:.2f}\t{KE+PE:.3f}\t{current_T:.1f}\t{H:.3f}\t{(new_time - old_time) / md_steps:.4f}')    
                else:
                    logger.info(f'{current_cycle*md_steps}\t{KE:.2f}\t{PE:.2f}\t{KE+PE:.3f}\t{(new_time - old_time) / md_steps:.4f}')
                    
            if box is None:
                positions.append(np.array(state.position))
            else:
                positions.append(
                    jax_md.space.transform(
                        box=box,
                        R=state.position,   
                    )
                )  
            velocities.append(np.array(state.momentum))
            if box is not None:
                boxs.append(
                    np.array(box)
                )
                
            
            if (
                (len(positions) % hdf5_buffer_size == 0 and
                len(positions) > 0) or
                (len(positions) == md_cycles*md_steps)
            ):
                # Saving the trajectory to the hdf5 file 
                positions, velocities, boxs = write_to_hdf5(
                    hdf5_store,
                    velocities,
                    positions,
                    boxs,
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
        
    if logger:
        logger.info('Total_time: {}'.format(time.time()-total_time))

def create_min_fn(
    lr: bool,
    min_apply: callable,
    box = None
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
)-> Tuple[callable, callable]:
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
        dt_start = min_start_dt, 
        dt_max = min_max_dt, 
        n_min = min_n_min
    )

    fire_init = jax.jit(fire_init)
    fire_apply = jax.jit(fire_apply)
    
    step_min_fn = create_min_fn(lr, fire_apply, box)

    return fire_init, step_min_fn

def perform_min(
    all_settings: Dict,
    logger: logging.Logger = None
    )-> jnp.ndarray:
    """
    
    Reads the settings and performs a minimization using the FIRE algorithm.
    TODO: Save the state of the minimization to a file and enable restart.

    Args:
        all_settings (Dict): Dictionary containing the settings for the 
                                minimization.
        logger (logging.Logger, optional): Logger object. Defaults to None.

    Raises:
        ValueError: If model_path is not defined and use_so3lr is set to False.
    
    Returns:
        jnp.ndarray: Relaxed geometry.
    """
    
    # Model path, settings and precision
    model_path = all_settings.get('model_path')
    use_so3lr = all_settings.get('use_so3lr', True)
    
    if model_path is None and not use_so3lr:
        raise ValueError(
            'Model path must be defined or use_so3lr must be set to True'
        )
    
    if use_so3lr and model_path is not None:
        logger.warning('Model path is ignored when use_so3lr is set to True')

    precision = all_settings.get('precision', 'float32')
    precision = jnp.float32 if precision == 'float32' else jnp.float64
    lr_cutoff = all_settings.get('lr_cutoff', 12.0)
    dispersion_energy_cutoff = all_settings.get('dispersion_energy_cutoff_lr_damping', 2.0)
    
    # Buffer for the neighbor lists
    buffer_size_multiplier_lr = all_settings.get('buffer_size_multiplier_lr', 1.25)
    buffer_size_multiplier_sr = all_settings.get('buffer_size_multiplier_sr', 1.25)
    
    # Settings for the minimization
    min_cycles = all_settings.get('min_cycles', 10)
    min_steps = all_settings.get('min_steps', 10)
    min_start_dt = all_settings.get('min_start_dt', 0.05)
    min_max_dt = all_settings.get('min_max_dt', 0.1)
    min_n_min = all_settings.get('min_n_min', 2)

    # Geometry, cell and charge
    total_charge = all_settings.get('total_charge', 0.)
    initial_geometry = all_settings.get('initial_geometry')
    if initial_geometry is None:
        raise ValueError('Initial geometry must be defined')
    initial_geometry = read(initial_geometry)
    cell = initial_geometry.get_cell()
    if np.all(cell == 0):
        cell = None

    if cell is not None:
        shift_displacement = 'periodic'
        initial_geometry.wrap()
    else:
        shift_displacement = 'free'

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)
    
    (
    initial_geometry_dict['positions'],
    box,
    displacement,
    shift, 
    fractional_coordinates
    ) = handle_box(
        shift_displacement,
        initial_geometry_dict['positions'],
        cell = cell
    )
        
    # Loading the model
    if use_so3lr:
        potential = So3lrPotential(
            dtype=precision,
            lr_cutoff=lr_cutoff,
            dispersion_energy_lr_cutoff_damping=dispersion_energy_cutoff
        )
    else:
        potential = load_model(
            model_path,
            precision= jnp.float32 if precision == 'float32' else jnp.float64,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff
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
        
    nbrs = neighbor_fn.allocate(
        initial_geometry_dict['positions'],
        box=box
    )

    if lr:
        nbrs_lr = neighbor_fn_lr.allocate(
            initial_geometry_dict['positions'],
            box=box
        )
    else:
        nbrs_lr = None
    
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
            initial_geometry_dict['positions'],
            box=box,
            neighbor=nbrs.idx,
            neighbor_lr=nbrs_lr.idx
        )
    else:
        min_state = min_init_fn(
            initial_geometry_dict['positions'],
            box=box,
            neighbor=nbrs.idx
        )
    # Start of minimization
    if logger:
        logger.info('Starting minimization')
        logger.info('Step\tE\tFmax')
        logger.info('----------------------------------------')
    for i in range(min_cycles):
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
        
        if logger:
            logger.info('{}\t{:.2f}\t{:.2f}'.format(i*min_steps, E, np.abs(F).max()))

    return min_state.position

def create_logger(
    log_file: str,
    logging_level: int = logging.INFO,
    logger_name: str = 'so3lr'
) -> logging.Logger:
    """
    Creates a logger object.

    Args:
        log_file (str): Path to the log file.
        logging_level (int, optional): Sets the log level. 
                                        Defaults to logging.INFO.
        logger_name (str, optional): Name for the logger.
                                        Defaults to 'so3lr'.

    Returns:
        logging.Logger: Logger object.
    """
    
    logger = logging.getLogger(logger_name)
    
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging_level)
    logger.addHandler(fh)

    logger.setLevel(logging_level)
    return logger

def save_state(
    state,
    box: jnp.array,
    cycle: int,
    path_to_save: str,
    ensemble: str = 'nvt'
)-> None:
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
)-> Tuple:
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
            chain = jax_md.simulate.NoseHooverChain(
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
    
    box = None if np.any(loaded_state['box'] == None) else loaded_state['box']
    cycle = loaded_state['step']
    
    return state, box, cycle

def run(
    settings: dict
    ) -> None:
    """
    Run the MD simulation using the provided settings. Relaxation before the
    MD run is optional. Creates a logger and sets the precision of the 
    simulation.
    
    Default precision is float32. If float64 is desired, set the precision
    key in the settings dictionary to 'float64'.
    
    Default behavior is to not relax the system before the MD run. To relax
    the system before the MD run, set the relax_before_run key in the settings
    dictionary to True.

    Args:
        settings (dict): Dictionary containing the settings for the MD 
                            simulation and relaxation.
    """
    
    relax_before_run = settings.get('relax_before_run', False)

    if settings.get('precision', "float32").lower() == 'float64':
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    logger = create_logger(settings.get('log_file', "./so3lr_md.log"))

    if relax_before_run:
        opt_structure = perform_min(settings, logger)
    else:
        opt_structure = None
    perform_md(settings, opt_structure, logger)

@click.command(help=f"Run MD using a YAML config file.\n\nYAML format example and default values:\n{HELP_STRING}")
@click.option('--settings', default='md_settings.yaml', help='Path to settings file.')
def main(settings):
    try:
        settings_dict = yaml.safe_load(open(settings, 'r'))
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not find settings file at {settings}!')
    
    run(settings_dict)

if __name__ == '__main__':
    main()


