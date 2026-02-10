import jax
import jax.numpy as jnp
import numpy as np
import logging

from collections import namedtuple
from functools import partial, partialmethod
from typing import Any, Optional

from ase.calculators.calculator import Calculator
from glp import atoms_to_system
from glp import System
from glp.calculators.utils import get_strain
from glp.calculators.utils import strain_system
from glp.graph import system_to_graph
from glp.neighborlist import quadratic_neighbor_list

from .potential import PotentialSparse


SpatialPartitioning = namedtuple(
    "SpatialPartitioning",
    (
        "allocate_fn",
        "update_fn",
        "cutoff",
        "lr_cutoff",
        "skin",
        "capacity_multiplier",
        "buffer_size_multiplier"
    )
)

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)

StackNet = Any


class AseCalculatorSparse(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    @classmethod
    def create_from_workdir(
            cls,
            workdir: str,
            calculate_stress: bool = False,
            lr_cutoff: Optional[float] = 10.,
            dispersion_energy_cutoff_lr_damping: Optional[float] = 2.,
            capacity_multiplier: float = 1.25,
            buffer_size_multiplier: float = 1.25,
            use_cell_list: bool = False,
            skin: float = 0.,
            add_energy_shift: bool = False,
            dtype: np.dtype = np.float64,
            model: str = 'so3krates',
            has_aux: bool = False,
            from_file: bool = False
    ):

        potential = PotentialSparse.create_from_workdir(
            workdir=workdir,
            add_shift=add_energy_shift,
            long_range_kwargs=dict(
                cutoff_lr=lr_cutoff,
                dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
                neighborlist_format_lr='sparse',  # ASECalculator has sparse format.
            ),
            dtype=dtype,
            model=model,
            from_file=from_file
        )

        return cls(
            potential=potential,
            calculate_stress=calculate_stress,
            capacity_multiplier=capacity_multiplier,
            buffer_size_multiplier=buffer_size_multiplier,
            use_cell_list=use_cell_list,
            skin=skin,
            dtype=dtype,
            has_aux=has_aux
        )

    def __init__(
            self,
            potential,
            calculate_stress: bool,
            capacity_multiplier: float,
            buffer_size_multiplier: float,
            use_cell_list: bool,
            skin: float,
            dtype: np.dtype,
            has_aux: bool,
            *args,
            **kwargs
    ):
        super(AseCalculatorSparse, self).__init__(*args, **kwargs)

        if calculate_stress:
            def energy_fn(system, strain: jnp.ndarray, neighbors):
                system = strain_system(system, strain)
                graph = system_to_graph(system, neighbors)

                out = potential(graph, has_aux=has_aux)
                if isinstance(out, tuple):
                    atomic_energy = out[0]
                    aux = out[1]
                    return atomic_energy.sum(), aux
                else:
                    atomic_energy = out
                    return atomic_energy.sum()

            @jax.jit
            def calculate_fn(system: System, neighbors):
                strain = get_strain()
                out, grads = jax.value_and_grad(
                    energy_fn,
                    argnums=(0, 1),
                    allow_int=True,
                    has_aux=has_aux
                )(
                    system,
                    strain,
                    neighbors
                )

                forces = - grads[0].R
                volume = jnp.abs(jnp.dot(jnp.cross(system.cell[0], system.cell[1]), system.cell[2]))
                stress = grads[1] / volume
                stress = matrix_to_voigt(stress)

                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    return {'energy': out[0], 'forces': forces, 'stress': stress, 'aux': out[1]}
                else:
                    return {'energy': out, 'forces': forces, 'stress': stress}

        else:
            def energy_fn(system, neighbors):
                graph = system_to_graph(system, neighbors)
                out = potential(graph, has_aux=has_aux)
                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    atomic_energy = out[0]
                    aux = out[1]
                    return atomic_energy.sum(), aux
                else:
                    atomic_energy = out
                    return atomic_energy.sum()

            @jax.jit
            def calculate_fn(system, neighbors):
                out, grads = jax.value_and_grad(
                    energy_fn,
                    allow_int=True,
                    has_aux=has_aux
                )(
                    system,
                    neighbors
                )
                forces = - grads.R

                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    return {'energy': out[0], 'forces': forces, 'aux': out[1]}
                else:
                    return {'energy': out, 'forces': forces}

        self.calculate_fn = calculate_fn
        self.neighbors = None
        self.spatial_partitioning = None
        self.capacity_multiplier = capacity_multiplier
        self.buffer_size_multiplier = buffer_size_multiplier
        self.skin = skin
        self.cutoff = potential.cutoff  # cutoff for the local neighbor list
        self.use_cell_list = use_cell_list

        # Check if the ML potential has long-range components
        long_range_bool = potential.long_range_bool

        # Determine the cutoff for the neighborlist.
        if long_range_bool is False:
            # Corresponds to having a (semi)-local ML potential as constructed by MPNN.
            logging.mlff(
                f'Running a local model with local cutoff {potential.cutoff}.'
            )
            self.lr_cutoff = -1.
            # Setting neighborlist cutoff to -1 corresponds to long range indices which equal the local indices.
            # Currently, NL list implementation does not allow to skip the calculation of lr indices as a whole.
            # TODO(kabylda): Maybe fix this? Not sure about the overhead due to this for a local model.
        else:
            # Corresponds to having a (semi)-local ML potential as constructed by MPNN and a long-range part
            # of electrostatics and/or dispersion energy.

            if potential.long_range_cutoff is None:
                logging.mlff(
                    f'Running a model with long-range corrections. The local cutoff is {potential.cutoff} Ang and '
                    f'no explicit long-range cutoff.'
                )
                # Take all atoms into account for long range NL list calculation if the potential has no cutoff
                # but is long-ranged. Can only be applied for structures in vacuum.
                self.lr_cutoff = 1e6
            else:
                logging.mlff(
                    f'Running a model with long-range corrections. The local cutoff is {potential.cutoff} Ang and '
                    f'the long-range cutoff is {potential.long_range_cutoff}.'
                )
                # Take all atoms up to long range cutoff for long range NL list calculation into account.
                # Common setting for simulations in a box of water.
                self.lr_cutoff = potential.long_range_cutoff

        self.dtype = dtype

    def calculate(self, atoms=None, *args, **kwargs):
        super(AseCalculatorSparse, self).calculate(atoms, *args, **kwargs)

        system = atoms_to_system(atoms, dtype=self.dtype)

        if atoms.get_pbc().any():
            cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype).T  # (3, 3)
        else:
            cell = None

        if self.spatial_partitioning is None:
            self.neighbors, self.spatial_partitioning = neighbor_list(
                positions=system.R,
                cell=cell,
                cutoff=self.cutoff,
                skin=self.skin,
                capacity_multiplier=self.capacity_multiplier,
                buffer_size_multiplier=self.buffer_size_multiplier,
                lr_cutoff=self.lr_cutoff,
                use_cell_list=self.use_cell_list
            )

        neighbors = self.spatial_partitioning.update_fn(
            system.R,
            self.neighbors,
            new_cell=cell
        )

        # Check if the neighbor list overflowed.
        if neighbors.overflow:
            # If so, do re-allocation. This will also re-allocate the cell list if there is one.
            logging.mlff('Re-allocating neighbours due to buffer size overflow in neighborlist.')

            self.neighbors, self.spatial_partitioning = neighbor_list(
                positions=system.R,
                cell=cell,
                cutoff=self.cutoff,
                skin=self.skin,
                capacity_multiplier=self.capacity_multiplier,
                buffer_size_multiplier=self.buffer_size_multiplier,
                lr_cutoff=self.lr_cutoff,
                use_cell_list=self.use_cell_list
            )

            neighbors = self.spatial_partitioning.update_fn(
                system.R,
                self.neighbors,
                new_cell=cell
            )

            assert not neighbors.overflow, f'Something went badly wrong, likely due to a bug. Please report.'

            self.neighbors = neighbors

        else:
            # Else, update the neighbors.
            self.neighbors = neighbors

        # Check if a cell list is present.
        if neighbors.cell_list is not None:
            # Check if cell needs re-allocation.
            if neighbors.cell_list.reallocate:
                logging.mlff('Re-allocating neighbours due to buffer size overflow in cell list.')

                # Re-allocate cell list.
                self.neighbors, self.spatial_partitioning = neighbor_list(
                    positions=system.R,
                    cell=cell,
                    cutoff=self.cutoff,
                    skin=self.skin,
                    capacity_multiplier=self.capacity_multiplier,
                    buffer_size_multiplier=self.buffer_size_multiplier,
                    lr_cutoff=self.lr_cutoff,
                    use_cell_list=self.use_cell_list
                )

                assert not neighbors.overflow, f'Something went badly wrong, likely due to a bug. Please report.'
                assert not neighbors.cell_list.reallocate, f'Something went badly wrong, likely due to a bug. Please report.'

        # Calculate the results.
        output = self.calculate_fn(
            system,
            neighbors
        )

        self.results = jax.tree_util.tree_map(lambda x: np.array(x, self.dtype), output)


def matrix_to_voigt(
        matrix: jnp.ndarray
):
    """
    Convert a 3x3 matrix to a 6-component stress vector in Voigt notation.

    Args:
        matrix (jnp.ndarray): A 3x3 matrix.

    Returns:
        jnp.ndarray: A 6-component stress vector in Voigt notation.
    """

    # Check input
    if matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix. Shape is ", matrix.shape)

    # Form Voigt vector
    voigt_vector = jnp.array(
        [
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            (matrix[1, 2] + matrix[2, 1]) / 2,
            (matrix[0, 2] + matrix[2, 0]) / 2,
            (matrix[0, 1] + matrix[1, 0]) / 2
        ]
    )

    return voigt_vector


@jax.jit
def add_batch_dim(tree):
    return jax.tree_util.tree_map(lambda x: x[None], tree)


def neighbor_list(
        positions: jnp.ndarray,
        cutoff: float,
        lr_cutoff: float,
        skin: float = 0.,
        cell: jnp.ndarray = None,
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier: float = 1.25,
        use_cell_list: bool = False
):
    """

    Args:
        positions ():
        cutoff ():
        lr_cutoff ():
        skin ():
        cell (): ASE cell.
        capacity_multiplier ():
        buffer_size_multiplier ():
        use_cell_list ():

    Returns:

    """

    allocate, update = quadratic_neighbor_list(
        cell,
        cutoff,
        skin,
        capacity_multiplier=capacity_multiplier,
        use_cell_list=use_cell_list,
        lr_cutoff=lr_cutoff,
        buffer_size_multiplier=buffer_size_multiplier
    )

    neighbors = allocate(positions)

    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=jax.jit(update),
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier,
                                          buffer_size_multiplier=buffer_size_multiplier,
                                          lr_cutoff=lr_cutoff
                                          )
