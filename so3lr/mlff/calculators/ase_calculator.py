import jax
import jax.numpy as jnp
import numpy as np
import logging

from collections import namedtuple
from functools import partial
from typing import Optional, Sequence

from ase.calculators.calculator import Calculator

from so3lr.graph import Graph as So3lrGraph
from so3lr.mlff.calculators import exact_neighbors
from so3lr.mlff.geometric.cutoff import add_cell_offsets_sparse
from so3lr.mlff.calculators.potential import PotentialSparse
from so3lr.mlff.nn.observable.dispersion_ref_data import check_atomic_numbers_supported
from so3lr.pbc import is_fully_periodic, lattice_rows

from glp.calculators.utils import get_strain, strain_system
from glp import System, atoms_to_system
from glp.graph import system_to_graph

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

logger = logging.getLogger("SO3LR")


def matrix_to_voigt(matrix):
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


class AseCalculatorSparse(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy', 'hessian']

    @classmethod
    def create_from_workdir(
            cls,
            workdir: str,
            calculate_stress: bool = False,
            calculate_hessian: bool = False,
            lr_neighbors_bool: bool = True,
            lr_cutoff: float = 10.,
            dispersion_energy_cutoff_lr_damping: float = 2.,
            capacity_multiplier: float = 1.25,
            buffer_size_multiplier: float = 1.25,
            skin: float = 0.,
            add_energy_shift: bool = False,
            dtype: np.dtype = np.float64,
            model: str = 'so3krates',
            head_index: int = 5,
            has_aux: bool = False,
            from_file: bool = False,
            observables: Optional[Sequence[str]] = None,
            output_atom_indices: Optional[Sequence[int]] = None,
            kspace_electrostatics: Optional[str] = None,
            kspace_smearing: Optional[float] = None,
            kspace_spacing: Optional[float] = None,
            charge_embed_total_charge: Optional[float] = None,
            charge_equilibration_total_charge: Optional[float] = None,
            neighbor_backend: str = 'auto',
            **kwargs
    ):
        if observables is not None and len(observables) > 0:
            output_intermediate_quantities = [o.replace('_grad','').replace('_jac','') for o in (observables)]
        else:
            output_intermediate_quantities = None

        long_range_kwargs = dict(
            cutoff_lr=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            neighborlist_format_lr='sparse',  # ASECalculator has sparse format.
            **kwargs
        )

        # ``kspace_electrostatics`` selects the k-space method for BOTH the grid
        # built in ``calculate`` and the solver inside the model. Only the grid
        # read it before: the solver is configured from
        # ``long_range_kwargs['coulomb_kspace_method']`` (see
        # load_model_from_workdir), which defaults to 'pme'. Asking for 'ewald'
        # therefore built an Ewald k-point set and fed it to the PME solver --
        # silently wrong energies rather than an error. The CLI never had this
        # problem because it sets coulomb_kspace_method itself (so3lr_md.py).
        #
        # An explicit coulomb_kspace_method (passed through **kwargs) still wins,
        # so the lower-level spelling keeps working unchanged.
        if kspace_electrostatics is not None:
            long_range_kwargs.setdefault('coulomb_kspace_method', kspace_electrostatics)

        mlff_potential = PotentialSparse.create_from_workdir(
            workdir=workdir,
            add_shift=add_energy_shift,
            long_range_kwargs=long_range_kwargs,
            dtype=dtype,
            model=model,
            from_file=from_file,
            output_intermediate_quantities=output_intermediate_quantities,
            # Named explicitly rather than left to **kwargs: **kwargs is folded
            # into long_range_kwargs above, where this would be silently dropped.
            charge_embed_total_charge=charge_embed_total_charge,
            charge_equilibration_total_charge=charge_equilibration_total_charge
        )

        return cls(potential=mlff_potential,
                   head_index=head_index,
                   calculate_stress=calculate_stress,
                   calculate_hessian=calculate_hessian,
                   capacity_multiplier=capacity_multiplier,
                   buffer_size_multiplier=buffer_size_multiplier,
                   skin=skin,
                   lr_neighbors_bool=lr_neighbors_bool,
                   lr_cutoff=lr_cutoff,
                   dtype=dtype,
                   has_aux=has_aux,
                   observables=observables,
                   output_atom_indices=output_atom_indices,
                   kspace_electrostatics=kspace_electrostatics,
                   kspace_smearing=kspace_smearing,
                   kspace_spacing=kspace_spacing,
                   neighbor_backend=neighbor_backend,
                   )

    def __init__(
            self,
            potential,
            capacity_multiplier: float,
            buffer_size_multiplier: float,
            skin: float,
            calculate_stress: bool,
            calculate_hessian: bool,
            dtype: np.dtype,
            has_aux: bool,
            head_index: int = 5,
            observables: Optional[Sequence[str]] = None,
            output_atom_indices: Optional[Sequence[int]] = None,
            kspace_electrostatics: Optional[str] = None,
            kspace_smearing: Optional[float] = None,
            kspace_spacing: Optional[float] = None,
            neighbor_backend: str = 'auto',
            *args,
            **kwargs
    ):
        """
        ASE calculator given a StackNet and parameters.

        neighbor_backend:
            'glp'   glp's minimum-image neighbour list. Exact only for orthorhombic
                    cells whose heights all exceed 2 * (lr_cutoff + skin).
            'exact' explicit periodic images (matscipy/ASE ``neighbor_list('ijS')``),
                    exact for any cell size and shape, rebuilt every call.
            'auto'  (default) 'exact' whenever the minimum image would be wrong
                    (non-orthorhombic cell, or any height <= 2 * cutoff), else 'glp'.
                    Finite systems always use 'glp'.
        """
        if neighbor_backend not in ('auto', 'glp', 'exact'):
            raise ValueError(f"neighbor_backend must be 'auto', 'glp' or 'exact', got {neighbor_backend!r}.")
        self.neighbor_backend = neighbor_backend
        self._exact_nb = None
        self._exact_nb_natoms = None

        super(AseCalculatorSparse, self).__init__(*args, **kwargs)

        calculate_obs_grads = False
        if observables is not None and any(obs.endswith('_grad') or obs.endswith('_jac') for obs in observables):
            calculate_obs_grads = True
            print("Calculating observables gradients")
            if output_atom_indices is None:
                raise ValueError("When calculating observables gradients, output_atom_indices must be provided.")
            print(f'Observables: {observables}, output_atom_indices: {output_atom_indices}')

        assert not (calculate_stress and calculate_hessian), "Calculating stress and hessian at the same time is not supported."
        assert not (calculate_stress and calculate_obs_grads), "Calculating stress and observables gradients at the same time is not supported."
        assert not (calculate_hessian and calculate_obs_grads), "Calculating hessian and observables gradients at the same time is not supported."

        if calculate_stress:
            def energy_fn(system, strain: jnp.ndarray, neighbors):
                system = strain_system(system, strain)
                graph = _system_to_graph(system, neighbors)
                
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
                    neighbors,
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

        elif calculate_hessian:

                def energy_fn(R, system, neighbors):
                    local_system = System(R, system.Z, system.cell, system.total_charge, system.num_unpaired_electrons, system.k_grid, system.k_smearing, system.theory_mask)
                    graph = _system_to_graph(local_system, neighbors)
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

                if output_atom_indices is not None:

                    assert len(output_atom_indices) > 0, "output_atom_indices should be non-empty if specified."
                    assert np.all(np.diff(output_atom_indices) == 1), "For partial hessian calculation, output_atom_indices should be a contiguous range of indices."

                    start_index = output_atom_indices[0]
                    end_index = output_atom_indices[-1] + 1  # exclusive

                    @partial(jax.jit, static_argnames=('start_index', 'end_index'))
                    def energy_hessian_sub_fn(R, system, neighbors, start_index, end_index):
                        penergy_fn = partial(energy_fn, system=system, neighbors=neighbors)

                        @partial(jax.jit, static_argnames=('start_index',))
                        def energy_fn_for_slice(slice_positions, all_position, start_index ):
                            position = jax.lax.dynamic_update_slice_in_dim(
                                all_position, slice_positions, start_index, axis=0
                            )
                            return penergy_fn(
                                    position,
                                )

                        return jax.hessian(energy_fn_for_slice, has_aux=has_aux)(R[start_index:end_index],  R, start_index)

                    @jax.jit
                    def calculate_fn(system, neighbors):
                        penergy_fn = partial(energy_fn, system=system, neighbors=neighbors)

                        out, grads = jax.value_and_grad(
                            penergy_fn,
                            allow_int=True,
                            has_aux=has_aux
                        )(
                            system.R
                        )
                        forces = -grads[start_index:end_index]

                        hessian = energy_hessian_sub_fn(system.R, system, neighbors, start_index, end_index)

                        if isinstance(out, tuple):
                            if not has_aux:
                                raise ValueError

                            return {'energy': out[0], 'forces': forces, 'hessian': hessian, 'aux': out[1]}
                        else:
                            return {'energy': out, 'forces': forces, 'hessian': hessian }

                else:

                    @jax.jit
                    def calculate_fn(system, neighbors):
                        penergy_fn = partial(energy_fn, system=system, neighbors=neighbors)

                        out, grads = jax.value_and_grad(
                            penergy_fn,
                            allow_int=True,
                            has_aux=has_aux
                        )(
                            system.R
                        )
                        forces = - grads

                        hessian = jax.hessian(penergy_fn, has_aux=has_aux)(system.R)

                        if isinstance(out, tuple):
                            if not has_aux:
                                raise ValueError

                            return {'energy': out[0], 'forces': forces, 'hessian': hessian, 'aux': out[1]}
                        else:
                            return {'energy': out, 'forces': forces, 'hessian': hessian }

        
        elif calculate_obs_grads:

            # Elementary functions:

            def energy_fn(R, system, neighbors, has_aux=False):
                local_system = System(R, system.Z, system.cell, system.total_charge, system.num_unpaired_electrons, system.k_grid, system.k_smearing, system.theory_mask)
                graph = _system_to_graph(local_system, neighbors)
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

            @partial(jax.jit, static_argnames=('obs_name',))
            def obs_fn_for_name(position, system, neighbors, obs_name):
                return energy_fn(
                        position,
                        system,
                        neighbors,
                        has_aux=True
                    )[1][obs_name]

            # Masking functions to calculate gradients/Jacobians for specific atoms only:

            @partial(jax.jit, static_argnames=('obs_name',))
            def summed_obs_fn_for_name_and_index(index_position, all_position, system, neighbors, obs_name, index):
                position = jax.lax.dynamic_update_index_in_dim(
                    all_position, index_position, index, axis=0
                )
                return energy_fn(
                        position,
                        system,
                        neighbors,
                        has_aux=True
                    )[1][obs_name].sum()

            @partial(jax.jit, static_argnames=('obs_name',))
            def obs_fn_for_name_and_index(index_position, all_position, system, neighbors, obs_name, index):
                position = jax.lax.dynamic_update_index_in_dim(
                    all_position, index_position, index, axis=0
                )
                return energy_fn(
                        position,
                        system,
                        neighbors,
                        has_aux=True
                    )[1][obs_name]

            # Value and gradient of a scalar-valued function:

            @partial(jax.jit, static_argnames=('obs_name',))
            def summed_obs_value_and_grad_fn( system, neighbors, obs_name, index):
                return jax.value_and_grad(
                            summed_obs_fn_for_name_and_index,
                        )(system.R[index],  system.R, system, neighbors, obs_name, index)

            @partial(jax.jit, static_argnames=('obs_name',))
            def obs_jacobian_fn( system, neighbors, obs_name, index):
                return jax.jacobian(
                            obs_fn_for_name_and_index,
                        )(system.R[index],  system.R, system, neighbors, obs_name, index)

            def calculate_fn(system, neighbors):
                obs_grad_dict = {}
                for o in observables:
                    obase=o.replace('_grad','').replace('_jac','')
                    values = obs_fn_for_name(system.R, system, neighbors, obase)
                    if o.endswith('_grad'):
                        obs_grad_dict[o] = {index: [values[index], summed_obs_value_and_grad_fn(system, neighbors, obase, index)[1]] for index in output_atom_indices}
                        # The gradient of the summed observable, not of the atomic one: atomic
                        # energies would need a factor 2 to give the total-energy gradient.
                    else:
                        if o.endswith('_jac'):
                            if 'energy' in o:
                                raise NotImplementedError("Energy is a scalar observable, its Jacobian is not defined.")
                            obs_grad_dict[o] = {index: [values[index], obs_jacobian_fn(system, neighbors, obase, index)] for index in output_atom_indices}
                        else:
                            obs_grad_dict[o] = values[jnp.array(output_atom_indices)]

                return {'energy': None, 'aux': obs_grad_dict}


        else:

                def energy_fn(system, neighbors):
                    graph = _system_to_graph(system, neighbors)
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
        # Elements the model may be asked about (None if the potential was built
        # without that information); checked per structure in ``calculate``.
        self.supported_atomic_numbers = getattr(
            potential, 'supported_atomic_numbers', None)

        # Check if the ML potential has long-range components
        long_range_bool = potential.long_range_bool

        # Determine the cutoff for the neighborlist.
        if long_range_bool is False:
            # Corresponds to having a (semi)-local ML potential as constructed by MPNN.
            logger.info(
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
                logger.info(
                    f'Running a model with long-range corrections. The local cutoff is {potential.cutoff} Ang and '
                    f'no explicit long-range cutoff.'
                )
                # Take all atoms into account for long range NL list calculation if the potential has no cutoff
                # but is long-ranged. Can only be applied for structures in vacuum.
                self.lr_cutoff = 1e6
            else:
                logger.info(
                    f'Running a model with long-range corrections. The local cutoff is {potential.cutoff} Ang and '
                    f'the long-range cutoff is {potential.long_range_cutoff}.'
                )
                # Take all atoms up to long range cutoff for long range NL list calculation into account.
                # Common setting for simulations in a box of water.
                self.lr_cutoff = potential.long_range_cutoff

        self.dtype = dtype
        self.head_index = head_index
        self.kspace_electrostatics = kspace_electrostatics
        self.kspace_smearing = kspace_smearing
        self.kspace_spacing = kspace_spacing
        self._kspace_reported_source = None

        if self.kspace_electrostatics is not None:
            if self.kspace_smearing is None or self.kspace_spacing is None:
                raise ValueError('kspace_electrostatics requires both kspace_smearing and '
                                 'kspace_spacing when passed to the ASE calculator.')

    def calculate(self, atoms=None, *args, **kwargs):
        super(AseCalculatorSparse, self).calculate(atoms, *args, **kwargs)

        pbc_periodic = is_fully_periodic(
            atoms.get_pbc(), context=f'{type(self).__name__}.calculate')

        # Reject elements the free-atom reference tables do not cover, here where
        # the structure is still eager and nameable. Past this point the lookups are
        # jnp.take under jit, which resolves an out-of-range Z by JAX's
        # out-of-bounds policy instead of raising -- a plausible number taken from
        # the wrong element, with nothing to indicate it happened.
        check_atomic_numbers_supported(
            atoms.get_atomic_numbers(), self.supported_atomic_numbers,
            context=f'{type(self).__name__}.calculate',
        )

        system = atoms_to_system(atoms, dtype=self.dtype)

        # Select the inference level-of-theory head. glp's atoms_to_system seeds
        # theory_mask with a default head that is not necessarily the production
        # head; pin it explicitly so multi-head models predict the right level.
        if getattr(system, 'theory_mask', None) is not None:
            num_theory_levels = system.theory_mask.shape[-1]
            system = system._replace(
                theory_mask=jnp.eye(num_theory_levels, dtype=self.dtype)[self.head_index].reshape(1, -1)
            )

        has_atoms_k_grid = 'k_grid' in atoms.info
        has_atoms_k_smearing = 'k_smearing' in atoms.info
        if has_atoms_k_grid != has_atoms_k_smearing:
            raise ValueError(
                'atoms.info must define both k_grid and k_smearing together '
                'for ASE k-space electrostatics.'
            )

        kspace_source = None
        kspace_message = None
        if has_atoms_k_grid:
            if not pbc_periodic:
                raise ValueError(
                    'atoms.info k_grid/k_smearing require a fully periodic system.'
                )
            if self.kspace_electrostatics is not None:
                kspace_source = 'atoms.info.override'
                kspace_message = ('Using k-space grid from atoms.info; it overrides ASE '
                                  'calculator k-space kwargs.')
            else:
                kspace_source = 'atoms.info'
                kspace_message = 'Using k-space grid from atoms.info.'

        if self.kspace_electrostatics is not None and not has_atoms_k_grid:
            if not pbc_periodic:
                kspace_source = 'kwargs.ignored'
                kspace_message = ('Ignoring ASE calculator k-space kwargs because the system '
                                  'is non-periodic.')
            else:
                kspace_source = 'kwargs.generated'
                kspace_message = (f'Generating {self.kspace_electrostatics.upper()} k-space '
                                  'grid from ASE calculator kwargs.')
                import jaxpme
                k_smearing = jnp.array([self.kspace_smearing], dtype=self.dtype)
                k_spacing = jnp.array([self.kspace_spacing])
                # jaxpme takes the lattice vectors as rows, as ASE stores them.
                cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype)
                if self.kspace_electrostatics == 'ewald':
                    k_grid = jaxpme.kspace.get_kgrid_ewald(cell, lr_wavelength=k_spacing)
                else:
                    k_grid = jaxpme.kspace.get_kgrid_mesh(cell, mesh_spacing=k_spacing)
                system = system._replace(k_grid=k_grid, k_smearing=k_smearing)

        if kspace_source is not None and self._kspace_reported_source != kspace_source:
            logger.info(kspace_message)
            self._kspace_reported_source = kspace_source

        if pbc_periodic:
            cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype).T  # (3, 3)
        else:
            cell = None
            # glp keeps any nonzero atoms.cell and would minimum-image the
            # edges of a pbc=False molecule that carries a bounding box.
            system = system._replace(cell=None)

        if pbc_periodic and self._use_exact(atoms):
            neighbors = self._exact_neighbors(atoms)
        else:
            neighbors = self._glp_neighbors(system, cell)

        output = self.calculate_fn(system, neighbors)
        def _to_numpy(x):
            arr = np.array(x, self.dtype)
            return arr.item() if arr.ndim == 0 else arr
        self.results = jax.tree_util.tree_map(_to_numpy, output)


    def _glp_neighbors(self, system, cell):
        """glp's minimum-image neighbour list, (re)allocated on overflow."""
        if self.spatial_partitioning is None:
            self.neighbors, self.spatial_partitioning = neighbor_list(
                positions=system.R,
                cell=cell,
                cutoff=self.cutoff,
                skin=self.skin,
                capacity_multiplier=self.capacity_multiplier,
                buffer_size_multiplier=self.buffer_size_multiplier,
                lr_cutoff=self.lr_cutoff,
            )

        neighbors = self.spatial_partitioning.update_fn(system.R, self.neighbors, new_cell=cell)
        if neighbors.overflow:
            logger.info('Re-allocating neighbours. ') 
            self.neighbors, self.spatial_partitioning = neighbor_list(
                        positions=system.R,
                        cell=cell,
                        cutoff=self.cutoff,
                        skin=self.skin,
                        capacity_multiplier=self.capacity_multiplier,
                        buffer_size_multiplier=self.buffer_size_multiplier,
                        lr_cutoff=self.lr_cutoff,
            )
            neighbors = self.spatial_partitioning.update_fn(system.R, self.neighbors, new_cell=cell)
            assert not neighbors.overflow
            self.neighbors = neighbors
        else:
            self.neighbors = neighbors
        if neighbors.cell_list is not None:
            # If cell list needs to be reallocated, then reallocate neighbors
            if neighbors.cell_list.reallocate:
                # self.neighbors now contains Neighbors namedtuple with idx_i_lr etc.
                self.neighbors, self.spatial_partitioning = neighbor_list(
                    positions=system.R,
                    cell=cell,
                    cutoff=self.cutoff,
                    skin=self.skin,
                    capacity_multiplier=self.capacity_multiplier,
                    buffer_size_multiplier=self.buffer_size_multiplier,
                    lr_cutoff=self.lr_cutoff
                )
        return neighbors

    # ------------------------------------------------------------------ exact PBC
    def _long_range_cutoff(self):
        """The long-range cutoff for the exact list; None for a local model."""
        if self.lr_cutoff <= 0:
            return None
        if self.lr_cutoff >= 1e5:
            raise ValueError(
                'This long-range model has no explicit long-range cutoff, which is only '
                'defined in vacuum; a periodic system needs a finite lr_cutoff.')
        return self.lr_cutoff

    def _use_exact(self, atoms):
        """Whether a fully periodic structure needs explicit periodic images."""
        if self.neighbor_backend != 'auto':
            return self.neighbor_backend == 'exact'
        return exact_neighbors.needs_exact(
            np.array(atoms.get_cell()), self.cutoff, self._long_range_cutoff(), self.skin)

    def _exact_neighbors(self, atoms):
        prev = self._exact_nb if self._exact_nb_natoms == len(atoms) else None
        self._exact_nb = exact_neighbors.build(atoms, self.cutoff, self._long_range_cutoff(),
                                               prev, multiplier=self.capacity_multiplier)
        self._exact_nb_natoms = len(atoms)
        return jax.device_put(self._exact_nb)


def _system_to_graph(system, neighbors):
    """glp's minimum-image graph, or explicit images for ExactNeighbors.

    With explicit images r_ij = R_j - R_i + S @ cell_rows, so an atom may meet
    several images of one partner, or its own image. glp's System carries the
    lattice as columns; under strain_system both R and the cell deform together,
    so the integer shifts stay valid and every calculate_fn variant (stress,
    hessian, observables) works unchanged on either neighbour type.
    """
    if not isinstance(neighbors, exact_neighbors.ExactNeighbors):
        return system_to_graph(system, neighbors)
    R, rows = system.R, lattice_rows(system.cell)
    ci, cj, ci_lr, cj_lr = neighbors.centers, neighbors.others, neighbors.idx_i_lr, neighbors.idx_j_lr
    edges = add_cell_offsets_sparse(R[cj] - R[ci], rows, neighbors.shifts.astype(R.dtype))
    edges_lr = add_cell_offsets_sparse(R[cj_lr] - R[ci_lr], rows, neighbors.shifts_lr.astype(R.dtype))
    return So3lrGraph(
        positions=R, edges=edges, nodes=system.Z, centers=ci, others=cj,
        mask=ci != R.shape[0], total_charge=system.total_charge,
        num_unpaired_electrons=system.num_unpaired_electrons,
        edges_lr=edges_lr, idx_i_lr=ci_lr, idx_j_lr=cj_lr, cell=system.cell,
        k_grid=system.k_grid, k_smearing=system.k_smearing,
        theory_mask=system.theory_mask)


def neighbor_list(
        positions: jnp.ndarray,
        cutoff: float,
        lr_cutoff: float,
        skin: float = 0.,
        cell: jnp.ndarray = None,
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier: float = 1.25
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

    Returns:

    """
    try:
        from glp.neighborlist import quadratic_neighbor_list
    except ImportError:
        raise ImportError('For neighborhood list, please install the glp package from ...')

    allocate, update = quadratic_neighbor_list(
        cell,
        cutoff,
        skin,
        capacity_multiplier=capacity_multiplier,
        use_cell_list=True,
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
