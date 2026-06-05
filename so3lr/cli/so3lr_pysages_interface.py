"""
SO3LR interface to PySAGES

This module provides the interface to PySAGES to perform MDs with enhanced sampling
and collective variables. All PySAGES imports are guarded so that SO3LR can be used
without PySAGES installed.
"""

import logging

import numpy as np
import jax.numpy as jnp
import jax_md.simulate
import jax_md.quantity
import jax_md.units
from jax_md.space import transform
import yaml
from jax import jit

try:
    import pysages
    import pysages.backends as pb
    import pysages.colvars
    import pysages.methods
    import pysages.serialization
    HAS_PYSAGES = True
except ImportError:
    HAS_PYSAGES = False

logger = logging.getLogger("SO3LR")


def create_pysages_interface_fns(lr, state, box, step_md_fn, md_dt, nbrs,
                                  ensemble='nvt', nbrs_lr=None, k_grid=None,
                                  k_grid_disp=None):
    """
    Create the PySAGES context generator function that wraps SO3LR's MD step.

    Args:
        lr: Whether long-range interactions are used.
        state: Current MD state.
        box: Simulation box.
        step_md_fn: The MD step function.
        md_dt: MD timestep.
        nbrs: Short-range neighbor list.
        ensemble: Ensemble type ('nvt', 'nve', or 'npt').
        nbrs_lr: Long-range neighbor list (optional).
        k_grid: K-space grid for Ewald/PME electrostatics (optional).
        k_grid_disp: K-space grid for dispersion PME (optional).

    Returns:
        A callable that generates a PySAGES JaxMDContext.
    """
    if np.all(box == 0) or box is None:
        box = jnp.array([0.0, 0.0, 0.0])

    if lr:
        def init_fn_pysages():
            state0 = state
            if ensemble == 'npt':
                state0 = state0.set(position=transform(jnp.array(box), state0.position))
            return pb.JaxMDContextState(
                state0,
                dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box,
                     k_grid=k_grid, k_grid_disp=k_grid_disp)
            )
    else:
        def init_fn_pysages():
            state0 = state
            if ensemble == 'npt':
                state0 = state0.set(position=transform(jnp.array(box), state0.position))
            return pb.JaxMDContextState(
                state0,
                dict(nbrs=nbrs, box=box,
                     k_grid=k_grid, k_grid_disp=k_grid_disp)
            )

    if lr:
        def step_fn_pysages(context_state):
            state = context_state.state

            nbrs = context_state.extras["nbrs"]
            nbrs_lr = context_state.extras["nbrs_lr"]
            box = context_state.extras["box"]
            k_grid = context_state.extras["k_grid"]
            k_grid_disp = context_state.extras["k_grid_disp"]
            i = 0

            if ensemble == 'npt':
                inv_box_vec = 1.0 / jnp.array(box)  # orthorhombic inverse
                frac_positions = transform(inv_box_vec, state.position)  # Cartesian -> fractional
                state = state.set(position=frac_positions)


            state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = step_md_fn(
                i, (state, nbrs, nbrs_lr, box, k_grid, k_grid_disp)
            )

            if ensemble == 'npt':
                box_vec_out = jnp.array(box)
                cart_positions_out = transform(box_vec_out, state.position)
                state = state.set(position=cart_positions_out)


            return pb.JaxMDContextState(
                state,
                dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box,
                     k_grid=k_grid, k_grid_disp=k_grid_disp)
            )
    else:
        def step_fn_pysages(context_state):
            state = context_state.state

            nbrs = context_state.extras["nbrs"]
            box = context_state.extras["box"]
            k_grid = context_state.extras["k_grid"]
            k_grid_disp = context_state.extras["k_grid_disp"]
            i = 0

            if ensemble == 'npt':
                inv_box_vec = 1.0 / jnp.array(box)  # orthorhombic inverse
                frac_positions = transform(inv_box_vec, state.position)  # Cartesian -> fractional
                state = state.set(position=frac_positions)

            state, nbrs, box, k_grid, k_grid_disp = step_md_fn(
                i, (state, nbrs, box, k_grid, k_grid_disp)
            )

            if ensemble == 'npt':
                box_vec_out = jnp.array(box)
                cart_positions_out = transform(box_vec_out, state.position)
                state = state.set(position=cart_positions_out)

            return pb.JaxMDContextState(
                state,
                dict(nbrs=nbrs, box=box,
                     k_grid=k_grid, k_grid_disp=k_grid_disp)
            )

    def generate_context_pysages():
        return pb.JaxMDContext(init_fn_pysages, step_fn_pysages, box, md_dt)

    return generate_context_pysages


@jit
def re_init_so3lr_state(chain_pos, chain_mom, chain_mass, chain_tau,
                         chain_ekin, chain_dof, pos, velocities, forces, mass):
    """Reconstruct an SO3LR NVT Nose-Hoover state from PySAGES snapshot data."""
    new_state = jax_md.simulate.NVTNoseHooverState(
        position=pos,
        momentum=None,
        force=forces,
        mass=mass,
        chain=None
    )

    new_state = jax_md.simulate.canonicalize_mass(new_state)
    new_state = new_state.set(momentum=new_state.mass * velocities)

    dof = jax_md.quantity.count_dof(pos)
    new_thermostat = jax_md.simulate.NoseHooverChain(
        position=chain_pos,
        momentum=chain_mom,
        mass=chain_mass,
        tau=chain_tau,
        kinetic_energy=chain_ekin,
        degrees_of_freedom=dof
    )

    new_state = new_state.set(chain=new_thermostat)
    return new_state


@jit
def re_init_so3lr_state_nve(pos, velocities, forces, mass):
    """Reconstruct an SO3LR NVE state from PySAGES snapshot data."""
    new_state = jax_md.simulate.NVEState(
        position=pos,
        momentum=None,
        force=forces,
        mass=mass,
    )
    new_state = jax_md.simulate.canonicalize_mass(new_state)
    new_state = new_state.set(momentum=new_state.mass * velocities)
    return new_state


#@jit
def re_init_so3lr_state_npt(chain_pos, chain_mom, chain_mass, chain_tau,
                              chain_ekin, chain_dof,
                              baro_pos, baro_mom, baro_mass, baro_tau,
                              baro_ekin, baro_dof,
                              ref_box, box_pos, box_mom, box_mass_val,
                              pos, velocities, forces, mass):
    """Reconstruct an SO3LR NPT Nose-Hoover state from PySAGES snapshot data."""
    dof = jax_md.quantity.count_dof(pos)

    # NoseHooverChain.degrees_of_freedom is a static (metadata) dataclass field
    # and must be a hashable Python int, not a JAX array. PySAGES snapshots round-trip
    # these as jnp arrays, which breaks pytree-equality checks on recompile.
    thermostat = jax_md.simulate.NoseHooverChain(
        position=chain_pos,
        momentum=chain_mom,
        mass=chain_mass,
        tau=chain_tau,
        kinetic_energy=chain_ekin,
        degrees_of_freedom=int(dof),
    )
    barostat = jax_md.simulate.NoseHooverChain(
        position=baro_pos,
        momentum=baro_mom,
        mass=baro_mass,
        tau=baro_tau,
        kinetic_energy=baro_ekin,
        degrees_of_freedom=int(baro_dof),
    )

    new_state = jax_md.simulate.NPTNoseHooverState(
        position=pos,
        momentum=None,
        force=forces,
        mass=mass,
        reference_box=ref_box,
        box_position=box_pos,
        box_momentum=box_mom,
        box_mass=box_mass_val,
        barostat=barostat,
        thermostat=thermostat,
    )
    new_state = jax_md.simulate.canonicalize_mass(new_state)
    new_state = new_state.set(momentum=new_state.mass * velocities)
    return new_state


def update_so3lr_after_pysages(raw_result, lr, init_fn, rng_key, md_T, dt,
                                nbrs, ensemble='nvt', nbrs_lr=None,
                                k_grid=None, k_grid_disp=None):
    """
    Extract the final state from PySAGES results and rebuild SO3LR state.

    Args:
        raw_result: PySAGES Result object.
        lr: Whether long-range interactions are used.
        init_fn: SO3LR init function.
        rng_key: JAX random key.
        md_T: Target temperature.
        dt: MD timestep (used to apply the post-snapshot bias momentum kick).
        nbrs: Short-range neighbor list.
        ensemble: Ensemble type ('nvt', 'nve', or 'npt').
        nbrs_lr: Long-range neighbor list (optional).
        k_grid: K-space grid (passed through unchanged).
        k_grid_disp: Dispersion-PME k-space grid (passed through unchanged).

    Returns:
        Tuple of (new_state, nbrs, nbrs_lr, new_box, k_grid, k_grid_disp).
    """
    def get_velocities(snapshot):
        V, _ = snapshot.vel_mass
        return V

    def get_masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    final_snapshot = raw_result.snapshots[-1]
    new_box = final_snapshot.box.H

    velocities = get_velocities(final_snapshot)
    masses = get_masses(final_snapshot)

    if ensemble == 'nvt':
        new_state = re_init_so3lr_state(
            final_snapshot.chain_data["position"],
            final_snapshot.chain_data["momentum"],
            final_snapshot.chain_data["mass"],
            final_snapshot.chain_data["tau"],
            final_snapshot.chain_data["kinetic_energy"],
            final_snapshot.chain_data["degrees_of_freedom"],
            final_snapshot.positions,
            velocities,
            final_snapshot.forces,
            masses,
        )
    elif ensemble == 'nve':
        new_state = re_init_so3lr_state_nve(
            final_snapshot.positions,
            velocities,
            final_snapshot.forces,
            masses,
        )
    elif ensemble == 'npt':

        # Convert Cartesian positions from PySAGES → fractional for JAX-MD
        frac_positions = transform(1.0/jnp.array(new_box), final_snapshot.positions)

        new_state = re_init_so3lr_state_npt(
            final_snapshot.chain_data["position"],
            final_snapshot.chain_data["momentum"],
            final_snapshot.chain_data["mass"],
            final_snapshot.chain_data["tau"],
            final_snapshot.chain_data["kinetic_energy"],
            final_snapshot.chain_data["degrees_of_freedom"],
            final_snapshot.barostat_data["position"],
            final_snapshot.barostat_data["momentum"],
            final_snapshot.barostat_data["mass"],
            final_snapshot.barostat_data["tau"],
            final_snapshot.barostat_data["kinetic_energy"],
            final_snapshot.barostat_data["degrees_of_freedom"],
            final_snapshot.npt_box_data["reference_box"],
            final_snapshot.npt_box_data["box_position"],
            final_snapshot.npt_box_data["box_momentum"],
            final_snapshot.npt_box_data["box_mass"],
            #final_snapshot.positions,
            frac_positions,
            velocities,
            final_snapshot.forces,
            masses,
        )
        # For NPT, derive the box from the reconstructed state (authoritative)
        # rather than relying on snapshot.box which may lag behind.
        new_box = jax_md.simulate.npt_box(new_state)

        frac_positions = transform(1.0 / jnp.array(new_box), final_snapshot.positions)
        new_state = new_state.set(position=frac_positions)

    else:
        raise ValueError(f"Unsupported ensemble for PySAGES: '{ensemble}'")

    # Update neighbor lists with the correct box
    if ensemble == 'npt':
        nbrs = nbrs.update(frac_positions, neighbor=nbrs.idx, box=new_box)
        nbrs_lr = nbrs_lr.update(
            frac_positions, neighbor=nbrs_lr.idx, box=new_box
        ) if lr else None

    else:
        nbrs = nbrs.update(final_snapshot.positions, neighbor=nbrs.idx, box=new_box)
        nbrs_lr = nbrs_lr.update(
            final_snapshot.positions, neighbor=nbrs_lr.idx, box=new_box
        ) if lr else None

    # Restore bias forces to avoid a discontinuity at the cycle boundary.
    # The snapshot stores unbiased forces (captured before PySAGES adds bias),
    # so the last bias must be re-applied to keep forces consistent with what
    # the integrator saw at the end of the PySAGES run.
    last_bias = raw_result.states[0].bias
    if last_bias is not None:
        new_state = new_state.set(force=new_state.force + last_bias)

        # The pysages jax-md backend patch unconditionally adds 0.5*dt*bias to
        # state.momentum after taking the snapshot (regardless of ensemble), so
        # the snapshot velocity is missing that kick for any non-trivial bias.
        p = new_state.momentum + 0.5 * dt * last_bias
        new_state = new_state.set(momentum=p)

        # The post-step bias kick breaks the invariant chain.kinetic_energy ==
        # KE(state.momentum) that jax-md's apply_fn establishes at line 640
        # (NVT) / 962-966 (NPT) and that thermostat.half_step's substep_fn
        # relies on (it scales both KE and momentum consistently rather than
        # recomputing KE). Restore it here. The barostat is untouched because
        # the bias does not modify box_momentum.
        from jax_md.quantity import kinetic_energy
        KE = kinetic_energy(momentum=p, mass=new_state.mass)
        if ensemble == "nvt":
            new_state = new_state.set(
                chain=new_state.chain.set(kinetic_energy=KE)
            )
        elif ensemble == "npt":
            new_state = new_state.set(
                thermostat=new_state.thermostat.set(kinetic_energy=KE)
            )


    return new_state, nbrs, nbrs_lr, new_box, k_grid, k_grid_disp


def parse_pysages_input(input_path):
    """Read settings for choice of CVs, restraints, grid, sampling method."""
    with open(input_path, 'r') as in_file:
        settings_dict = yaml.load(in_file, Loader=yaml.SafeLoader)

    logger.info(f'Parsed settings for PySAGES: {settings_dict}')
    return settings_dict


def process_cv(cv_dict, nbrs=None, species=None, box=None):
    """Create a PySAGES collective variable from a settings dictionary."""
    name = cv_dict['type']
    arguments = cv_dict

    name_lower = name.lower()
    if name_lower == 'distance':
        return pysages.colvars.Distance(
            indices=[tuple(arguments['grp1']), tuple(arguments['grp2'])]
        )
    elif name_lower == 'distance_pbc':
        if box is None:
            raise ValueError("distance_pbc requires the simulation box.")
        return pysages.colvars.DistancePBC(
            indices=[tuple(arguments['grp1']), tuple(arguments['grp2'])],
            box=box,
        )
    elif name_lower == 'differenceofdistances':
        return pysages.colvars.DifferenceOfDistances(indices=arguments['indices'])
    elif name_lower == 'differenceofdistances_pbc':
        if box is None:
            raise ValueError("differenceofdistances_pbc requires the simulation box.")
        return pysages.colvars.DifferenceOfDistancesPBC(
            indices=arguments['indices'],
            box=box,
        )
    elif name_lower == 'angle':
        return pysages.colvars.Angle(indices=arguments['indices'])
    elif name_lower == 'dihedral':
        return pysages.colvars.DihedralAngle(indices=arguments['indices'])
    elif name_lower == 'gyrrad':
        return pysages.colvars.RadiusOfGyration(indices=arguments['indices'])
    elif name_lower == 'princmom':
        return pysages.colvars.PrincipalMoment(
            indices=arguments['indices'], axis=arguments['axis']
        )
    elif name_lower == 'asphericity':
        return pysages.colvars.Asphericity(indices=arguments['indices'])
    elif name_lower == 'acylindricity':
        return pysages.colvars.Acylindricity(indices=arguments['indices'])
    elif name_lower == 'shapeanisotropy':
        return pysages.colvars.ShapeAnisotropy(indices=arguments['indices'])
    elif name_lower == 'coordinationnumber':
        n_atoms = nbrs.reference_position.shape[0]
        return pysages.colvars.coordinates.CoordinationNumber(
            indices=[
                tuple(np.arange(0, n_atoms).tolist()),
                tuple(arguments['indices'])
            ],
            nbrs=nbrs,
            species=species,
            box=box,
            species_nn=arguments.get('species'),
            cn_exponents=arguments.get('cn_exps'),
            indices_nn=arguments.get('indices_nn'),
        )
    else:
        raise ValueError(f"Unknown collective variable type: '{name}'")


def process_grid(cv_settings):
    """Create a PySAGES Grid from the CV settings."""
    grid_mins = [cv['grid_min'] for cv in cv_settings]
    grid_maxs = [cv['grid_max'] for cv in cv_settings]
    grid_strides = [cv['grid_stride'] for cv in cv_settings]
    periodic = [cv['periodic'] for cv in cv_settings]
    return pysages.Grid(
        lower=tuple(grid_mins),
        upper=tuple(grid_maxs),
        shape=tuple(grid_strides),
        periodic=all(periodic)
    )


def get_pysages_method(settings_dict, nbrs=None, species=None, box=None):
    """Process the parsed settings and return the enhanced sampling method object."""
    cvs = [process_cv(cv, nbrs=nbrs, species=species, box=box)
           for cv in settings_dict['cv']]
    grid = process_grid(settings_dict['cv'])

    metal_units = jax_md.units.metal_unit_system()

    if 'restraints' in settings_dict:
        restraints_dict = settings_dict['restraints']
        logger.info(f'Restraints: {restraints_dict}')
        restraints = pysages.CVRestraints(
            lower=restraints_dict['lower'],
            upper=restraints_dict['upper'],
            kl=restraints_dict['kl'],
            ku=restraints_dict['ku']
        )
    else:
        restraints = None

    method_name = settings_dict['method'].lower()
    if method_name == 'abf':
        N=settings_dict['method_args'].get('N', 500) if 'method_args' in settings_dict else 500
        logger.info(f'N(ABF): {N}')
        return pysages.methods.ABF(cvs, grid, restraints=restraints, N=N)
    elif method_name == 'spectralabf':
        return pysages.methods.SpectralABF(cvs, grid, restraints=restraints)
    elif method_name == 'metad':
        args = settings_dict['method_args']
        return pysages.methods.Metadynamics(
            cvs,
            args['height'],
            args['sigma'],
            args['stride'],
            args['ngauss'],
            deltaT=args['deltaT'],
            kB=metal_units['temperature'],
            grid=grid,
            restraints=restraints
        )
    elif method_name == 'unbiased':
        return pysages.methods.Unbiased(cvs)
    elif method_name == 'harmonic':
        args = settings_dict['method_args']
        return pysages.methods.HarmonicBias(
            cvs,
            kspring=args['kspring'],
            center=args['center']
        )
    else:
        raise ValueError(
            f"Unknown PySAGES method: '{settings_dict['method']}'. "
            "Supported methods: abf, spectralabf, metad, harmonic, unbiased."
        )


def save_pysages_state(pysages_result, path_to_save):
    """Save PySAGES result for restart."""
    pysages.serialization.save(pysages_result, path_to_save)


def load_pysages_state(path_to_load):
    """Load PySAGES result from restart file."""
    return pysages.serialization.load(path_to_load)
