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
                                  nbrs_lr=None, k_grid=None):
    """
    Create the PySAGES context generator function that wraps SO3LR's MD step.

    Args:
        lr: Whether long-range interactions are used.
        state: Current MD state.
        box: Simulation box.
        step_md_fn: The MD step function.
        md_dt: MD timestep.
        nbrs: Short-range neighbor list.
        nbrs_lr: Long-range neighbor list (optional).
        k_grid: K-space grid for Ewald/PME electrostatics (optional).

    Returns:
        A callable that generates a PySAGES JaxMDContext.
    """
    if np.all(box == 0) or box is None:
        box = jnp.array([0.0, 0.0, 0.0])

    if lr:
        def init_fn_pysages():
            return pb.JaxMDContextState(
                state,
                dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box, k_grid=k_grid)
            )
    else:
        def init_fn_pysages():
            return pb.JaxMDContextState(
                state,
                dict(nbrs=nbrs, box=box, k_grid=k_grid)
            )

    if lr:
        def step_fn_pysages(context_state):
            state = context_state.state
            nbrs = context_state.extras["nbrs"]
            nbrs_lr = context_state.extras["nbrs_lr"]
            box = context_state.extras["box"]
            k_grid = context_state.extras["k_grid"]
            i = 0
            state, nbrs, nbrs_lr, box, k_grid = step_md_fn(
                i, (state, nbrs, nbrs_lr, box, k_grid)
            )
            return pb.JaxMDContextState(
                state,
                dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box, k_grid=k_grid)
            )
    else:
        def step_fn_pysages(context_state):
            state = context_state.state
            nbrs = context_state.extras["nbrs"]
            box = context_state.extras["box"]
            k_grid = context_state.extras["k_grid"]
            i = 0
            state, nbrs, box, k_grid = step_md_fn(
                i, (state, nbrs, box, k_grid)
            )
            return pb.JaxMDContextState(
                state,
                dict(nbrs=nbrs, box=box, k_grid=k_grid)
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

    # dof = jax_md.quantity.count_dof(pos)
    new_thermostat = jax_md.simulate.NoseHooverChain(
        position=chain_pos,
        momentum=chain_mom,
        mass=chain_mass,
        tau=chain_tau,
        kinetic_energy=chain_ekin,
        # degrees_of_freedom=dof
        degrees_of_freedom=chain_dof
    )

    new_state = new_state.set(chain=new_thermostat)
    return new_state


def update_so3lr_after_pysages(raw_result, lr, init_fn, rng_key, md_T,
                                nbrs, nbrs_lr=None, k_grid=None):
    """
    Extract the final state from PySAGES results and rebuild SO3LR state.

    Args:
        raw_result: PySAGES Result object.
        lr: Whether long-range interactions are used.
        init_fn: SO3LR init function.
        rng_key: JAX random key.
        md_T: Target temperature.
        nbrs: Short-range neighbor list.
        nbrs_lr: Long-range neighbor list (optional).
        k_grid: K-space grid (passed through unchanged).

    Returns:
        Tuple of (new_state, nbrs, nbrs_lr, new_box, k_grid).
    """
    def get_velocities(snapshot):
        V, _ = snapshot.vel_mass
        return V

    def get_masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    final_snapshot = raw_result.snapshots[-1]
    new_box = final_snapshot.box.H

    nbrs = nbrs.update(final_snapshot.positions, neighbor=nbrs.idx, box=new_box)
    nbrs_lr = nbrs_lr.update(
        final_snapshot.positions, neighbor=nbrs_lr.idx, box=new_box
    ) if lr else None

    new_state = re_init_so3lr_state(
        final_snapshot.chain_data["position"],
        final_snapshot.chain_data["momentum"],
        final_snapshot.chain_data["mass"],
        final_snapshot.chain_data["tau"],
        final_snapshot.chain_data["kinetic_energy"],
        final_snapshot.chain_data["degrees_of_freedom"],
        final_snapshot.positions,
        get_velocities(final_snapshot),
        final_snapshot.forces,
        get_masses(final_snapshot)
    )

    # Restore bias forces to avoid a discontinuity at the cycle boundary.
    # The snapshot stores unbiased forces (captured before PySAGES adds bias),
    # so the last bias must be re-applied to keep forces consistent with what
    # the integrator saw at the end of the PySAGES run.
    last_bias = raw_result.states[0].bias
    if last_bias is not None:
        new_state = new_state.set(force=new_state.force + last_bias)

    return new_state, nbrs, nbrs_lr, new_box, k_grid


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
    elif name_lower == 'differenceofdistances':
        return pysages.colvars.DifferenceOfDistances(indices=arguments['indices'])
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
        return pysages.methods.ABF(cvs, grid, restraints=restraints)
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
    else:
        raise ValueError(
            f"Unknown PySAGES method: '{settings_dict['method']}'. "
            "Supported methods: abf, spectralabf, metad, unbiased."
        )


def save_pysages_state(pysages_result, path_to_save):
    """Save PySAGES result for restart."""
    pysages.serialization.save(pysages_result, path_to_save)


def load_pysages_state(path_to_load):
    """Load PySAGES result from restart file."""
    return pysages.serialization.load(path_to_load)
