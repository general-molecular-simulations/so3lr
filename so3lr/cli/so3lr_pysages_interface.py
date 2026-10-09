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


def create_pysages_interface_fns(state, box, step_md_fn, md_dt, nbrs,
                                  ensemble='nvt', nbrs_lr=None, k_grid=None,
                                  k_grid_disp=None):
    """
    Create the PySAGES context generator function that wraps SO3LR's MD step.

    Args:
        state: Current MD state.
        box: Simulation box.
        step_md_fn: The MD step function, over the carry
            (state, nbrs, nbrs_lr, box, k_grid, k_grid_disp).
        md_dt: MD timestep.
        nbrs: Short-range neighbor list.
        ensemble: Ensemble type ('nvt', 'nve', or 'npt').
        nbrs_lr: Long-range neighbor list.
        k_grid: K-space grid for Ewald/PME electrostatics (optional).
        k_grid_disp: K-space grid for dispersion PME (optional).

    Returns:
        A callable that generates a PySAGES JaxMDContext.
    """
    if np.all(box == 0) or box is None:
        box = jnp.array([0.0, 0.0, 0.0])

    def init_fn_pysages():
        state0 = state
        if ensemble == 'npt':
            state0 = state0.set(position=transform(jnp.array(box), state0.position))
        return pb.JaxMDContextState(
            state0,
            dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box,
                 k_grid=k_grid, k_grid_disp=k_grid_disp)
        )

    def step_fn_pysages(context_state):
        state = context_state.state
        extras = context_state.extras

        if ensemble == 'npt':
            # PySAGES works in Cartesian coordinates, the NPT integrator in fractional.
            state = state.set(position=transform(1.0 / jnp.array(extras["box"]), state.position))

        state, nbrs, nbrs_lr, box, k_grid, k_grid_disp = step_md_fn(
            0, (state, extras["nbrs"], extras["nbrs_lr"], extras["box"],
                extras["k_grid"], extras["k_grid_disp"])
        )

        if ensemble == 'npt':
            state = state.set(position=transform(jnp.array(box), state.position))

        return pb.JaxMDContextState(
            state,
            dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box,
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
        # dUdV is the cached box-strain derivative dE/deps, a required field on
        # jax-md's isotropic NPTNoseHooverState. For npt_nose_hoover it is a SCALAR
        # (gradient w.r.t. the scalar perturbation eps), not a (3,3) tensor — that
        # shape belongs to the flex/anisotropic state's stress_tensor. PySAGES
        # snapshots do not carry dUdV, so seed it scalar-zero; the integrator
        # consumes it on the first half-step (box_force) before recomputing it.
        dUdV=jnp.zeros(()),
    )
    new_state = jax_md.simulate.canonicalize_mass(new_state)
    new_state = new_state.set(momentum=new_state.mass * velocities)
    return new_state


def update_so3lr_after_pysages(raw_result, init_fn, rng_key, md_T, dt,
                                nbrs, ensemble='nvt', nbrs_lr=None,
                                k_grid=None, k_grid_disp=None):
    """
    Extract the final state from PySAGES results and rebuild SO3LR state.

    Args:
        raw_result: PySAGES Result object.
        init_fn: SO3LR init function.
        rng_key: JAX random key.
        md_T: Target temperature.
        dt: MD timestep (used to apply the post-snapshot bias momentum kick).
        nbrs: Short-range neighbor list.
        ensemble: Ensemble type ('nvt', 'nve', or 'npt').
        nbrs_lr: Long-range neighbor list.
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
    position = frac_positions if ensemble == 'npt' else final_snapshot.positions
    nbrs = nbrs.update(position, neighbor=nbrs.idx, box=new_box)
    nbrs_lr = nbrs_lr.update(position, neighbor=nbrs_lr.idx, box=new_box)

    # Restore bias forces to avoid a discontinuity at the cycle boundary.
    # The snapshot stores unbiased forces (captured before PySAGES adds bias),
    # so the last bias must be re-applied to keep forces consistent with what
    # the integrator saw at the end of the PySAGES run.
    last_bias = raw_result.states[0].bias
    if last_bias is not None:
        # PySAGES computes the bias in float64 (it forces jax_enable_x64 on); cast it
        # down to the integrator dtype so a float32 run is not promoted to float64 here
        # and carried into the next MD cycle (no-op for a float64 run).
        last_bias = last_bias.astype(new_state.force.dtype)
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


def _expand_indices(spec):
    """Expand a YAML index specification into a flat list of ints.

    Supported forms:
      - int:                       5             -> [5]
      - list / tuple (any mix):    [0, 1, 2]     -> [0, 1, 2]
                                   [0, "3-5"]    -> [0, 3, 4, 5]
      - inclusive range string:    "0-19"        -> [0, 1, ..., 19]
      - multi-token string:        "0-2, 5, 8-9" -> [0, 1, 2, 5, 8, 9]
        (tokens separated by commas and/or whitespace; each token is a single
        index or an inclusive "a-b" range with a <= b)

    Ranges "a-b" include BOTH endpoints. Negative indices are not supported.
    """
    def expand_token(tok):
        if isinstance(tok, bool):
            raise ValueError(f"Invalid index: {tok!r}")
        if isinstance(tok, int):
            return [int(tok)]
        if isinstance(tok, str):
            out = []
            for piece in tok.replace(',', ' ').split():
                if '-' in piece:
                    a, b = piece.split('-')
                    a, b = int(a), int(b)
                    if b < a:
                        raise ValueError(
                            f"Invalid index range '{piece}': end < start."
                        )
                    out.extend(range(a, b + 1))
                else:
                    out.append(int(piece))
            return out
        raise ValueError(f"Unsupported index token: {tok!r}")

    if isinstance(spec, (list, tuple)):
        result = []
        for item in spec:
            result.extend(expand_token(item))
        return result
    return expand_token(spec)


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
    elif name_lower == 'meanofdistances':
        return pysages.colvars.MeanOfDistances(indices=arguments['indices'])
    elif name_lower == 'meanofdistances_pbc':
        if box is None:
            raise ValueError("meanofdistances_pbc requires the simulation box.")
        return pysages.colvars.MeanOfDistancesPBC(
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
    elif name_lower == 'component':
        # Cartesian component (axis 0/1/2 = x/y/z) of the barycenter of a group.
        # axis=2 gives the z-coordinate of the solute centroid (interfacial transfer CV).
        return pysages.colvars.Component(
            indices=arguments['indices'], axis=arguments['axis']
        )
    elif name_lower == 'relativecomponent':
        # Cartesian component (axis 0/1/2 = x/y/z) of the barycenter of grp1
        # relative to the barycenter of grp2: xi = bary(grp1)[axis] - bary(grp2)[axis].
        # Translation-invariant -> a bias on it injects no net COM force (no drift).
        # grp1 = solute, grp2 = reference (e.g. the solvent range); axis=2 -> z.
        # Both groups are explicit index specs (range string / list); see
        # _expand_indices. For zero self-coupling (slope exactly 1) make grp2 the
        # solvent only (exclude the solute atoms in grp1).
        if arguments.get('grp1') is None or arguments.get('grp2') is None:
            raise ValueError(
                "relativecomponent requires explicit 'grp1' and 'grp2' index "
                "specs (range string or list)."
            )
        return pysages.colvars.RelativeComponent(
            indices=[tuple(_expand_indices(arguments['grp1'])),
                     tuple(_expand_indices(arguments['grp2']))],
            axis=arguments['axis'],
        )
    elif name_lower == 'relativecomponent_pbc':
        # PBC-robust RelativeComponent: grp1 made whole (anchor min-image),
        # grp2 plain mean (box-spanning reference), separation wrapped to
        # (-L/2, L/2]. Correct even if the solute reaches the periodic seam.
        # Both groups are explicit index specs (range string / list).
        if box is None:
            raise ValueError("relativecomponent_pbc requires the simulation box.")
        if arguments.get('grp1') is None or arguments.get('grp2') is None:
            raise ValueError(
                "relativecomponent_pbc requires explicit 'grp1' and 'grp2' index "
                "specs (range string or list)."
            )
        return pysages.colvars.RelativeComponentPBC(
            indices=[tuple(_expand_indices(arguments['grp1'])),
                     tuple(_expand_indices(arguments['grp2']))],
            axis=arguments['axis'],
            box=box,
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
    elif name_lower == 'ringphase':
        return pysages.colvars.angles.RingPhaseAngle(indices=arguments['indices'])
    elif name_lower == 'ringamplitude':
        return pysages.colvars.angles.RingAmplitude(indices=arguments['indices'])
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
        method_args = settings_dict.get('method_args', {})
        N=method_args.get('N', 500) if 'method_args' in settings_dict else 500
        use_pinv = method_args.get('use_pinv', False)
        logger.info(f'N(ABF): {N}')
        logger.info(f'use_pinv(ABF): {use_pinv}')
        return pysages.methods.ABF(cvs, grid, restraints=restraints, N=N, use_pinv=use_pinv)
    elif method_name == 'spectralabf':
        method_args = settings_dict.get('method_args', {})
        N=method_args.get('N', 500) if 'method_args' in settings_dict else 500
        use_pinv = method_args.get('use_pinv', False)
        logger.info(f'N(SpectralABF): {N}')
        logger.info(f'use_pinv(SpectralABF): {use_pinv}')
        return pysages.methods.SpectralABF(cvs, grid, restraints=restraints, N=N, use_pinv=use_pinv)
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
