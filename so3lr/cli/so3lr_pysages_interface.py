"""
SO3LR interface to PySAGES

This module provides the interface to PySAGES to perform MDs with enhanced sampling
and collective variables. All PySAGES imports are guarded so that SO3LR can be used
without PySAGES installed.
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
import jax_md.simulate
import jax_md.quantity
import jax_md.units
from jax_md.space import transform
from so3lr.md_settings_validation import load_settings_yaml
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


def run_pysages_cycle(method, pysages_result, generate_context, md_steps):
    """One MD cycle under PySAGES: continues ``pysages_result`` if given, else starts
    ``method``. Returns the new PySAGES result."""
    # A fresh dict each call: pysages.run stores its context in context_args, and its
    # default dict would carry that into the next run started in the same process.
    return jax.block_until_ready(
        pysages.run(pysages_result or method, generate_context, md_steps, context_args={})
    )


def reset_pysages_runner():
    """Drop the run loop that the PySAGES jax-md backend compiles once and caches for
    the process, so the next cycle compiles it again. Needed when the force inside the
    MD step changes while its inputs keep their shapes (the flash engine's wider tables
    after an overflow)."""
    import importlib
    backend = importlib.import_module("pysages.backends.jax-md")
    backend.jax_fn_container.update(is_defined=False, run_fn=None)


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
                                nbrs, nbrs_lr, ensemble='nvt',
                                k_grid=None, k_grid_disp=None):
    """
    Extract the final state from PySAGES results and rebuild SO3LR state.

    Args:
        raw_result: PySAGES Result object.
        init_fn: SO3LR init function.
        rng_key: JAX random key.
        md_T: Target temperature.
        dt: MD timestep (used to apply the post-snapshot bias momentum kick).
        nbrs: Short-range neighbor list (None for the flash engine, which keeps none).
        ensemble: Ensemble type ('nvt', 'nve', or 'npt').
        nbrs_lr: Long-range neighbor list (None with nbrs).
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
    if nbrs is not None:
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
    settings_dict = load_settings_yaml(input_path)

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


def _water_oxygens(arguments, nbrs, species, box):
    """Water oxygen indices for the proton-transfer CVs (chargeseparation,
    ionsitedistance): 'water_indices' if given, otherwise all O atoms without another
    heavy atom within 1.7 A in the starting structure (same rule as the pKa analysis).
    """
    if arguments.get('water_indices') is not None:
        return np.asarray(_expand_indices(arguments['water_indices']), dtype=int)
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    species = np.asarray(species)
    box = np.asarray(box, dtype=float)
    if box.ndim == 2:
        box = np.diag(box)
    pos = np.asarray(nbrs.reference_position, dtype=float)
    if np.abs(pos).max() <= 1.0 + 1e-6:      # fractional coordinates
        pos = pos * box
    # O(N) pair search instead of a dense (n_O, n_heavy, 3) array (GBs at 10k waters)
    i, j = neighbor_list('ij', Atoms(numbers=species, positions=pos, cell=box, pbc=True), 1.7)
    heavy = species != 1
    bonded = np.zeros(len(species), dtype=bool)
    bonded[i[heavy[i] & heavy[j]]] = True    # has another heavy atom within 1.7 A
    o_idx = np.flatnonzero(species == 8)
    return o_idx[~bonded[o_idx]]


def _neutral_h_counts(arguments, input_atoms, species, h_idx, owner_idx, ref_idx,
                      ow_idx, site_idx, sigma):
    """Neutral H counts of the chargeseparation charge sites: the CV's own Gaussian
    sharing, rounded, in the undissociated structure ('neutral_reference' if given,
    otherwise the input structure). Waters must hold 2 H and every count must be an
    integer there, otherwise the structure is not a valid reference."""
    from pysages.colvars.coordinates import charge_separation
    path = arguments.get('neutral_reference')
    if path is not None:
        from ase.io import read
        atoms, source = read(path), f"neutral_reference {path}"
    elif input_atoms is not None:
        atoms, source = input_atoms, "the input structure"
    else:
        raise ValueError("chargeseparation: charge_sites needs the input structure or "
                         "'neutral_reference' (an undissociated structure) for the "
                         "neutral H counts.")
    if not np.array_equal(np.asarray(atoms.numbers), np.asarray(species)):
        raise ValueError(f"chargeseparation: {source} does not have the atoms of the "
                         f"simulated system.")
    cell = np.asarray(atoms.cell[:], dtype=float)
    L = np.diag(cell)
    if not (np.all(atoms.pbc) and np.all(L > 0) and np.allclose(cell, np.diag(L))):
        raise ValueError(f"chargeseparation: {source} needs an orthorhombic periodic cell.")
    pos = np.asarray(atoms.positions, dtype=float)
    water_sites = np.searchsorted(owner_idx, ow_idx)
    # jitted, so XLA fuses the (n_H, n_owner, 3) temporaries instead of materialising
    # each one on the device
    _, n_owned = jax.jit(lambda h, o, r: charge_separation(
        h, o, r, jnp.asarray(L), water_sites, sigma, 2.0))(
        jnp.asarray(pos[h_idx]), jnp.asarray(pos[owner_idx]), jnp.asarray(pos[ref_idx]))
    n = np.asarray(n_owned, dtype=float)[np.searchsorted(owner_idx, site_idx)]
    n0 = np.round(n)
    off = np.abs(n - n0) > 0.1
    if np.any(off):
        raise ValueError(f"chargeseparation: in {source} atoms {site_idx[off].tolist()} "
                         f"own {n[off].round(2).tolist()} H (not an integer), so it gives "
                         f"no neutral H counts; set 'neutral_reference' to an undissociated "
                         f"structure.")
    charged = np.isin(site_idx, ow_idx) & (n0 != 2)
    if np.any(charged):
        raise ValueError(f"chargeseparation: {source} contains a charged water (O "
                         f"{site_idx[charged].tolist()} owns {n0[charged].tolist()} H), "
                         f"i.e. it is not the undissociated state; set 'neutral_reference' "
                         f"to an undissociated structure.")
    return n0, source


def process_cv(cv_dict, nbrs=None, species=None, box=None, input_atoms=None):
    """Create a PySAGES collective variable from a settings dictionary.

    `input_atoms` (ase.Atoms of the input structure) is used by chargeseparation with
    charge_sites for the neutral H counts."""
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
            r0_multiplier=arguments.get('r0_multiplier'),
        )
    elif name_lower == 'chargeseparation':
        # Signed distance between the solute (reference atoms) and the H3O+ (> 0) or
        # OH- (< 0) in the solvent, Xu et al. arXiv:2610.13265. Every H is shared among
        # the owner sites (heavy atoms of the elements in owner_species, default
        # C, N, O, P, S; monatomic ions such as Na+ / Cl- are left out). Water O
        # default: O atoms without another heavy atom within 1.7 A in the starting
        # structure (same rule as the pKa analysis).
        if box is None or not np.any(box):
            raise ValueError("chargeseparation requires a periodic box.")
        species_arr = np.asarray(species)
        box_arr = np.asarray(box, dtype=float)
        ref_spec = arguments.get('reference', arguments.get('indices'))
        if ref_spec is None:
            raise ValueError("chargeseparation requires 'reference' (atom indices "
                             "of the acid/base centre).")
        ref_idx = np.asarray(_expand_indices(ref_spec), dtype=int)
        h_idx = np.flatnonzero(species_arr == 1)
        owner_species = arguments.get('owner_species', [6, 7, 8, 15, 16])
        owner_idx = np.flatnonzero(np.isin(species_arr, owner_species))
        ow_idx = np.setdiff1d(_water_oxygens(arguments, nbrs, species_arr, box_arr),
                              ref_idx)
        if not np.all(np.isin(ow_idx, owner_idx)):
            raise ValueError("chargeseparation: water oxygens must be owner sites "
                             "(check owner_species).")
        water_sites = np.searchsorted(owner_idx, ow_idx)
        # charge_sites: whose charge enters s. 'water' (default): the water O only;
        # 'all': also every other owner site except the reference atoms, so a proton
        # taken from / given to a solute site (amide N-H, carbonyl O) keeps s at the
        # distance of the charge; or a list of atoms to add to the water O. Their
        # neutral H counts come from the input structure or 'neutral_reference'. Every
        # atom that can carry the titratable proton (both carboxyl O, both His ring N,
        # the guanidinium N) belongs in 'reference', else its proton counts as a charge.
        sigma = arguments.get('sigma', 0.3)
        charge_spec = arguments.get('charge_sites', 'water')
        charge_sites = n_ref_sites = None
        if not (isinstance(charge_spec, str) and charge_spec.lower() == 'water'):
            if isinstance(charge_spec, str) and charge_spec.lower() == 'all':
                extra = np.setdiff1d(owner_idx, np.union1d(ow_idx, ref_idx))
            elif isinstance(charge_spec, str) and not any(c.isdigit() for c in charge_spec):
                raise ValueError(f"chargeseparation: charge_sites must be 'water', 'all' "
                                 f"or atom indices, got {charge_spec!r}.")
            else:
                extra = np.setdiff1d(np.asarray(_expand_indices(charge_spec), dtype=int), ow_idx)
                if not np.all(np.isin(extra, owner_idx)):
                    raise ValueError("chargeseparation: charge_sites must be owner sites "
                                     "(heavy atoms of owner_species).")
                if np.any(np.isin(extra, ref_idx)):
                    raise ValueError("chargeseparation: the reference atoms cannot be "
                                     "charge_sites.")
            site_idx = np.union1d(ow_idx, extra)
            n_ref_sites, source = _neutral_h_counts(
                arguments, input_atoms, species_arr, h_idx, owner_idx, ref_idx, ow_idx,
                site_idx, sigma)
            charge_sites = np.searchsorted(owner_idx, site_idx)
            other = ~np.isin(site_idx, ow_idx)
            charge_log = (f'charge on {len(ow_idx)} water O + {int(other.sum())} other sites, '
                          f'neutral H counts from {source}')
            logger.info('chargeseparation: neutral H counts of the non-water charge sites '
                        '(atom:H): ' + ' '.join(f'{i}:{int(k)}' for i, k in
                                                zip(site_idx[other], n_ref_sites[other])))
        # 'pair_guard: true' maps configurations with an H3O+ and an OH- in the solvent
        # (water autoionisation) onto s = 0, so the bias cannot drive autoionisation
        cv = pysages.colvars.coordinates.ChargeSeparation(
            indices=[tuple(h_idx.tolist()), tuple(owner_idx.tolist()),
                     tuple(ref_idx.tolist())],
            box=box_arr,
            water_sites=water_sites,
            sigma=sigma,
            pair_guard=bool(arguments.get('pair_guard', False)),
            n_hydronium=arguments.get('n_hydronium'),
            n_hydroxide=arguments.get('n_hydroxide'),
            guard_width=arguments.get('guard_width', 0.8),
            guard_x0=arguments.get('guard_x0', 0.5),
            charge_sites=charge_sites,
            n_ref_sites=n_ref_sites,
        )
        guard = (f'pair guard on (n_hydronium {cv.n_hydronium}, n_hydroxide '
                 f'{cv.n_hydroxide}, width {cv.guard_width}, x0 {cv.guard_x0})'
                 if cv.pair_guard else 'pair guard off')
        logger.info(f'chargeseparation: reference {ref_idx.tolist()}, {len(h_idx)} H, '
                    f'{len(owner_idx)} owner sites, {len(ow_idx)} water O, '
                    f'sigma {cv.sigma} A, {guard}'
                    + (f', {charge_log}' if charge_sites is not None else ''))
        return cv
    elif name_lower == 'ionsitedistance':
        # Distance between the site atoms ('indices') and the H3O+ (mode 'acid') or
        # OH- (mode 'base') in the solvent; the distance to the nearest water while no
        # such ion exists. Meant to be biased together with a coordinationnumber of the
        # same site. Water O as for chargeseparation ('water_indices' or automatic).
        # Every H is shared among all heavy atoms of the system (owners).
        if box is None or not np.any(box):
            raise ValueError("ionsitedistance requires a periodic box.")
        removed = [k for k in ('d_null', 'r0_water', 'water_cn_exps', 'p') if k in arguments]
        if removed:
            raise ValueError(f"ionsitedistance: option(s) {removed} were removed (H counts "
                             "now use Gaussian ownership, 'sigma'; the ion share is a smooth "
                             "step of width 'a'; without an ion the CV is the distance to the "
                             "nearest water, 'tau').")
        site_spec = arguments.get('indices', arguments.get('reference'))
        if site_spec is None:
            raise ValueError("ionsitedistance requires 'indices' (atom indices of the "
                             "ionizable site).")
        species_arr = np.asarray(species)
        box_arr = np.asarray(box, dtype=float)
        site_idx = np.asarray(_expand_indices(site_spec), dtype=int)
        if np.any(species_arr[site_idx] == 1):
            raise ValueError("ionsitedistance: the site atoms must be heavy atoms.")
        h_idx = np.flatnonzero(species_arr == 1)
        own_idx = np.flatnonzero(species_arr != 1)
        ow_idx = np.setdiff1d(_water_oxygens(arguments, nbrs, species_arr, box_arr),
                              site_idx)
        if len(ow_idx) == 0 or not np.all(species_arr[ow_idx] == 8):
            raise ValueError("ionsitedistance: the water indices must be O atoms.")
        cv = pysages.colvars.coordinates.IonSiteDistance(
            indices=[tuple(site_idx.tolist()), tuple(own_idx.tolist()),
                     tuple(h_idx.tolist())],
            box=box_arr,
            water_sites=np.searchsorted(own_idx, ow_idx),
            mode=arguments.get('mode', 'acid'),
            n0=arguments.get('n0'),
            a=arguments.get('a', 0.4),
            sigma=arguments.get('sigma', 0.25),
            tau=arguments.get('tau', 0.2),
        )
        logger.info(f'ionsitedistance: site {site_idx.tolist()}, mode {cv.mode}, '
                    f'{len(ow_idx)} water O, {len(own_idx)} owners, {len(h_idx)} H, '
                    f'n0 {cv.n0}, a {cv.a}, sigma {cv.sigma} A, tau {cv.tau} A')
        return cv
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


def get_pysages_method(settings_dict, nbrs=None, species=None, box=None, input_atoms=None):
    """Process the parsed settings and return the enhanced sampling method object."""
    cvs = [process_cv(cv, nbrs=nbrs, species=species, box=box, input_atoms=input_atoms)
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


# CV options compared on a PySAGES restart; other CVs: type and atom indices only (some
# keep values derived from the starting positions, e.g. CoordinationNumber.number_nn_max)
_RESTART_CV_OPTIONS = {
    'ChargeSeparation': ('water_sites', 'sigma', 'n_ref_water', 'pair_guard', 'n_hydronium',
                         'n_hydroxide', 'guard_width', 'guard_x0', 'charge_sites',
                         'n_ref_sites'),
}


def _method_settings(method):
    """The settings of a PySAGES method that come from pysages_input: plain attributes
    and kwargs of the method, grid, restraints, and per CV its type, atom indices and
    options (_RESTART_CV_OPTIONS). Missing attributes (older pickles) read as None."""
    def plain(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (np.ndarray, jax.Array)) or (
                isinstance(v, (list, tuple)) and all(np.isscalar(x) for x in v)):
            return np.asarray(v)
        return plain   # marker: not a plain value

    out = {'method': type(method).__name__}
    for k, v in list(vars(method).items()) + list(getattr(method, 'kwargs', {}).items()):
        if k not in ('cvs', 'cv', 'grid', 'restraints', 'kwargs') and plain(v) is not plain:
            out[k] = plain(v)
    grid = getattr(method, 'grid', None)
    if grid is not None:
        out['grid'] = (np.asarray(grid.lower), np.asarray(grid.upper),
                       np.asarray(grid.shape), bool(grid.is_periodic))
    r = getattr(method, 'restraints', None)
    out['restraints'] = None if r is None else tuple(np.asarray(x) for x in r)
    for i, cv in enumerate(method.cvs):
        name = type(cv).__name__
        out[f'cv[{i}]'] = name
        out[f'cv[{i}].indices'] = np.asarray(cv.indices)
        for k in _RESTART_CV_OPTIONS.get(name, ()):
            out[f'cv[{i}].{k}'] = plain(getattr(cv, k, None))
    return out


def _same(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if a is None or b is None:
            return False
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':
            return bool(np.allclose(a, b, rtol=1e-9, atol=1e-12))
        return bool(np.all(a == b))
    return a == b


def warn_restart_method_mismatch(method, restored_method):
    """Warn about every pysages_input setting in which the method built from it differs
    from the one in the PySAGES restart file: a restart continues with the restart
    file's method (CVs, grid, restraints, method arguments), so such input changes have
    no effect. Returns the differing setting names. Only a diagnostic: a comparison that
    fails is logged and never stops the run."""
    try:
        new, old = _method_settings(method), _method_settings(restored_method)
        differ = [k for k in sorted(set(new) | set(old)) if not _same(new.get(k), old.get(k))]
    except Exception as e:      # noqa: BLE001
        logger.warning(f'PySAGES restart: could not compare pysages_input with the restart '
                       f'file\'s method ({type(e).__name__}: {e}).')
        return []

    def show(v):
        if isinstance(v, np.ndarray) and v.size > 6:
            return f'array{v.shape} [{", ".join(map(str, v.ravel()[:3]))}, ...]'
        if isinstance(v, tuple):
            return '(' + ', '.join(show(x) for x in v) + ')'
        return str(v.tolist() if isinstance(v, np.ndarray) else v)
    for k in differ:
        logger.warning(f'PySAGES restart: {k} in pysages_input is {show(new.get(k))}, but '
                       f'{show(old.get(k))} in the restart file. The run continues with the '
                       f'restart file\'s value; the pysages_input value has no effect.')
    return differ


def save_pysages_state(pysages_result, path_to_save):
    """Save PySAGES result for restart."""
    pysages.serialization.save(pysages_result, path_to_save)


def load_pysages_state(path_to_load):
    """Load PySAGES result from restart file."""
    return pysages.serialization.load(path_to_load)
