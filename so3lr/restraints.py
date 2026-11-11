"""Restraint potentials added to the model's energy in MD.

The cylindrical (funnel) restraint confines the mass-weighted centre of mass of a group
of atoms, for example a guest in a host, to a flat-bottom cylinder with optional axial
bounds. It is a term of the potential, so it works with or without enhanced sampling;
a PySAGES method adds its own bias on top.
"""

import logging

import jax.numpy as jnp
import numpy as np

from so3lr.md_settings_validation import expand_indices
from so3lr.pbc import box_to_diag

logger = logging.getLogger("SO3LR")


def parse_cylinder_settings(all_settings):
    """Extract and validate the scalar cylindrical-restraint settings.

    Returns ``None`` when the restraint is disabled, otherwise a dict of validated
    parameters: ``atom_indices`` (a list of distinct ints from an index, a list or a
    range string such as "0-19"; range-checked later against n_atoms),
    ``axis`` (np.ndarray shape (3,), non-zero, un-normalized), ``origin`` (np.ndarray
    (3,) or None -> box center), ``radius`` (float>0), ``k_r`` (float>=0),
    ``axial_min``/``axial_max`` (float or None), ``k_z`` (float>=0, defaults to k_r).

    Raises ``ValueError`` on any invalid combination so mistakes fail fast at setup
    rather than producing a silently wrong ensemble.
    """
    if not bool(all_settings.get('cylinder_restraint_enabled', False)):
        return None

    spec = all_settings.get('cylinder_restraint_atom_indices')
    try:
        indices = [] if spec is None else expand_indices(spec)
    except (TypeError, ValueError) as e:
        raise ValueError(f'cylinder_restraint_atom_indices: {e} (got {spec!r}).') from e
    if not indices:
        raise ValueError(
            'cylinder_restraint_enabled=true requires cylinder_restraint_atom_indices '
            '(0-based indices of the guest atoms defining the COM).'
        )
    if len(set(indices)) != len(indices):
        dup = sorted({i for i in indices if indices.count(i) > 1})
        raise ValueError(f'cylinder_restraint_atom_indices repeats atoms {dup}: each atom '
                         f'enters the centre of mass once.')
    radius = all_settings.get('cylinder_restraint_radius')
    if radius is None:
        raise ValueError('cylinder_restraint_enabled=true requires cylinder_restraint_radius (Å).')
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError(f'cylinder_restraint_radius must be > 0, got {radius}.')

    k_r = all_settings.get('cylinder_restraint_force_constant')
    k_r = 5.0 if k_r is None else float(k_r)          # null: the default
    if k_r < 0.0:
        raise ValueError(f'cylinder_restraint_force_constant must be >= 0, got {k_r}.')

    axis = all_settings.get('cylinder_restraint_axis')
    axis = np.asarray([0.0, 0.0, 1.0] if axis is None else axis, dtype=float).reshape(-1)
    if axis.shape != (3,) or not np.any(axis):
        raise ValueError(
            f"cylinder_restraint_axis must be a non-zero length-3 vector, "
            f"got {all_settings.get('cylinder_restraint_axis')!r}."
        )

    origin_setting = all_settings.get('cylinder_restraint_axis_origin')
    if origin_setting is not None:
        origin = np.asarray(origin_setting, dtype=float).reshape(-1)
        if origin.shape != (3,):
            raise ValueError(
                f'cylinder_restraint_axis_origin must be a length-3 vector, got {origin_setting!r}.'
            )
    else:
        origin = None

    zmin = all_settings.get('cylinder_restraint_axial_min')
    zmax = all_settings.get('cylinder_restraint_axial_max')
    zmin = None if zmin is None else float(zmin)
    zmax = None if zmax is None else float(zmax)
    if zmin is not None and zmax is not None and zmin > zmax:
        raise ValueError(
            f'cylinder_restraint_axial_min ({zmin}) must not exceed '
            f'cylinder_restraint_axial_max ({zmax}).'
        )

    kz_setting = all_settings.get('cylinder_restraint_axial_force_constant')
    k_z = k_r if kz_setting is None else float(kz_setting)
    if k_z < 0.0:
        raise ValueError(f'cylinder_restraint_axial_force_constant must be >= 0, got {k_z}.')

    return {
        'atom_indices': indices,
        'axis': axis,
        'origin': origin,
        'radius': radius,
        'k_r': k_r,
        'axial_min': zmin,
        'axial_max': zmax,
        'k_z': k_z,
    }


def cylindrical_restraint_energy(
    cartesian_position,
    idx,
    weights,
    axis_hat,
    origin,
    radius,
    k_r,
    axial_min=None,
    axial_max=None,
    k_z=None,
    box=None,
    periodic: bool = False,
):
    """Flat-bottom cylindrical (funnel) restraint on a mass-weighted center of mass.

    Confines the COM of the atoms ``idx`` to a cylinder of ``radius`` about the line
    through ``origin`` with direction ``axis_hat`` (assumed unit-length), with optional
    flat-bottom axial bounds turning it into a one-sided funnel:

        E = 0.5*k_r*relu(r_perp - radius)^2
          + 0.5*k_z*relu(axial_min - s)^2  (if axial_min given)
          + 0.5*k_z*relu(s - axial_max)^2  (if axial_max given)

    where ``s`` is the signed coordinate along the axis and ``r_perp`` the perpendicular
    distance to it. Zero energy/force inside the cylinder and within the axial band.

    Parameters
    ----------
    cartesian_position : (n_atoms, 3) array in Cartesian Å.
    idx : integer index array selecting the restrained atoms.
    weights : (len(idx), 1) array; mass fractions summing to 1 (mass-weighted COM).
    axis_hat : (3,) unit vector along the cylinder axis.
    origin : (3,) point on the axis, Cartesian Å.
    radius : scalar R_cyl in Å.
    k_r : radial force constant (eV/Å²).
    axial_min, axial_max : optional flat-bottom axial bounds (Å along the axis from origin).
    k_z : axial force constant; falls back to ``k_r`` if None.
    box : current (possibly NPT-scaled) box for the minimum-image convention.
    periodic : if True and ``box`` given, make the guest whole around its first atom
        before taking the COM (the integrator wraps atoms one at a time, so a guest
        that crosses a box face would otherwise get a COM between its two halves), and
        apply an orthorhombic minimum image to the origin->COM displacement, so the
        result is independent of the periodic images. The guest must span less than
        half the box along each axis.
    """
    x = cartesian_position[idx]
    if periodic and box is not None:
        diag = box_to_diag(jnp.asarray(box))
        d = x - x[0]
        com = x[0] + jnp.sum(weights * (d - diag * jnp.round(d / diag)), axis=0)
        p = com - origin
        p = p - diag * jnp.round(p / diag)
    else:
        p = jnp.sum(weights * x, axis=0) - origin
    s = jnp.dot(p, axis_hat)
    perp = p - s * axis_hat
    # Safe norm: jnp.linalg.norm has a NaN gradient at the axis (perp == 0), which is
    # exactly where a confined guest sits most of the time. The double-`where` gives
    # r_perp == 0 there with a well-defined zero gradient.
    sq = jnp.sum(perp ** 2)
    r_perp = jnp.where(sq > 0.0, jnp.sqrt(jnp.where(sq > 0.0, sq, 1.0)), 0.0)
    kz = k_r if k_z is None else k_z
    E = 0.5 * k_r * jnp.maximum(r_perp - radius, 0.0) ** 2
    if axial_min is not None:
        E = E + 0.5 * kz * jnp.maximum(axial_min - s, 0.0) ** 2
    if axial_max is not None:
        E = E + 0.5 * kz * jnp.maximum(s - axial_max, 0.0) ** 2
    return E


def build_cylinder_restraint(cyl_settings, masses, box, dtype):
    """The constant arguments of ``cylindrical_restraint_energy`` for a run, from the
    validated ``parse_cylinder_settings`` dict, the per-atom ``masses`` and the run's
    ``box``. Kept as jnp constants / Python floats so the energy-fn wrapper closure
    stays pure and JIT-friendly."""
    masses = np.asarray(masses)
    idx = np.asarray(cyl_settings['atom_indices'], dtype=int)
    if np.any(idx < 0) or np.any(idx >= masses.shape[0]):
        raise ValueError('cylinder_restraint_atom_indices contains out-of-range indices')
    m = masses[idx]
    if not np.all(m > 0):
        raise ValueError('cylinder_restraint_atom_indices includes atoms with non-positive mass')
    axis_hat = cyl_settings['axis'] / np.linalg.norm(cyl_settings['axis'])
    # Static periodicity flag: a real periodic box is a vector/matrix with non-zero
    # entries (free BC passes a scalar/zero box).
    b = np.asarray(box)
    periodic = b.ndim >= 1 and bool(np.any(b))
    origin = cyl_settings['origin']        # validated (3,) or None -> box centre
    if origin is None:
        if b.shape not in ((3,), (3, 3)):
            # free BC or a scalar/degenerate box: no meaningful center. Require an
            # explicit origin rather than guessing a wrong-shaped default.
            raise ValueError(
                'cylinder_restraint_axis_origin must be given explicitly for this '
                'boundary condition (no periodic box to derive a center from).'
            )
        origin = 0.5 * (b.sum(axis=0) if b.ndim == 2 else b)  # half the sum of lattice vectors
    zmin, zmax = cyl_settings['axial_min'], cyl_settings['axial_max']
    logger.info(
        f'Cylindrical restraint: {len(idx)} guest atoms, '
        f'R_cyl={cyl_settings["radius"]} Å, k_r={cyl_settings["k_r"]} eV/Å²'
        + ('' if (zmin is None and zmax is None)
           else f', axial=[{zmin}, {zmax}] Å, k_z={cyl_settings["k_z"]} eV/Å²')
        + f', axis={axis_hat.tolist()}, origin={np.asarray(origin).tolist()}'
    )
    return dict(
        idx=jnp.asarray(idx),
        weights=jnp.asarray((m / m.sum())[:, None], dtype=dtype),
        axis_hat=jnp.asarray(axis_hat, dtype=dtype),
        origin=jnp.asarray(origin, dtype=dtype),
        radius=cyl_settings['radius'],
        k_r=cyl_settings['k_r'],
        axial_min=zmin,
        axial_max=zmax,
        k_z=cyl_settings['k_z'],
        periodic=periodic,
    )
