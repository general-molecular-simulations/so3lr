"""
Tuning of the long-range parameters (Ewald, PME/P3M, or a plain cutoff) for SO3LR on one
structure: every candidate's force error is estimated, and the fastest candidate within
the accuracy is recommended.
"""
from jax.ops import segment_sum
from so3lr.mlff.masking.mask import safe_scale, safe_mask
from so3lr.mlff.nn.observable.observable_sparse import mixing_rules, gamma_cubic_fit, vdw_QDO_disp_damp, switching_fn
from so3lr.mlff.nn.observable.observable_sparse import coulomb_erf_shifted_force_smooth, coulomb_erf_shifted_force_smooth_pme, coulomb_erf
import numpy as np
import jax
import jax.numpy as jnp
import math
import time
import logging
from typing import Any
from warnings import warn
from itertools import product
from functools import partial

from jax_md.space import distance, periodic_general, DisplacementFn
from jax_md.quantity import force

import jaxpme

from ase.io import read
from ase.units import Bohr

from so3lr.pbc import lattice_rows, minimum_image_problem
from so3lr.cli.so3lr_md import load_model, atoms_to_jnp, process_model, dispersion_ratios
from .so3lr_md import setup_logger

# Setup logging
logger = logging.getLogger("SO3LR")


def filter_neighbor_idx(
    cutoff: float, neighbor_indices: jnp.array, neighbor_distances: jnp.array
):
    filter_idx = jnp.where((neighbor_distances < cutoff) & (
        neighbor_distances > 0.0) & (~jnp.isnan(neighbor_distances)))[0]
    return neighbor_indices[:, filter_idx]


def get_forces_rs(energy, positions, q, displacement, **params):
    d = jax.vmap(partial(displacement))

    @jax.jit
    def total_energy(positions):
        r = distance(d(positions[params['idx_i']], positions[params['idx_j']]))
        num_nodes = len(positions)
        atomic_energy = segment_sum(
            energy(q, r, **params),
            segment_ids=params['idx_i'],
            num_segments=num_nodes
        )
        return -jnp.sum(atomic_energy)

    @jax.jit
    def forces(pos):
        # Get the force by differentiating the energy with respect to pos
        return jax.value_and_grad(total_energy)(pos)[1]

    return forces(positions)


@jax.jit
def dispersion_vdw_QDO_energy(hirshfeld_ratios, r, idx_i, idx_j,  atomic_numbers, sigma, cutoff_lr, cutoff_lr_damping, c6_ratios=None):
    input_dtype = hirshfeld_ratios.dtype

    # Calculate alpha_ij and C6_ij using mixing rules
    alpha_ij, C6_ij = mixing_rules(
        atomic_numbers,
        idx_i,
        idx_j,
        hirshfeld_ratios,
        c6_ratios=c6_ratios,
    )
    # Use cubic fit for gamma
    gamma_ij = gamma_cubic_fit(alpha_ij)

    # Get dispersion energy, positions are converted to to a.u.
    dispersion_energy_ij = vdw_QDO_disp_damp(
        r / jnp.asarray(Bohr, dtype=input_dtype),
        gamma_ij,
        C6_ij,
        alpha_ij,
        jnp.asarray(sigma, dtype=input_dtype),
        'ordered_sparse'
    )

    if cutoff_lr is not None and cutoff_lr_damping is not None:

        _cutoff_lr = jnp.asarray(cutoff_lr, dtype=input_dtype)
        _cutoff_lr_damping = jnp.asarray(cutoff_lr_damping, dtype=input_dtype)
        w = safe_mask(
            r > 0,
            partial(switching_fn, x_on=_cutoff_lr -
                    _cutoff_lr_damping, x_off=_cutoff_lr),
            r,
            0.
        )
        dispersion_energy_ij = safe_scale(dispersion_energy_ij, w, 0.)

    return dispersion_energy_ij


COULOMB_CONSTANT = 14.399645351950548  # e^2 / (4 pi eps_0) in eV A

# Petersen's constants for the PME reciprocal-space error, by interpolation order p
PETERSEN_CONSTANTS = {3: 0.246, 4: 0.404, 5: 0.950, 6: 2.51, 7: 8.42}


def _charge_prefactor(charges, n_atoms):
    """2 k_e Q / sqrt(N) with Q the sum of squared charges: the scale of every RMS force
    error estimate below."""
    return 2 * COULOMB_CONSTANT * float(jnp.sum(jnp.square(charges))) / math.sqrt(n_atoms)


def _alpha(smearing):
    """The Ewald splitting parameter of a Gaussian charge smearing sigma."""
    return 1 / (math.sqrt(2) * smearing)


def _rms_difference(forces, reference):
    """Root mean square over atoms of the force deviation."""
    return jnp.sqrt(jnp.sum(jnp.square(forces - reference)) / len(reference))


def _combined(**parts):
    """Add independent error contributions in quadrature, logging each."""
    total = math.sqrt(sum(float(v) ** 2 for v in parts.values()))
    logger.info("Error estimates: "
                + ", ".join(f"{k}={float(v):.4g}" for k, v in parts.items())
                + f", total={total:.4g}")
    return total


def _describe(candidate):
    return ", ".join(f"{k}: {v}" for k, v in candidate.items() if v is not None)


def force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn,
                     kspace_electrostatics=None, n_warmup=4, n_repeat=20, seed=42):
    """A function of one candidate's parameters returning the mean wall time (s) of a
    force call with them, after n_warmup untimed calls."""
    # one small displacement, reused, so every call does the same work
    perturbed = positions + 1e-3 * jax.random.normal(jax.random.key(seed), positions.shape)

    def mean_time(cutoff, smearing=None, spacing=None, **_):
        k_grid, k_smearing = None, None
        if kspace_electrostatics == "ewald":
            k_grid = jaxpme.kspace.get_kgrid_ewald(cell, jnp.array([spacing]))
        elif kspace_electrostatics in ("pme", "p3m"):
            k_grid = jaxpme.kspace.get_kgrid_mesh(cell, jnp.array([spacing]))
        if k_grid is not None:
            k_smearing = jnp.array([smearing])
        neighbors = filter_neighbor_idx(cutoff, nbrs_lr, nbrs_lr_distances)

        def call():
            # block, or asynchronous dispatch would time the launch instead of the work
            jax.block_until_ready(force_fn(perturbed, neighbor_lr=neighbors,
                                           k_grid=k_grid, k_smearing=k_smearing))
        for _ in range(n_warmup):
            call()
        start = time.monotonic()
        for _ in range(n_repeat):
            call()
        return (time.monotonic() - start) / n_repeat

    return mean_time


def grid_search(candidates, error_fn, time_fn, accuracy):
    """Estimate every candidate's force error, then time those within the accuracy.

    Returns the errors and the timings (inf where the error is too large), in the order
    of the candidates."""
    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
    logger.info("Force-error estimates:")
    errors = []
    for candidate in candidates:
        logger.info(f"  {_describe(candidate)}")
        errors.append(float(error_fn(**candidate)))
    logger.info("Timings of the candidates within the accuracy:")
    timings = []
    for candidate, error in zip(candidates, errors):
        timings.append(time_fn(**candidate) if error <= accuracy else float("inf"))
        logger.info(f"  {_describe(candidate)}: {timings[-1]} s per force call, "
                    f"estimated error {error}")
    return errors, timings


def _check_exponent(exponent):
    if exponent != 1:
        raise NotImplementedError(f"Only exponent = 1 is supported but got {exponent}.")


def tune_ewald(
    charges: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    displacement: DisplacementFn,
    nbrs_lr: jnp.ndarray,
    nbrs_lr_distances: jnp.ndarray,
    force_fn: callable,
    cutoff_lo: float = 7.0,
    cutoff_hi: float = 15.0,
    cutoff_it: float = 2.0,
    smear_lo: float = 2.0,
    smear_hi: float = 6.0,
    smear_it: float = 2.0,
    ns_lo: int = 4,
    ns_hi: int = 10,
    accuracy: float = 2e-2,
    model_kwargs: dict[str, Any] = {},
) -> tuple[list[dict], list[float], list[float]]:
    """Ewald candidates (smearing, real-space cutoff, wavelength) with their estimated force
    errors and timings. The wavelengths divide the shortest cell side into ns_lo..ns_hi
    parts."""
    shortest = float(np.min(np.linalg.norm(cell, axis=1)))
    candidates = [
        {"smearing": smear, "cutoff": cut, "spacing": shortest / ns}
        for smear, cut, ns in product(
            np.arange(smear_lo, smear_hi + smear_it, smear_it),
            np.arange(cutoff_hi, cutoff_lo - cutoff_it, -cutoff_it),
            range(ns_hi, ns_lo - 1, -1),
        )
    ]
    logger.info(f"Testing {len(candidates)} parameter combinations.")
    bounds = EwaldErrorBounds_SR(charges=charges, cell=cell, positions=positions,
                                 displacement=displacement, nbrs_lr=nbrs_lr,
                                 nbrs_lr_distances=nbrs_lr_distances, **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn, "ewald")
    errors, timings = grid_search(candidates, bounds, timer, accuracy)
    return candidates, errors, timings


class EwaldErrorBounds:
    """RMS force-error estimates (eV/A) for Ewald summation with Gaussian smearing sigma,
    i.e. splitting parameter alpha = 1 / (sqrt(2) sigma), from Kolafa and Perram,
    Mol. Simul. 9, 351 (1992), https://doi.org/10.1080/08927029208049126."""

    def __init__(self, charges: jnp.ndarray, cell: jnp.ndarray, positions: jnp.ndarray):
        self.prefactor = _charge_prefactor(charges, len(positions))
        self.volume = float(abs(jnp.linalg.det(cell)))

    def err_kspace(self, smearing: float, spacing: float) -> float:
        """The reciprocal sum kept to |k| <= 2 pi / spacing (LAMMPS ``Ewald::rms`` evaluates
        the same estimate)."""
        alpha = _alpha(smearing)
        return (self.prefactor * alpha / math.sqrt(math.pi) * math.sqrt(spacing / self.volume)
                * math.exp(-(math.pi / (alpha * spacing)) ** 2))

    def err_rspace(self, smearing: float, cutoff: float) -> float:
        """The erfc-screened real-space sum truncated at the cutoff."""
        return (self.prefactor / math.sqrt(cutoff * self.volume)
                * math.exp(-(_alpha(smearing) * cutoff) ** 2))


def _pme_kernel_error(positions, charges, displacement, nbrs_lr, nbrs_lr_distances, *,
                      smearing, cutoff, cutoff_sr, sigma):
    """RMS force error of SO3LR's real-space PME kernel (short-range corrected, shifted
    force) against the erf-screened Coulomb pair sum, measured on this structure."""
    nbrs = filter_neighbor_idx(cutoff, nbrs_lr, nbrs_lr_distances)
    pairs = dict(idx_i=nbrs[0], idx_j=nbrs[1], ke=COULOMB_CONSTANT, sigma=sigma)
    exact = get_forces_rs(coulomb_erf, positions, charges, displacement,
                          cutoff=smearing, **pairs)
    kernel = get_forces_rs(coulomb_erf_shifted_force_smooth_pme, positions, charges,
                           displacement, cuton=cutoff_sr, cutoff=cutoff, smearing=smearing,
                           **pairs)
    return _rms_difference(kernel, exact)


class EwaldErrorBounds_SR(EwaldErrorBounds):
    """Ewald's estimates plus the measured error of SO3LR's real-space kernel."""

    def __init__(self, charges, cell, positions, displacement: DisplacementFn, nbrs_lr,
                 nbrs_lr_distances, cutoff_sr: float = 4.5,
                 electrostatic_energy_scale: float = 1.0):
        super().__init__(charges, cell, positions)
        self.kernel_error = partial(_pme_kernel_error, positions, charges, displacement,
                                    nbrs_lr, nbrs_lr_distances, cutoff_sr=cutoff_sr,
                                    sigma=electrostatic_energy_scale)

    def __call__(self, smearing: float, spacing: float, cutoff: float) -> float:
        return _combined(kspace=self.err_kspace(smearing, spacing),
                         rspace_sr=self.kernel_error(smearing=smearing, cutoff=cutoff),
                         rspace_lr=self.err_rspace(smearing, cutoff))


def tune_pme(
    charges: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    displacement: DisplacementFn,
    nbrs_lr: jnp.ndarray,
    nbrs_lr_distances: jnp.ndarray,
    force_fn: callable,
    cutoff_lo: float = 7.0,
    cutoff_hi: float = 15.0,
    cutoff_it: float = 2.0,
    smear_lo: float = 2.0,
    smear_hi: float = 6.0,
    smear_it: float = 2.0,
    exponent: int = 1,
    nodes_lo: int = 3,
    nodes_hi: int = 5,
    mesh_lo: int = 2,
    mesh_hi: int = 6,
    accuracy: float = 2e-2,
    model_kwargs: dict[str, Any] = {},
) -> tuple[list[dict], list[float], list[float]]:
    """PME candidates (smearing, real-space cutoff, interpolation order, mesh spacing) with
    their estimated force errors and timings. The meshes have 2**m points along the
    shortest cell side, m = mesh_lo..mesh_hi."""
    _check_exponent(exponent)
    shortest = float(np.min(jnp.linalg.norm(cell, axis=1)))
    candidates = [
        {
            "smearing": smear,
            "cutoff": cut,
            "interpolation_nodes": nodes,
            # a side L gets 2 L / spacing + 1 mesh points (PMEErrorBounds.err_kspace)
            "spacing": 2 * shortest / (2**m - 1),
        }
        for smear, cut, nodes, m in product(
            np.arange(smear_lo, smear_hi + smear_it, smear_it),
            np.arange(cutoff_hi, cutoff_lo - cutoff_it, -cutoff_it),
            range(nodes_hi, nodes_lo - 1, -1),
            range(mesh_hi, mesh_lo - 1, -1),
        )
    ]
    logger.info(f"Testing {len(candidates)} parameter combinations.")
    bounds = PMEErrorBounds_SR(charges=charges, cell=cell, positions=positions,
                               displacement=displacement, nbrs_lr=nbrs_lr,
                               nbrs_lr_distances=nbrs_lr_distances, **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn, "pme")
    errors, timings = grid_search(candidates, bounds, timer, accuracy)
    return candidates, errors, timings


class PMEErrorBounds(EwaldErrorBounds):
    """RMS force-error estimates (eV/A) for PME: the reciprocal part from Petersen,
    J. Chem. Phys. 103, 3668 (1995), https://doi.org/10.1063/1.470043, for interpolation
    order p and mesh spacing h; the real-space part as for Ewald."""

    def __init__(self, charges: jnp.ndarray, cell: jnp.ndarray, positions: jnp.ndarray):
        super().__init__(charges, cell, positions)
        self.sides = np.asarray(jnp.linalg.norm(cell, axis=1))

    def err_kspace(self, smearing: float, spacing: float, interpolation_nodes: int) -> float:
        p = int(interpolation_nodes)
        if p not in PETERSEN_CONSTANTS:
            raise ValueError(f"PME error estimates cover interpolation orders "
                             f"{sorted(PETERSEN_CONSTANTS)}, not {p}")
        alpha = _alpha(smearing)
        h = float(np.prod(self.sides / (2 * self.sides / spacing + 1)) ** (1 / 3))
        return (self.prefactor * math.pi ** 0.25 * math.sqrt(6 * alpha / (2 * p + 1))
                / self.volume ** (2 / 3) * (2 * alpha * h) ** p / math.factorial(p)
                * (p / (2 * math.e)) ** (p / 2) * PETERSEN_CONSTANTS[p])


class PMEErrorBounds_SR(PMEErrorBounds):
    """PME's estimates plus the measured error of SO3LR's real-space kernel."""

    def __init__(self, charges, cell, positions, displacement: DisplacementFn, nbrs_lr,
                 nbrs_lr_distances, cutoff_sr: float = 4.5,
                 electrostatic_energy_scale: float = 1.0):
        super().__init__(charges, cell, positions)
        self.kernel_error = partial(_pme_kernel_error, positions, charges, displacement,
                                    nbrs_lr, nbrs_lr_distances, cutoff_sr=cutoff_sr,
                                    sigma=electrostatic_energy_scale)

    def __call__(self, cutoff: float, smearing: float, spacing: float,
                 interpolation_nodes: int) -> float:
        return _combined(kspace=self.err_kspace(smearing, spacing, interpolation_nodes),
                         rspace_sr=self.kernel_error(smearing=smearing, cutoff=cutoff),
                         rspace_lr=self.err_rspace(smearing, cutoff))


def tune_sr(
    charges: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    displacement: DisplacementFn,
    nbrs_lr: jnp.ndarray,
    nbrs_lr_distances: jnp.ndarray,
    force_fn: callable,
    cutoff_lo: float = 10.0,
    cutoff_hi: float = 15.0,
    cutoff_it: float = 2.0,
    exponent: int = 1,
    accuracy: float = 1.0,
    model_kwargs: dict[str, Any] = {},
) -> tuple[list[dict], list[float], list[float]]:
    """Cutoff electrostatics (no k-space): candidate cutoffs with their measured force
    errors and timings."""
    _check_exponent(exponent)
    candidates = [{"cutoff": cut}
                  for cut in np.arange(cutoff_hi, cutoff_lo - cutoff_it, -cutoff_it)]
    logger.info(f"Testing {len(candidates)} parameter combinations.")
    bounds = NativeErrorBounds(charges=charges, cell=cell, positions=positions,
                               displacement=displacement, nbrs_lr=nbrs_lr,
                               nbrs_lr_distances=nbrs_lr_distances, **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn)
    errors, timings = grid_search(candidates, bounds, timer, accuracy)
    return candidates, errors, timings


class NativeErrorBounds:
    """Force errors of cutoff electrostatics, measured on the structure: SO3LR's
    shifted-force kernel against the erf-damped Coulomb sum within the cutoff (rspace_sr),
    and that sum against the one over every long-range pair (rspace_lr)."""

    def __init__(self, charges, cell, positions, displacement: DisplacementFn, nbrs_lr,
                 nbrs_lr_distances, cutoff_sr: float = 4.5,
                 electrostatic_energy_scale: float = 1.0):
        self.positions, self.charges, self.displacement = positions, charges, displacement
        self.nbrs_lr, self.nbrs_lr_distances = nbrs_lr, nbrs_lr_distances
        self.sigma = electrostatic_energy_scale
        self.reference = self._forces(coulomb_erf, nbrs_lr, cutoff=None)

    def _forces(self, energy, nbrs, **kwargs):
        return get_forces_rs(energy, self.positions, self.charges, self.displacement,
                             idx_i=nbrs[0], idx_j=nbrs[1], ke=COULOMB_CONSTANT,
                             sigma=self.sigma, **kwargs)

    def __call__(self, cutoff: float) -> float:
        nbrs = filter_neighbor_idx(cutoff, self.nbrs_lr, self.nbrs_lr_distances)
        truncated = self._forces(coulomb_erf, nbrs, cutoff=None)
        kernel = self._forces(coulomb_erf_shifted_force_smooth, nbrs,
                              cuton=0.45 * cutoff, cutoff=cutoff)
        return _combined(rspace_sr=_rms_difference(kernel, truncated),
                         rspace_lr=_rms_difference(truncated, self.reference))


def tune_dispersion_sr(
    hirshfeld_ratios: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    displacement: DisplacementFn,
    nbrs_lr: jnp.ndarray,
    nbrs_lr_distances: jnp.ndarray,
    force_fn: callable,
    cutoff_lo: float = 10.0,
    cutoff_hi: float = 15.0,
    cutoff_it: float = 2.0,
    damp_lo: float = 2.0,
    damp_hi: float = 4.0,
    damp_it: float = 1.0,
    exponent: int = 1,
    accuracy: float = 0.001,
    model_kwargs: dict[str, Any] = {},
    c6_ratios: jnp.ndarray = None,
) -> tuple[list[dict], list[float], list[float]]:
    """Cutoff dispersion: candidate cutoffs and damping widths with their measured force
    errors and timings."""
    _check_exponent(exponent)
    candidates = [
        {"cutoff": cut, "cutoff_lr_damping": damp}
        for cut, damp in product(
            np.arange(cutoff_hi, cutoff_lo - cutoff_it, -cutoff_it),
            np.arange(damp_lo, damp_hi + damp_it, damp_it),
        )
    ]
    logger.info(f"Testing {len(candidates)} parameter combinations.")
    bounds = NativeErrorBounds_Dispersion(hirshfeld_ratios=hirshfeld_ratios,
                                          c6_ratios=c6_ratios, cell=cell, positions=positions,
                                          displacement=displacement, nbrs_lr=nbrs_lr,
                                          nbrs_lr_distances=nbrs_lr_distances, **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn)
    errors, timings = grid_search(candidates, bounds, timer, accuracy)
    return candidates, errors, timings


class NativeErrorBounds_Dispersion:
    """Force errors of cutoff dispersion, measured on the structure: the switched-off
    dispersion against the plainly truncated one (rspace_sr), and that against the sum
    over every long-range pair (rspace_lr)."""

    def __init__(self, hirshfeld_ratios, cell, positions, displacement: DisplacementFn,
                 nbrs_lr, nbrs_lr_distances, atomic_numbers,
                 dispersion_energy_scale: float = 1.2, c6_ratios: jnp.ndarray = None):
        self.positions, self.hirshfeld_ratios = positions, hirshfeld_ratios
        self.displacement = displacement
        self.nbrs_lr, self.nbrs_lr_distances = nbrs_lr, nbrs_lr_distances
        self.energy = partial(dispersion_vdw_QDO_energy, atomic_numbers=atomic_numbers,
                              sigma=dispersion_energy_scale, c6_ratios=c6_ratios)
        self.reference = self._forces(nbrs_lr, cutoff_lr=None, cutoff_lr_damping=None)

    def _forces(self, nbrs, **kwargs):
        return get_forces_rs(partial(self.energy, **kwargs), self.positions,
                             self.hirshfeld_ratios, self.displacement,
                             idx_i=nbrs[0], idx_j=nbrs[1])

    def __call__(self, cutoff: float, cutoff_lr_damping: float) -> float:
        nbrs = filter_neighbor_idx(cutoff, self.nbrs_lr, self.nbrs_lr_distances)
        truncated = self._forces(nbrs, cutoff_lr=cutoff, cutoff_lr_damping=None)
        switched = self._forces(nbrs, cutoff_lr=cutoff, cutoff_lr_damping=cutoff_lr_damping)
        return _combined(rspace_sr=_rms_difference(switched, truncated),
                         rspace_lr=_rms_difference(truncated, self.reference))


def cell_is_present(cell) -> bool:
    """True if the geometry carries a usable (not all-zero) cell.

    The predecessor `not np.all(cell != 0)` was inverted: it PASSED for an
    all-zero (missing) cell and RAISED for a fully dense triclinic cell. Only
    orthorhombic cells, whose off-diagonal zeros make `np.all(cell != 0)`
    False, behaved correctly -- which is why it went unnoticed. The test is
    simply "the cell is not all zeros".
    """
    return cell is not None and not np.all(np.asarray(cell) == 0.0)


def tune(
    settings: dict
) -> None:
    """
    Tune Ewald parameters for running SO3LR model on a given structure.

    Args:
        settings (dict): Dictionary containing the settings for the tuning process.
    """

    # Configure JAX process-globals (x64 + matmul precision). Centralised in
    # setup_jax_globals so that every entry point sets the same things up
    # consistently. Previously this only toggled x64 and never set the matmul
    # precision, so tune-ewald drifted from the MD path on the same hardware.
    from .so3lr_md import setup_jax_globals
    setup_jax_globals(
        precision=settings.get('precision', 'float32'),
        matmul_precision=settings.get('matmul_precision', 'highest'),
    )

    # Setup logger
    log_file = settings.get('log_file')
    setup_logger(log_file)

    # Model parameters
    model_path = settings.get('model_path')
    precision = settings.get('precision')
    lr_cutoff = settings.get('lr_cutoff')
    total_charge = settings.get('total_charge')
    dispersion_damping = settings.get('dispersion_damping')
    # Match DEFAULT_BUFFER_MULTIPLIER in so3lr_cli.py (1.25). The CLI wrapper
    # always seeds these keys, but a Python-API caller invoking tune() with a
    # bare settings dict previously got 1.0, diverging from the rest of the
    # codebase. Hardcoded here (rather than imported from so3lr_cli) to avoid
    # a circular import (so3lr_cli imports `tune` from this module).
    buffer_size_multiplier_lr = settings.get('buffer_size_multiplier_lr', 1.25)
    buffer_size_multiplier_sr = settings.get('buffer_size_multiplier_sr', 1.25)
    jit_compile = settings.get('jit_compile', True)

    # Ewald electrostatics settings
    kspace_electrostatics = settings.get('kspace_electrostatics', None)
    kspace_dispersion = settings.get('kspace_dispersion', None)
    # Check before the model and neighbour lists are built.
    if kspace_electrostatics not in ('ewald', 'pme', 'p3m', 'cutoff') and kspace_dispersion != 'cutoff':
        raise ValueError(
            f"Unknown kspace_electrostatics type: {kspace_electrostatics}. Must be 'ewald', 'pme', 'p3m', or 'cutoff'.")

    # Tuning parameters
    # Note that all the tuning parameters are loaded further below

    input_file_path = settings.get('input_file')
    if input_file_path is None:
        raise ValueError(
            'Initial geometry file path must be provided in settings.')
    try:
        initial_geometry = read(input_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find initial geometry file: {input_file_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to read geometry file {input_file_path}: {e}")

    cell = initial_geometry.get_cell().T
    assert cell_is_present(cell), "Cell must be provided in the input geometry."
    problem = minimum_image_problem(initial_geometry.get_cell(), lr_cutoff)
    if problem:
        raise ValueError(f'tune-ewald: {problem}.')

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)

    # We don't want fractional coordinates here

    box = jnp.array(cell)
    if box.shape == (3, 3) and np.all(box[~np.eye(box.shape[0], dtype=bool)] == 0):
        # orthorhombic boxes are reduced to 3 entries
        box = jnp.diag(box)
    fractional_coordinates = False

    # Create displacement and shift functions for periodic boundary conditions
    displacement, shift = periodic_general(
        box=box,
        fractional_coordinates=fractional_coordinates
    )
    position = initial_geometry_dict['positions']

    # A bundled name, a path, or None for the default model; load_model builds
    # the solver for the method being timed.
    # The PME/P3M solver is built for one interpolation order, so the timings are
    # for nodes_lo; the error estimates cover every order in the grid.
    nodes_lo, nodes_hi = settings.get('nodes_lo', 3), settings.get('nodes_hi', 3)
    if kspace_electrostatics in ('pme', 'p3m') and nodes_hi != nodes_lo:
        logger.warning(f'Timings use {nodes_lo} interpolation nodes for every candidate '
                       f'(nodes_lo={nodes_lo}, nodes_hi={nodes_hi}).')
    potential = load_model(
        model_path,
        precision=jnp.float64 if precision == 'float64' else jnp.float32,
        lr_cutoff=lr_cutoff,
        dispersion_damping=dispersion_damping,
        kspace_electrostatics=kspace_electrostatics if kspace_electrostatics != 'cutoff' else None,
        kspace_interp_nodes=nodes_lo,
        output_intermediate_quantities=['partial_charges'],
    )

    model_kwargs = dict(
        cutoff_sr=potential.cutoff,
        # 4.0 is the default value, TODO load from model_config!
        electrostatic_energy_scale=4.0
    )

    dispersion_model_kwargs = dict(
        dispersion_energy_scale=1.2,  # 1.2 is the default value, TODO load from model_config!
        atomic_numbers=initial_geometry.get_atomic_numbers()
    )

    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = process_model(
        potential=potential,
        species=initial_geometry_dict['species'],
        displacement=displacement,
        box=box,
        total_charge=total_charge,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_sr=buffer_size_multiplier_sr,
        buffer_size_multiplier_lr=buffer_size_multiplier_lr,
        has_aux=True,
    )

    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box)

    d = jax.vmap(partial(displacement))
    nbrs_lr_distances = distance(
        d(position[nbrs_lr.idx[0]], position[nbrs_lr.idx[1]]))

    if jit_compile:
        energy_fn = jax.jit(
            partial(energy_or_obs_fn, has_aux=False, neighbor=nbrs.idx, box=box))
        force_fn = jax.jit(force(energy_fn))
    else:
        energy_fn = partial(energy_or_obs_fn, has_aux=False,
                            neighbor=nbrs.idx, box=box)
        force_fn = force(energy_fn)

    obs_data = energy_or_obs_fn(
        position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box,
        has_aux=True
    )
    charges = obs_data[1]['partial_charges']
    hirshfeld_ratios, c6_ratios = dispersion_ratios(obs_data[1])

    if box.shape == (3, 3):
        cell = lattice_rows(box)  # error bounds and jaxpme k-grids use rows
    elif box.shape == (3, ):
        cell = jnp.diag(box)
    elif box.shape == (1,):
        cell = jnp.diag(jnp.repeat(box, 3))
    else:
        raise ValueError(
            f"k-space electrostatics: Invalid box shape {box.shape}. Expected (3, 3), (3,), or (1,).")

    accuracy = float(settings.get('accuracy', 1.0 if kspace_electrostatics == 'cutoff' else 1e-3))
    common = dict(
        cell=cell, positions=position, displacement=displacement, nbrs_lr=nbrs_lr.idx,
        nbrs_lr_distances=nbrs_lr_distances, force_fn=force_fn, accuracy=accuracy,
        cutoff_it=settings.get('cutoff_it', 2.0), cutoff_hi=settings.get('cutoff_hi', lr_cutoff),
    )
    smearing = dict(
        cutoff_lo=settings.get('cutoff_lo', 7.0), smear_lo=settings.get('smear_lo', 2.0),
        smear_hi=settings.get('smear_hi', 6.0), smear_it=settings.get('smear_it', 2.0),
    )
    if kspace_electrostatics == "ewald":
        logger.info("Tuning Ewald parameters...")
        params, errs, timings = tune_ewald(
            charges=charges, **common, **smearing, model_kwargs=model_kwargs,
            ns_lo=settings.get('ns_lo', 4), ns_hi=settings.get('ns_hi', 8),
        )
    elif kspace_electrostatics in ("pme", "p3m"):
        logger.info(f"Tuning {kspace_electrostatics.upper()} parameters...")
        params, errs, timings = tune_pme(
            charges=charges, **common, **smearing, model_kwargs=model_kwargs,
            nodes_lo=nodes_lo, nodes_hi=nodes_hi,
            mesh_lo=settings.get('mesh_lo', 4), mesh_hi=settings.get('mesh_hi', 6),
        )
    elif kspace_electrostatics == 'cutoff':
        logger.info("No k-space electrostatics tuning, only tuning the cutoff.")
        params, errs, timings = tune_sr(
            charges=charges, **common, cutoff_lo=settings.get('cutoff_lo', 10.0),
            exponent=1, model_kwargs=model_kwargs,
        )
    else:  # kspace_dispersion == 'cutoff'
        logger.info("No k-space dispersion tuning, only tuning the cutoff.")
        params, errs, timings = tune_dispersion_sr(
            hirshfeld_ratios=hirshfeld_ratios, c6_ratios=c6_ratios, **common,
            cutoff_lo=settings.get('cutoff_lo', 10.0),
            damp_lo=settings.get('dispersion_damping_lo', 2.0),
            damp_hi=settings.get('dispersion_damping_hi', 4.0),
            damp_it=settings.get('dispersion_damping_it', 1.0),
            exponent=1, model_kwargs=dispersion_model_kwargs,
        )

    recommended_param = None
    if any(err < accuracy for err in errs):
        param, timing, error = params[timings.index(min(timings))], min(
            timings), errs[timings.index(min(timings))]
        recommended_param = param  # fastest param that still meets the accuracy target
        if kspace_electrostatics == 'cutoff':
            logger.info(
                f"Optimally timed parameters for cutoff electrostatics with an estimated force error below {accuracy}eV/A:")
            logger.info(f"  lr_cutoff: {param['cutoff']}")
        elif kspace_electrostatics in ['ewald', 'pme', 'p3m']:
            logger.info(
                f"Optimally timed {kspace_electrostatics.upper()} parameters with an estimated force error below {accuracy}eV/A:")
            logger.info(f"  lr_cutoff: {param['cutoff']}")
            logger.info(f"  kspace_smearing: {param['smearing']}")
            logger.info(f"  kspace_spacing: {param['spacing']}")
            if 'interpolation_nodes' in param:
                logger.info(
                    f"  kspace_interp_nodes: {param['interpolation_nodes']}")
        elif kspace_dispersion == 'cutoff':
            logger.info(
                f"Optimally timed parameters for cutoff disperion with an estimated force error below {accuracy}eV/A:")
            logger.info(f"  lr_cutoff: {param['cutoff']}")
            logger.info(f"  dispersion_damping: {param['cutoff_lr_damping']}")
        logger.info(f"  timing: {timing}")
        logger.info(f"  force error estimate [eV/A]: {error}")

    else:
        warn("No parameter meets the accuracy requirement.\n", stacklevel=1)
    param, timing, error = params[errs.index(
        min(errs))], timings[errs.index(min(errs))], min(errs)
    logger.info("The parameters with the smallest error estimate are:")
    if kspace_electrostatics == 'cutoff':
        logger.info(f"  lr_cutoff: {param['cutoff']}")
    elif kspace_electrostatics in ['ewald', 'pme', 'p3m']:
        logger.info(f"Tuned {kspace_electrostatics.upper()} parameters: ")
        logger.info(f"  lr_cutoff: {param['cutoff']}")
        logger.info(f"  kspace_smearing: {param['smearing']}")
        logger.info(f"  kspace_spacing: {param['spacing']}")
        if 'interpolation_nodes' in param:
            logger.info(
                f"  kspace_interp_nodes: {param['interpolation_nodes']}")
    elif kspace_dispersion == 'cutoff':
        logger.info("Tuned cutoff dispersion parameters: ")
        logger.info(f"  lr_cutoff: {param['cutoff']}")
        logger.info(f"  dispersion_damping: {param['cutoff_lr_damping']}")
    logger.info(f"  timing: {timing}")
    logger.info(f"  force error estimate [eV/A]: {error}")

    # Print a ready-to-run MD command using the tuned parameters so users know
    # exactly how to launch a simulation next. Prefer the fastest param that met
    # the accuracy target; otherwise fall back to the smallest-error param.
    rec = recommended_param if recommended_param is not None else param
    if kspace_electrostatics in ('ewald', 'pme', 'p3m'):
        _interp = rec.get('interpolation_nodes')
        _interp_flag = f" --kspace-interp-nodes {int(_interp)}" if _interp is not None else ""
        logger.info("=" * 60)
        logger.info("To run MD with these k-space parameters, e.g. NPT (CLI):")
        logger.info(
            f"  so3lr npt --input {input_file_path} --lr-cutoff {rec['cutoff']} "
            f"--kspace-electrostatics {kspace_electrostatics} "
            f"--kspace-smearing {rec['smearing']} --kspace-spacing {rec['spacing']}{_interp_flag} "
            f"--temperature 300 --pressure 1.0"
        )
        logger.info("Or add the following to your settings.yaml:")
        logger.info(f"  lr_cutoff: {rec['cutoff']}")
        logger.info(f"  kspace_electrostatics: {kspace_electrostatics}")
        logger.info(f"  kspace_smearing: {rec['smearing']}")
        logger.info(f"  kspace_spacing: {rec['spacing']}")
        if _interp is not None:
            logger.info(f"  kspace_interp_nodes: {int(_interp)}")
        logger.info("=" * 60)
    elif kspace_electrostatics == 'cutoff' or kspace_dispersion == 'cutoff':
        logger.info("=" * 60)
        logger.info("To run MD with this cutoff, e.g. NVT (CLI):")
        logger.info(
            f"  so3lr nvt --input {input_file_path} --lr-cutoff {rec['cutoff']} --temperature 300"
        )
        logger.info("Or add the following to your settings.yaml:")
        logger.info(f"  lr_cutoff: {rec['cutoff']}")
        logger.info("=" * 60)
