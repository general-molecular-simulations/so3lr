"""
Tuning of the long-range parameters (Ewald, PME/P3M, or a plain cutoff) for SO3LR on one
structure: every candidate's force error is estimated, and the fastest candidate within
the accuracy is recommended.

The approach (analytical error estimates for every candidate, then timings of those
within the accuracy) follows the tuner of torch-pme, https://github.com/lab-cosmo/torch-pme
(Loche et al., J. Chem. Phys. 162 (2025), https://doi.org/10.1063/5.0251713).
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
from itertools import product
from functools import lru_cache, partial

from jax_md.space import distance, periodic_general, DisplacementFn
from jax_md.quantity import force

from so3lr.kspace_defaults import DEFAULT_KSPACE_INTERP_NODES, KSPACE_INTERP_NODES
from so3lr.mlff.utils.kspace_utils import setup_kspace_grid

from ase.io import read
from ase.units import Bohr

from so3lr.model_config import load_model_config
from so3lr.model_registry import resolve_model
from so3lr.pbc import lattice_rows, minimum_image_problem
from so3lr.cli.so3lr_md import load_model, atoms_to_jnp, process_model, dispersion_ratios
from .so3lr_md import run_dtypes, setup_logger, setup_run
from .presentation import get_reporter

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


@partial(jax.jit, static_argnames=('legacy',))
def dispersion_vdw_QDO_energy(hirshfeld_ratios, r, idx_i, idx_j,  atomic_numbers, sigma, cutoff_lr, cutoff_lr_damping, c6_ratios=None,
                              legacy=False):
    input_dtype = hirshfeld_ratios.dtype

    # Calculate alpha_ij and C6_ij using mixing rules
    alpha_ij, C6_ij = mixing_rules(
        atomic_numbers,
        idx_i,
        idx_j,
        hirshfeld_ratios,
        c6_ratios=c6_ratios,
        legacy=legacy,
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


def _inclusive(start, stop, step):
    """np.arange(start, stop + step, step): reaches stop when it lies on the step grid from
    start, and otherwise goes up to one step past it (step may be negative)."""
    return np.arange(start, stop + step, step)


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
    pairs = lru_cache(maxsize=None)(partial(filter_neighbor_idx, neighbor_indices=nbrs_lr,
                                            neighbor_distances=nbrs_lr_distances))

    def mean_time(cutoff, smearing=None, spacing=None, **_):
        k_grid, k_smearing = (setup_kspace_grid(kspace_electrostatics, smearing, spacing, cell)
                              if kspace_electrostatics else (None, None))
        neighbors = pairs(cutoff)

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

    Returns the candidates, their errors and their timings (inf where the error is too
    large), in the order of the candidates."""
    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
    logger.info(f"Testing {len(candidates)} parameter combinations.")
    logger.info("Force-error estimates:")
    reporter = get_reporter()
    reporter.phase('Estimating', 'Force-error estimate of each candidate')
    errors = []
    for candidate in candidates:
        logger.info(f"  {_describe(candidate)}")
        errors.append(float(error_fn(**candidate)))
        reporter.task_progress(len(errors), len(candidates), 'candidates')
    logger.info("Timings of the candidates within the accuracy:")
    reporter.phase('Timing', f'Force calls of the candidates within {accuracy:g} eV/Å')
    within = sum(error <= accuracy for error in errors)
    timings = []
    for candidate, error in zip(candidates, errors):
        timings.append(time_fn(**candidate) if error <= accuracy else float("inf"))
        if error <= accuracy:
            reporter.task_progress(sum(t != float("inf") for t in timings), within, 'candidates timed')
        logger.info(f"  {_describe(candidate)}: {timings[-1]} s per force call, "
                    f"estimated error {error}")
    return candidates, errors, timings


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
    *,
    model_kwargs: dict[str, Any],
) -> tuple[list[dict], list[float], list[float]]:
    """Ewald candidates (smearing, real-space cutoff, wavelength) with their estimated force
    errors and timings. The wavelengths divide the shortest cell side into ns_lo..ns_hi
    parts."""
    shortest = float(np.min(np.linalg.norm(cell, axis=1)))
    candidates = [
        {"smearing": smear, "cutoff": cut, "spacing": shortest / ns}
        for smear, cut, ns in product(
            _inclusive(smear_lo, smear_hi, smear_it),
            _inclusive(cutoff_hi, cutoff_lo, -cutoff_it),
            range(ns_hi, ns_lo - 1, -1),
        )
    ]
    bounds = EwaldErrorBounds_SR(charges=charges, cell=cell, positions=positions,
                                 displacement=displacement, nbrs_lr=nbrs_lr,
                                 nbrs_lr_distances=nbrs_lr_distances, **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn, "ewald")
    return grid_search(candidates, bounds, timer, accuracy)


class EwaldErrorBounds:
    """RMS force-error estimates (eV/A) for Ewald summation with Gaussian smearing sigma,
    i.e. splitting parameter alpha = 1 / (sqrt(2) sigma), from Kolafa and Perram,
    Mol. Simul. 9, 351 (1992), https://doi.org/10.1080/08927029208049126."""

    def __init__(self, charges: jnp.ndarray, cell: jnp.ndarray, positions: jnp.ndarray):
        self.prefactor = _charge_prefactor(charges, len(positions))
        self.volume = float(abs(jnp.linalg.det(cell)))

    def err_kspace(self, smearing: float, spacing: float) -> float:
        """The reciprocal sum kept to |k| <= 2 pi / spacing (LAMMPS ``Ewald::rms`` evaluates
        the same estimate). jax-pme's Ewald grid keeps |k| up to about pi / spacing along
        each axis, so for its grid this underestimates the error."""
        # Like every Coulomb force error, linear in the prefactor (q^2). torch-pme's tuner
        # takes its square root, which scales as |q| and underestimates the error 4-36x
        # for typical systems.
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


class _WithKernelError:
    """A k-space method's estimates plus the measured error of SO3LR's real-space kernel."""

    def __init__(self, charges, cell, positions, displacement: DisplacementFn, nbrs_lr,
                 nbrs_lr_distances, cutoff_sr: float = 4.5, *,
                 electrostatic_energy_scale: float):
        super().__init__(charges, cell, positions)
        # depends on (smearing, cutoff) only, so it is computed once for all spacings
        self.kernel_error = lru_cache(maxsize=None)(partial(
            _pme_kernel_error, positions, charges, displacement, nbrs_lr, nbrs_lr_distances,
            cutoff_sr=cutoff_sr, sigma=electrostatic_energy_scale))

    def __call__(self, smearing: float, spacing: float, cutoff: float, **order) -> float:
        return _combined(kspace=self.err_kspace(smearing, spacing, **order),
                         rspace_sr=self.kernel_error(smearing=smearing, cutoff=cutoff),
                         rspace_lr=self.err_rspace(smearing, cutoff))


class EwaldErrorBounds_SR(_WithKernelError, EwaldErrorBounds):
    """Ewald's estimates plus the measured error of SO3LR's real-space kernel."""


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
    nodes_lo: int = DEFAULT_KSPACE_INTERP_NODES,
    nodes_hi: int = DEFAULT_KSPACE_INTERP_NODES,
    mesh_lo: int = 2,
    mesh_hi: int = 6,
    accuracy: float = 2e-2,
    *,
    model_kwargs: dict[str, Any],
    kspace_electrostatics: str = "pme",
) -> tuple[list[dict], list[float], list[float]]:
    """PME/P3M candidates (smearing, real-space cutoff, interpolation order, mesh spacing)
    with their estimated force errors and timings. The spacings are 2 L / (2**m - 1),
    m = mesh_lo..mesh_hi, L the shortest cell side."""
    shortest = float(np.min(jnp.linalg.norm(cell, axis=1)))
    candidates = [
        {
            "smearing": smear,
            "cutoff": cut,
            "interpolation_nodes": nodes,
            "spacing": 2 * shortest / (2**m - 1),
        }
        for smear, cut, nodes, m in product(
            _inclusive(smear_lo, smear_hi, smear_it),
            _inclusive(cutoff_hi, cutoff_lo, -cutoff_it),
            range(nodes_hi, nodes_lo - 1, -1),
            range(mesh_hi, mesh_lo - 1, -1),
        )
    ]
    bounds = PMEErrorBounds_SR(charges=charges, cell=cell, positions=positions,
                               displacement=displacement, nbrs_lr=nbrs_lr,
                               nbrs_lr_distances=nbrs_lr_distances, **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn,
                             kspace_electrostatics)
    return grid_search(candidates, bounds, timer, accuracy)


class PMEErrorBounds(EwaldErrorBounds):
    """RMS force-error estimates (eV/A) for PME: the reciprocal part from Petersen,
    J. Chem. Phys. 103, 3668 (1995), https://doi.org/10.1063/1.470043, for interpolation
    order p and mesh spacing h; the real-space part as for Ewald."""

    def __init__(self, charges: jnp.ndarray, cell: jnp.ndarray, positions: jnp.ndarray):
        super().__init__(charges, cell, positions)
        self.sides = np.asarray(jnp.linalg.norm(cell, axis=1))

    def err_kspace(self, smearing: float, spacing: float, interpolation_nodes: int) -> float:
        """h is the geometric mean of L / (2 L / spacing + 1) over the sides L. jax-pme
        pads its mesh to a power of two, so the mesh it builds is at least as fine and
        the estimate errs high."""
        p = int(interpolation_nodes)
        if p not in PETERSEN_CONSTANTS:
            raise ValueError(f"PME error estimates cover interpolation orders "
                             f"{sorted(PETERSEN_CONSTANTS)}, not {p}")
        alpha = _alpha(smearing)
        h = float(np.prod(self.sides / (2 * self.sides / spacing + 1)) ** (1 / 3))
        return (self.prefactor * math.pi ** 0.25 * math.sqrt(6 * alpha / (2 * p + 1))
                / self.volume ** (2 / 3) * (2 * alpha * h) ** p / math.factorial(p)
                * (p / (2 * math.e)) ** (p / 2) * PETERSEN_CONSTANTS[p])


class PMEErrorBounds_SR(_WithKernelError, PMEErrorBounds):
    """PME's estimates plus the measured error of SO3LR's real-space kernel."""


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
    accuracy: float = 1.0,
    *,
    model_kwargs: dict[str, Any],
) -> tuple[list[dict], list[float], list[float]]:
    """Cutoff electrostatics (no k-space): candidate cutoffs with their measured force
    errors and timings."""
    candidates = [{"cutoff": cut} for cut in _inclusive(cutoff_hi, cutoff_lo, -cutoff_it)]
    bounds = NativeErrorBounds(charges=charges, positions=positions, displacement=displacement,
                               nbrs_lr=nbrs_lr, nbrs_lr_distances=nbrs_lr_distances,
                               **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn)
    return grid_search(candidates, bounds, timer, accuracy)


class NativeErrorBounds:
    """Force errors of cutoff electrostatics, measured on the structure: SO3LR's
    shifted-force kernel against the erf-damped Coulomb sum within the cutoff (rspace_sr),
    and that sum against the one over every long-range pair (rspace_lr)."""

    def __init__(self, charges, positions, displacement: DisplacementFn, nbrs_lr,
                 nbrs_lr_distances, cutoff_sr: float = 4.5, *,
                 electrostatic_energy_scale: float):
        # cutoff_sr comes with the other bounds' arguments and is unused: this kernel
        # switches on at 0.45 * cutoff
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
    accuracy: float = 0.001,
    *,
    model_kwargs: dict[str, Any],
    c6_ratios: jnp.ndarray = None,
) -> tuple[list[dict], list[float], list[float]]:
    """Cutoff dispersion: candidate cutoffs and damping widths with their measured force
    errors and timings."""
    candidates = [
        {"cutoff": cut, "cutoff_lr_damping": damp}
        for cut, damp in product(
            _inclusive(cutoff_hi, cutoff_lo, -cutoff_it),
            _inclusive(damp_lo, damp_hi, damp_it),
        )
    ]
    bounds = NativeErrorBounds_Dispersion(hirshfeld_ratios=hirshfeld_ratios,
                                          c6_ratios=c6_ratios, positions=positions,
                                          displacement=displacement, nbrs_lr=nbrs_lr,
                                          nbrs_lr_distances=nbrs_lr_distances, **model_kwargs)
    timer = force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn)
    return grid_search(candidates, bounds, timer, accuracy)


class NativeErrorBounds_Dispersion:
    """Force errors of cutoff dispersion, measured on the structure: the switched-off
    dispersion against the plainly truncated one (rspace_sr), and that against the sum
    over every long-range pair (rspace_lr)."""

    def __init__(self, hirshfeld_ratios, positions, displacement: DisplacementFn,
                 nbrs_lr, nbrs_lr_distances, atomic_numbers,
                 *, dispersion_energy_scale: float, c6_ratios: jnp.ndarray = None,
                 legacy: bool = False):
        self.positions, self.hirshfeld_ratios = positions, hirshfeld_ratios
        self.displacement = displacement
        self.nbrs_lr, self.nbrs_lr_distances = nbrs_lr, nbrs_lr_distances
        self.energy = partial(dispersion_vdw_QDO_energy, atomic_numbers=atomic_numbers,
                              sigma=dispersion_energy_scale, c6_ratios=c6_ratios,
                              legacy=legacy)
        self.reference = self._forces(nbrs_lr, cutoff_lr=None, cutoff_lr_damping=None)
        # depends on the cutoff only, so it is computed once for all damping widths
        self.truncated = lru_cache(maxsize=None)(self._truncated)

    def _forces(self, nbrs, **kwargs):
        return get_forces_rs(partial(self.energy, **kwargs), self.positions,
                             self.hirshfeld_ratios, self.displacement,
                             idx_i=nbrs[0], idx_j=nbrs[1])

    def _truncated(self, cutoff):
        nbrs = filter_neighbor_idx(cutoff, self.nbrs_lr, self.nbrs_lr_distances)
        return nbrs, self._forces(nbrs, cutoff_lr=cutoff, cutoff_lr_damping=None)

    def __call__(self, cutoff: float, cutoff_lr_damping: float) -> float:
        nbrs, truncated = self.truncated(cutoff)
        switched = self._forces(nbrs, cutoff_lr=cutoff, cutoff_lr_damping=cutoff_lr_damping)
        return _combined(rspace_sr=_rms_difference(switched, truncated),
                         rspace_lr=_rms_difference(truncated, self.reference))


def model_pair_settings(model_path) -> dict:
    """The settings of the model's own pair functions, which the error estimates evaluate
    (model_path is a name, a path, or None for the default): the electrostatic and
    dispersion widths, and whether the dispersion uses the legacy so3lr's free-atom
    tables."""
    model = load_model_config(resolve_model(model_path).dir)['model']
    return dict(electrostatic_energy_scale=float(model['electrostatic_energy_scale']),
                dispersion_energy_scale=float(model['dispersion_energy_scale']),
                legacy=model.get('legacy_so3lr_bool', False) is True)


def cell_is_present(cell) -> bool:
    """True if the geometry carries a usable (not all-zero) cell.

    The predecessor `not np.all(cell != 0)` was inverted: it PASSED for an
    all-zero (missing) cell and RAISED for a fully dense triclinic cell. Only
    orthorhombic cells, whose off-diagonal zeros make `np.all(cell != 0)`
    False, behaved correctly -- which is why it went unnoticed. The test is
    simply "the cell is not all zeros".
    """
    return cell is not None and not np.all(np.asarray(cell) == 0.0)


def _report_recommendation(rec, error, timing, accuracy, *, met, method, command):
    """The chosen parameters as the live display's result table (settings-file names)."""
    names = (('cutoff', 'lr_cutoff', ' Å'), ('smearing', 'kspace_smearing', ' Å'),
             ('spacing', 'kspace_spacing', ' Å'), ('interpolation_nodes', 'kspace_interp_nodes', ''),
             ('cutoff_lr_damping', 'dispersion_damping', ' Å'))
    rows = [(name, f'{rec[key]:g}{unit}') for key, name, unit in names if rec.get(key) is not None]
    if method in ('ewald', 'pme', 'p3m'):
        rows.insert(1, ('kspace_electrostatics', method))
    rows.append(('Force error', f'{error:.3g} eV/Å estimated · target ≤ {accuracy:g}'))
    if timing != float('inf'):
        rows.append(('Force call', f'{timing * 1e3:.3g} ms'))
    if command:
        rows.append(('Run MD', command))
    title = ('Fastest settings within the target' if met
             else 'No candidate meets the target: smallest error')
    reporter = get_reporter()
    reporter.results(title, rows)
    if met:
        reporter.outcome(f'force error {error:.2g} ≤ {accuracy:g} eV/Å')
    else:
        reporter.outcome(f'smallest force error {error:.2g} eV/Å',
                         warning=f'no candidate meets the {accuracy:g} eV/Å target')


def tune(
    settings: dict
) -> None:
    """
    Tune Ewald parameters for running SO3LR model on a given structure.

    Args:
        settings (dict): Dictionary containing the settings for the tuning process.
    """

    # Setup logger (first: setup_run may warn of a deprecated setting)
    log_file = settings.get('log_file')
    setup_logger(log_file)
    # the run's precision and JAX's globals (x64, matmul precision), as on every entry point
    settings = setup_run(dict(settings))

    # Model parameters
    model_path = settings.get('model_path')
    model_dtype, state_dtype = run_dtypes(settings)
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
    # only orders jax-pme's solver implements and the PME error estimate covers; any
    # other fails in one of them, after the model is built
    nodes_lo = settings.get('nodes_lo', DEFAULT_KSPACE_INTERP_NODES)
    nodes_hi = settings.get('nodes_hi', DEFAULT_KSPACE_INTERP_NODES)
    if kspace_electrostatics in KSPACE_INTERP_NODES:
        supported = sorted(set(KSPACE_INTERP_NODES[kspace_electrostatics])
                           & set(PETERSEN_CONSTANTS))
        if not (nodes_lo <= nodes_hi and set(range(nodes_lo, nodes_hi + 1)) <= set(supported)):
            raise ValueError(
                f"tune-ewald tunes {kspace_electrostatics.upper()} with "
                f"{', '.join(map(str, supported))} interpolation nodes "
                f"(received nodes_lo={nodes_lo}, nodes_hi={nodes_hi}).")
        # the solver is built for one order, so the timings are for nodes_lo; the error
        # estimates cover every order in the grid
        if nodes_hi != nodes_lo:
            logger.warning(f'Timings use {nodes_lo} interpolation nodes for every candidate '
                           f'(nodes_lo={nodes_lo}, nodes_hi={nodes_hi}).')

    # Tuning parameters
    # Note that all the tuning parameters are loaded further below

    reporter = get_reporter()
    reporter.phase('Loading', 'Reading the structure')
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

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=state_dtype)

    # We don't want fractional coordinates here

    box = jnp.array(cell, dtype=state_dtype)
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

    reporter.phase('Loading', 'Loading the force field')
    # A bundled name, a path, or None for the default model; load_model builds
    # the solver for the method being timed.
    potential = load_model(
        model_path,
        precision=model_dtype,
        lr_cutoff=lr_cutoff,
        dispersion_damping=dispersion_damping,
        kspace_electrostatics=kspace_electrostatics if kspace_electrostatics != 'cutoff' else None,
        kspace_interp_nodes=nodes_lo,
        output_intermediate_quantities=['partial_charges'],
    )

    pair = model_pair_settings(model_path)
    model_kwargs = dict(
        cutoff_sr=potential.cutoff,
        electrostatic_energy_scale=pair['electrostatic_energy_scale'],
    )

    dispersion_model_kwargs = dict(
        dispersion_energy_scale=pair['dispersion_energy_scale'],
        legacy=pair['legacy'],
        atomic_numbers=initial_geometry.get_atomic_numbers()
    )

    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = process_model(
        potential=potential,
        species=initial_geometry_dict['species'],
        displacement=displacement,
        box=box,
        total_charge=total_charge,
        precision=model_dtype,
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
            kspace_electrostatics=kspace_electrostatics,
        )
    elif kspace_electrostatics == 'cutoff':
        logger.info("No k-space electrostatics tuning, only tuning the cutoff.")
        params, errs, timings = tune_sr(
            charges=charges, **common, cutoff_lo=settings.get('cutoff_lo', 10.0),
            model_kwargs=model_kwargs,
        )
    else:  # kspace_dispersion == 'cutoff'
        logger.info("No k-space dispersion tuning, only tuning the cutoff.")
        params, errs, timings = tune_dispersion_sr(
            hirshfeld_ratios=hirshfeld_ratios, c6_ratios=c6_ratios, **common,
            cutoff_lo=settings.get('cutoff_lo', 10.0),
            damp_lo=settings.get('dispersion_damping_lo', 2.0),
            damp_hi=settings.get('dispersion_damping_hi', 4.0),
            damp_it=settings.get('dispersion_damping_it', 1.0),
            model_kwargs=dispersion_model_kwargs,
        )

    recommended_param = None
    if any(err <= accuracy for err in errs):
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
        logger.warning(f"No parameter meets the accuracy requirement ({accuracy:g} eV/Å).")
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
    command = None
    if kspace_electrostatics in ('ewald', 'pme', 'p3m'):
        _interp = rec.get('interpolation_nodes')
        _interp_flag = f" --kspace-interp-nodes {int(_interp)}" if _interp is not None else ""
        logger.info("=" * 60)
        logger.info("To run MD with these k-space parameters, e.g. NPT (CLI):")
        command = (f"so3lr npt --input {input_file_path} --lr-cutoff {rec['cutoff']:g} "
                   f"--kspace-electrostatics {kspace_electrostatics} "
                   f"--kspace-smearing {rec['smearing']:g} --kspace-spacing {rec['spacing']:g}{_interp_flag} "
                   f"--temperature 300 --pressure 1.0")
        logger.info(f"  {command}")
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
        command = f"so3lr nvt --input {input_file_path} --lr-cutoff {rec['cutoff']:g} --temperature 300"
        logger.info(f"  {command}")
        logger.info("Or add the following to your settings.yaml:")
        logger.info(f"  lr_cutoff: {rec['cutoff']}")
        logger.info("=" * 60)
    _report_recommendation(rec, errs[params.index(rec)], timings[params.index(rec)], accuracy,
                           met=recommended_param is not None, command=command,
                           method=kspace_electrostatics if kspace_dispersion != 'cutoff' else 'dispersion')
