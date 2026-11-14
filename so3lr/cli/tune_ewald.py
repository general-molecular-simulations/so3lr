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
import gc
import time
import logging
from types import SimpleNamespace
from typing import Any
from warnings import warn
from itertools import product
from functools import lru_cache, partial

from jax_md.space import distance, periodic_general, DisplacementFn
from jax_md.quantity import force

from so3lr.kspace_defaults import DEFAULT_KSPACE_INTERP_NODES, KSPACE_INTERP_NODES
from so3lr.mlff.utils.kspace_utils import kspace_grid_shape, setup_kspace_grid

from ase.io import read
from ase.units import Bohr

from so3lr.model_config import load_model_config
from so3lr.model_registry import resolve_model
from so3lr.pbc import lattice_rows, minimum_image_problem
from so3lr.cli.so3lr_md import load_model, atoms_to_jnp, process_model, dispersion_ratios, run_on_flash
from .so3lr_md import run_dtypes, setup_logger, setup_run

# Setup logging
logger = logging.getLogger("SO3LR")


def filter_neighbor_idx(
    cutoff: float, neighbor_indices: jnp.array, neighbor_distances: jnp.array
):
    filter_idx = jnp.where((neighbor_distances < cutoff) & (
        neighbor_distances > 0.0) & (~jnp.isnan(neighbor_distances)))[0]
    return neighbor_indices[:, filter_idx]


def get_forces_rs(energy, positions, q, displacement, **params):
    """Forces on every atom from the pair energy over the pairs params['idx_i'],
    params['idx_j'] (the dispersion estimates)."""
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


# the atoms the measured real-space errors are averaged over: all of them up to this many,
# else a random sample of this many (an RMS over 2048 atoms is within a few per cent of
# the full one, and the pairs of every atom do not fit a GPU at tens of thousands of atoms)
ERROR_SAMPLE_ATOMS = 2048


class SamplePairs:
    """The pairs within cutoff of a sample of the atoms (every atom up to n_sample): for
    each sampled atom i (by its place in the sample, seg) every partner j, both by their
    index in the structure (idx_i, idx_j), with distance r and unit vector r_i - r_j."""

    def __init__(self, positions, displacement, cutoff, n_sample=ERROR_SAMPLE_ATOMS, seed=0, chunk=256):
        n = len(positions)
        self.sample = (np.arange(n) if n <= n_sample
                       else np.sort(np.random.default_rng(seed).choice(n, n_sample, replace=False)))
        disp = jax.vmap(jax.vmap(displacement, (None, 0)), (0, None))

        @jax.jit
        def rows_to_all(rows, positions):
            dvec = disp(positions[rows], positions)
            r = jnp.sqrt(jnp.sum(dvec ** 2, axis=-1))
            return dvec, (r < cutoff) & (r > 0)
        seg, idx_j, vec = [], [], []
        for start in range(0, len(self.sample), chunk):
            rows = self.sample[start:start + chunk]
            # padded to the chunk, so every chunk runs one compiled program
            dvec, near = rows_to_all(np.concatenate([rows, np.repeat(rows[-1:], chunk - len(rows))]), positions)
            # only the mask and the pairs within the cutoff come to the host
            a, b = np.nonzero(np.asarray(near)[:len(rows)])
            seg.append(start + a)
            idx_j.append(b)
            vec.append(np.asarray(dvec[a, b]))
        self.seg = jnp.asarray(np.concatenate(seg))
        self.idx_i = jnp.asarray(self.sample[np.concatenate(seg)])
        self.idx_j = jnp.asarray(np.concatenate(idx_j))
        vec = np.concatenate(vec)
        r = np.linalg.norm(vec, axis=-1)
        self.r = jnp.asarray(r, positions.dtype)
        self.unit = jnp.asarray(vec / r[:, None], positions.dtype)
        self.n = len(self.sample)

    def forces(self, energy, q, r_max, **params):
        """Forces on the sampled atoms from the pair energy over the pairs closer than r_max."""
        return _pair_forces(energy, jnp.asarray(q, self.r.dtype).reshape(-1), self.r, self.unit,
                            self.idx_i, self.idx_j, self.seg, self.n, r_max, params)


@partial(jax.jit, static_argnames=("energy", "n"))
def _pair_forces(energy, q, r, unit, idx_i, idx_j, seg, n, r_max, params):
    """Forces on n atoms from a pair energy with so3lr's sparse convention (each ordered
    pair carries half of the pair's energy, so F_i = -2 de/dr (r_i - r_j)/r), the pairs
    from r_max on left out. One program per energy, pair count and set of parameters given
    as None: r_max and the kernel's other parameters are traced (#42 item 7)."""
    def total(r):
        return jnp.sum(jnp.where(r < r_max, energy(q, r, idx_i, idx_j, **params), 0.0))
    de = jax.grad(total)(r)
    return segment_sum(-2.0 * de[:, None] * unit, seg, num_segments=n)


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
    """start, start + step, ... up to stop (step may be negative), never past it (#42 item 4:
    np.arange(start, stop + step, step) went one step beyond a stop off the step grid)."""
    values = np.arange(start, stop + step, step)
    lo, hi = min(start, stop), max(start, stop)
    return values[(values >= lo - 1e-9) & (values <= hi + 1e-9)]


def _rms_difference(forces, reference):
    """Root mean square over atoms of the force deviation."""
    return float(jnp.sqrt(jnp.sum(jnp.square(forces - reference)) / len(reference)))


def _combined(**parts):
    """Add independent error contributions in quadrature, logging each."""
    total = math.sqrt(sum(float(v) ** 2 for v in parts.values()))
    logger.info("Error estimates: "
                + ", ".join(f"{k}={float(v):.4g}" for k, v in parts.items())
                + f", total={total:.4g}")
    return total


def _describe(candidate):
    return ", ".join(f"{k}: {v}" for k, v in candidate.items() if v is not None)


def _timed(call, n_warmup, n_repeat):
    """The median wall time (s) of call() over n_repeat calls, after n_warmup untimed ones."""
    for _ in range(n_warmup):
        call()
    samples = []
    for _ in range(n_repeat):
        start = time.monotonic()
        call()
        samples.append(time.monotonic() - start)
    return float(np.median(samples))


def force_call_timer(cell, positions, nbrs_lr, nbrs_lr_distances, force_fn,
                     kspace_electrostatics=None, n_warmup=4, n_repeat=20, seed=42):
    """A function of one candidate's parameters returning the wall time (s) of a
    neighbour-list force call with them. Timed once per cutoff and k-space grid shape:
    the smearing and the spacing within one grid shape change no work (#42 item 8), so
    such candidates share a timing and the smaller error wins the tie."""
    # one small displacement, reused, so every call does the same work
    perturbed = positions + 1e-3 * jax.random.normal(jax.random.key(seed), positions.shape)
    pairs = lru_cache(maxsize=None)(partial(filter_neighbor_idx, neighbor_indices=nbrs_lr,
                                            neighbor_distances=nbrs_lr_distances))
    times = {}

    def mean_time(cutoff, smearing=None, spacing=None, **_):
        key = (float(cutoff), kspace_grid_shape(kspace_electrostatics, spacing, cell))
        if key not in times:
            k_grid, k_smearing = (setup_kspace_grid(kspace_electrostatics, smearing, spacing, cell)
                                  if kspace_electrostatics else (None, None))
            neighbors = pairs(cutoff)
            # block, or asynchronous dispatch would time the launch instead of the work
            times[key] = _timed(lambda: jax.block_until_ready(force_fn(
                perturbed, neighbor_lr=neighbors, k_grid=k_grid, k_smearing=k_smearing)), n_warmup, n_repeat)
        return times[key]

    return mean_time


class FlashTimer:
    """time_fn of one PME candidate's parameters: the wall time (s) of a flash force call
    with them (``FlashSO3LR.energy_forces``, as MD calls it). Each cutoff and PME mesh shape
    is an engine of its own (the cutoff and smearing are in its tables, the mesh in its
    programs) and is built, compiled and timed once: candidates that differ only in
    smearing, or in spacing within one mesh shape, do the same work. ``charges`` takes the
    model's charges from the engine of a candidate it times as well."""

    def __init__(self, atoms, settings, build_engine, n_warmup=2, n_repeat=5):
        self.atoms, self.settings, self.build_engine = atoms, settings, build_engine
        self.n_warmup, self.n_repeat = n_warmup, n_repeat
        self.dtype, position_dtype = run_dtypes(settings)
        self.L = np.diag(np.asarray(atoms.cell[:]))
        self.R = jnp.asarray(np.mod(atoms.positions, self.L), position_dtype)
        self.times = {}

    def charges(self, cutoff, smearing, spacing):
        """The model's charges (they come from the short-range model: no long-range setting
        changes them), from this candidate's engine, which is timed while it exists."""
        return self._measure(cutoff, smearing, spacing, charges=True)

    def __call__(self, cutoff, smearing, spacing, **_):
        key = self._key(cutoff, spacing)
        if key not in self.times:
            self._measure(cutoff, smearing, spacing)
        return self.times[key]

    def _key(self, cutoff, spacing):
        from so3lr.flash.engine import pme_mesh_shape
        return float(cutoff), pme_mesh_shape(self.L, float(spacing))

    def _measure(self, cutoff, smearing, spacing, charges=False):
        gc.collect()                     # the last engine's arrays, before the next one's
        engine, staged = self.build_engine(
            self.atoms, dict(self.settings, lr_cutoff=float(cutoff), kspace_smearing=float(smearing),
                             kspace_spacing=float(spacing)), self.dtype, box_margin=0.0)
        q = jnp.asarray(engine.observables(self.R)[0]) if charges else None
        energy_forces = engine.energy_forces_staged if staged else engine.energy_forces
        key = self._key(cutoff, spacing)
        self.times[key] = _timed(lambda f=energy_forces: jax.block_until_ready(f(self.R)),
                                 self.n_warmup, self.n_repeat)
        logger.info(f"  flash, lr_cutoff {key[0]}, PME mesh {key[1]}: {self.times[key]:.4g} s per force call")
        del engine, energy_forces
        gc.collect()                     # the engine's programs hold it in a cycle
        return q


def grid_search(candidates, error_fn, time_fn, accuracy):
    """Estimate every candidate's force error, then time those within the accuracy.

    Returns the candidates, their errors and their timings (inf where the error is too
    large), in the order of the candidates."""
    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
    logger.info(f"Testing {len(candidates)} parameter combinations.")
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
    return candidates, errors, timings


# timings this close to the fastest count as equally fast (the median of a few calls varies
# by about 1 %): on flash the PME mesh costs next to nothing beside the model, so without a
# tolerance timing noise picks a candidate just inside the accuracy over one with a few
# times less error at the same cost (32,512-atom POPC: 0.0764 vs 0.0764 s, 9.7e-4 vs 3.8e-4)
TIMING_TOLERANCE = 0.03


def recommended_index(errors, timings, accuracy, tolerance=TIMING_TOLERANCE):
    """Of the candidates within the accuracy, those within tolerance of the fastest's
    timing, the one with the smallest error; None if none is within the accuracy."""
    within = [i for i, e in enumerate(errors) if e <= accuracy]
    if not within:
        return None
    fastest = min(timings[i] for i in within)
    return min((i for i in within if timings[i] <= fastest * (1 + tolerance)),
               key=lambda i: (errors[i], timings[i]))


def tune_ewald(
    charges: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    pairs: "SamplePairs",
    time_fn: callable,
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
    bounds = EwaldErrorBounds_SR(charges=charges, cell=cell, positions=positions, pairs=pairs,
                                 **model_kwargs)
    return grid_search(candidates, bounds, time_fn, accuracy)


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


def _pme_kernel_error(pairs, charges, *, smearing, cutoff, cutoff_sr, sigma):
    """RMS force error of SO3LR's real-space PME kernel (short-range corrected, shifted
    force) against the erf-screened Coulomb pair sum, measured on the sampled atoms."""
    common = dict(ke=COULOMB_CONSTANT, sigma=sigma)
    exact = pairs.forces(coulomb_erf, charges, cutoff, cutoff=smearing, **common)
    kernel = pairs.forces(coulomb_erf_shifted_force_smooth_pme, charges, cutoff,
                          cuton=cutoff_sr, cutoff=cutoff, smearing=smearing, **common)
    return _rms_difference(kernel, exact)


class _WithKernelError:
    """A k-space method's estimates plus the measured error of SO3LR's real-space kernel."""

    def __init__(self, charges, cell, positions, pairs: "SamplePairs", cutoff_sr: float = 4.5, *,
                 electrostatic_energy_scale: float):
        super().__init__(charges, cell, positions)
        # depends on (smearing, cutoff) only, so it is computed once for all spacings
        self.kernel_error = lru_cache(maxsize=None)(partial(
            _pme_kernel_error, pairs, charges, cutoff_sr=cutoff_sr, sigma=electrostatic_energy_scale))

    def __call__(self, smearing: float, spacing: float, cutoff: float, **order) -> float:
        return _combined(kspace=self.err_kspace(smearing, spacing, **order),
                         rspace_sr=self.kernel_error(smearing=float(smearing), cutoff=float(cutoff)),
                         rspace_lr=self.err_rspace(smearing, cutoff))


class EwaldErrorBounds_SR(_WithKernelError, EwaldErrorBounds):
    """Ewald's estimates plus the measured error of SO3LR's real-space kernel."""


def pme_mesh_spacing(side: float, m: int) -> float:
    """A mesh spacing for which jax-pme builds 2**m points along a side: it takes
    ceil(side / spacing) wavelengths and 2 ceil(...) + 1 points, padded to a power of two
    (kspace.get_kgrid_mesh_shape). Half a point below the edge, so rounding cannot tip it
    to the next power."""
    if m < 2:
        raise ValueError(f"PME meshes have at least 4 points per side (2**m, m >= 2), not 2**{m}")
    return side / (2 ** (m - 1) - 1.5)


def tune_pme(
    charges: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    pairs: "SamplePairs",
    time_fn: callable,
    cutoff_lo: float = 7.0,
    cutoff_hi: float = 15.0,
    cutoff_it: float = 2.0,
    smear_lo: float = 2.0,
    smear_hi: float = 6.0,
    smear_it: float = 2.0,
    nodes_lo: int = DEFAULT_KSPACE_INTERP_NODES,
    nodes_hi: int = DEFAULT_KSPACE_INTERP_NODES,
    mesh_lo: int = 5,
    mesh_hi: int = 7,
    accuracy: float = 2e-2,
    *,
    model_kwargs: dict[str, Any],
) -> tuple[list[dict], list[float], list[float]]:
    """PME/P3M candidates (smearing, real-space cutoff, interpolation order, mesh spacing)
    with their estimated force errors and timings. The spacings give 2**m mesh points
    along the shortest cell side, m = mesh_lo..mesh_hi (longer sides get as many or more)."""
    shortest = float(np.min(jnp.linalg.norm(cell, axis=1)))
    # m means 2**m points here; before 2026-11 it meant 2**(m+1) (spacing 2 L / (2**m - 1))
    logger.info(f"PME meshes tried along the shortest side ({shortest:.2f} A): "
                + ", ".join(str(2 ** m) for m in range(mesh_lo, mesh_hi + 1))
                + f" points (mesh_lo {mesh_lo} .. mesh_hi {mesh_hi}, 2**m each)")
    candidates = [
        {
            "smearing": smear,
            "cutoff": cut,
            "interpolation_nodes": nodes,
            "spacing": pme_mesh_spacing(shortest, m),
        }
        for smear, cut, nodes, m in product(
            _inclusive(smear_lo, smear_hi, smear_it),
            _inclusive(cutoff_hi, cutoff_lo, -cutoff_it),
            range(nodes_hi, nodes_lo - 1, -1),
            range(mesh_hi, mesh_lo - 1, -1),
        )
    ]
    bounds = PMEErrorBounds_SR(charges=charges, cell=cell, positions=positions, pairs=pairs,
                               **model_kwargs)
    return grid_search(candidates, bounds, time_fn, accuracy)


class PMEErrorBounds(EwaldErrorBounds):
    """RMS force-error estimates (eV/A) for PME: the reciprocal part from Petersen,
    J. Chem. Phys. 103, 3668 (1995), https://doi.org/10.1063/1.470043, for interpolation
    order p and mesh spacing h; the real-space part as for Ewald."""

    def __init__(self, charges: jnp.ndarray, cell: jnp.ndarray, positions: jnp.ndarray):
        super().__init__(charges, cell, positions)
        self.cell = np.asarray(cell)
        self.sides = np.linalg.norm(self.cell, axis=1)

    def err_kspace(self, smearing: float, spacing: float, interpolation_nodes: int) -> float:
        """h is the geometric mean of side / points over the mesh jax-pme builds for this
        spacing (padded to a power of two per side: up to twice as fine as the spacing,
        #42 item 2)."""
        p = int(interpolation_nodes)
        if p not in PETERSEN_CONSTANTS:
            raise ValueError(f"PME error estimates cover interpolation orders "
                             f"{sorted(PETERSEN_CONSTANTS)}, not {p}")
        alpha = _alpha(smearing)
        mesh = np.asarray(kspace_grid_shape("pme", spacing, self.cell), np.float64)
        h = float(np.prod(self.sides / mesh) ** (1 / 3))
        return (self.prefactor * math.pi ** 0.25 * math.sqrt(6 * alpha / (2 * p + 1))
                / self.volume ** (2 / 3) * (2 * alpha * h) ** p / math.factorial(p)
                * (p / (2 * math.e)) ** (p / 2) * PETERSEN_CONSTANTS[p])


class PMEErrorBounds_SR(_WithKernelError, PMEErrorBounds):
    """PME's estimates plus the measured error of SO3LR's real-space kernel."""


def tune_sr(
    charges: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    pairs: "SamplePairs",
    time_fn: callable,
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
    bounds = NativeErrorBounds(charges=charges, pairs=pairs, **model_kwargs)
    return grid_search(candidates, bounds, time_fn, accuracy)


class NativeErrorBounds:
    """Force errors of cutoff electrostatics, measured on the sampled atoms: SO3LR's
    shifted-force kernel against the erf-damped Coulomb sum within the cutoff (rspace_sr),
    and that sum against the one over every pair the sample has (rspace_lr: up to the
    largest cutoff tried)."""

    def __init__(self, charges, pairs: "SamplePairs", cutoff_sr: float = 4.5, *,
                 electrostatic_energy_scale: float):
        # cutoff_sr comes with the other bounds' arguments and is unused: this kernel
        # switches on at 0.45 * cutoff
        self.charges, self.pairs = charges, pairs
        self.common = dict(ke=COULOMB_CONSTANT, sigma=electrostatic_energy_scale)
        self.reference = pairs.forces(coulomb_erf, charges, math.inf, cutoff=None, **self.common)

    def __call__(self, cutoff: float) -> float:
        cutoff = float(cutoff)
        truncated = self.pairs.forces(coulomb_erf, self.charges, cutoff, cutoff=None, **self.common)
        kernel = self.pairs.forces(coulomb_erf_shifted_force_smooth, self.charges, cutoff,
                                   cuton=0.45 * cutoff, cutoff=cutoff, **self.common)
        return _combined(rspace_sr=_rms_difference(kernel, truncated),
                         rspace_lr=_rms_difference(truncated, self.reference))


def tune_dispersion_sr(
    hirshfeld_ratios: jnp.ndarray,
    cell: jnp.ndarray,
    positions: jnp.ndarray,
    displacement: DisplacementFn,
    nbrs_lr: jnp.ndarray,
    nbrs_lr_distances: jnp.ndarray,
    time_fn: callable,
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
    return grid_search(candidates, bounds, time_fn, accuracy)


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


def _flash_setup(settings, atoms, cutoff, smearing, spacing):
    """(charges, timer) on the flash engine where ``run_on_flash`` takes the tune task (PME
    on a structure flash runs; engine auto or flash), else None: the neighbour-list engine
    tunes. The charges come from the engine of the candidate (cutoff, smearing, spacing),
    which is timed with them, and no neighbour list's per-edge arrays are built (they do not
    fit a GPU at tens of thousands of atoms)."""
    # the charge and spin the neighbour-list path gives the model, so both tune the same
    # charges (neither takes the charge-embedding overrides: what is tuned is grid
    # convergence); one GPU (the charges take the staged forward); the model's own forces,
    # as single points'
    flash_settings = dict(settings, total_charge=float(settings.get('total_charge') or 0),
                          num_unpaired_electrons=float(settings.get('num_unpaired_electrons') or 0),
                          charge_embed_total_charge=None, charge_equilibration_total_charge=None,
                          flash_devices=1, kspace_remove_net_force=False)

    def start(flash_md):
        timer = FlashTimer(atoms, flash_settings, flash_md.build_engine)
        try:
            return timer.charges(cutoff, smearing, spacing), timer
        except Exception as exc:
            raise flash_md.FlashUnavailable(f"{type(exc).__name__}: {exc}") from exc

    ran, result = run_on_flash(dict(flash_settings, lr_cutoff=float(cutoff)), atoms, 'tune', start)
    return result if ran else None


def _neighbour_list_setup(settings, geometry, box, cell, displacement, model_dtype,
                          kspace_electrostatics, nodes_lo):
    """The neighbour-list model's charges and dispersion ratios, a timer of its force calls
    (time_fn), its short-range cutoff (cutoff_sr) and its long-range list (nbrs_lr_idx,
    nbrs_lr_distances)."""
    position = geometry['positions']
    method = kspace_electrostatics if kspace_electrostatics != 'cutoff' else None
    # A bundled name, a path, or None for the default model; load_model builds
    # the solver for the method being timed.
    potential = load_model(
        settings.get('model_path'),
        precision=model_dtype,
        lr_cutoff=settings.get('lr_cutoff'),
        dispersion_damping=settings.get('dispersion_damping'),
        kspace_electrostatics=method,
        kspace_interp_nodes=nodes_lo,
        output_intermediate_quantities=['partial_charges'],
    )
    # Match DEFAULT_BUFFER_MULTIPLIER in so3lr_cli.py (1.25). The CLI wrapper always seeds
    # these keys; hardcoded here (rather than imported from so3lr_cli) to avoid a circular
    # import (so3lr_cli imports `tune` from this module).
    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = process_model(
        potential=potential,
        species=geometry['species'],
        displacement=displacement,
        box=box,
        total_charge=settings.get('total_charge') or 0.,
        num_unpaired_electrons=settings.get('num_unpaired_electrons') or 0.,
        precision=model_dtype,
        fractional_coordinates=False,
        buffer_size_multiplier_sr=settings.get('buffer_size_multiplier_sr', 1.25),
        buffer_size_multiplier_lr=settings.get('buffer_size_multiplier_lr', 1.25),
        has_aux=True,
    )
    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box)
    d = jax.vmap(partial(displacement))
    nbrs_lr_distances = distance(d(position[nbrs_lr.idx[0]], position[nbrs_lr.idx[1]]))

    energy_fn = partial(energy_or_obs_fn, has_aux=False, neighbor=nbrs.idx, box=box)
    force_fn = force(energy_fn)
    if settings.get('jit_compile', True):
        force_fn = jax.jit(force_fn)

    obs_data = energy_or_obs_fn(position, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box, has_aux=True)
    charges = obs_data[1]['partial_charges']
    hirshfeld_ratios, c6_ratios = dispersion_ratios(obs_data[1])
    return SimpleNamespace(
        charges=charges, hirshfeld_ratios=hirshfeld_ratios, c6_ratios=c6_ratios,
        time_fn=force_call_timer(cell, position, nbrs_lr.idx, nbrs_lr_distances, force_fn, method),
        cutoff_sr=potential.cutoff, nbrs_lr_idx=nbrs_lr.idx, nbrs_lr_distances=nbrs_lr_distances)


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
    cutoff_hi = settings.get('cutoff_hi', lr_cutoff)

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
    # the measured errors take the minimum image up to the largest cutoff tried
    problem = minimum_image_problem(initial_geometry.get_cell(), max(lr_cutoff, cutoff_hi))
    if problem:
        raise ValueError(f'tune-ewald: {problem}.')

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=state_dtype)

    # We don't want fractional coordinates here

    box = jnp.array(cell, dtype=state_dtype)
    if box.shape == (3, 3) and np.all(box[~np.eye(box.shape[0], dtype=bool)] == 0):
        # orthorhombic boxes are reduced to 3 entries
        box = jnp.diag(box)
    # Create displacement and shift functions for periodic boundary conditions
    displacement, _ = periodic_general(box=box, fractional_coordinates=False)
    position = initial_geometry_dict['positions']

    if box.shape == (3, 3):
        cell = lattice_rows(box)  # error bounds and jaxpme k-grids use rows
    elif box.shape == (3, ):
        cell = jnp.diag(box)
    elif box.shape == (1,):
        cell = jnp.diag(jnp.repeat(box, 3))
    else:
        raise ValueError(
            f"k-space electrostatics: Invalid box shape {box.shape}. Expected (3, 3), (3,), or (1,).")

    pair = model_pair_settings(model_path)
    smear_lo, mesh_hi = settings.get('smear_lo', 2.0), settings.get('mesh_hi', 7)
    # the charges on flash come with the first candidate's engine (its largest cutoff and
    # finest mesh, the likeliest within the accuracy), which is timed too
    first_spacing = (pme_mesh_spacing(float(np.min(np.linalg.norm(cell, axis=1))), mesh_hi)
                     if kspace_electrostatics == 'pme' else None)
    on_flash = _flash_setup(settings, initial_geometry, cutoff_hi, smear_lo, first_spacing)
    if on_flash is not None:
        from so3lr.flash.check import model_cutoff
        charges, time_fn = on_flash
        cutoff_sr, nl = model_cutoff(model_path), None
    else:
        nl = _neighbour_list_setup(settings, initial_geometry_dict, box, cell, displacement, model_dtype,
                                   kspace_electrostatics, nodes_lo)
        charges, time_fn, cutoff_sr = nl.charges, nl.time_fn, nl.cutoff_sr

    model_kwargs = dict(
        cutoff_sr=cutoff_sr,
        electrostatic_energy_scale=pair['electrostatic_energy_scale'],
    )

    accuracy = float(settings.get('accuracy', 1.0 if kspace_electrostatics == 'cutoff' else 1e-3))
    common = dict(
        cell=cell, positions=position, time_fn=time_fn, accuracy=accuracy,
        cutoff_it=settings.get('cutoff_it', 2.0), cutoff_hi=cutoff_hi,
    )
    smearing = dict(
        cutoff_lo=settings.get('cutoff_lo', 7.0), smear_lo=smear_lo,
        smear_hi=settings.get('smear_hi', 6.0), smear_it=settings.get('smear_it', 2.0),
    )
    if kspace_electrostatics in ("ewald", "pme", "p3m", "cutoff"):
        # the measured real-space errors: the pairs of a sample of the atoms, up to the
        # largest cutoff tried and lr_cutoff (the cutoff tuning's reference sum covers the
        # long-range pairs the model sees)
        reach = float(max(cutoff_hi, lr_cutoff))
        pairs = SamplePairs(position, displacement, reach)
        logger.info(f"Real-space errors measured on {pairs.n} of {len(position)} atoms "
                    f"({len(pairs.r)} pairs within {reach} A).")
    if kspace_electrostatics == "ewald":
        logger.info("Tuning Ewald parameters...")
        params, errs, timings = tune_ewald(
            charges=charges, pairs=pairs, **common, **smearing, model_kwargs=model_kwargs,
            ns_lo=settings.get('ns_lo', 4), ns_hi=settings.get('ns_hi', 8),
        )
    elif kspace_electrostatics in ("pme", "p3m"):
        logger.info(f"Tuning {kspace_electrostatics.upper()} parameters...")
        params, errs, timings = tune_pme(
            charges=charges, pairs=pairs, **common, **smearing, model_kwargs=model_kwargs,
            nodes_lo=nodes_lo, nodes_hi=nodes_hi,
            mesh_lo=settings.get('mesh_lo', 5), mesh_hi=mesh_hi,
        )
    elif kspace_electrostatics == 'cutoff':
        logger.info("No k-space electrostatics tuning, only tuning the cutoff.")
        params, errs, timings = tune_sr(
            charges=charges, pairs=pairs, **common, cutoff_lo=settings.get('cutoff_lo', 10.0),
            model_kwargs=model_kwargs,
        )
    else:  # kspace_dispersion == 'cutoff'
        logger.info("No k-space dispersion tuning, only tuning the cutoff.")
        dispersion_model_kwargs = dict(
            dispersion_energy_scale=pair['dispersion_energy_scale'],
            legacy=pair['legacy'],
            atomic_numbers=initial_geometry.get_atomic_numbers()
        )
        params, errs, timings = tune_dispersion_sr(
            hirshfeld_ratios=nl.hirshfeld_ratios, c6_ratios=nl.c6_ratios, displacement=displacement,
            nbrs_lr=nl.nbrs_lr_idx, nbrs_lr_distances=nl.nbrs_lr_distances, **common,
            cutoff_lo=settings.get('cutoff_lo', 10.0),
            damp_lo=settings.get('dispersion_damping_lo', 2.0),
            damp_hi=settings.get('dispersion_damping_hi', 4.0),
            damp_it=settings.get('dispersion_damping_it', 1.0),
            model_kwargs=dispersion_model_kwargs,
        )

    recommended_param = None
    best = recommended_index(errs, timings, accuracy)
    if best is not None:
        param, timing, error = params[best], timings[best], errs[best]
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
