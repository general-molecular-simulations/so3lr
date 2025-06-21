"""
Tuning module for Ewald summation parameters.

This tuning module is refactored from:
https://github.com/lab-cosmo/torch-pme/tree/main/src/torchpme/tuning
"""
import numpy as np
import jax
import jax.numpy as jnp
import os
import math
import time
import logging
from typing import Any, Optional
from warnings import warn
from itertools import product
from functools import partial

#from jax_md import partition
from jax_md.space import distance, periodic_general, DisplacementFn
from jax_md.quantity import force

from jaxpme import get_kgrid_ewald, get_kgrid_mesh

from ase import Atoms
from ase.io import read

from so3lr import So3lrPotential
from so3lr.cli.so3lr_md import handle_box, load_model, atoms_to_jnp, process_model, load_state
from .so3lr_md import setup_logger

# Setup logging
logger = logging.getLogger("SO3LR")

from mlff.nn.observable.observable_sparse import coulomb_erf_shifted_force_smooth, coulomb_erf_shifted_force_smooth_pme, coulomb_erf
from jax.ops import segment_sum

def filter_neighbors(
    cutoff: float, neighbor_indices: jnp.array, neighbor_distances: jnp.array
):
    filter_idx = jnp.where((neighbor_distances < cutoff) & (neighbor_distances > 0.0) & (~jnp.isnan(neighbor_distances)))
    return neighbor_indices[:,filter_idx], neighbor_distances[filter_idx]

def filter_neighbor_idx(
    cutoff: float, neighbor_indices: jnp.array, neighbor_distances: jnp.array
):
    filter_idx = jnp.where((neighbor_distances < cutoff) & (neighbor_distances > 0.0) & (~jnp.isnan(neighbor_distances)))[0]
    return neighbor_indices[:,filter_idx]

def get_forces(energy, rij, q, **params):

    @jax.jit
    def total_energy(r):
        num_nodes=len(q)
        atomic_energy = segment_sum(
                energy(q, r, **params),
                segment_ids=params['idx_i'],
                num_segments=num_nodes
            )
        return -jnp.sum(atomic_energy)
    
    @jax.jit
    def forces(r):
        return jax.value_and_grad(total_energy)(r)[1]  # Get the force by differentiating the energy with respect to rij

    return forces(rij)

def get_forces_rs(energy, positions, q, displacement, **params):
    d = jax.vmap(partial(displacement))

    @jax.jit
    def total_energy(positions):
        r = distance(d(positions[params['idx_i']], positions[params['idx_j']]))
        num_nodes=len(positions)
        atomic_energy = segment_sum(
                energy(q, r, **params),
                segment_ids=params['idx_i'],
                num_segments=num_nodes
            )
        return -jnp.sum(atomic_energy)
    
    @jax.jit
    def forces(pos):
        return jax.value_and_grad(total_energy)(pos)[1]  # Get the force by differentiating the energy with respect to pos

    return forces(positions)


def get_forces_rs_at_fixed_distance(energy, positions, q, fixed_distance, **params):

    @jax.jit
    def total_energy(positions):
        r = jnp.ones_like(params['idx_i'],dtype=positions.dtype)*fixed_distance
        num_nodes=len(positions)
        atomic_energy = segment_sum(
                energy(q, r, **params),
                segment_ids=params['idx_i'],
                num_segments=num_nodes
            )
        return -jnp.sum(atomic_energy)
    
    @jax.jit
    def forces(pos):
        return jax.value_and_grad(total_energy)(pos)[1]  # Get the force by differentiating the energy with respect to pos

    return forces(positions)

class TuningErrorBounds:
    def __init__(self, charges: jnp.ndarray, cell: jnp.ndarray, 
                 positions: jnp.ndarray, scale: float = 14.399645351950548):
        self._charges = charges
        self._cell = cell
        self._positions = positions
        self._scale = scale

    def __call__(self, *args, **kwargs):
        return self.error(*args, **kwargs)

    def error(self, *args, **kwargs):
        raise NotImplementedError


class TunerBase:

    def __init__(
        self,
        charges: jnp.ndarray,
        cell: jnp.ndarray,
        positions: jnp.ndarray,
        scale: float = 14.399645351950548,
        exponent: int = 1,
    ):
        if exponent != 1:
            raise NotImplementedError(
                f"Only exponent = 1 is supported but got {exponent}."
            )

        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.exponent = exponent

        self._prefac = 2 * scale * float((charges**2).sum()) / math.sqrt(len(positions))

    def tune(self, accuracy: float = 2e-2):
        raise NotImplementedError

    def estimate_smearing(self, cutoff: float, accuracy: float) -> float:
        if not isinstance(accuracy, float):
            raise ValueError(f"'{accuracy}' is not a float.")
        ratio = math.sqrt(
            -2
            * math.log(
                accuracy
                / 2
                / self._prefac
                * math.sqrt(cutoff * abs(np.linalg.det(self.cell)))
            )
        )
        smearing = cutoff / ratio

        return float(smearing)

class GridSearchTuner(TunerBase):
    def __init__(
        self,
        charges: jnp.ndarray,
        cell: jnp.ndarray,
        positions: jnp.ndarray,
        error_bounds: type[TuningErrorBounds],
        params: list[dict],
        potential_kwargs: dict,
        exponent: int = 1,
        ):
        super().__init__(
            charges=charges,
            cell=cell,
            positions=positions,
            exponent=exponent,
        )
        self.error_bounds = error_bounds
        self.params = params
        self.time_func = TuningTimings(
            charges=charges,
            cell=cell,
            positions=positions,
            **potential_kwargs
        )

    def tune(self, accuracy: float = 1e-3) -> tuple[list[float], list[float]]:
        if not isinstance(accuracy, float):
            raise ValueError(f"'{accuracy}' is not a float.")
        #smearing = self.estimate_smearing(accuracy)
        param_errors = []
        param_timings = []
        param_memorys = []
        logger.info("---------------------------ERROR ESTIMATION:")
        for param in self.params:
            logger.info("parameters: {}".format(", ".join(f"{k}: {v}" for k, v in param.items() if v is not None)))
            error = self.error_bounds(**param)
            param_errors.append(float(error))
        
        logger.info("---------------------------TIMING ESTIMATION:")
        for param, error in zip(self.params, param_errors):
            logger.info("parameters: {}".format(", ".join(f"{k}: {v}" for k, v in param.items() if v is not None)))
            # if param['smearing']
            # logger.info(f"kspace_spacing: {param['spacing']}")
            # if 'interpolation_nodes' in param:
            #     logger.info(f"  kspace_interp_nodes: {param['interpolation_nodes']}")               
            if error <= accuracy:
               time, mem = self._timing(param)
            else:
                time = float("inf")
                mem = {"not tested":float("inf")}
            param_timings.append( time )
            param_memorys.append(mem)
            
            #logger.info("Results: ")
            logger.info(f"measured timing: {param_timings[-1]}, total error estimate: {error}")
            #print('memory', sum(param_memorys[-1].values()))
        return param_errors, param_timings, param_memorys

    def _timing(self, k_space_params: dict):  
        # calculator = self.calculator(
        #     potential=InversePowerLawPotential(
        #         exponent=self.exponent,  # but only exponent = 1 is supported
        #         smearing=smearing,
        #     ),
        #     **k_space_params,
        # )
        # calculator.to(device=self.positions.device, dtype=self.positions.dtype)
        #return self.time_func(calculator)        
        return self.time_func(**k_space_params)

class TuningTimings:
    """
    This class is refactored from:
    https://github.com/lab-cosmo/torch-pme/tree/main/src/torchpme/tuning

    Class for timing a calculator.

    The class estimates the average execution time of a given calculator after several
    warmup runs. The class takes the information of the structure that one wants to
    benchmark on, and the configuration of the timing process as inputs.

    :param charges: numpy.ndarray of shape ``(len(positions), 1)`` containing the atomic
        (pseudo-)charges
    :param cell: numpy.ndarray of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: numpy.ndarray of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param neighbor_indices: numpy.ndarray with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: numpy.ndarray with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param n_repeat: number of times to repeat to estimate the average timing
    :param n_warmup: number of warmup runs, recommended to be at least 4
    :param run_backward: whether to run the backward pass (not applicable for numpy)
    """

    def __init__(
        self,
        charges: jnp.ndarray,
        cell: jnp.ndarray,
        positions: jnp.ndarray,
        nbrs_lr: jnp.ndarray,
        nbrs_lr_distances: jnp.ndarray,
        force_fn: callable,
        kspace_electrostatics: Optional[str] = None,
        n_repeat: int = 20,
        n_warmup: int = 4,
        RND_SEED: int = 42,        
    ):

        self.n_repeat = n_repeat
        self.n_warmup = n_warmup

        self.force_fn = force_fn
        self.positions = positions
        self.cell = cell

        self.nbrs_lr = nbrs_lr
        self.nbrs_lr_distances = nbrs_lr_distances
        self.rnd_key = jax.random.key(RND_SEED)

        # self.atoms = Atoms(
        #     positions=positions,
        #     cell=cell,
        #     charges=charges,
        #     pbc=True,
        # )

        self.kspace_electrostatics = kspace_electrostatics

    def __call__(self, **input_params) -> tuple[float, dict[str, float]]:
        """
        Estimate the execution time of a given calculator for the structure
        to be used as benchmark.

        :param calculator: the calculator to be tuned
        :return: a float, the average execution time
        """
        execution_time = 0.0

        cutoff = input_params.get('cutoff')

        if self.kspace_electrostatics == "ewald":
            k_spacing = jnp.array([input_params.get('spacing')])
            k_smearing = jnp.array([input_params.get('smearing')])
            k_grid = get_kgrid_ewald(self.cell, k_spacing)
        elif self.kspace_electrostatics == "pme":
            k_spacing = jnp.array([input_params.get('spacing')])
            k_smearing = jnp.array([input_params.get('smearing')])
            k_grid = get_kgrid_mesh(self.cell, k_spacing)
        else:
            k_grid = None
            k_smearing = None

        nbrs_idx = filter_neighbor_idx(cutoff, self.nbrs_lr, self.nbrs_lr_distances)
        

        for i in range(self.n_repeat + self.n_warmup):
            if i == self.n_warmup:
                execution_time = 0.0
            pos = self.positions+jax.random.normal(self.rnd_key,self.positions.shape)*0.001
            start_time = time.monotonic()
            _ = self.force_fn(
                pos,
                neighbor_lr=nbrs_idx,
                k_grid=k_grid,
                k_smearing=k_smearing,
            )            
            #value=0.0
            #forces = self.atoms.get_forces()            
            #result = calculator.forward(
            #    positions=positions,
            #    charges=charges,
            #    cell=cell,
            #    neighbor_indices=self.neighbor_indices,
            #    neighbor_distances=self.neighbor_distances,
            #)
            #value = result.sum()
            execution_time += time.monotonic() - start_time

        memory_estimate = {'None':0.0}
        # memory_estimate = {
        #     'lr_neighbor_list':
        #         atoms.calc.neighbors.idx_i_lr.shape[0] * 2 * 8 / 1024 / 1024,
        #     'positions':
        #         atoms.get_positions().shape[0] * 3 * 8 / 1024 / 1024,
        #     'k_grid':
        #         np.prod(atoms.info['k_grid'].shape) * 8 / 1024 / 1024,
        # }

        return execution_time / self.n_repeat, memory_estimate
    
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
) -> tuple[float, dict[str, Any], float]:
    r"""
    This class is refactored from:
    https://github.com/lab-cosmo/torch-pme/tree/main/src/torchpme/tuning

    Find the optimal parameters for Ewald summation`.

    :param charges: numpy array of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: numpy array of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: numpy array of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent: :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1`
        is supported
    :param neighbor_indices: numpy array with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: numpy array with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param ns_lo: Minimum number of spatial resolution along each axis
    :param ns_hi: Maximum number of spatial resolution along each axis
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, and a dictionary with the parameters for
        :class:`EwaldCalculator`, and the timing of this set of parameters.
    """
    min_dimension = float(np.min(np.linalg.norm(cell, axis=1)))

    params = [
        {
            "smearing": smear,
            "cutoff": cut,
            "spacing":  min_dimension / ns,
        }
         for smear, cut, ns in product(
             np.arange(smear_lo, smear_hi + smear_it, smear_it),
             np.arange(cutoff_hi, cutoff_lo -cutoff_it, -cutoff_it), 
             range(ns_hi, ns_lo-1,-1)
        )
    ]    
    
    logger.info(f"Testing {len(params)} parameter combinations.")

    tuner = GridSearchTuner(
        charges=charges,
        cell=cell,
        positions=positions,
        # exponent=exponent,
        error_bounds=EwaldErrorBounds_SR(charges=charges, cell=cell, positions=positions,
            displacement=displacement,
            nbrs_lr=nbrs_lr,nbrs_lr_distances=nbrs_lr_distances,
            **model_kwargs),
        params=params,
        potential_kwargs=dict(
            nbrs_lr=nbrs_lr,
            nbrs_lr_distances=nbrs_lr_distances,
            force_fn=force_fn,
            kspace_electrostatics="ewald",  # Ewald summation
        ),
    )
    #smearing = tuner.estimate_smearing(accuracy)
    #logger.info(f"Estimated smearing: {smearing}")
    errs, timings, memorys = tuner.tune(accuracy)
    #timings = [0.0]*len(errs)
    #print(errs)

    if any(err < accuracy for err in errs):
        return params[timings.index(min(timings))], min(timings)
    warn(
        f"No parameter meets the accuracy requirement.\n"
        f"Returning the parameter with the smallest error, which is {min(errs)}.\n",
        stacklevel=1,
    )
    return params[errs.index(min(errs))], timings[errs.index(min(errs))]


class EwaldErrorBounds(TuningErrorBounds):
    r"""
    This class is refactored from:
    https://github.com/lab-cosmo/torch-pme/tree/main/src/torchpme/tuning
        
    Error bounds for :class:`torchpme.calculators.ewald.EwaldCalculator`.

    :param charges: numpy array of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: numpy array of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: numpy array of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    """

    def __init__(
        self,
        charges: jnp.ndarray,
        cell: jnp.ndarray,
        positions: jnp.ndarray,
    ):
        super().__init__(charges, cell, positions)
        self.volume = abs(jnp.linalg.det(cell))
        self.sum_squared_charges = jnp.sum(charges**2)
        self.prefac = 2 * self._scale * self.sum_squared_charges / math.sqrt(len(positions))

    def err_kspace(
        self, smearing: float, spacing: float
    ) -> float:
        """
        The Fourier space error of Ewald.

        :param smearing: see :class:`torchpme.EwaldCalculator` for details
        :param spacing: see :class:`torchpme.EwaldCalculator` for details
        """
        # return (
        #     self.prefac**0.5
        #     / smearing
        #     / np.sqrt((2 * np.pi) ** 2 * self.volume / (spacing) ** 0.5)
        #     * np.exp(-((2 * np.pi) ** 2) * smearing**2 / (spacing))
        # )
        return (
            self.prefac*0.5
            / smearing
            / np.sqrt((2 * np.pi) ** 2 * self.volume / (spacing) ** 0.5)
            * np.exp(-((2 * np.pi) ** 2) * smearing**2 / (spacing))
        )

    def err_rspace(self, smearing: float, cutoff: float) -> float:
        """
        The real space error of Ewald.

        :param smearing: see :class:`torchpme.EwaldCalculator` for details
        :param spacing: see :class:`torchpme.EwaldCalculator` for details
        """
        return (
            self.prefac
            / np.sqrt(cutoff * self.volume)
            * np.exp(-(cutoff**2) / 2 / smearing**2)
        )

class EwaldErrorBounds_SR(EwaldErrorBounds):
    def __init__(
        self, 
        charges: jnp.ndarray, 
        cell: jnp.ndarray, 
        positions: jnp.ndarray,
        displacement: DisplacementFn,
        nbrs_lr: jnp.ndarray, 
        nbrs_lr_distances: jnp.ndarray,
        cutoff_sr: float = 4.5,
        electrostatic_energy_scale: float = 1.0
    ):
        super().__init__(charges, cell, positions)
        self.displacement = displacement

        self.nbrs_lr=nbrs_lr
        self.nbrs_lr_distances= nbrs_lr_distances

        self.cutoff_sr = cutoff_sr
        self.electrostatic_energy_scale=electrostatic_energy_scale
        

    def err_rspace(self, smearing: float, cutoff: float) -> float:
        ke  = self._scale
        sigma = self.electrostatic_energy_scale
        cuton = self.cutoff_sr

        nbrs = filter_neighbor_idx(cutoff, self.nbrs_lr, self.nbrs_lr_distances)
        idx_i, idx_j = nbrs[0], nbrs[1]

        # full_force = get_forces(coulomb_erf, rij, q, idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=smearing)
        # interp_force = get_forces(coulomb_erf_shifted_force_smooth_pme, rij, q, 
        #                           idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cuton=cuton, cutoff=cutoff, smearing=smearing)
        # return jnp.sqrt(
        #     jnp.sum(
        #         jnp.square(full_force - interp_force)
        #     )
        # ) / jnp.sqrt(len(full_force))
        full_force = get_forces_rs(coulomb_erf,
                                    self._positions, self._charges,self.displacement, 
                                    idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=smearing)
        interp_force = get_forces_rs(coulomb_erf_shifted_force_smooth_pme, 
                                     self._positions, self._charges,self.displacement, 
                                     idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cuton=cuton, cutoff=cutoff, smearing=smearing)

        return jnp.sqrt(jnp.sum(jnp.square(full_force - interp_force))/ (len(self._positions)))

    def error(
        self, smearing: float, spacing: float, cutoff: float
    ) -> float:
        r"""
        Calculate the error bound of Ewald.

        :param smearing: see :class:`torchpme.EwaldCalculator` for details
        :param spacing: see :class:`torchpme.EwaldCalculator` for details
        :param cutoff: see :class:`torchpme.EwaldCalculator` for details
        """
        kspace = self.err_kspace(smearing, spacing)
        rspace = self.err_rspace(smearing, cutoff)
        rspace_ana = super().err_rspace(smearing, self.cutoff_sr)
        logger.info(
            f"Error bounds: kspace={kspace}, rspace={rspace}, rspace (estimate)={rspace_ana} total={np.sqrt(kspace**2 + rspace**2)}"
        )

        return np.sqrt(kspace**2 + rspace**2)

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
) -> tuple[float, dict[str, Any], float]:
    min_dimension = float(np.min(jnp.linalg.norm(cell, axis=1)))

    params = [
        {
            "smearing": smear,
            "cutoff": cut,
            "interpolation_nodes": interpolation_nodes,
            "spacing": 2 * min_dimension / (2**ns - 1),
        }
         for smear, cut, interpolation_nodes, ns in product(
             np.arange(smear_lo, smear_hi + smear_it, smear_it),
             np.arange(cutoff_hi, cutoff_lo-cutoff_it, -cutoff_it), 
            range(nodes_hi, nodes_lo-1,-1), 
            range(mesh_hi, mesh_lo-1, -1)
        )
    ]
    logger.info(f"Testing {len(params)} parameter combinations.")

    tuner = GridSearchTuner(
        charges=charges,
        cell=cell,
        positions=positions,
        exponent=exponent,
        error_bounds=PMEErrorBounds_SR(charges=charges, cell=cell, positions=positions,
                                        displacement=displacement,
                                       nbrs_lr=nbrs_lr, nbrs_lr_distances=nbrs_lr_distances,
                                       **model_kwargs),
        params=params,
        potential_kwargs=dict(
            nbrs_lr=nbrs_lr,
            nbrs_lr_distances=nbrs_lr_distances,
            force_fn=force_fn,
            kspace_electrostatics="pme",  # PME summation
        ),
    )
    #smearing = tuner.estimate_smearing(accuracy)
    #logger.info(f"Estimated smearing: {smearing}")
    errs, timings, memorys = tuner.tune(accuracy)
    #timings = [0.0]*len(errs)
    #print(errs)
    
    if any(err < accuracy for err in errs):
        return params[timings.index(min(timings))], min(timings)
    warn(
        f"No parameter meets the accuracy requirement.\n"
        f"Returning the parameter with the smallest error, which is {min(errs)}.\n",
        stacklevel=1,
    )
    return params[errs.index(min(errs))], timings[errs.index(min(errs))]


class PMEErrorBounds(TuningErrorBounds):
    def __init__(
        self, charges: jnp.ndarray, cell: jnp.ndarray, positions: jnp.ndarray
    ):
        super().__init__(charges, cell, positions)

        self.volume = abs(jnp.linalg.det(cell))
        self.sum_squared_charges = jnp.sum(charges**2)
        self.prefac = 2 * self._scale * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell_dimensions = jnp.linalg.norm(cell, axis=1)

    def err_kspace(
        self,
        smearing: float,
        spacing: float,
        interpolation_nodes: int,
    ) -> float:
        actual_spacing = self.cell_dimensions / (
            2 * self.cell_dimensions / spacing + 1
        )
        h = np.prod(actual_spacing) ** (1 / 3)
        i_n_factorial = math.factorial(interpolation_nodes)
        RMS_phi = [None, None, 0.246, 0.404, 0.950, 2.51, 8.42]

        return (
            self.prefac
            * math.pi**0.25
            * (6 * (1 / 2**0.5 / smearing) / (2 * interpolation_nodes + 1)) ** 0.5
            / self.volume ** (2 / 3)
            * (2**0.5 / smearing * h) ** interpolation_nodes
            / i_n_factorial
            * np.exp(
                interpolation_nodes * (np.log(interpolation_nodes / 2) - 1) / 2
            )
            * RMS_phi[interpolation_nodes - 1]
        )

    def err_rspace(self, smearing: float, cutoff: float) -> float:
        return (
            self.prefac
            / math.sqrt(cutoff * self.volume)
            * np.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def error(
        self,
        cutoff: float,
        smearing: float,
        spacing: float,
        interpolation_nodes: int,
    ) -> float:

        kspace = self.err_kspace(smearing, spacing, interpolation_nodes)
        rspace = self.err_rspace(smearing, cutoff)
        rspace_ana = super().err_rspace(smearing, self.cutoff_sr)
        logger.info(
            f"Error bounds: kspace={kspace}, rspace={rspace}, rspace (estimate)={rspace_ana}, total={np.sqrt(kspace**2 + rspace**2)}"
        )

        return np.sqrt(kspace**2 + rspace**2)


class PMEErrorBounds_SR(PMEErrorBounds):
    def __init__(
        self, 
        charges: jnp.ndarray, 
        cell: jnp.ndarray, 
        positions: jnp.ndarray,
        displacement: DisplacementFn,
        nbrs_lr: jnp.ndarray, 
        nbrs_lr_distances: jnp.ndarray,
        cutoff_sr: float = 4.5,
        electrostatic_energy_scale: float = 1.0
    ):
        super().__init__(charges, cell, positions)
        self.nbrs_lr=nbrs_lr
        self.nbrs_lr_distances= nbrs_lr_distances
        self._positions = positions
        self._charges = charges
        self.displacement = displacement
        self.electrostatic_energy_scale=electrostatic_energy_scale
        self.cutoff_sr = cutoff_sr
        

    def err_rspace(self, smearing: float, cutoff: float) -> float:
        ke  = self._scale
        sigma = self.electrostatic_energy_scale
        cuton = self.cutoff_sr

        nbrs = filter_neighbor_idx(cutoff, self.nbrs_lr, self.nbrs_lr_distances)
        idx_i, idx_j = nbrs[0], nbrs[1]

        # full_force = get_forces(coulomb_erf, rij, q, idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=smearing)
        # interp_force = get_forces(coulomb_erf_shifted_force_smooth_pme, rij, q, 
        #                           idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cuton=cuton, cutoff=cutoff, smearing=smearing)
        # return jnp.sqrt(
        #     jnp.sum(
        #         jnp.square(full_force - interp_force)
        #     )
        # ) / jnp.sqrt(len(full_force))
        full_force = get_forces_rs(coulomb_erf,
                                    self._positions, self._charges,self.displacement, 
                                    idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=smearing)
        interp_force = get_forces_rs(coulomb_erf_shifted_force_smooth_pme, 
                                     self._positions, self._charges,self.displacement, 
                                     idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cuton=cuton, cutoff=cutoff, smearing=smearing)

        return jnp.sqrt(jnp.sum(jnp.square(full_force - interp_force))/ (len(self._positions)))


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
) -> tuple[float, dict[str, Any], float]:
    min_dimension = float(np.min(jnp.linalg.norm(cell, axis=1)))

    params = [
        {
            "cutoff": cut,
        }
         for cut in np.arange(cutoff_hi, cutoff_lo -cutoff_it, -cutoff_it)
    ]

    logger.info(f"Testing {len(params)} parameter combinations.")

    tuner = GridSearchTuner(
        charges=charges,
        cell=cell,
        positions=positions,
        exponent=exponent,
        error_bounds=NativeErrorBounds(charges=charges, cell=cell, positions=positions,
                                        displacement=displacement,
                                       nbrs_lr=nbrs_lr, nbrs_lr_distances=nbrs_lr_distances,
                                       **model_kwargs),
        params=params,
        potential_kwargs=dict(
            nbrs_lr=nbrs_lr,
            nbrs_lr_distances=nbrs_lr_distances,
            force_fn=force_fn,
            kspace_electrostatics=None,  # PME summation
        ),
    )
    #smearing = tuner.estimate_smearing(accuracy)
    #logger.info(f"Estimated smearing: {smearing}")
    errs, timings, memorys = tuner.tune(accuracy)
    #timings = [0.0]*len(errs)
    #print(errs)
    
    if any(err < accuracy for err in errs):
        return params[timings.index(min(timings))], min(timings)
    warn(
        f"No parameter meets the accuracy requirement.\n"
        f"Returning the parameter with the smallest error, which is {min(errs)}.\n",
        stacklevel=1,
    )
    return params[errs.index(min(errs))], timings[errs.index(min(errs))]


class NativeErrorBounds(TuningErrorBounds):
    def __init__(
        self, 
        charges: jnp.ndarray, 
        cell: jnp.ndarray, 
        positions: jnp.ndarray,
        displacement: DisplacementFn,
        nbrs_lr: jnp.ndarray, 
        nbrs_lr_distances: jnp.ndarray,
        cutoff_sr: float = 4.5,
        electrostatic_energy_scale: float = 1.0
    ):
        super().__init__(charges, cell, positions)
        self.nbrs_lr=nbrs_lr
        self.nbrs_lr_distances= nbrs_lr_distances
        self._positions = positions
        self._charges = charges
        self.displacement = displacement
        self.electrostatic_energy_scale=electrostatic_energy_scale


        ke = self._scale
        sigma = self.electrostatic_energy_scale
        idx_i, idx_j = self.nbrs_lr[0], self.nbrs_lr[1]
        self.best_full_force = get_forces_rs(coulomb_erf,
                                   self._positions, self._charges,self.displacement, 
                                   idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=None)        
        

    def err_rspace(self, cutoff: float) -> float:
        ke  = self._scale
        sigma = self.electrostatic_energy_scale
        cuton = 0.45 * cutoff

        nbrs = filter_neighbor_idx(cutoff, self.nbrs_lr, self.nbrs_lr_distances)
        idx_i, idx_j = nbrs[0], nbrs[1]

        # full_force = get_forces(coulomb_erf, rij, q, idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=smearing)
        # interp_force = get_forces(coulomb_erf_shifted_force_smooth_pme, rij, q, 
        #                           idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cuton=cuton, cutoff=cutoff, smearing=smearing)
        # return jnp.sqrt(
        #     jnp.sum(
        #         jnp.square(full_force - interp_force)
        #     )
        # ) / jnp.sqrt(len(full_force))
        #full_force = get_forces_rs(coulomb_erf,
        #                            self._positions, self._charges,self.displacement, 
        #                            idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=None)
        #full_force = self.all_force_errs

        # full_force_at_cut = get_forces_rs_at_fixed_distance(coulomb_erf,
        #                             self._positions, self._charges, fixed_distance=cutoff, 
        #                             idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=None) 
        # lr_error = jnp.sqrt(jnp.sum(jnp.square(full_force_at_cut))/ (3*len(self._positions)))
        # print(lr_error)

        full_force = get_forces_rs(coulomb_erf,
                                    self._positions, self._charges,self.displacement, 
                                    idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cutoff=None)         
        interp_force = get_forces_rs(coulomb_erf_shifted_force_smooth, 
                                     self._positions, self._charges,self.displacement, 
                                     idx_i=idx_i, idx_j=idx_j, ke=ke, sigma=sigma, cuton=cuton, cutoff=cutoff)
        return full_force, jnp.sqrt(jnp.sum(jnp.square(full_force - interp_force))/ (len(self._positions)))

    def err_kspace(self, cutoff: float) -> float:
        # Currently this give the error per interaction, not per atom, this needs to be improved
        ke  = self._scale
        sigma = self.electrostatic_energy_scale
        positions = self.positions
        nbr_distances = self.nbrs_lr_distances

        r = jnp.ones(positions.shape[0],dtype=positions.dtype)*cutoff
        fake_idx = jnp.arange(len(positions), dtype=jnp.int32)
        full_force_at_cut = get_forces(coulomb_erf, r, q, idx_i=fake_idx,  idx_j=fake_idx, ke=ke, sigma=sigma, cutoff=None)
        return jnp.sqrt(jnp.sum(full_force_at_cut**2)/(len(full_force_at_cut)))    


    def error(
        self,
        cutoff: float,
    ) -> float:

        #kspace = self.err_kspace(cutoff)
        full_force, rspace = self.err_rspace(cutoff)
        kspace =  jnp.sqrt(jnp.sum(jnp.square(full_force - self.best_full_force))/ (len(self._positions)))
        logger.info(
            f"Error bounds: kspace={kspace}, rspace={rspace}, total={np.sqrt(kspace**2 + rspace**2)}"
        )

        return np.sqrt(kspace**2 + rspace**2)

def tune(
    settings: dict
) -> None:
    """
    Tune Ewald parameters for running SO3LR model on a given structure.

    Args:
        settings (dict): Dictionary containing the settings for the tuning process.
    """

    # Set the precision
    if settings.get('precision', "float32").lower() == 'float64':
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    # Setup logger
    log_file = settings.get('log_file')
    setup_logger(log_file)

    # Model parameters
    model_path = settings.get('model_path')
    precision = settings.get('precision')
    lr_cutoff = settings.get('lr_cutoff')
    total_charge = settings.get('total_charge')
    dispersion_damping = settings.get('dispersion_damping')
    buffer_size_multiplier_lr = settings.get('buffer_size_multiplier_lr', 1.0)
    buffer_size_multiplier_sr = settings.get('buffer_size_multiplier_sr', 1.0)
    jit_compile = settings.get('jit_compile', True)

    # Ewald electrostatics settings
    kspace_electrostatics = settings.get('kspace_electrostatics',None)
    #assert kspace_electrostatics is not None, "type of kspace_electrostatics (\"ewald\" or \"pme\") must be provided for tuning."

    # Tuning parameters
    # Note that all the tuning parameters are loaded further below

    input_file_path = settings.get('input_file')
    if input_file_path is None:
        raise ValueError('Initial geometry file path must be provided in settings.')
    try:
        initial_geometry = read(input_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find initial geometry file: {input_file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to read geometry file {input_file_path}: {e}")

    cell = initial_geometry.get_cell().T
    assert cell is not None and not np.all(cell != 0.0), "Cell must be provided in the input geometry."

    initial_geometry_dict = atoms_to_jnp(initial_geometry, precision=precision)

    # We don't want fractional coordinates here
    #(position, box, displacement, shift, fractional_coordinates) = handle_box(
    #    'periodic', initial_geometry_dict['positions'], cell)
    
    box = jnp.array(cell)
    if box.shape == (3,3) and np.all( box[~np.eye(box.shape[0],dtype=bool)]==0):
        # orthorhombic boxes are reduced to 3 entries
        box = jnp.diag(box)
    fractional_coordinates = False

    # Create displacement and shift functions for periodic boundary conditions
    displacement, shift = periodic_general(
        box=box,
        fractional_coordinates=fractional_coordinates
    )
    position = initial_geometry_dict['positions']

    # Loading the model
    if model_path is None:
        # Load default SO3LR
        #logger.info("Using default SO3LR potential.")
        potential = So3lrPotential(
            dtype=jnp.float64 if precision == 'float64' else jnp.float32,
            lr_cutoff=lr_cutoff,
            dispersion_energy_cutoff_lr_damping=dispersion_damping,
            output_intermediate_quantities=['partial_charges'],
        )
    else:
        # Verify model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Custom model path not found: {model_path}")
        #logger.info(f"Using custom MLFF potential from: {model_path}")
        potential = load_model(
            model_path,
            precision=jnp.float64 if precision == 'float64' else jnp.float32,
            lr_cutoff=lr_cutoff,
            dispersion_damping=dispersion_damping,
            kspace_electrostatics=kspace_electrostatics,
            #kspace_interp_nodes=kspace_interp_nodes,
            output_intermediate_quantities=['partial_charges'],
        )

    model_kwargs=dict(
        cutoff_sr=potential.cutoff,
        electrostatic_energy_scale=4.0 # 4.0 is the default value, TODO load from model_config!
    )


    neighbor_fn, neighbor_fn_lr, energy_or_obs_fn = process_model(
        potential=potential,
        species=initial_geometry_dict['species'],
        displacement=displacement,
        box=box,
        total_charge=total_charge,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
        buffer_size_multiplier_lr=buffer_size_multiplier_sr,
        buffer_size_multiplier_sr=buffer_size_multiplier_lr,
        has_aux=True,
    )

    nbrs = neighbor_fn.allocate(position, box=box)
    nbrs_lr = neighbor_fn_lr.allocate(position, box=box)

    d = jax.vmap(partial(displacement))
    nbrs_lr_distances = distance(d(position[nbrs_lr.idx[0]], position[nbrs_lr.idx[1]]))

    if jit_compile:
        energy_fn = jax.jit(partial(energy_or_obs_fn,has_aux=False,neighbor=nbrs.idx, box=box))
        force_fn = jax.jit(force(energy_fn))
    else:
        energy_fn = partial(energy_or_obs_fn,has_aux=False,neighbor=nbrs.idx, box=box)
        force_fn = force(energy_fn)

    obs_data = energy_or_obs_fn(
                        position,
                        neighbor=nbrs.idx,
                        neighbor_lr=nbrs_lr.idx,
                        box=box,
                        has_aux=True
                    )
    charges = obs_data[1]['partial_charges']

    if box.shape == (3, 3):
        box = box
    elif box.shape == (3, ):
        box = jnp.diag(box)
    elif box.shape == (1,):
        box = jnp.diag(jnp.repeat(box, 3))
    else:
        raise ValueError(f"k-space electrostatics: Invalid box shape {box.shape}. Expected (3, 3), (3,), or (1,).") 

    if kspace_electrostatics == "ewald":
        # Tune Ewald parameters
        logger.info("Tuning Ewald parameters...")
        params, timing = tune_ewald(
            charges=charges,
            cell=box,
            positions=position,
            displacement=displacement,
            nbrs_lr=nbrs_lr.idx,
            nbrs_lr_distances=nbrs_lr_distances,
            force_fn=force_fn,
            cutoff_lo=settings.get('cutoff_lo', 7.0),  # Load from settings
            cutoff_it=settings.get('cutoff_it', 2.0),  # Load from settings  
            cutoff_hi=settings.get('cutoff_hi', lr_cutoff),  # Default to lr_cutoff
            smear_lo= settings.get('smear_lo', 2.0),  # Load from settings
            smear_hi= settings.get('smear_hi', 6.0),  # Load from settings
            smear_it= settings.get('smear_it', 2.0),  # Load from settings
            ns_lo=    settings.get('ns_lo', 4),  # Load from settings
            ns_hi=    settings.get('ns_hi', 8),  # Load from settings
            accuracy= settings.get('accuracy', 2e-2),  # Load from settings
            model_kwargs=model_kwargs,
        )

    elif kspace_electrostatics == "pme":
        logger.info("Tuning PME parameters...")
        # Tune PME parameters
        params, timing = tune_pme(
            charges=charges,
            cell=box,
            positions=position,
            displacement=displacement,
            nbrs_lr=nbrs_lr.idx,
            nbrs_lr_distances=nbrs_lr_distances,
            force_fn=force_fn,
            cutoff_lo=settings.get('cutoff_lo', 7.0),  # Load from settings
            cutoff_it=settings.get('cutoff_it', 2.0),  # Load from settings
            cutoff_hi=settings.get('cutoff_hi', lr_cutoff),  # Default to lr_cutoff 
            smear_lo= settings.get('smear_lo', 2.0),  # Load from settings
            smear_hi= settings.get('smear_hi', 6.0),  # Load from settings
            smear_it= settings.get('smear_it', 2.0),  # Load from settings
            nodes_lo= settings.get('nodes_lo', 3),  # Load from settings
            nodes_hi= settings.get('nodes_hi', 3),  # Load from settings
            mesh_lo=  settings.get('mesh_lo', 4),  # Load from settings
            mesh_hi=  settings.get('mesh_hi', 6),  # Load from settings
            accuracy= settings.get('accuracy', 2e-2),  # Load from settings
            model_kwargs=model_kwargs,
        )
    elif kspace_electrostatics is None:
        logger.info("No k-space electrostatics tuning, only tuning the cutoff.")
        # If no k-space electrostatics is specified, use native short-range tuning
        # No k-space electrostatics, use native short-range tuning
        params, timing = tune_sr(
            charges=charges,
            cell=box, 
            positions=position,
            displacement=displacement,
            nbrs_lr=nbrs_lr.idx,
            nbrs_lr_distances=nbrs_lr_distances,
            force_fn=force_fn,
            cutoff_lo=settings.get('cutoff_lo', 10.0),  # Load from settings
            cutoff_it=settings.get('cutoff_it', 2.0),  # Load from settings
            cutoff_hi=settings.get('cutoff_hi', lr_cutoff),  # Default to lr_cutoff
            accuracy= settings.get('accuracy', 1.0),  # Load from settings
            exponent=1,  # Native short-range uses 1/r potential
            model_kwargs=model_kwargs,
        )
    else:
        raise ValueError(f"Unknown kspace_electrostatics type: {kspace_electrostatics}. Must be 'ewald' or 'pme'.")

    if kspace_electrostatics is None:
        logger.info("Tuned parameters for cutoff electrostatics:")
        logger.info(f"  lr_cutoff: {params['cutoff']}")
    else:
        logger.info(f"Tuned {kspace_electrostatics.upper()} parameters: ")
        logger.info(f"  kspace_smearing: {params['smearing']}")
        logger.info(f"  kspace_spacing: {params['spacing']}")
        if 'interpolation_nodes' in params:
            logger.info(f"  kspace_interp_nodes: {params['interpolation_nodes']}")        

