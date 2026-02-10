import jax.numpy as jnp
import jax

from typing import (Any, Callable, Dict, Tuple)
from flax.core.frozen_dict import FrozenDict
from so3lr.mlff.masking.mask import safe_scale

Array = Any
StackNetSparse = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]


def get_observable_fn_sparse(model: StackNetSparse, observable: str = None):
    """
    Get the observable function of a `model`. If no `observable_key` is specified, values for all implemented
    observables of the model are returned.

    Args:
        model (StackNet): A `StackNetSparse` module or a module that returns a dictionary of observables.
        observable (str): Observable name.

    Returns: Observable function,

    """
    if observable is None:
        def observable_fn(
                params,
                positions: jnp.ndarray,
                atomic_numbers: jnp.ndarray,
                idx_i: jnp.ndarray,
                idx_j: jnp.ndarray,
                cell: jnp.ndarray = None,
                cell_offset: jnp.ndarray = None,
                batch_segments: jnp.ndarray = None,
                node_mask: jnp.ndarray = None,
                graph_mask: jnp.ndarray = None,
                displacements: jnp.ndarray = None,
                displacements_lr: jnp.ndarray = None,
                total_charge: jnp.ndarray = None,
                num_unpaired_electrons: jnp.ndarray = None,
                idx_i_lr: jnp.ndarray = None,
                idx_j_lr: jnp.ndarray = None,
                theory_mask: jnp.ndarray = None,
                k_grid: jnp.ndarray = None,
                k_smearing: jnp.ndarray = None,
                residue_segments: jnp.ndarray = None,
                residue_charge: jnp.ndarray = None,
                **kwargs
        ):
            if batch_segments is None:
                assert graph_mask is None
                assert node_mask is None

                graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
                node_mask = jnp.ones((len(atomic_numbers),)).astype(jnp.bool_)  # (num_nodes)
                batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)

            inputs = dict(
                positions=positions,
                atomic_numbers=atomic_numbers,
                idx_i=idx_i,
                idx_j=idx_j,
                cell=cell,
                cell_offset=cell_offset,
                batch_segments=batch_segments,
                node_mask=node_mask,
                graph_mask=graph_mask,
                displacements=displacements,
                displacements_lr=displacements_lr,
                total_charge=total_charge,
                num_unpaired_electrons=num_unpaired_electrons,
                idx_i_lr=idx_i_lr,
                idx_j_lr=idx_j_lr,
                theory_mask=theory_mask,
                k_grid = k_grid,
                k_smearing = k_smearing,
                residue_segments=residue_segments,
                residue_charge=residue_charge,
            )
            return model.apply(params, inputs)
    else:
        def observable_fn(
                params,
                positions: jnp.ndarray,
                atomic_numbers: jnp.ndarray,
                idx_i: jnp.ndarray,
                idx_j: jnp.ndarray,
                cell: jnp.ndarray = None,
                cell_offset: jnp.ndarray = None,
                batch_segments: jnp.ndarray = None,
                node_mask: jnp.ndarray = None,
                graph_mask: jnp.ndarray = None,
                displacements: jnp.ndarray = None,
                displacements_lr: jnp.ndarray = None,
                total_charge: jnp.ndarray = None,
                num_unpaired_electrons: jnp.ndarray = None,
                idx_i_lr: jnp.ndarray = None,
                idx_j_lr: jnp.ndarray = None,
                theory_mask: jnp.ndarray = None,
                k_grid: jnp.ndarray = None,
                k_smearing: jnp.ndarray = None,
                residue_segments: jnp.ndarray = None,
                residue_charge: jnp.ndarray = None,
                **kwargs
        ):
            if batch_segments is None:
                assert graph_mask is None
                assert node_mask is None

                graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
                node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)  # (num_nodes)
                batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)   

            inputs = dict(
                positions=positions,
                atomic_numbers=atomic_numbers,
                idx_i=idx_i,
                idx_j=idx_j,
                cell=cell,
                cell_offset=cell_offset,
                batch_segments=batch_segments,
                node_mask=node_mask,
                graph_mask=graph_mask,
                displacements=displacements,
                displacements_lr=displacements_lr,
                total_charge=total_charge,
                num_unpaired_electrons=num_unpaired_electrons,
                idx_i_lr=idx_i_lr,
                idx_j_lr=idx_j_lr,
                theory_mask=theory_mask,
                k_grid = k_grid,
                k_smearing = k_smearing,
                residue_segments=residue_segments,
                residue_charge=residue_charge,
            )
            return dict(observable=model.apply(params, inputs)[observable])

    return observable_fn


def get_energy_and_force_fn_sparse(model: StackNetSparse):
    def energy_fn(params,
                  positions: jnp.ndarray,
                  atomic_numbers: jnp.ndarray,
                  idx_i: jnp.ndarray,
                  idx_j: jnp.ndarray,
                  cell: jnp.ndarray = None,
                  cell_offset: jnp.ndarray = None,
                  batch_segments: jnp.ndarray = None,
                  node_mask: jnp.ndarray = None,
                  graph_mask: jnp.ndarray = None,
                  displacements: jnp.ndarray = None,
                  displacements_lr: jnp.ndarray = None,
                  total_charge: jnp.ndarray = None,
                  num_unpaired_electrons: jnp.ndarray = None,
                  idx_i_lr: jnp.ndarray = None,
                  idx_j_lr: jnp.ndarray = None,
                  theory_mask: jnp.ndarray = None,
                  k_grid: jnp.ndarray = None,
                  k_smearing: jnp.ndarray = None,
                  residue_segments: jnp.ndarray = None,
                  residue_charge: jnp.ndarray = None,
                  ):
        if batch_segments is None:
            assert graph_mask is None
            assert node_mask is None

            graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
            node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)  # (num_nodes)
            batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)

        inputs = dict(positions=positions,
                      atomic_numbers=atomic_numbers,
                      idx_i=idx_i,
                      idx_j=idx_j,
                      cell=cell,
                      cell_offset=cell_offset,
                      batch_segments=batch_segments,
                      node_mask=node_mask,
                      graph_mask=graph_mask,
                      displacements=displacements,
                      displacements_lr=displacements_lr,
                      total_charge=total_charge,
                      num_unpaired_electrons=num_unpaired_electrons,
                      idx_i_lr=idx_i_lr,
                      idx_j_lr=idx_j_lr,
                      theory_mask=theory_mask,
                      k_grid = k_grid,
                      k_smearing = k_smearing,
                      residue_segments=residue_segments,
                      residue_charge=residue_charge,
                      )

        energy = model.apply(params, inputs)['energy']  # (num_graphs)
        energy = safe_scale(energy, graph_mask)
        return -jnp.sum(energy), energy  # (), (num_graphs)

    def energy_and_force_and_dipole_and_hirsh_fn(
            params,
            positions: jnp.ndarray,
            atomic_numbers: jnp.ndarray,
            idx_i: jnp.ndarray,
            idx_j: jnp.ndarray,
            cell: jnp.ndarray = None,
            cell_offset: jnp.ndarray = None,
            batch_segments: jnp.ndarray = None,
            node_mask: jnp.ndarray = None,
            graph_mask: jnp.ndarray = None,
            displacements: jnp.ndarray = None,
            displacements_lr: jnp.ndarray = None,
            total_charge: jnp.ndarray = None,
            num_unpaired_electrons: jnp.ndarray = None,
            idx_i_lr: jnp.ndarray = None,
            idx_j_lr: jnp.ndarray = None,
            theory_mask: jnp.ndarray = None,
            k_grid: jnp.ndarray = None,
            k_smearing: jnp.ndarray = None,
            residue_segments: jnp.ndarray = None,
            residue_charge: jnp.ndarray = None,
            *args,
            **kwargs
    ):
        (_, energy), forces = jax.value_and_grad(
            energy_fn,
            argnums=1,
            has_aux=True)(params,
                          positions,
                          atomic_numbers,
                          idx_i,
                          idx_j,
                          cell,
                          cell_offset,
                          batch_segments,
                          node_mask,
                          graph_mask,
                          displacements,
                          displacements_lr,
                          total_charge,
                          num_unpaired_electrons,
                          idx_i_lr,
                          idx_j_lr,
                          theory_mask,
                          k_grid = k_grid,
                          k_smearing = k_smearing,
                          residue_segments=residue_segments,
                          residue_charge=residue_charge,
                          )

        if batch_segments is None:
            assert graph_mask is None
            assert node_mask is None

            graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
            node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)  # (num_nodes)
            batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes) 

        inputs = dict(positions=positions,
                      atomic_numbers=atomic_numbers,
                      idx_i=idx_i,
                      idx_j=idx_j,
                      cell=cell,
                      cell_offset=cell_offset,
                      batch_segments=batch_segments,
                      node_mask=node_mask,
                      graph_mask=graph_mask,
                      displacements=displacements,
                      displacements_lr=displacements_lr,
                      total_charge=total_charge,
                      num_unpaired_electrons=num_unpaired_electrons,
                      idx_i_lr=idx_i_lr,
                      idx_j_lr=idx_j_lr,
                      theory_mask=theory_mask,
                      k_grid = k_grid,
                      k_smearing = k_smearing,
                      residue_segments=residue_segments,
                      residue_charge=residue_charge,
                      )

        # _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts=True, size=len(graph_mask))

        result = dict(
            energy=energy,
            forces=forces
        )

        model_output = model.apply(params, inputs)

        if 'dipole_vec' in model_output:
            dipole_vec = model_output['dipole_vec']  # (num_graphs)
            dipole_vec = safe_scale(dipole_vec, graph_mask[:, None])  # (num_graphs)
            result['dipole_vec'] = dipole_vec

        if 'hirshfeld_ratios' in model_output:
            hirshfeld_ratios = model_output['hirshfeld_ratios']  # (num_graphs)
            hirshfeld_ratios = safe_scale(hirshfeld_ratios, node_mask)  # (num_graphs)
            result['hirshfeld_ratios'] = hirshfeld_ratios

        return result

    return energy_and_force_and_dipole_and_hirsh_fn


def get_hybrid_energy_force_fn_sparse(model: StackNetSparse, force_mode: str = 'constrained'):
    """
    Get energy and force function with switchable force computation mode.
    
    This enables pretraining with unconstrained forces (faster) and fine-tuning
    with constrained forces (energy-conserving) using the same model.
    
    Args:
        model: StackNetSparse model. For unconstrained mode, the EnergySparse 
               observable must have predict_forces_directly=True.
        force_mode: 'constrained' (forces via -∇E, energy conserving) or 
                    'unconstrained' (direct NN prediction + physics gradients)
    
    Returns:
        Function computing energy and forces
    """
    if force_mode == 'constrained':
        # Use existing gradient-based force computation
        return get_energy_and_force_fn_sparse(model)
    
    elif force_mode == 'unconstrained':
        def physics_energy_fn(positions, params, inputs_template):
            """Helper: compute only physics-based energy contributions."""
            inputs = inputs_template.copy()
            inputs['positions'] = positions
            outputs = model.apply(params, inputs)
            
            # Sum up physics-based atomic energy contributions
            physics_energy = jnp.zeros(1)
            if 'electrostatic_energy' in outputs:
                physics_energy += outputs.get('electrostatic_energy', jnp.zeros(1)).sum()
            if 'dispersion_energy' in outputs:
                physics_energy += outputs.get('dispersion_energy', jnp.zeros(1)).sum()
            if 'zbl_repulsion' in outputs:
                physics_energy += outputs.get('zbl_repulsion', jnp.zeros(1)).sum()
            return physics_energy
        
        def energy_and_force_fn_unconstrained(
                params,
                positions: jnp.ndarray,
                atomic_numbers: jnp.ndarray,
                idx_i: jnp.ndarray,
                idx_j: jnp.ndarray,
                cell: jnp.ndarray = None,
                cell_offset: jnp.ndarray = None,
                batch_segments: jnp.ndarray = None,
                node_mask: jnp.ndarray = None,
                graph_mask: jnp.ndarray = None,
                displacements: jnp.ndarray = None,
                displacements_lr: jnp.ndarray = None,
                total_charge: jnp.ndarray = None,
                num_unpaired_electrons: jnp.ndarray = None,
                idx_i_lr: jnp.ndarray = None,
                idx_j_lr: jnp.ndarray = None,
                theory_mask: jnp.ndarray = None,
                k_grid: jnp.ndarray = None,
                k_smearing: jnp.ndarray = None,
                residue_segments: jnp.ndarray = None,
                residue_charge: jnp.ndarray = None,
                *args,
                **kwargs
        ):
            if batch_segments is None:
                assert graph_mask is None
                assert node_mask is None
                graph_mask = jnp.ones((1,)).astype(jnp.bool_)
                node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)
                batch_segments = jnp.zeros_like(atomic_numbers)

            inputs = dict(
                positions=positions,
                atomic_numbers=atomic_numbers,
                idx_i=idx_i,
                idx_j=idx_j,
                cell=cell,
                cell_offset=cell_offset,
                batch_segments=batch_segments,
                node_mask=node_mask,
                graph_mask=graph_mask,
                displacements=displacements,
                displacements_lr=displacements_lr,
                total_charge=total_charge,
                num_unpaired_electrons=num_unpaired_electrons,
                idx_i_lr=idx_i_lr,
                idx_j_lr=idx_j_lr,
                theory_mask=theory_mask,
                k_grid=k_grid,
                k_smearing=k_smearing,
                residue_segments=residue_segments,
                residue_charge=residue_charge,
            )

            # Get model outputs (energy + direct NN forces if predict_forces_directly=True)
            outputs = model.apply(params, inputs)
            energy = outputs['energy']
            energy = safe_scale(energy, graph_mask)
            
            # Get NN forces (direct prediction)
            nn_forces = outputs.get('nn_forces')
            if nn_forces is None:
                raise ValueError(
                    "Model does not output 'nn_forces'. Make sure EnergySparse has "
                    "predict_forces_directly=True for unconstrained mode."
                )
            
            # Compute physics force contributions via gradient
            # (electrostatics, dispersion, ZBL - these are energy-based)
            has_physics = any(k in outputs for k in ['electrostatic_energy', 'dispersion_energy', 'zbl_repulsion'])
            if has_physics:
                physics_forces = jax.grad(
                    lambda p: -physics_energy_fn(p, params, inputs)
                )(positions)
                physics_forces = safe_scale(physics_forces, node_mask[:, None])
            else:
                physics_forces = jnp.zeros_like(nn_forces)
            
            # Combine NN forces + physics forces
            total_forces = nn_forces + physics_forces
            total_forces = safe_scale(total_forces, node_mask[:, None])

            # Get optional outputs
            result = dict(energy=energy, forces=total_forces)
            
            if 'dipole_vec' in outputs:
                result['dipole_vec'] = safe_scale(outputs['dipole_vec'], graph_mask[:, None])
            if 'hirshfeld_ratios' in outputs:
                result['hirshfeld_ratios'] = safe_scale(outputs['hirshfeld_ratios'], node_mask)
            
            return result
        
        return energy_and_force_fn_unconstrained
    
    else:
        raise ValueError(
            f"force_mode must be 'constrained' or 'unconstrained', got '{force_mode}'"
        )
