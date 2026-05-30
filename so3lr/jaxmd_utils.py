import jax
import jax.numpy as jnp

from functools import partial
from jax_md import partition
from jax_md.space import DisplacementOrMetricFn, Box, transform

from so3lr.graph import Graph


def neighbor_list_featurizer(
        displacement_fn,
        species,
        fractional_coordinates=True,
        total_charge: float = 0.0,
        num_unpaired_electrons: float = 0.0,
):
    def featurize(R, neighbor, neighbor_lr, **kwargs):
        idx_i = neighbor[0]  # shape: P
        idx_j = neighbor[1]  # shape: P
        idx_i_lr = neighbor_lr[0]  # shape: P
        idx_j_lr = neighbor_lr[1]  # shape: P

        Ra = R[idx_i]
        Rb = R[idx_j]
        Ra_lr = R[idx_i_lr]
        Rb_lr = R[idx_j_lr]

        box = None
        positions = None
        k_smearing = None
        k_grid = None
        k_grid_disp = None
        k_smearing_disp = None
        if 'box' in kwargs:
            box = kwargs.get('box')
            if box is not None:
                if box.shape == (3, 3):
                    box = box
                elif box.shape == (3, ):
                    box = jnp.diag(box)
                elif box.shape == (1,):
                    box = jnp.diag(jnp.repeat(box, 3))
                else:
                    raise ValueError(f"k-space electrostatics: Invalid box shape {box.shape}. Expected (3, 3), (3,), or (1,).")

        if 'k_grid' in kwargs:
            k_grid = kwargs.get('k_grid')
            if fractional_coordinates:
                positions = transform(box, R) # transform to non-fractional coordinates
            else:
                positions = R
        if 'k_smearing' in kwargs:
            k_smearing = kwargs.get('k_smearing')

        if 'k_grid_disp' in kwargs:
            k_grid_disp = kwargs.get('k_grid_disp')
            if positions is None:
                if fractional_coordinates:
                    positions = transform(box, R)
                else:
                    positions = R
        if 'k_smearing_disp' in kwargs:
            k_smearing_disp = kwargs.get('k_smearing_disp')

        if 'perturbation' in kwargs:
            pert = kwargs.get('perturbation')
            if positions is not None:
                positions = transform(pert, positions)
            box = transform(pert, box)

        d = jax.vmap(partial(displacement_fn, **kwargs))
        dR = d(Ra, Rb)
        dR_lr = d(Ra_lr, Rb_lr)

        graph = Graph(
            positions=positions,
            nodes=species,
            edges=dR,
            centers=idx_i,
            others=idx_j,
            mask=None,
            total_charge=jnp.asarray([total_charge], dtype=R.dtype),
            num_unpaired_electrons=jnp.asarray([num_unpaired_electrons], dtype=R.dtype),
            edges_lr=dR_lr,
            idx_i_lr=idx_i_lr,
            idx_j_lr=idx_j_lr,
            cell=box,  # will raise an error if box not in kwargs.
            k_grid=k_grid,
            k_smearing=k_smearing,
            k_grid_disp=k_grid_disp,
            k_smearing_disp=k_smearing_disp,
            theory_mask=jnp.eye(16)[5].reshape(1, -1)
        )

        return graph

    return featurize


def to_jax_md(
        potential,  # the mlff potential
        displacement_or_metric: DisplacementOrMetricFn,
        box_size: Box,  # box if it exists, check md jax documentation for conventions
        species: jnp.ndarray = None,  # the atomic species, jnp.ndarray of shape n_atoms
        dr_threshold: float = 0.,  # currently dr_threshold > 0 is experimental
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        minimum_cell_size_multiplier_sr: float = 1.0,
        minimum_cell_size_multiplier_lr: float = 1.0,
        disable_cell_list: bool = False,
        fractional_coordinates: bool = True,
        total_charge: float = 0.0,  # net charge of the system (e)
        num_unpaired_electrons: float = 0.0,  # spin: number of unpaired electrons
        **neighbor_kwargs
):
    # Handle vacuum systems (box_size=None) by using a large effective box
    # JAX-MD doesn't accept None for box_size, so we use a very large value
    if box_size is None:
        effective_box_size = jnp.array(1e7)  # Very large box for vacuum
    else:
        effective_box_size = box_size

    # create the neighbor_fn
    neighbor_fn = partition.neighbor_list(
        displacement_or_metric,
        effective_box_size,
        r_cutoff=potential.cutoff,
        dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier,
        buffer_size_multiplier=buffer_size_multiplier_sr,
        minimum_cell_size_multiplier=minimum_cell_size_multiplier_sr,
        fractional_coordinates=fractional_coordinates,
        format=partition.NeighborListFormat(1),  # only sparse is supported in mlff
        disable_cell_list=disable_cell_list,
        **neighbor_kwargs)

    # create the neighbor_fn for long-range cutoff
    neighbor_fn_lr = partition.neighbor_list(
        displacement_or_metric,
        effective_box_size,
        r_cutoff=potential.long_range_cutoff,
        dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier,
        buffer_size_multiplier=buffer_size_multiplier_lr,
        minimum_cell_size_multiplier=minimum_cell_size_multiplier_lr,
        fractional_coordinates=fractional_coordinates,
        format=partition.NeighborListFormat(2),  # long-range modules can handle OrderedSparse.
        disable_cell_list=disable_cell_list,
        **neighbor_kwargs)

    featurizer = neighbor_list_featurizer(
        displacement_or_metric,
        species,
        fractional_coordinates=fractional_coordinates,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
    )

    # create an energy_fn that is compatible with jax_md
    def energy_fn(
            R,
            neighbor,
            neighbor_lr,
            has_aux=False,
            **energy_fn_kwargs
    ):
        graph = featurizer(R, neighbor, neighbor_lr, **energy_fn_kwargs)
        if has_aux:
            return potential(graph, has_aux=True)
        else:
            return potential(graph).sum()

    return neighbor_fn, neighbor_fn_lr, energy_fn
