import jax
import jax.numpy as jnp

from functools import partial
from jax_md import partition
from jax_md.space import DisplacementOrMetricFn, Box, transform

from so3lr.graph import Graph


def neighbor_list_featurizer(
        displacement_fn,
        species,
        total_charge: float = 0.0,
        precision=None,
        fractional_coordinates: bool = True,
        num_unpaired_electrons: float = 0.0,
):
    """Build an MLFF :class:`Graph` from positions + short/long-range neighbors.

    This is the single, shared featurizer used both by the public
    :func:`to_jax_md` helper and by the CLI MD engine (which imports it as
    ``neighbor_list_featurizer_custom``).

    Args:
        precision: dtype for the auxiliary arrays (``total_charge``,
            ``num_unpaired_electrons``, ``theory_mask``). When ``None`` these
            follow the positions array ``R.dtype`` (the historical behaviour of
            the public helper); pass an explicit dtype (e.g. ``jnp.float64``) to
            force it (the CLI MD behaviour).
    """
    def featurize(R, neighbor, neighbor_lr, **kwargs):
        idx_i = neighbor[0]  # shape: P
        idx_j = neighbor[1]  # shape: P
        idx_i_lr = neighbor_lr[0]  # shape: P
        idx_j_lr = neighbor_lr[1]  # shape: P

        Ra = R[idx_i]
        Rb = R[idx_j]
        Ra_lr = R[idx_i_lr]
        Rb_lr = R[idx_j_lr]

        aux_dtype = precision if precision is not None else R.dtype

        positions = None
        k_smearing = None
        k_smearing_disp = None
        box = kwargs.get('box', None)
        k_grid = kwargs.get('k_grid', None)
        k_grid_disp = kwargs.get('k_grid_disp', None)

        if box is not None:
            if box.shape == (3, 3):
                box = box
            elif box.shape == (3, ):
                box = jnp.diag(box)
            elif box.shape == (1,):
                box = jnp.diag(jnp.repeat(box, 3))
            else:
                raise ValueError(f"k-space electrostatics: Invalid box shape {box.shape}. Expected (3, 3), (3,), or (1,).")

        # Cartesian positions are needed by real-space observables (dipole_vec)
        # too, not only by the k-space solvers.
        if fractional_coordinates and box is not None:
            positions = transform(box, R)
        else:
            positions = R

        if k_grid is not None:
            k_smearing = kwargs.get('k_smearing')
            assert box is not None, "k_grid requires a box to be defined."
            assert k_smearing is not None, "k_grid requires k_smearing to be defined."
        if k_grid_disp is not None:
            k_smearing_disp = kwargs.get('k_smearing_disp')
            assert box is not None, "k_grid_disp requires a box to be defined."
            assert k_smearing_disp is not None, "k_grid_disp requires k_smearing_disp to be defined."

        if 'perturbation' in kwargs:
            pert = kwargs.get('perturbation')
            if positions is not None:
                positions = transform(pert, positions)
            if box is not None:
                # box holds the lattice vectors as COLUMNS (Graph.cell convention);
                # transform() would act on its rows (box @ pert.T), which deforms
                # the transposed lattice once PotentialSparse hands rows to PME.
                # Isotropic NPT passes a scalar perturbation (1 + eps).
                box = pert * box if jnp.ndim(pert) == 0 else pert @ box

        d = jax.vmap(partial(displacement_fn, **kwargs))
        dR = d(Ra, Rb)
        dR_lr = d(Ra_lr, Rb_lr)

        theory_mask = (jnp.eye(16, dtype=precision) if precision is not None else jnp.eye(16))[5].reshape(1, -1)

        graph = Graph(
            positions=positions,
            nodes=species,
            edges=dR,
            centers=idx_i,
            others=idx_j,
            mask=None,
            total_charge=jnp.asarray([total_charge], dtype=aux_dtype),
            num_unpaired_electrons=jnp.asarray([num_unpaired_electrons], dtype=aux_dtype),
            edges_lr=dR_lr,
            idx_i_lr=idx_i_lr,
            idx_j_lr=idx_j_lr,
            cell=box,  # None for free boundary conditions
            k_grid=k_grid,
            k_smearing=k_smearing,
            k_grid_disp=k_grid_disp,
            k_smearing_disp=k_smearing_disp,
            theory_mask=theory_mask
        )

        return graph

    return featurize


def to_jax_md(
        potential,  # the mlff potential
        displacement_or_metric: DisplacementOrMetricFn,
        box_size: Box,  # box if it exists, check md jax documentation for conventions
        species: jnp.ndarray = None,  # the atomic species, jnp.ndarray of shape n_atoms
        dr_threshold: float = 0.,  # currently dr_threshold > 0 is experimental
        capacity_multiplier: float = None,
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        minimum_cell_size_multiplier_sr: float = 1.0,
        minimum_cell_size_multiplier_lr: float = 1.0,
        disable_cell_list: bool = False,
        fractional_coordinates: bool = True,
        total_charge: float = 0.0,  # net charge of the system (e)
        num_unpaired_electrons: float = 0.0,  # spin: number of unpaired electrons
        precision=None,  # dtype for auxiliary arrays; None -> follow positions dtype
        **neighbor_kwargs
):
    """Build the (short-range, long-range) neighbor functions and a jax-md
    compatible energy function for an MLFF ``potential``.

    Neighbor-list capacity can be specified in two equivalent ways:

    * ``capacity_multiplier=None`` (default): the per-list
      ``buffer_size_multiplier_sr``/``_lr`` are used directly as the
      ``capacity_multiplier`` of each neighbor list. This matches the CLI MD
      engine, which calls this helper via :func:`to_jax_md_custom`.
    * ``capacity_multiplier=<float>``: a single capacity multiplier is applied to
      both lists and the per-list values are forwarded as ``buffer_size_multiplier``
      (the legacy behaviour of this helper).
    """
    # Handle vacuum systems (box_size=None) by using a large effective box
    # JAX-MD doesn't accept None for box_size, so we use a very large value
    if box_size is None:
        effective_box_size = jnp.array(1e7)  # Very large box for vacuum
    else:
        effective_box_size = box_size

    def _make_neighbor_fn(r_cutoff, buffer_mult, min_cell_mult, nl_format):
        extra = {}
        if capacity_multiplier is None:
            cap = buffer_mult
        else:
            cap = capacity_multiplier
            extra['buffer_size_multiplier'] = buffer_mult
        return partition.neighbor_list(
            displacement_or_metric,
            effective_box_size,
            r_cutoff=r_cutoff,
            dr_threshold=dr_threshold,
            capacity_multiplier=cap,
            minimum_cell_size_multiplier=min_cell_mult,
            fractional_coordinates=fractional_coordinates,
            format=nl_format,
            disable_cell_list=disable_cell_list,
            **extra,
            **neighbor_kwargs)

    # create the neighbor_fn (only sparse is supported in mlff)
    neighbor_fn = _make_neighbor_fn(
        potential.cutoff, buffer_size_multiplier_sr,
        minimum_cell_size_multiplier_sr, partition.NeighborListFormat(1))

    # create the neighbor_fn for long-range cutoff (handles OrderedSparse)
    neighbor_fn_lr = _make_neighbor_fn(
        potential.long_range_cutoff, buffer_size_multiplier_lr,
        minimum_cell_size_multiplier_lr, partition.NeighborListFormat(2))

    featurizer = neighbor_list_featurizer(
        displacement_or_metric,
        species,
        total_charge=total_charge,
        precision=precision,
        fractional_coordinates=fractional_coordinates,
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
