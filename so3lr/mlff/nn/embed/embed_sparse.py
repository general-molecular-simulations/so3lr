import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import (Any, Dict, Sequence)

import flax.linen as nn
import e3x
import logging

from so3lr.mlff.nn.sub_module import BaseSubModule
from so3lr.mlff.nn.mlp import Residual
from so3lr.mlff.masking.mask import safe_mask
from so3lr.mlff.masking.mask import safe_norm
from so3lr.mlff.geometric import add_cell_offsets_sparse
from so3lr.mlff import utils
from so3lr.mlff.sph_ops import init_sph_fn


class GeometryEmbedE3x(BaseSubModule):
    prop_keys: Dict
    max_degree: int
    radial_basis_fn: str
    num_radial_basis_fn: int
    cutoff_fn: str
    cutoff: float
    input_convention: str = 'positions'
    module_name: str = 'geometry_embed_e3x'

    def __call__(self, inputs, *args, **kwargs):

        idx_i = inputs['idx_i']  # shape: (num_pairs)
        idx_j = inputs['idx_j']  # shape: (num_pairs)
        idx_i_lr = inputs.get('idx_i_lr')  # shape: (num_pairs_lr)
        idx_j_lr = inputs.get('idx_j_lr')  # shape: (num_pairs_lr)
        cell = inputs.get('cell')  # shape: (n_graphs, 3, 3) or (3, 3)
        cell_offsets = inputs.get('cell_offset')  # shape: (num_pairs, 3)
        cell_offsets_lr = inputs.get('cell_offset_lr')  # shape: (num_pairs_lr, 3)
        batch_segments = inputs.get('batch_segments')  # shape: (N,)

        if self.input_convention == 'positions':
            positions = inputs['positions']  # (N, 3)

            # Calculate pairwise distance vectors.
            r_ij = jax.vmap(
                lambda i, j: positions[j] - positions[i]
            )(idx_i, idx_j)  # (num_pairs, 3)

            r_ij_lr = None
            long_range_indices_present = False

            if idx_i_lr is not None:
                assert idx_j_lr is not None
                if len(idx_i_lr) > 0:
                    assert len(idx_j_lr) > 0
                    long_range_indices_present = True

            # If indices for long range corrections are present they are used.
            if long_range_indices_present is True:
                # Calculate pairwise distance vectors on long range indices.
                r_ij_lr = jax.vmap(
                    lambda i, j: positions[j] - positions[i]
                )(idx_i_lr, idx_j_lr)  # (num_pairs_lr, 3)

            # Apply minimal image convention if needed.
            if cell is not None:
                r_ij = add_cell_offsets_sparse(
                    r_ij=r_ij,
                    cell=cell,
                    cell_offsets=cell_offsets,
                    batch_segments=batch_segments,
                    idx_i=idx_i,
                )  # shape: (num_pairs,3)

                if long_range_indices_present and cell_offsets_lr is not None:
                    logging.warning(
                        'The use of long range indices with PBCs has not been tested thoroughly yet, so use with care!'
                    )

                    r_ij_lr = add_cell_offsets_sparse(
                        r_ij=r_ij_lr,
                        cell=cell,
                        cell_offsets=cell_offsets_lr,
                        batch_segments=batch_segments,
                        idx_i=idx_i_lr,
                    )  # shape: (num_pairs_lr,3)

        # Here it is assumed that PBC (if present) have already been respected in displacement calculation.
        elif self.input_convention == 'displacements':
            positions = inputs['positions']  # we need positions for dipole moment calculation
            r_ij = inputs['displacements']
            r_ij_lr = inputs.get('displacements_lr')  # shape : (num_pairs_lr, 3)
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Calculate pairwise distances.
        d_ij = safe_norm(r_ij, axis=-1)  # shape : (num_pairs)

        if r_ij_lr is not None:
            d_ij_lr = safe_norm(r_ij_lr, axis=-1)  # shape : (num_pairs_lr)
            del r_ij_lr
        else:
            d_ij_lr = None

        basis, cut = e3x.nn.basis(
            r=r_ij,
            max_degree=self.max_degree,
            radial_fn=getattr(e3x.nn, self.radial_basis_fn),
            num=self.num_radial_basis_fn,
            cutoff_fn=partial(getattr(e3x.nn, self.cutoff_fn), cutoff=self.cutoff),
            return_cutoff=True,

        )  # (N, 1, (max_degree+1)^2, num_radial_basis_fn), (N, )

        geometric_data = {'positions': positions,
                          'basis': basis,
                          'r_ij': r_ij,
                          'd_ij': d_ij,
                          'd_ij_lr': d_ij_lr,
                          'cut': cut,
                          }

        return geometric_data

    def reset_input_convention(self, input_convention: str) -> None:
        self.input_convention = input_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'max_degree': self.max_degree,
                                   'radial_basis_fn': self.radial_basis_fn,
                                   'num_radial_basis_fn': self.num_radial_basis_fn,
                                   'cutoff_fn': self.cutoff_fn,
                                   'cutoff': self.cutoff,
                                   'input_convention': self.input_convention,
                                   'prop_keys': self.prop_keys}
                }


# --- descriptor distance floor -------------------------------------------------
# Below the closest approach it was trained on, the network extrapolates wildly:
# a steric clash makes the charge head emit >100 e and the energy head >1e10 eV,
# and on so3lr-l a compressed S-S pair returns NaN outright. The learned heads
# are the only part that fails this way -- NLH/ZBL repulsion, electrostatics and
# dispersion are analytic and remain valid at any separation. So the distance
# fed to the RADIAL BASIS is floored, while every physics term keeps the true
# geometry and still diverges at real contact.
#
# The floor is per element pair, because the edge of validity is: for so3lr-l,
# S-S degrades below ~1.0 A while H-H never does. The scale comes from the model
# itself -- NLH repulsion is ke*Zi*Zj/r * sum_k a_k exp(-b_k r), whose b_k are
# inverse lengths, so the distance at which the tabulated repulsion reaches
# PAIR_FLOOR_ENERGY is a per-pair length. It lands essentially on the covalent
# bond length (O-H 0.97, C-H 1.08, S-S 2.03 A) with nothing to calibrate, and it
# is by construction consistent with the repulsion that takes over there.
PAIR_FLOOR_ENERGY = 1.0    # [eV] NLH repulsion level defining the pair length scale
PAIR_FLOOR_SCALE = 0.92    # floor at this fraction of it
PAIR_FLOOR_WIDTH = 0.05    # [A] handover; the floor is exactly inert above it
# SCALE is bracketed from both sides by measurement, not chosen. Too small and
# the floor sits below the network's own edge of validity (measured per pair
# from diatomic curves: so3lr-l degrades below 0.61 A for O-H and 1.01 A for
# S-S), so the descriptor still gets garbage. Too large and the handover reaches
# real bonds -- equilibrated water at 673 K / 250 bar samples O-H down to
# 0.863 A. Across all 18 pairs the feasible window is 0.86-0.98; 0.92 is its
# midpoint. WIDTH is small for the same reason: O-H leaves only ~0.2 A of room.

_NLH_REFERENCE_DISTANCE = {}


def nlh_reference_distance(target_energy: float = PAIR_FLOOR_ENERGY):
    """Distance at which the tabulated NLH repulsion reaches ``target_energy``.

    Mirrors NLHRepulsionSparse, including its ``switching_fn(d, 0, 1.5)`` factor
    -- without that the repulsion looks like it extends to 2 A when the shipped
    term is switched off entirely beyond 1.5 A. The ``cut`` envelope is omitted:
    it is 0.97-1.0 over this range, which moves the root by <0.5%. Verified
    against the model's own ``nlh_repulsion`` output to within 5%.

    Returns a plain numpy ``(n_elements, n_elements)`` array indexed
    ``[min(Z)-1, max(Z)-1]``. Cached as numpy, never as a jax array: the first
    caller may be inside a jit trace, and caching a value created there would
    let a tracer escape its scope and fail every subsequent call.
    """
    if target_energy in _NLH_REFERENCE_DISTANCE:
        return _NLH_REFERENCE_DISTANCE[target_energy]
    from so3lr.mlff.nn.observable.dispersion_ref_data import NLH_AA, NLH_BB
    ke = 14.399645351950548

    def _sigma(x):
        return np.where(x > 0, np.exp(-1.0 / np.where(x > 0, x, 1.0)), 0.0)

    def _switch(x, x_on, x_off):            # 1 below x_on, exactly 0 above x_off
        c = (x - x_on) / (x_off - x_on)
        return _sigma(1 - c) / (_sigma(1 - c) + _sigma(c))

    n = NLH_AA.shape[0]
    z = np.arange(1, n + 1)
    zi, zj = np.meshgrid(z, z, indexing='ij')
    a, b = np.asarray(NLH_AA), np.asarray(NLH_BB)
    lo, hi = np.full((n, n), 0.02), np.full((n, n), 1.5)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        with np.errstate(over='ignore', invalid='ignore'):
            y = np.sum(a * np.exp(-b * mid[..., None]), axis=-1)
            e = _switch(mid, 0.0, 1.5) * ke * zi * zj / mid * y
        above = np.nan_to_num(e, nan=0.0) > target_energy
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    _NLH_REFERENCE_DISTANCE[target_energy] = 0.5 * (lo + hi)
    return _NLH_REFERENCE_DISTANCE[target_energy]


def floor_descriptor_distance(d_ij, atomic_numbers, idx_i, idx_j):
    """Floor the distance handed to the radial basis, per element pair.

    Exactly the identity at and above ``rmin + PAIR_FLOOR_WIDTH`` -- which sits
    below every real bond -- so ordinary chemistry is bit-for-bit untouched.
    C1 at the handover, monotone, and finite-gradient at d = 0. Apply to the
    descriptor only: the repulsion and the electrostatics need the true distance.
    """
    table = jnp.asarray(nlh_reference_distance())
    zi = atomic_numbers[idx_i] - 1
    zj = atomic_numbers[idx_j] - 1
    rmin = PAIR_FLOOR_SCALE * table[jnp.minimum(zi, zj), jnp.maximum(zi, zj)]
    d_off = rmin + PAIR_FLOOR_WIDTH
    u = (d_off - d_ij) / (d_off - rmin)
    return jnp.where(d_ij >= d_off, d_ij, d_off - (d_off - rmin) * jnp.tanh(u))


class GeometryEmbedSparse(BaseSubModule):
    prop_keys: Dict
    degrees: Sequence[int]
    radial_basis_fn: str
    num_radial_basis_fn: int
    cutoff_fn: str
    cutoff: float
    input_convention: str = 'positions'
    module_name: str = 'geometry_embed_sparse'

    def setup(self):
        self.ylm_fns = [init_sph_fn(y) for y in self.degrees]

        self.rbf_fn = getattr(
            utils.radial_basis_fn, self.radial_basis_fn
        )(n_rbf=self.num_radial_basis_fn, r_cut=self.cutoff)

        self.cut_fn = partial(getattr(utils.cutoff_fn, self.cutoff_fn), r_cut=self.cutoff)

    def __call__(self, inputs: Dict):
        """
        Embed geometric information from the atomic positions and its neighboring atoms.

        Args:
            inputs (Dict): Input dictionary, with key - entry pairs:
                positions (Array): Atomic positions, (N, 3)
                idx_i (Array): Index centering atom, (num_pairs)
                idx_j (Array): Index neighboring atom, (num_pairs)
                cell (Array): Unit or super cell, (n_graphs, 3, 3) or (3, 3)
                cell_offset: Cell offsets for short-range PBCs, (num_pairs, 3)
                cell_offset_lr: Cell offsets for long-range PBCs, (num_pairs_lr, 3)
                batch_segments (Array): Graph index for each atom, (N,)

        Returns:
        """
        idx_i = inputs['idx_i']  # shape: (num_pairs)
        idx_j = inputs['idx_j']  # shape: (num_pairs)
        idx_i_lr = inputs.get('idx_i_lr')  # shape: (num_pairs_lr)
        idx_j_lr = inputs.get('idx_j_lr')  # shape: (num_pairs_lr)
        cell = inputs.get('cell')  # shape: (n_graphs, 3, 3) or (3, 3)
        cell_offsets = inputs.get('cell_offset')  # shape: (num_pairs, 3)
        cell_offsets_lr = inputs.get('cell_offset_lr')  # shape: (num_pairs_lr, 3)
        batch_segments = inputs.get('batch_segments')  # shape: (N,)
        if self.input_convention == 'positions':
            positions = inputs['positions']  # (N, 3)

            # Calculate pairwise distance vectors.
            r_ij = jax.vmap(
                lambda i, j: positions[j] - positions[i]
            )(idx_i, idx_j)  # (num_pairs, 3)

            # Apply minimal image convention if needed.
            if cell is not None:
                r_ij = add_cell_offsets_sparse(
                    r_ij=r_ij,
                    cell=cell,
                    cell_offsets=cell_offsets,
                    batch_segments=batch_segments,
                    idx_i=idx_i,
                )  # shape: (num_pairs,3)

            r_ij_lr = None
            long_range_indices_present = False

            # If indices for long range corrections are present they are used.
            if idx_i_lr is not None:
                assert idx_j_lr is not None
                if len(idx_i_lr) > 0:
                    assert len(idx_j_lr) > 0
                    long_range_indices_present = True

            # Calculate pairwise distance vectors on long range indices.
            if long_range_indices_present is True:
                # Calculate raw distance vectors.
                r_ij_lr = jax.vmap(
                    lambda i, j: positions[j] - positions[i]
                )(idx_i_lr, idx_j_lr)  # (num_pairs_lr, 3)

                # Apply cell offsets for periodic boundary conditions if needed.
                if cell is not None and cell_offsets_lr is not None:
                    logging.warning(
                        'The use of long range indices with PBCs has not been tested thoroughly yet, so use with care!'
                    )
                    r_ij_lr = add_cell_offsets_sparse(
                        r_ij=r_ij_lr,
                        cell=cell,
                        cell_offsets=cell_offsets_lr,
                        batch_segments=batch_segments,
                        idx_i=idx_i_lr,
                    )  # shape: (num_pairs_lr, 3)

        # Here it is assumed that PBC (if present) have already been respected in displacement calculation.
        elif self.input_convention == 'displacements':
            positions = inputs['positions']
            r_ij = inputs['displacements']  # shape : (num_pairs, 3)
            r_ij_lr = inputs.get('displacements_lr')  # shape : (num_pairs_lr, 3)
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Calculate pairwise distances.
        d_ij = safe_norm(r_ij, axis=-1)  # shape : (num_pairs)

        if r_ij_lr is not None:
            d_ij_lr = safe_norm(r_ij_lr, axis=-1)  # shape : (num_pairs_lr)
            del r_ij_lr
        else:
            d_ij_lr = None

        # Never ask the radial basis to extrapolate below the closest approach the
        # model was trained on. Only the DESCRIPTOR is floored: the angular part
        # comes from ``unit_r_ij`` (untouched, so equivariance is unaffected) and
        # ``cut``/``d_ij`` below stay on the true geometry, so the NLH/ZBL
        # repulsion and the electrostatics still see the real contact and still
        # diverge there. Without this the learned heads extrapolate wildly at a
        # steric clash -- on so3lr-m a 0.2 A contact puts >100 e on atoms a bond
        # away, and the resulting Coulomb term buries the repulsion (2.5e4 eV of
        # electrostatics against 200 eV of repulsion).
        d_ij_basis = floor_descriptor_distance(
            d_ij, inputs['atomic_numbers'], idx_i, idx_j)

        # Gaussian basis expansion of distances.
        rbf_ij = self.rbf_fn(jnp.expand_dims(d_ij_basis, axis=-1))  # shape: (num_pairs, num_radial_basis_fn)

        # Radial cutoff function.
        cut = self.cut_fn(d_ij)  # shape: (num_pairs)

        # Normalized distance vectors.
        unit_r_ij = safe_mask(mask=jnp.expand_dims(d_ij, axis=-1) > 0,
                              operand=r_ij,
                              fn=lambda y: y / jnp.expand_dims(d_ij, axis=-1),
                              placeholder=0
                              )  # shape: (num_pairs, 3)

        # Spherical harmonics.
        ylm_ij = [ylm_fn(unit_r_ij) for ylm_fn in self.ylm_fns]
        ylm_ij = jnp.concatenate(ylm_ij, axis=-1) if len(ylm_ij) > 0 else None  # shape: (num_pairs, m_tot)

        # Determine number of atoms from positions or atomic_numbers
        num_atoms = len(inputs.get('atomic_numbers', positions))

        geometric_data = {'positions': positions,
                          'r_ij': r_ij,
                          'unit_r_ij': unit_r_ij,
                          'd_ij': d_ij,
                          'd_ij_lr': d_ij_lr,
                          'rbf_ij': rbf_ij,
                          'cut': cut,
                          'ylm_ij': ylm_ij,
                          'ev': jnp.zeros((num_atoms, ylm_ij.shape[-1]), dtype=r_ij.dtype)
                          }

        return geometric_data

    def reset_input_convention(self, input_convention: str) -> None:
        self.input_convention = input_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'radial_basis_fn': self.radial_basis_fn,
                                   'num_radial_basis_fn': self.num_radial_basis_fn,
                                   'cutoff_fn': self.cutoff_fn,
                                   'cutoff': self.cutoff,
                                   'input_convention': self.input_convention,
                                   'prop_keys': self.prop_keys}
                }


class AtomTypeEmbedSparse(BaseSubModule):
    num_features: int
    prop_keys: Dict
    zmax: int = 118
    module_name: str = 'atom_type_embed_sparse'

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the atomic types.

        Args:
            inputs (Dict):
                atomic_numbers (Array): atomic types, shape: (num_nodes)
            *args (Tuple):
            **kwargs (Dict):

        Returns: Atomic embeddings, shape: (num_nodes,num_features)

        """
        atomic_numbers = inputs['atomic_numbers']
        return nn.Embed(num_embeddings=self.zmax + 1, features=self.num_features)(atomic_numbers)

    def __dict_repr__(self):
        return {self.module_name: {'num_features': self.num_features,
                                   'zmax': self.zmax,
                                   'prop_keys': self.prop_keys}}


class ChargeSpinEmbedSparse(nn.Module):
    num_features: int
    activation_fn: str = 'silu'
    zmax: int = 118

    @nn.compact
    def __call__(self,
                 atomic_numbers: jnp.ndarray,
                 psi: jnp.ndarray,
                 batch_segments: jnp.ndarray,
                 graph_mask: jnp.ndarray,
                 *args,
                 **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the total charge or the number of unpaired spins in the system, following the
        embedding procedure introduced in SpookyNet. Returns per atom embeddings of dimension F.

        Args:
            z (Array): Atomic types, shape: (N)
            psi (Array): Total charge or number of unpaired spins, shape: (num_graphs)
            batch_segment (Array): (N)
            graph_mask (Array): Mask for atom-wise operations, shape: (num_graphs)
            *args ():
            **kwargs ():

        Returns: Per atom embedding, shape: (n,F)

        """
        assert psi.ndim == 1

        q = nn.Embed(
            num_embeddings=self.zmax + 1,
            features=self.num_features
        )(atomic_numbers)  # shape: (N,F)

        psi_ = psi // jnp.inf  # -1 if psi < 0 and 0 otherwise
        psi_ = psi_.astype(jnp.int32)  # shape: (num_graphs)

        k = nn.Embed(
            num_embeddings=2,
            features=self.num_features
        )(psi_)[batch_segments]  # shape: (N, F)

        v = nn.Embed(
            num_embeddings=2,
            features=self.num_features
        )(psi_)[batch_segments]  # shape: (N, F)

        q_x_k = (q*k).sum(axis=-1) / jnp.sqrt(self.num_features)  # shape: (N)

        y = nn.softplus(q_x_k)  # shape: (N)
        denominator = jax.ops.segment_sum(
            y,
            segment_ids=batch_segments,
            num_segments=len(graph_mask)
        )  # (num_graphs)

        denominator = jnp.where(
            graph_mask,
            denominator,
            jnp.asarray(1., dtype=q.dtype)
        )  # (num_graphs)

        a = psi[batch_segments] * y / denominator[batch_segments]  # shape: (N)
        e_psi = Residual(
            use_bias=False,
            activation_fn=getattr(jax.nn, self.activation_fn) if self.activation_fn != 'identity' else lambda u: u
        )(jnp.expand_dims(a, axis=-1) * v)  # shape: (N, F)

        return e_psi


class ChargeEmbedSparse(BaseSubModule):
    prop_keys: Dict
    num_features: int
    activation_fn: str = 'silu'
    zmax: int = 118
    module_name: str = 'charge_embed_sparse'

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                atomic_numbers (Array): atomic types, shape: (N)
                total_charge (Array): total charge, shape: (num_graphs)
                graph_mask (Array): (num_graphs)
                batch_segments (Array): (N)
            *args ():
            **kwargs ():

        Returns:

        """
        atomic_numbers = inputs['atomic_numbers']
        Q = inputs['total_charge']
        graph_mask = inputs['graph_mask']
        batch_segments = inputs['batch_segments']

        if Q is None:
            raise ValueError(
                f'ChargeEmbedSparse requires to pass `total_charge != None`.'
            )

        return ChargeSpinEmbedSparse(
            zmax=self.zmax,
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(
            atomic_numbers=atomic_numbers,
            psi=Q,
            batch_segments=batch_segments,
            graph_mask=graph_mask
            )

    def __dict_repr__(self):
        return {self.module_name: {'num_features': self.num_features,
                                   'zmax': self.zmax,
                                   'prop_keys': self.prop_keys}}


class SpinEmbedSparse(BaseSubModule):
    prop_keys: Dict
    num_features: int
    activation_fn: str = 'silu'
    zmax: int = 118
    module_name: str = 'spin_embed_sparse'

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                atomic_numbers (Array): atomic types, shape: (N)
                num_unpaired_electrons (Array): total charge, shape: (num_graphs)
                graph_mask (Array): (num_graphs)
                batch_segments (Array): (N)
            *args ():
            **kwargs ():

        Returns:

        """
        atomic_numbers = inputs['atomic_numbers']
        S = inputs['num_unpaired_electrons']
        graph_mask = inputs['graph_mask']
        batch_segments = inputs['batch_segments']

        if S is None:
            raise ValueError(
                f'SpinEmbedSparse requires to pass `num_unpaired_electrons != None`.'
            )

        return ChargeSpinEmbedSparse(
            zmax=self.zmax,
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(
            atomic_numbers=atomic_numbers,
            psi=S,
            batch_segments=batch_segments,
            graph_mask=graph_mask
            )

    def __dict_repr__(self):
        return {self.module_name: {'num_features': self.num_features,
                                   'zmax': self.zmax,
                                   'prop_keys': self.prop_keys}}
