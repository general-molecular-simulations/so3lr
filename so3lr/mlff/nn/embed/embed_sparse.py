import jax
import jax.numpy as jnp

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

        # Gaussian basis expansion of distances.
        rbf_ij = self.rbf_fn(jnp.expand_dims(d_ij, axis=-1))  # shape: (num_pairs, num_radial_basis_fn)

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
