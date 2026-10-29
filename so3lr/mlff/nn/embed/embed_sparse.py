import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import (Any, Dict, Optional, Sequence)

import flax.linen as nn
import logging

from so3lr.mlff.nn.sub_module import BaseSubModule
from so3lr.mlff.nn.mlp import Residual
from so3lr.mlff.masking.mask import safe_mask
from so3lr.mlff.masking.mask import safe_norm
from so3lr.mlff.geometric import add_cell_offsets_sparse
from so3lr.mlff import utils
from so3lr.mlff.sph_ops import init_sph_fn


# --- descriptor distance floor -------------------------------------------------
# Pushed far enough inside its trained range, the learned part of the model does
# not merely become inaccurate, it leaves the physical regime: the energy head
# returns >100 eV for a two-atom system, forces exceed 1e3 eV/A, and on so3lr-tauri-l a
# compressed S-S pair returns NaN outright. The analytic terms never do this --
# NLH/ZBL repulsion, electrostatics and dispersion are closed-form and stay valid
# at any separation. So only the distance handed to the RADIAL BASIS is floored.
# Every physics term, the cutoff and the angular part (from ``unit_r_ij``, so
# equivariance is untouched) keep the true geometry and still diverge at contact.
#
# WHERE the floor engages is the whole question, because the model was trained
# with the existing repulsion and switching function and that behaviour must not
# be touched while the model is still interpolating. The handover is therefore
# set by the pair's OWN repulsion: the floor is exactly the identity until the
# tabulated NLH repulsion for that element pair already exceeds
# PAIR_FLOOR_ENERGY. That is a per-pair length with nothing to hand-tune, and it
# scales the way the failures do -- the heavier the pair, the further out.
PAIR_FLOOR_ENERGY = 5.0    # [eV] engage only where the pair's repulsion exceeds this
PAIR_FLOOR_WIDTH = 0.05    # [A] handover; the floor is exactly inert above it
# 5 eV is bounded from both sides by measurement: the lower bound from dimer
# scans of the three -2606 checkpoints, the upper bound from the 73 labelled test
# sets. It is the value the SO3LR-2 paper's test-set errors were computed with
# (SI, "Behaviour at close contact"); the per-pair numbers are MEASURED in
# tests/mlff_tests/test_descriptor_floor.py, from floor_analysis/floor_pairs.csv
# in the paper's evaluation data.
#
#   lower bound  the failure onset per pair -- the outermost separation where the
#                UN-floored network diverges (|F| > 1e3 eV/A or NaN) on dimer
#                scans. rmin must reach past it.
#   upper bound  the closest REAL contact per pair across every labelled test set.
#                The training data goes far closer than intuition suggests: the
#                scaled-separation sets label down to 0.4 A, and real contacts
#                reach O-H 0.666, N-H 0.668, C-H 0.691, H-H 0.407. Where the
#                floor engages on those it is neutral-to-harmful, because the
#                model is inaccurate but NOT diverging there -- it is
#                interpolating, and flooring distorts what it learned.
#
# At 5 eV rmin sits below the closest real contact for EVERY pair, which is why
# it beats the 3 eV this started at: on the affected subsets forces improve, e.g.
# so3lr-tauri-l on geom_orca6 RMSE 4.55 -> 2.98 kcal/mol/A, and 48/63 model x
# subset entries improve while 4 are negligibly worse. 23 of 26 pairs are also
# protected. C-H is the price: 4.4-4.6 eV would protect it too while keeping
# every rmin below its real contact, but was never evaluated.
#
# THREE PAIRS sit outside rmin's reach at this setting: P-P (onset 1.30 A),
# P-S (1.26 A) and C-H (0.680, against a closest real contact of 0.691). Those
# onsets were measured on BARE DIMERS, which are themselves deeply out of
# distribution -- the model never saw two isolated P atoms -- so they overstate
# the risk. Re-measured with the atoms carrying their normal valence
# (PH3...PH3, PH3...H2S, and one C-H bond of CH4 compressed), only one of the
# three is a real problem:
#
#   C-H  clean in CH4. Compressing one bonded C-H gives a monotone wall on all
#        four checkpoints (legacy so3lr and the three -2606 models); the
#        divergence survives in the bare C-H radical dimer on so3lr-tauri-l.
#        Non-bonded C...H clashes (docking poses, overlapping solvent) were not
#        scanned, and there the floor now clamps inside the measured band.
#   P-P  fixed. Clean in PH3...PH3 on all four; the bare dimer on so3lr-tauri-l
#        keeps bounded oscillations of a few tens of eV.
#   P-S  NOT fixed, and worse than the dimer suggested. In PH3...H2S the energy
#        dives to -1.8 keV on so3lr-tauri-m and -42 keV on so3lr-tauri-l around
#        0.9-1.2 A, with forces reaching 4e5 eV/A -- i.e. inside a realistic
#        geometry, only ~0.3 A below the 1.534 A closest P-S contact in the
#        labelled data.
#        A global energy cannot reach it: rmin is 1.07 A at 5 eV and 1.17 A at
#        1 eV, and reaching the 1.26 A onset would take ~0.13 eV. It needs a
#        per-pair override or a stronger P repulsion. Until then, treat P/S
#        systems near contact as unguarded.
#
# WIDTH is still 0.05 because that is what the 73-set evaluation above was run
# with. It buys ZERO protection -- that is set by rmin alone -- and adds
# intrusion through the handover shoulder: at 0.05 it reaches the closest real
# B-H, C-H, F-H, N-H and O-H contacts, moving them by under 0.005 A but scaling
# the radial derivative there by 0.65-0.95. A narrower width intrudes less but
# bends more sharply (curvature ~1/width), so 0.02 is a trade-off to evaluate,
# not a free improvement.
#
# Deliberately NOT covered: below ~1.3 A the S-S curve has a shallow spurious
# minimum behind a ~2 eV barrier on legacy so3lr and so3lr-tauri-s. It is bounded
# and finite, and at 1.31 A the S-S repulsion is 0.007 eV, so flooring there would
# mean overriding the model where its own repulsion is negligible -- exactly the
# interpolating regime this constant exists to protect. Reported, not patched.
#
# Scope: the floor runs in every model built on GeometryEmbedSparse -- legacy
# so3lr (whose repulsion is ZBL, not the NLH table used here) and training and
# fine-tuning forward passes included -- while the calibration above is for the
# three -2606 checkpoints only (not -2602). Editing either constant here changes
# all of them. Overriding them at runtime does not: nlh_reference_distance binds
# the energy as its default at import, and so3lr.flash copies the width.

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

    Exactly the identity at and above ``rmin + PAIR_FLOOR_WIDTH``, so ordinary
    chemistry is bit-for-bit untouched. Below it the distance is bent smoothly
    onto ``rmin`` -- C1 at the handover, monotone, finite-gradient at d = 0.
    Apply to the descriptor only: repulsion and electrostatics need the true
    distance, and flooring those would change the physics rather than protect it.
    """
    table = jnp.asarray(nlh_reference_distance())
    zi = atomic_numbers[idx_i] - 1
    zj = atomic_numbers[idx_j] - 1
    rmin = table[jnp.minimum(zi, zj), jnp.maximum(zi, zj)]
    d_off = rmin + PAIR_FLOOR_WIDTH
    u = (d_off - d_ij) / PAIR_FLOOR_WIDTH
    return jnp.where(d_ij >= d_off, d_ij, d_off - PAIR_FLOOR_WIDTH * jnp.tanh(u))


def pair_displacements(inputs: Dict):
    """(r_ij, r_ij_lr): R_j - R_i for the short-range pairs (idx_i, idx_j) and the long-range ones
    (idx_i_lr, idx_j_lr; None without them), each plus its periodic image's offset when the
    inputs carry a cell (long-range: and cell_offset_lr). In the positions' dtype."""
    positions, cell, batch_segments = inputs['positions'], inputs.get('cell'), inputs.get('batch_segments')

    def displacements(idx_i, idx_j, cell_offsets):
        r_ij = positions[idx_j] - positions[idx_i]  # (num_pairs, 3)
        if cell is None:
            return r_ij
        return add_cell_offsets_sparse(r_ij=r_ij, cell=cell, cell_offsets=cell_offsets,
                                       batch_segments=batch_segments, idx_i=idx_i)

    r_ij = displacements(inputs['idx_i'], inputs['idx_j'], inputs.get('cell_offset'))
    idx_i_lr, idx_j_lr, cell_offsets_lr = inputs.get('idx_i_lr'), inputs.get('idx_j_lr'), inputs.get('cell_offset_lr')
    if idx_i_lr is None or len(idx_i_lr) == 0:
        return r_ij, None
    assert idx_j_lr is not None and len(idx_j_lr) > 0
    if cell_offsets_lr is None:
        return r_ij, positions[idx_j_lr] - positions[idx_i_lr]
    if cell is not None:
        logging.warning(
            'The use of long range indices with PBCs has not been tested thoroughly yet, so use with care!')
    return r_ij, displacements(idx_i_lr, idx_j_lr, cell_offsets_lr)


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
        if self.input_convention == 'positions':
            positions = inputs['positions']  # (N, 3)
            r_ij, r_ij_lr = pair_displacements(inputs)

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

        # Floor ONLY the distance the radial basis sees, so a clash cannot drive
        # the learned heads outside the regime they were trained in. The angular
        # part comes from ``unit_r_ij`` (untouched, so equivariance is unaffected)
        # and ``cut``/``d_ij`` below stay on the true geometry, so the NLH/ZBL
        # repulsion and the electrostatics still diverge at real contact.
        d_ij_basis = floor_descriptor_distance(
            d_ij, inputs['atomic_numbers'], idx_i, idx_j)

        # Gaussian basis expansion of distances.
        rbf_ij = self.rbf_fn(jnp.expand_dims(d_ij_basis, axis=-1))  # shape: (num_pairs, num_radial_basis_fn)

        # Radial cutoff function, zero on d = 0 edges. Every padding convention
        # (jraph's self-loops, index N in jax-md and the exact-image list) puts its
        # edges at d = 0, where phys(0) = 1: ~10^5 such edges overflowed the force
        # loss's second-order gradient to NaN. Real atoms never see them.
        cut = jnp.where(d_ij > 0, self.cut_fn(d_ij), 0.)  # shape: (num_pairs)

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
    # Inference-time override for the charge fed to THIS module only. None ->
    # use the graph's ``total_charge`` (the default; behaviour unchanged). A
    # float -> embed that value instead, while charge equilibration in
    # ElectrostaticEnergySparse keeps using the true ``total_charge``.
    #
    # Deliberately not ``use_charge_embed``: that flag decides whether this
    # module exists at all (see make_embedding_modules), so flipping it changes
    # the parameter tree and a released checkpoint no longer loads. Overriding
    # the *input* keeps every trained weight in place. At 0.0 the contribution
    # is exactly zero -- ``a = psi * y / denominator`` in ChargeSpinEmbedSparse
    # vanishes and its Residual carries no bias.
    #
    # Not exposed in __dict_repr__ on purpose: this is a per-run choice, not
    # architecture, and must not be written into hyperparameters.json.
    total_charge_override: Optional[float] = None
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

        if self.total_charge_override is not None:
            # full_like keeps psi at (num_graphs) and floating dtype, which
            # ChargeSpinEmbedSparse asserts on.
            Q = jnp.full_like(Q, self.total_charge_override)

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
