import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn

from importlib.resources import files
import pickle
import itertools as it

from functools import partial
from typing import (Callable, Sequence)


indx_fn = lambda x: int((x+1)**2) if x >= 0 else 0

def load_cgmatrix():
    resource_path = files('so3lr.mlff.sph_ops') / 'cgmatrix.npz'
    with resource_path.open('rb') as f:
        return np.load(f)['cg']

def init_clebsch_gordan_matrix(degrees, l_out_max=None):
    """
    Initialize the Clebsch-Gordan matrix (coefficients for the Clebsch-Gordan expansion of spherical basis functions)
    for given ``degrees`` and a maximal output order ``l_out_max`` up to which the given all_degrees shall be
    expanded. Minimal output order is ``min(degrees)``.

    Args:
        degrees (List): Sequence of degrees l. The lowest order can be chosen freely. However, it should
            be noted that all following all_degrees must be exactly one order larger than the following one. E.g.
            [0,1,2,3] or [1,2,3] are valid but [0,1,3] or [0,2] are not.
        l_out_max (int): Maximal output order. Can be both, smaller or larger than maximal order in degrees.
            Defaults to the maximum value of the passed degrees.

    Returns: Clebsch-Gordan matrix,
        shape: (``(l_out_max+1)**2, (l_out_max+1)**2 - (l_in_min)**2, (l_out_max+1)**2 - (l_in_min)**2``)

    """
    if l_out_max is None:
        _l_out_max = max(degrees)
    else:
        _l_out_max = l_out_max

    l_in_max = max(degrees)
    l_in_min = min(degrees)

    offset_corr = indx_fn(l_in_min - 1)
    _cg = load_cgmatrix()
    return _cg[offset_corr:indx_fn(_l_out_max), offset_corr:indx_fn(l_in_max), offset_corr:indx_fn(l_in_max)]


def init_expansion_fn(degrees: Sequence[int], cg: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Initialize function that returns the expansion in CG coefficients for all possible combinations for given
    ``degrees`` up to arbitrary order ``l3``, which is determined by the order along the 0-th axis of CG matrix,
    stores in ``cg``. The returned function takes the outer product of different orders of the spherical basis to the
    respective expansions in arbitrary order ``l3``.

    Args:
        degrees (List): Sequence of harmonic orders. The lowest order can be chosen freely. However, it should
            be noted that all following orders must be exactly one order larger than the following one. E.g.
            [0,1,2,3] or [1,2,3] are valid but [0,1,3] or [0,2] are not.
        cg (Array): Matrix that contains the CG coefficients for given `degrees`. They can be constructed
            using the function `ramses.utils.geometric.init_clebsch_gordan_matrix`.
            shape: (``(l_out_max+1)**2, (l_out_max+1)**2 - (l_in_min)**2, (l_out_max+1)**2 - (l_in_min)**2``)

    Returns: Expansion function, which takes the outer product of spherical basis functions to the correct expansion
        for arbitrary order l3. Illustrated in more detail in the example below.

    Examples:
        >>> from ramses.utils.geometric.spherical_harmonics import init_sph_fn
        >>> from mlff.src. import coordinates_to_distance_vectors_normalized
        >>> # Calculate the normalized distance vectors given some points in Euclidean space.
        >>> n = 4
        >>> p0 = generate_points(n=n, d_min=1, d_max=3)
        >>> rij = coordinates_to_distance_vectors_normalized(p0)  # shape: (n,n,3)
        >>> l_out_max=3
        >>> all_degrees = [0,1,2]
        >>> # Construct SPH coordinates in gamma.
        >>> alpha = (jnp.eye(rij.shape[-2]) == 0)[..., None]  # shape: (n,n,1)
        >>> sph_harms = []
        >>> gamma = []
        >>> sph_fns = [init_sph_fn(l) for l in all_degrees]
        >>> for sph_fn in sph_fns:
            >>> sph_ij = sph_fn(rij)  # shape: (n,n,2l+1)
            >>> gamma += [(sph_ij*alpha).sum(axis=-2)]
        >>> gamma = jnp.concatenate(gamma, axis=-1)  # shape: (n,m_tot)
        >>> # Take outer product of SPH coordinates.
        >>> gammaxgamma = jnp.einsum('nm, nl -> nml', gamma, gamma)[:, None, :, :]  # shape: (n,1,m_in,m_in)
        >>> # Initialize the Clebsch-Gordan matrix.
        >>> cg = init_clebsch_gordan_matrix(degrees=all_degrees, l_out_max=l_out_max)
        >>> # shape: (m_out,m_in,m_in)
        >>> # Initialize the expansion_fn
        >>> expansion_fn = init_expansion_fn(degrees=all_degrees, cg=cg)
        >>> J = expansion_fn(gammaxgamma)
        >>> J.shape
        (4,16,3,3)
        >>> # One can now obtain the expansion of e.g. order l1=1 and l2=2 in the basis functions of order l3=3 as
        >>> l1 = 1
        >>> l2 = 2
        >>> l1l2_l3 = J[:, 9:, l1, l2]
        >>> l1l2_l3.shape
        (4,7)
        >>> l1 = 0
        >>> l2 = 2
        >>> # or similar the expansion of e.g. order l1=0 and l2=2 in the basis functions of order l3=2 as
        >>> l1l2_l3 = J[:, 4:9, l1, l2]
        >>> l1l2_l3.shape
        (4,5)
        >>> # the SelfMix layer without parametrization from PhisNet can then be written as. Since we now that l2 > l1,
        >>> # we just sum over the upper tri-diagonal part of the matrix.
        >>> gamma_contracted = jnp.triu(J, k=1).sum(axis=(-2, -1))  # shape: (n, m_out)
    """
    segment_ids = jnp.array(
        [y for y in it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])])
    num_segments = len(degrees)
    _segment_sum = partial(jax.ops.segment_sum, segment_ids=segment_ids, num_segments=num_segments)
    _v_segment_sum_l1 = jax.vmap(jax.vmap(jax.vmap(_segment_sum)))
    _v_segment_sum_l2 = jax.vmap(jax.vmap(_segment_sum))
    _expansion_fn = jax.jit(lambda x: _v_segment_sum_l1(_v_segment_sum_l2(x * cg)))
    return _expansion_fn


def make_l0_contraction_fn(degrees, dtype=jnp.float32):
    # get CG coefficients
    cg = np.diagonal(init_clebsch_gordan_matrix(degrees=list({0, *degrees}), l_out_max=0), axis1=1, axis2=2)[0]
    # shape: (m_tot**2)
    cg_rep = []
    for d, r in zip(*np.unique(np.array(degrees), return_counts=True)):
        cg_rep += [np.tile(cg[indx_fn(d - 1): indx_fn(d)], r)]

    cg_rep = np.concatenate(cg_rep)  # shape: (m_tot), m_tot = \sum_l 2l+1 for l in degrees
    cg_rep = jnp.array(cg_rep, dtype=dtype)  # shape: (m_tot)

    segment_ids = jnp.array(
        [y for y in it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])])
    num_segments = len(degrees)
    _segment_sum = jax.vmap(partial(jax.ops.segment_sum, segment_ids=segment_ids, num_segments=num_segments))

    def contraction_fn(sphc):
        """
        Args:
            sphc (Array): Spherical harmonic coordinates, shape: (n,m_tot)
        Returns: Contraction on degree l=0 for each degree up to l_max, shape: (n,|l|)
        """

        return _segment_sum(sphc*sphc*cg_rep[None, :])  # shape: (n,len(degrees))

    return contraction_fn

def load_u_matrix():
    resource_path = files('so3lr.mlff.sph_ops') / 'u_matrix.pickle'
    with resource_path.open('rb') as f:
        return pickle.load(f)


def degrees_to_str(x):
    _x = [str(y) for y in x]
    return ''.join(_x)


_u_matrix = load_u_matrix()


def get_U_matrix(degrees_in, degree_out, correlation):
    degrees_str = degrees_to_str(degrees_in)
    return _u_matrix[correlation][degrees_str][degree_out]


class SymmetricContraction(nn.Module):
    """r Class for building higher body-order representations. It uses `ContractionToIrrep` to create multiple
    concatenated higher body-order representations, each transforming according
    to different irrep.
    Args:
        degrees_out (Sequence[int]): the irreps of the concatenated output
            tensors.
        degrees_in (Sequence[int]): the irreps of the concatenated input
            tensors.
        n_feature (int): Feature dimension.
        max_body_order (int): output tensors up to body-order :data:`max_body_order`
            are calculated and their sum is returned.
        n_node_type (int): Number of different node types.
    """
    degrees_out: Sequence[int]
    degrees_in: Sequence[int]
    n_feature: int
    max_body_order: int
    n_node_type: int

    def setup(self) -> None:
        contractions = {}
        for degree_out in self.degrees_out:
            contractions[degree_out] = ContractionToIrrep(
                degree_out, self.degrees_in, self.n_feature, self.max_body_order, self.n_node_type
            )
        self.contractions = contractions

    @nn.compact
    def __call__(self, A, node_attrs=None):
        """
        Build higher body-order representations, from representations that transform SO(3) equivariant.

        Args:
            A (Array): Equivariant representations, shape: (*,F,m_tot)
            node_attrs ():

        Returns: Higher body-order representations for each irrep, shape: (*,F,m_tot)

        """
        Bs = []
        for degree_out in self.degrees_out:
            Bs.append(self.contractions[degree_out](A, node_attrs))
        return jnp.concatenate(Bs, axis=-1)


class ContractionToIrrep(nn.Module):
    r"""
    Create higher body-order tensors transforming according to some irrep.
    Taking as input concatenated 2-body tensors that transform according to
    :data:`degrees_in`, it calculates their tensor products using the generalized
    Clebsch--Gordan coefficients, to return a sum of higher body-order tensors
    that transforms as :data:`degrees_out`.
    Input array must have the shape
    [:math:`N_\text{batch}`, :math:`N_\text{feature}, :math:`\sum_{i}(2 l_i + 1)`],
    where :math:`i`, runs over :data:`irreps_in`. The output array has the shape
    [:math:`N_\text{batch}`, :math:`N_\text{feature}`, :math:`2 l_\text{out} + 1`].
    Args:
        irrep_out (e3nn_jax.Irrep): the irrep of the output tensor
        irreps_in (Sequence[e3nn_jax.Irrep]): the irreps of the concatenated input
            tensors.
        n_feature (int): the number of features of the input tensors.
        max_body_order (int): output tensors up to body-order :data:`max_body_order`
            are calculated and their sum is returned.
        n_node_type: (int) Number of different node types.
    """
    degree_out: int
    degrees_in: Sequence[int]
    n_feature: int
    max_body_order: int
    n_node_type: int

    def setup(self) -> None:
        if self.max_body_order < 2:
            raise ValueError(f"Maximal body order has to be larger than 2. Body order is {self.max_body_order}.")

        self.correlation = self.max_body_order - 2
        self.scalar_out = self.degree_out == 0

        U_matrices = []

        for nu in range(1, self.max_body_order):
            U = get_U_matrix(degrees_in=self.degrees_in,
                             degree_out=self.degree_out,
                             correlation=nu)
            if self.degrees_in == [0]:
                # U matrix for single scalar input is missing all but its
                # last dimension (all size 1), we need to add it manually
                for _ in range(nu):
                    U = U[None]

            U_matrices.append(U)

        self.U_matrices = U_matrices

        self.equation_init = "...ik,ekc,bci,be -> bc..."
        self.equation_weighting = "...k,ekc,be -> bc..."
        self.equation_contract = "bc...i,bci -> bc..."

        weights = []
        for nu in range(1, self.max_body_order):
            # number of ways irrep_out can be created from irreps_in at body order nu:
            n_coupling = self.U_matrices[nu - 1].shape[-1]
            weights.append(
                self.param(
                    f"coupling_weights_{nu}",
                    nn.initializers.lecun_normal(),
                    [self.n_node_type, n_coupling, self.n_feature],
                )
            )
        self.weights = weights

    @nn.compact
    def __call__(self, A, node_types):
        # node_types is onehot encoded, it selects the index of weights,
        # and is usually faster than indexing
        B = jnp.einsum(
            self.equation_init,
            jnp.asarray(self.U_matrices[self.correlation], dtype=A.dtype),
            self.weights[self.correlation],
            A,
            node_types,
        )
        for corr in reversed(range(self.correlation)):
            c_tensor = jnp.einsum(
                self.equation_weighting,
                jnp.asarray(self.U_matrices[corr], A.dtype),
                self.weights[corr],
                node_types,
            )
            c_tensor = c_tensor + B
            B = jnp.einsum(self.equation_contract, c_tensor, A)
        return B[..., None] if self.scalar_out else B
