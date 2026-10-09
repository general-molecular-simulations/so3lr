import itertools as it
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def make_l0_contraction_fn(degrees, dtype=jnp.float32):
    # Clebsch-Gordan coefficient for coupling two degree-l features to l=0: 1/sqrt(2l+1)
    # for every m. Written as sqrt(2l+1)/(2l+1), which is bit-identical to the
    # tabulated values the shipped models were trained with (1/sqrt differs in the last bit).
    cg_rep = np.concatenate([np.full(2 * d + 1, np.sqrt(2 * d + 1) / (2 * d + 1)) for d in degrees])
    cg_rep = jnp.array(cg_rep, dtype=dtype)  # shape: (m_tot), m_tot = \sum_l 2l+1 for l in degrees

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
