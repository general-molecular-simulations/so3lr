from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def make_l0_contraction_fn(degrees, dtype=jnp.float32):
    sizes = np.array([2 * d + 1 for d in degrees])  # m per degree
    # Clebsch-Gordan coefficient for coupling two degree-l features to l=0: 1/sqrt(2l+1)
    # for every m. Written as sqrt(2l+1)/(2l+1), which is bit-identical to the
    # tabulated values the shipped models were trained with (1/sqrt differs in the last bit).
    cg_rep = jnp.array(np.repeat(np.sqrt(sizes) / sizes, sizes), dtype=dtype)  # shape: (m_tot)
    segment_ids = jnp.array(np.repeat(np.arange(len(degrees)), sizes))
    _segment_sum = jax.vmap(partial(jax.ops.segment_sum, segment_ids=segment_ids, num_segments=len(degrees)))

    def contraction_fn(sphc):
        """
        Args:
            sphc (Array): Spherical harmonic coordinates, shape: (n,m_tot)
        Returns: Contraction on degree l=0 for each degree up to l_max, shape: (n,|l|)
        """

        return _segment_sum(sphc * sphc * cg_rep)  # shape: (n,len(degrees))

    return contraction_fn
