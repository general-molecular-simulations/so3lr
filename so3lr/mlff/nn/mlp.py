import jax.numpy as jnp

from jax.nn import silu
from flax import linen as nn
from typing import Callable


class Residual(nn.Module):
    num_blocks: int = 2
    activation_fn: Callable = silu
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        feat = x.shape[-1]
        for i in range(self.num_blocks):
            x = self.activation_fn(x)
            x = nn.Dense(feat, use_bias=self.use_bias, name=f'layers_{i}')(x)

        return inputs + x


