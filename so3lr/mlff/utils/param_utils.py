import jax.numpy as jnp
import numpy as np


def print_param_shapes(params, prefix=''):
    if isinstance(params, dict):
        for key, value in params.items():
            if isinstance(value, (dict, jnp.ndarray)):
                if isinstance(value, jnp.ndarray):
                    print(f"{prefix}{key}: {value.shape}")
                else:
                    print(f"{prefix}{key}:")
                    print_param_shapes(value, prefix + '  ')


def count_params(params_dict):
    total = 0
    if isinstance(params_dict, dict):
        for key, value in params_dict.items():
            if isinstance(value, jnp.ndarray):
                total += value.size
            elif isinstance(value, dict):
                total += count_params(value)
            elif isinstance(value, np.ndarray):
                total += value.size
    return total
