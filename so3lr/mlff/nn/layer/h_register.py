from typing import Dict
from .so3krates_layer_sparse import SO3kratesLayerSparse


def get_layer(name: str, h: Dict):
    if name == 'so3krates_layer_sparse':
        return SO3kratesLayerSparse(**h)
    elif name == 'spookynet_layer':
        raise NotImplementedError('SpookyNet not implemented!')
    elif name == 'painn_layer':
        raise NotImplementedError('PaiNN not implemented!')
    else:
        msg = f"Layer with `module_name={name}` is not implemented."
        raise NotImplementedError(msg)
