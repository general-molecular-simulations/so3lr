from typing import Dict
from functools import partial

from .lr_decay import exponential, on_plateau


def get_lr_decay(name: str, h: Dict):
    if name == 'exponential':
        return partial(exponential, **h)
    elif name == 'on_plateau':
        return partial(partial(on_plateau), **h)
    else:
        msg = "{} is not a known learning rate decay function.".format(name)
        raise ValueError(msg)

