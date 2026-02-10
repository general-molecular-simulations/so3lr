from typing import Dict
from functools import partial

from .stopping_fn import stop_by_lr, stop_by_metric


def get_stopping_fn(name: str, h: Dict):
    if name == 'stop_by_lr':
        return partial(stop_by_lr, **h)
    elif name == 'stop_by_metric':
        return partial(partial(stop_by_metric), **h)
    else:
        msg = "{} is not a known stopping function.".format(name)
        raise ValueError(msg)