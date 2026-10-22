"""A model directory's hyperparameters, completed from its checkpoint where an old config is silent.

``legacy_so3lr_bool`` (the v1 heads: attention Hirshfeld head, learnable ZBL, no C6 head)
and ``use_final_bias_bool`` (a bias in the energy readout's final layer) are absent from
configs written before they existed, the original v1 one among them, and the readers
default both to False. A v1 checkpoint loaded with those defaults is v1 for the network
and v2 for everything else that reads the config (element coverage, eval's ratio key,
fine-tuning, the flash guards), so every reader goes through :func:`load_model_config`.
An explicit value in the config is never overridden.

Kept import-cheap: the checkpoint is only read when a flag is missing.
"""
import copy
import json
import logging
import pathlib

logger = logging.getLogger("SO3LR")

# flags read from the checkpoint when a config leaves them out
_INFERRED = ('legacy_so3lr_bool', 'use_final_bias_bool')

_CACHE = {}


def infer_flags_from_params(params):
    """The flags a parameter tree implies: the v1 Hirshfeld head stores a query embedding
    ('Embed_1') under observables_2, which the v2 scalar head does not; the final bias is
    a 'bias' leaf in observables_0/energy_dense_final."""
    p = params.get('params', params)
    return {
        'legacy_so3lr_bool': 'Embed_1' in p.get('observables_2', {}),
        'use_final_bias_bool': 'bias' in p.get('observables_0', {}).get('energy_dense_final', {}),
    }


def load_model_config(workdir, params=None):
    """``hyperparameters.json`` of a model directory as a dict, with ``legacy_so3lr_bool``
    and ``use_final_bias_bool`` read from the checkpoint when the config does not set them.

    ``params`` (the directory's parameter tree, if already loaded) saves reading the
    checkpoint again. Results are cached per directory and file version; callers get a copy.
    """
    path = pathlib.Path(workdir).expanduser().resolve() / 'hyperparameters.json'
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _CACHE:
        with open(path) as fp:
            cfg = json.load(fp)
        model = cfg.setdefault('model', {})
        missing = [k for k in _INFERRED if k not in model]
        if missing:
            if params is None:
                from so3lr.mlff.utils.checkpoint_utils import load_params_from_workdir
                params = load_params_from_workdir(path.parent)
            inferred = infer_flags_from_params(params)
            for k in missing:
                model[k] = inferred[k]
            log = logger.info if any(inferred[k] for k in missing) else logger.debug
            log('%s does not set %s; read from the checkpoint: %s', path, ', '.join(missing),
                ', '.join(f'{k}={inferred[k]}' for k in missing))
        _CACHE[key] = cfg
    return copy.deepcopy(_CACHE[key])
