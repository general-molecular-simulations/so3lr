"""Training utilities for SO3LR models."""
import json
import logging
import pathlib
from typing import Optional

import yaml
from ml_collections import config_dict

from so3lr.mlff.config import from_config
from .so3lr_md import setup_logger

# Get logger
logger = logging.getLogger("SO3LR")


def train_so3lr(config_path: str, log_file: Optional[str] = None):
    """Train a SO3krates model from a config file.

    Args:
        config_path: Path to the training config file (YAML or JSON).
        log_file: Optional path to write logs to.
    """
    config = pathlib.Path(config_path).expanduser().absolute().resolve()
    if config.suffix == '.json':
        with open(config, mode='r') as fp:
            cfg = config_dict.ConfigDict(json.load(fp=fp))
    elif config.suffix == '.yaml':
        with open(config, mode='r') as fp:
            cfg = config_dict.ConfigDict(yaml.load(fp, Loader=yaml.FullLoader))
    else:
        raise ValueError(f"Config file must be .json or .yaml, got: {config.suffix}")

    # Default the log file to the config name with a .log extension
    # (e.g. `config.yaml` -> `config.log`) so that a directly-invoked training
    # run (e.g. from a notebook or script, not via `so3lr train`) still produces
    # its own log file rather than relying solely on captured stdout (SLURM).
    if log_file is None:
        log_file = str(pathlib.Path(config_path).with_suffix('.log'))

    setup_logger(log_file)

    from_config.check_config(cfg)
    from_config.run_training(cfg)
