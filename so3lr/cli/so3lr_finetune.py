"""Finetune utilities for SO3LR models."""
import jax
import jraph
import numpy as np
import jax.numpy as jnp
import time
import pprint
import pathlib
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm
import yaml

from ase.io import write
from mlff.utils import jraph_utils, evaluation_utils
from mlff.data import AseDataLoaderSparse
from mlff.utils import calculator_utils
from mlff.config import from_config

from ml_collections import config_dict
from pathlib import Path
import json
from collections import defaultdict

from ..jraph_utils import jraph_to_ase_atoms, unbatch_np
from ..base_calculator import make_so3lr
from .so3lr_md import load_model, setup_logger

# Get logger
logger = logging.getLogger("SO3LR")

def finetune_so3lr(
    workdir: str,
    datafile: str,
    num_train: Optional[int],
    num_valid: Optional[int],
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
    strategy: str = 'full',
    log_file: Optional[str] = None
):  
    package_dir = pathlib.Path(__file__).parent.parent.resolve()

    # Setup logging
    setup_logger(log_file)

    # Initialize the model
    if model_path is None:
        logger.info("Using default SO3LR potential")
        model_path = package_dir / 'params'
    else:
        logger.info(f"Using custom MLFF potential from: {model_path}")
    
    if config_path is None:
        # Use the default config for finetuning.
        config_path = package_dir / 'config/finetune.yaml'
        logger.info(f"Using default config for finetuning from: {config_path} ")
    else:
        logger.info(f"Using config for finetuning from: {config_path}")
    
    with open(config_path, mode='r') as fp:
        finetune_cfg = config_dict.ConfigDict(yaml.load(fp, Loader=yaml.FullLoader))
    
    finetune_cfg.workdir = workdir
    finetune_cfg.data.filepath = datafile

    if num_train is not None:
        finetune_cfg.training.num_train = num_train
    else:
        if finetune_cfg.training.num_train is None:
            raise ValueError(f'num_train must not be `None`. Either set it in your config at {config_path} or via the CLI argument `--num-train`.')
    
    if num_valid is not None:
        finetune_cfg.training.num_valid = num_valid
    else:
        if finetune_cfg.training.num_valid is None:
            raise ValueError(f'num_valid must not be `None`. Either set it in your config at {config_path} or via the CLI argument `--num-valid`.')

    from_config.run_fine_tuning(
        finetune_cfg,
        start_from_workdir=model_path,
        strategy=strategy
    )
