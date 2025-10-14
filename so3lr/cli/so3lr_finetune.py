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
    num_train: int,
    num_valid: int,
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
        model_path = package_dir / 'so3lr/params'
    else:
        logger.info(f"Using custom MLFF potential from: {model_path}")
    
    if config_path is None:
        # Use the default config for finetuning.
        config_path = package_dir / 'so3lr/config/finetune.yaml'
        logger.info(f"Using default config for finetuning from: {config_path} ")
    else:
        logger.info(f"Using config for finetuning from: {config_path}")
    
    with open(config_path, mode='r') as fp:
        finetune_cfg = config_dict.ConfigDict(json.load(fp=fp))
    
    finetune_cfg.workdir = workdir
    finetune_cfg.data.filepath = datafile
    finetune_cfg.training.num_train = num_train
    finetune_cfg.training.num_valid = num_valid

    from_config.run_fine_tuning(
        finetune_cfg,
        start_from_workdir=model_path,
        strategy=strategy
    )
