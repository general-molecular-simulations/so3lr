"""Utilities for the calculators."""
import json
from pathlib import Path

from ml_collections import config_dict
from typing import Any, Dict, Optional, Sequence

from ..config import from_config
from .checkpoint_utils import load_params_from_workdir


def load_hyperparameters(workdir: str):
    with open(Path(workdir) / "hyperparameters.json", "r") as fp:
        cfg = json.load(fp)

    cfg = config_dict.ConfigDict(cfg)
    return cfg



def load_model_from_workdir(
        workdir: str,
        model='so3krates',
        long_range_kwargs: Dict[str, Any] = None,
        output_intermediate_quantities: Optional[Sequence[str]] = None,
        from_file: bool = False
):
    """
    Load a neural network model from workdir.

    Args:
        workdir (str): The workdir. Can be any directory which contains `hyperparameters.json` and
            `hyperparameters.yaml` and `checkpoints` as a subdirectory.
        model (str): Model name.
        long_range_kwargs (): Dictionary with the following keys:
            ['cutoff_lr', 'dispersion_energy_cutoff_lr_damping', 'neighborlist_format_lr']. Can not be `None` when
            long-range electrostatic and/or long-range dispersion modules are used.  'cutoff_lr = None' will be treated
            like `cutoff_lr = np.inf`, i.e. infinite long-range cutoff. Neighborlist format can be `sparse` or
            `ordered_sparse`. `dispersion_energy_cutoff_lr_damping` specifies where to start the damping for the
            dispersion interactions. For `dispersion_energy_cutoff_lr_damping = 2.` damping starts at `lr_cutoff - 2.`.
        output_intermediate_quantities (Optional[Sequence[str]]): If not None, the model will return intermediate quantities for the keys given in the sequence.
        from_file (bool): Load parameters from file not from checkpoint directory.

    Returns:

    """

    cleaned_workdir = Path(workdir).expanduser().resolve()
    cfg = load_hyperparameters(cleaned_workdir)

    dispersion_energy_bool = cfg.model.dispersion_energy_bool
    electrostatic_energy_bool = cfg.model.electrostatic_energy_bool

    # For local model both are false.
    if (electrostatic_energy_bool is True) or (dispersion_energy_bool is True):
        if long_range_kwargs is None:
            raise ValueError(
                "For a potential with long-range electrostatic and/or dispersion corrections, long_range_kwargs must "
                f"be specified. Received {long_range_kwargs=}."
            )

        cutoff_lr = long_range_kwargs['cutoff_lr']
        neighborlist_format = long_range_kwargs['neighborlist_format_lr']
        if cutoff_lr is not None:
            if cutoff_lr < 0:
                raise ValueError(
                    f"For a potential with long range components the long range cutoff value must be greater "
                    f"than zero. received {cutoff_lr=}."
                )

        cfg.model.cutoff_lr = cutoff_lr
        cfg.neighborlist_format_lr = neighborlist_format

        if dispersion_energy_bool is True:
            dispersion_energy_cutoff_lr_damping = long_range_kwargs['dispersion_energy_cutoff_lr_damping']
            if cutoff_lr is not None:
                if dispersion_energy_cutoff_lr_damping is None:
                    raise ValueError(
                        f"dispersion_energy_cutoff_lr_damping must not be None if dispersion_energy_bool is True and "
                        f"cutoff_lr has a finite value. received {dispersion_energy_bool=}, {cutoff_lr=} and "
                        f"{dispersion_energy_cutoff_lr_damping=}."
                    )
            if cutoff_lr is None:
                if dispersion_energy_cutoff_lr_damping is not None:
                    raise ValueError(
                        f"dispersion_energy_cutoff_lr_damping must be None if dispersion_energy_bool is True and "
                        f"cutoff_lr is infinite (specified via lr_cutoff=None). received {dispersion_energy_bool=}, "
                        f"{dispersion_energy_cutoff_lr_damping=} and {cutoff_lr=}"
                    )
            cfg.model.dispersion_energy_cutoff_lr_damping = dispersion_energy_cutoff_lr_damping

    if from_file is True:
        import pickle

        with open(cleaned_workdir / 'params.pkl', 'rb') as f:
            params = pickle.load(f)
    else:
        params = load_params_from_workdir(
            workdir=cleaned_workdir
        )

    if model == 'so3krates':
        net = from_config.make_so3krates_sparse_from_config(
            cfg,
            output_intermediate_quantities=output_intermediate_quantities
        )
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )

    return net, params
