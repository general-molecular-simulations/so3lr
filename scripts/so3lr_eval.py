import argparse
import os
import pandas as pd
import sys
import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib
from typing import Optional, Sequence
import logging
import time


def evaluate_so3krates_sparse_on(
    workdir: str,
    filepath: str,
    length_unit: str = "Angstrom",
    energy_unit: str = "eV",
    max_num_graphs: int = 11,
    max_num_nodes: Optional[int] = None,
    max_num_edges: Optional[int] = None,
    max_num_pairs: int = 10000,
    num_test: Optional[int] = None,
    write_batch_metrics_to: Optional[str] = None,
    testing_targets: Optional[Sequence[str]] = None,
):
    """
    Loads a so3rkates model from a directory (direcotry where the model
    was traini)

    Args:
        workdir (str): _description_
        filepath (str): _description_
        length_unit (str, optional): _description_. Defaults to 'Angstrom'.
        energy_unit (str, optional): _description_. Defaults to 'eV'.
        max_num_graphs (int, optional): _description_. Defaults to 11.
        max_num_nodes (Optional[int], optional): _description_. Defaults to None.
        max_num_edges (Optional[int], optional): _description_. Defaults to None.
        max_num_pairs (int, optional): _description_. Defaults to 10000.
        num_test (Optional[int], optional): _description_. Defaults to None.
        write_batch_metrics_to (Optional[str], optional): _description_. Defaults to None.
        testing_targets (Optional[Sequence[str]], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        an: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if num_test is not None and write_batch_metrics_to is not None:
        raise ValueError(
            f"--num_test={num_test} is not `None` such that data is randomly sub-sampled from {filepath}. "
            f"At the same time `--write_batch_metrics_to={write_batch_metrics_to}` is specified. Due to the "
            f"random subsampling of data, there is no one-to-one correspondence between the lines in the file the "
            f"metrics are written to and the indices of the data point, so we raise an error here for security."
        )

    #workdir = pathlib.Path(workdir).expanduser().resolve()

    with open(workdir / "hyperparameters.json", "r") as fp:
        x = json.load(fp)
    cfg = config_dict.ConfigDict(x)

    # Overwrite the data information in config dict.
    cfg.data.filepath = filepath
    cfg.data.length_unit = length_unit
    cfg.data.energy_unit = energy_unit

    # Set the batching info.
    cfg.training.batch_max_num_graphs = max_num_graphs
    cfg.training.batch_max_num_edges = max_num_edges
    cfg.training.batch_max_num_nodes = max_num_nodes
    cfg.training.batch_max_num_pairs = max_num_pairs

    if (
        write_batch_metrics_to is not None
        and cfg.training.batch_max_num_graphs > 2
    ):
        raise ValueError(
            f"--write_batch_metrics_to={write_batch_metrics_to} is not None and `batch_max_num_graphs != 2.` "
            "Note, that the metrics are written per batch, so one-to-one correspondence to the original data set can "
            "only be achieved when `batch_max_num_nodes = 2` which allows one graph per batch, following the `jraph` "
            "logic that one graph in used as padding graph. Raising error for security here."
        )

    # Expand and resolve path for writing metrics.
    write_batch_metrics_to = (
        pathlib.Path(write_batch_metrics_to).expanduser().resolve()
        if write_batch_metrics_to is not None
        else None
    )

    if write_batch_metrics_to is not None:
        if write_batch_metrics_to.suffix == ".csv":
            pass
        else:
            write_batch_metrics_to = f"{write_batch_metrics_to}.csv"

    metrics = from_config.run_evaluation(
        config=cfg,
        num_test=num_test,
        pick_idx=None,
        write_batch_metrics_to=write_batch_metrics_to,
        testing_targets=testing_targets,
    )
    return metrics



#This script can evaluate a series of so3krates models on different datasets.
#Note that the models_folder should contain itself a folder for each model.
#Same applies to the data_folder.
#In order to acquire predictions per molecule, set `--max_num_graphs=2`.
#and `--write_batch_metrics_to` to a folder where the predictions should be saved.
#The script takes the following arguments:.


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--models_folder",
    type=str,
    required=True,
    help="Path to folder containing the models. The models should be stored in subfolders.",
    )
argparser.add_argument(
    "--data_folder",
    type=str,
    required=True,
    help="Path to folder containing the datasets. The datasets should be stored in subfolders.",
)
argparser.add_argument(
    "--length_unit",
    type=str,
    default="Angstrom",
    help="Unit of length used in the datasets. Defaults to 'Angstrom'.",
)
argparser.add_argument(
    "--energy_unit",
    type=str,
    default="eV",
    help="Unit of energy used in the datasets. Defaults to 'eV'.",
)
argparser.add_argument(
    "--max_num_graphs",
    type=int,
    default=2,
    help="Maximum number of graphs per batch. Defaults to 2.",
)
argparser.add_argument(
    "--max_num_nodes",
    type=int,
    default=None,
    help="Maximum number of nodes per batch. Defaults to None.",
)
argparser.add_argument(
    "--max_num_edges",
    type=int,
    default=None,
    help="Maximum number of edges per batch. Defaults to None.",
)
argparser.add_argument(
    "--max_num_pairs",
    type=int,
    default=10000,
    help="Maximum number of pairs per batch. Defaults to 10000.",
)
argparser.add_argument(
    "--num_test",
    type=int,
    default=None,
    help="Number of (random) test samples to use. Defaults to None.",
)
argparser.add_argument(
    "--write_batch_metrics_to",
    type=str,
    default=None,
    help="Path to file where batch metrics should be written. Defaults to None.",
)
argparser.add_argument(
    "--testing_targets",
    type=str,
    nargs="+",
    default=["dipole_vec", "energy", "forces", "hirshfeld_ratios"],
    help="List of targets to test. Defaults to ['dipole_vec', 'energy', 'forces', 'hirshfeld_ratios'].",
)
argparser.add_argument(
    "--logging_level",
    type=str,
    default="INFO",
    help="Logging level. Defaults to 'INFO'.",
)
argparser.add_argument(
    "--log_file",
    type=str,
    default="./so3lr_eval.log",
    help="Path to log file. Defaults to './so3lr_eval.log'.",
)
argparser.add_argument(
    "--timing",
    type=bool,
    default=True,
    help="Whether to log the time taken for each evaluation. Defaults to True.",
)


args = argparser.parse_args()

models_folder = args.models_folder
data_folder = args.data_folder
length_unit = args.length_unit
energy_unit = args.energy_unit
max_num_graphs = args.max_num_graphs
max_num_nodes = args.max_num_nodes
max_num_edges = args.max_num_edges
max_num_pairs = args.max_num_pairs
num_test = args.num_test
write_batch_metrics_to = args.write_batch_metrics_to
logging_level = args.logging_level
logging_level = logging.INFO if logging_level.lower() == "info" else logging.DEBUG
log_file = args.log_file
timing = args.timing

logger = logging.getLogger('so3_eval_logger')

formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
fh.setLevel(logging_level)
logger.addHandler(fh)

logger.setLevel(logging_level)

models_folder = pathlib.Path(models_folder).resolve()
data_folder = pathlib.Path(data_folder).resolve()

models = os.listdir(models_folder)
datasets = os.listdir(data_folder)


logger.info(f"Found {len(models)} models in {models_folder}")
logger.info(f"Found {len(datasets)} datasets in {data_folder}")


current_wdir = os.getcwd()
results_folder = current_wdir + "/results"

if os.path.exists(results_folder):
    raise ValueError(
        f"{results_folder} already exists."
        " Please remove it before running this script."
    )

os.mkdir(results_folder)

if write_batch_metrics_to is not None:
    batch_results_folder = results_folder + "/batch_results"

    if os.path.exists(batch_results_folder):
        raise ValueError(
            f"{batch_results_folder} already exists."
            " Please remove it before running this script"
        )
    os.mkdir(batch_results_folder)

logger.info(f"Writing results to {results_folder}")
if write_batch_metrics_to is not None:
    logger.info(
        f"Writing batch results to {batch_results_folder}"
        " which includes predictions for each batch."
    )

#TODO: Find out why we have to change the directory here
os.chdir(models_folder)

for model in models:
    model_path = models_folder / model
    model_results_folder = results_folder + f"/{model}"
    model_batch_results_folder = batch_results_folder + f"/{model}"

    if not os.path.exists(model_results_folder):
        os.mkdir(model_results_folder)

    if write_batch_metrics_to is not None:
        if not os.path.exists(model_batch_results_folder):
            os.mkdir(model_batch_results_folder)

    for dataset in datasets:
        logger.info('---------------------------------------------')
        logger.info(f"Running evaluation for {model} on {dataset}.")
        data_path = data_folder / dataset

        if write_batch_metrics_to is not None:
            write_batch_metrics_to = (
                f"{model_batch_results_folder}"
                + f"/{model}_{dataset}_batches.csv"
            )

        if timing:
            start = time.process_time()
        metrics = evaluate_so3krates_sparse_on(
            workdir=model_path,
            filepath=str(data_path),
            length_unit=length_unit,
            energy_unit=energy_unit,
            max_num_graphs=max_num_graphs,
            max_num_nodes=max_num_nodes,
            max_num_edges=max_num_edges,
            max_num_pairs=max_num_pairs,
            num_test=num_test,
            write_batch_metrics_to=write_batch_metrics_to,
        )
        if timing:
            end = time.process_time()
        logger.info(f"Evaluation for {model} on {dataset} complete.")
        if timing:
            logger.info(f"Time taken: {end-start:.4f} seconds (includes compilation!).")

        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info('---------------------------------------------'+'\n')
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(
            f"{model_results_folder}" + f"/{model}_{dataset}.csv"
        )
