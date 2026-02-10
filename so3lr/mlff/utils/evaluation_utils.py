import clu.metrics as clu_metrics
import itertools as it
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from so3lr.mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from so3lr.mlff.utils.jraph_utils import dynamically_batch_with_lr


def evaluate(
        model,
        params,
        graph_to_batch_fn,
        testing_data,
        testing_targets,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs,
        batch_max_num_pairs,
        write_batch_metrics_to: str = None
):
    """Evaluate a model given its params on the testing data.

    Args:
        model (): The FLAX model.
        params (): The model parameters as PyTree.
        graph_to_batch_fn (): Function that takes a `jraph.GraphsTuple` and returns the input to the
            observable function.
        testing_data (): The testing data as list of `jraph.GraphsTuple`.
        testing_targets (): The targets for which the metrics should be calculated.
        batch_max_num_nodes (): Maximal number of nodes per batch.
        batch_max_num_edges (): Maximal number of edges per batch.
        batch_max_num_graphs (): Maximal number of graphs oer batch.
        batch_max_num_pairs (int): Maximal number of pairs in long-range indices.
        write_batch_metrics_to (str): Path to file where metrics per batch should be written to. If not given,
            batch metrics are not written to a file. Note, that the metrics are written per batch, so one-to-one
            correspondence to the original data set can only be achieved when `batch_max_num_nodes = 2` which allows
            one graph per batch, following the `jraph` logic that one graph in used as padding graph.

    Returns:
        The metrics on testing data.
    """

    obs_fn = jax.jit(
        get_energy_and_force_fn_sparse(model)
    )

    iterator_testing = dynamically_batch_with_lr(
        iter(testing_data),
        n_node=batch_max_num_nodes,
        n_edge=batch_max_num_edges,
        n_graph=batch_max_num_graphs,
        n_pairs=batch_max_num_pairs,
    )

    # Create a collections object for the test targets.
    test_collection = clu_metrics.Collection.create(
        **{f'{t}_{m}': clu_metrics.Average.from_output(f'{t}_{m}') for (t, m) in it.product(testing_targets, ('mae', 'mse'))})

    # Start iteration over validation batches.
    row_metrics = []
    test_metrics: Any = None
    for graph_batch_testing, lr_batch_testing in tqdm(iterator_testing):
        batch_testing = graph_to_batch_fn(graph_batch_testing, lr_batch_testing)
        batch_testing = jax.tree_util.tree_map(jnp.array, batch_testing)

        node_mask = batch_testing['node_mask']
        graph_mask = batch_testing['graph_mask']

        inputs = {k: v for (k, v) in batch_testing.items() if k not in testing_targets}
        output_prediction = obs_fn(params, **inputs)

        metrics_dict = {}
        for t in testing_targets:
            if t == 'energy':
                msk = graph_mask
            elif t == 'forces':
                msk = node_mask
            elif t == 'stress':
                msk = graph_mask
            elif t == 'dipole_vec':
                msk = graph_mask
            elif t == 'hirshfeld_ratios':
                msk = node_mask
            elif t == 'dispersion_energy':
                msk = graph_mask
            elif t == 'electrostatic_energy':
                msk = graph_mask
            else:
                raise ValueError(
                    f"Evaluate not implemented for target={t}."
                )
            metrics_dict[f"{t}_mae"] = calculate_mae(
                y_predicted=output_prediction[t], y_true=batch_testing[t], msk=msk
            )
            metrics_dict[f"{t}_mse"] = calculate_mse(
                y_predicted=output_prediction[t], y_true=batch_testing[t], msk=msk
            )
            metrics_dict[f"{t}_true"] = batch_testing[t][msk]
            metrics_dict[f"{t}_predicted"] = output_prediction[t][msk]
     
        # Track the metrics per batch if they are written to file.
        if write_batch_metrics_to is not None:
            row_metrics += [jax.device_get(metrics_dict)]

        test_metrics = (
            test_collection.single_from_model_output(**metrics_dict)
            if test_metrics is None
            else test_metrics.merge(test_collection.single_from_model_output(**metrics_dict))
        )
    test_metrics = test_metrics.compute()

    if write_batch_metrics_to:
        df = pd.DataFrame(row_metrics)
        with open(write_batch_metrics_to, mode='w') as fp:
            df.to_csv(fp)

    test_metrics = {
        f'test_{k}': float(v) for k, v in test_metrics.items()
    }

    for t in testing_targets:
        test_metrics[f'test_{t}_rmse'] = float(np.sqrt(test_metrics[f'test_{t}_mse']))
    return test_metrics


def calculate_mse(y_predicted, y_true, msk):
    assert y_predicted.shape == y_true.shape
    assert len(y_predicted) == len(y_true) == len(msk)

    return jnp.square(
        y_predicted[msk] - y_true[msk]
    ).mean()


def calculate_mae(y_predicted, y_true, msk):
    assert y_predicted.shape == y_true.shape
    assert len(y_predicted) == len(y_true) == len(msk)

    return jnp.abs(
        y_predicted[msk] - y_true[msk]
    ).mean()
