"""Fine-tuning strategies resolve to keys that exist in the shipped checkpoints.

These are the fast counterpart to `tests/test_cli_finetune.py`: they pin the
strategy -> parameter-key mapping against every bundled model in milliseconds,
so an off-by-one in `layers_{num_layers - 1}` (or a head that was renamed) is
caught on every PR without paying for a subprocess training run.
"""
import json
import pathlib
import pickle

import pytest

from so3lr.finetuning_strategies import (
    FINETUNING_STRATEGY_NAMES,
    resolve_finetuning_strategy,
)
from so3lr.mlff.utils.param_utils import (
    count_params,
    count_trainable_params,
    unmatched_param_keys,
)

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / 'so3lr' / 'models'
BUNDLED_MODELS = sorted(p.name for p in MODELS_DIR.iterdir() if (p / 'params.pkl').exists())

# `so3lr` is the legacy checkpoint: it has no C6 head, so every strategy naming
# `observables_3` legitimately selects nothing there.
_OBS_3_STRATEGIES = {'obs_3', 'obs_23', 'obs_023', 'll_obs_023'}

# (model, strategy) pairs where the whitelist matches *nothing at all*, so the
# strategy would freeze the entire model. `run_fine_tuning` refuses these
# outright rather than running a training job that updates nothing.
UNSUPPORTED = {('so3lr', 'obs_3')}


def _load(model_name):
    model_dir = MODELS_DIR / model_name
    with open(model_dir / 'params.pkl', 'rb') as fp:
        params = pickle.load(fp)
    with open(model_dir / 'hyperparameters.json') as fp:
        num_layers = json.load(fp)['model']['num_layers']
    return params, num_layers


def test_cli_choices_match_the_strategy_table():
    """`--strategy` accepts exactly the strategies that are implemented.

    The two lists were maintained by hand and had drifted:
    `freeze_all_but_final_mlp` was implemented but rejected by click.
    """
    from so3lr.cli import so3lr_cli

    finetune_cmd = so3lr_cli.cli.commands['finetune']
    strategy_option = next(p for p in finetune_cmd.params if p.name == 'strategy')
    assert tuple(strategy_option.type.choices) == FINETUNING_STRATEGY_NAMES


def test_unknown_strategy_lists_every_valid_name():
    with pytest.raises(ValueError) as excinfo:
        resolve_finetuning_strategy('not_a_strategy', num_layers=3)
    message = str(excinfo.value)
    for name in FINETUNING_STRATEGY_NAMES:
        assert f'`{name}`' in message


@pytest.mark.parametrize('strategy', FINETUNING_STRATEGY_NAMES)
@pytest.mark.parametrize('model_name', BUNDLED_MODELS)
def test_strategy_selects_real_parameters(strategy, model_name):
    """Every strategy trains something, and less than everything (except `full`).

    This is the regression guard for the `layers_{num_layers - 1}` fix: pointing
    it at the *finetune* config's num_layers, or off by one, yields a key that
    matches no parameter -- which used to freeze the whole model and still exit
    0 after a full training run.
    """
    params, num_layers = _load(model_name)
    trainable_keys, frozen_keys = resolve_finetuning_strategy(strategy, num_layers)

    total = count_params(params)
    trainable = count_trainable_params(
        params, trainable_keys=trainable_keys, frozen_keys=frozen_keys
    )
    unmatched = unmatched_param_keys(params, trainable_keys or frozen_keys or [])

    if (model_name, strategy) in UNSUPPORTED:
        # No key matches -> everything would be frozen. run_fine_tuning turns
        # exactly this condition into a ValueError instead of a no-op run.
        assert trainable == 0
        assert len(unmatched) == len(trainable_keys)
        return

    if strategy == 'full':
        assert trainable_keys is None and frozen_keys is None
        assert trainable == total
    else:
        assert 0 < trainable < total, (
            f'{strategy} on {model_name} trains {trainable}/{total} parameters'
        )

    # Every key must select something, so a typo or a renamed head cannot pass
    # unnoticed. The legacy `so3lr` checkpoint genuinely has no C6 head.
    if model_name == 'so3lr' and (strategy in _OBS_3_STRATEGIES
                                  or strategy in ('freeze_long_range', 'forces_only')):
        assert unmatched == ['observables_3']
    else:
        assert unmatched == [], f'{strategy} on {model_name}: {unmatched} match nothing'


@pytest.mark.parametrize('model_name', BUNDLED_MODELS)
def test_forces_only_freezes_the_partial_charges(model_name):
    """`forces_only` must freeze the electrostatics head.

    It used to name a non-existent `observables_1`, so the partial charges under
    `observables_0/electrostatic_energy` kept training on data that carries no
    electrostatics labels -- destroying exactly the pretrained predictions the
    strategy exists to protect.
    """
    params, num_layers = _load(model_name)
    _, frozen_keys = resolve_finetuning_strategy('forces_only', num_layers)

    charges = params['params']['observables_0']['electrostatic_energy']
    n_charge_params = count_params(charges)
    assert n_charge_params > 0

    trainable = count_trainable_params(params, frozen_keys=frozen_keys)
    trainable_without_charges_frozen = count_trainable_params(
        params, frozen_keys=[k for k in frozen_keys if k != 'observables_0/electrostatic_energy']
    )
    assert trainable_without_charges_frozen - trainable == n_charge_params


def test_aliases_resolve_identically():
    assert (resolve_finetuning_strategy('forces_only', 3)
            == resolve_finetuning_strategy('freeze_long_range', 3))
    assert (resolve_finetuning_strategy('freeze_all_but_final_mlp', 3)
            == resolve_finetuning_strategy('final_mlp', 3))


def test_last_layer_strategies_track_num_layers():
    """The last-layer strategies name the block by index, so they must follow
    the *pretrained* model's depth rather than a hardcoded number."""
    for num_layers in (2, 3, 5):
        trainable, _ = resolve_finetuning_strategy('last_layer', num_layers)
        assert trainable == [f'layers_{num_layers - 1}']
