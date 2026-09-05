"""Security and privacy regressions for telemetry and serialized artifacts."""

import builtins
import inspect
import json
import pathlib
import pickle
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import yaml
from ml_collections import config_dict

from so3lr._serialization import (
    UnsafePickleWarning,
    _is_bundled_model_path,
    load_pickle_with_warning,
)
from so3lr.mlff.config.wandb_utils import (
    REDACTED_CREDENTIAL,
    REDACTED_LOCAL_PATH,
    initialize_wandb,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("config", [
    {"training": {}},
    {"training": {"use_wandb": False}},
    {"training": {"use_wandb": None}},
    {"training": {"use_wandb": "false"}},
    {"training": {"use_wandb": 0}},
])
def test_wandb_absent_or_false_never_imports_or_initializes(monkeypatch, config):
    real_import = builtins.__import__

    def reject_wandb_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("W&B must not be imported without explicit opt-in")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_wandb_import)
    assert initialize_wandb(config) is False


def test_wandb_opt_in_redacts_outbound_config_and_credential_args(caplog):
    calls = []
    fake_wandb = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs))
    config = {
        "workdir": "/private/research/run-17",
        "data": {
            "filepath": "/private/research/secret-dataset.extxyz",
            "relative_path": "datasets/public.extxyz",
            "datafiles": ["~/private-a.extxyz", "relative-b.extxyz"],
            "datasets": [
                {"path": "/mnt/private/train.tfrecord", "weight": 1.0},
                {"path": "s3://published-bucket/train", "weight": 0.5},
            ],
        },
        "service_password": "do-not-send",
        "training": {
            "use_wandb": True,
            "wandb_init_args": {
                "project": "privacy-test",
                "api_token": "do-not-pass-as-an-init-argument",
                "settings": {
                    "client_secret": "do-not-pass-nested-either",
                    "mode": "offline",
                },
                "config": {"unredacted": True},
            },
        },
    }

    assert initialize_wandb(
        config_dict.ConfigDict(config), wandb_module=fake_wandb
    ) is True
    assert len(calls) == 1
    init_call = calls[0]
    assert init_call["project"] == "privacy-test"
    assert "api_token" not in init_call
    assert init_call["settings"] == {"mode": "offline"}

    outbound = init_call["config"]
    assert outbound["workdir"] == REDACTED_LOCAL_PATH
    assert outbound["data"]["filepath"] == REDACTED_LOCAL_PATH
    assert outbound["data"]["relative_path"] == "datasets/public.extxyz"
    assert outbound["data"]["datafiles"] == [
        REDACTED_LOCAL_PATH,
        "relative-b.extxyz",
    ]
    assert outbound["data"]["datasets"][0]["path"] == REDACTED_LOCAL_PATH
    assert outbound["data"]["datasets"][1]["path"] == "s3://published-bucket/train"
    assert outbound["service_password"] == REDACTED_CREDENTIAL
    assert (
        outbound["training"]["wandb_init_args"]["api_token"]
        == REDACTED_CREDENTIAL
    )

    # Sanitization is copy-only: local hyperparameter persistence may retain
    # the original values, while W&B receives only the redacted copy.
    assert config["workdir"] == "/private/research/run-17"
    assert "PRIVACY NOTICE" in caplog.text
    assert "do-not-send" not in caplog.text


def test_wandb_opt_in_without_optional_dependency_is_actionable(monkeypatch):
    real_import = builtins.__import__

    def import_without_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ModuleNotFoundError(
                "No module named 'wandb'", name="wandb"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_wandb)
    with pytest.raises(ImportError, match=r"so3lr\[wandb\]"):
        initialize_wandb({"training": {"use_wandb": True}})


def test_wandb_broken_transitive_import_preserves_original_error(monkeypatch):
    real_import = builtins.__import__

    def import_broken_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ModuleNotFoundError(
                "No module named 'wandb_transitive_dependency'",
                name="wandb_transitive_dependency",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_broken_wandb)
    with pytest.raises(
        ModuleNotFoundError, match="wandb_transitive_dependency"
    ) as exc_info:
        initialize_wandb({"training": {"use_wandb": True}})

    assert exc_info.value.name == "wandb_transitive_dependency"


def test_bundled_train_and_finetune_configs_disable_wandb():
    for config_name in ("config.yaml", "finetune.yaml"):
        with open(PROJECT_ROOT / "so3lr" / "config" / config_name) as stream:
            config = yaml.safe_load(stream)
        assert config["training"]["use_wandb"] is False

    # The legacy JSON example omits the key; absence must also be opt-out.
    with open(PROJECT_ROOT / "so3lr" / "mlff" / "config" / "config.json") as stream:
        legacy_config = json.load(stream)
    assert initialize_wandb(legacy_config) is False


def test_low_level_training_entry_points_are_opt_in():
    from so3lr.mlff.training.coach import Coach
    from so3lr.mlff.training.run import run_training
    from so3lr.mlff.utils import training_utils

    assert inspect.signature(Coach.run).parameters["use_wandb"].default is False
    assert inspect.signature(run_training).parameters["use_wandb"].default is False
    assert inspect.signature(training_utils.fit).parameters["use_wandb"].default is False
    assert (
        inspect.signature(training_utils.fit_from_iterator)
        .parameters["use_wandb"].default is False
    )


def _plain_nve_state():
    return SimpleNamespace(
        position=np.array([[0.1, 0.2, 0.3]]),
        momentum=np.array([[0.4, 0.5, 0.6]]),
        force=np.array([[0.7, 0.8, 0.9]]),
        mass=np.array([[1.0]]),
    )


def test_so3lr_npz_restart_round_trip_needs_no_pickle(tmp_path):
    from so3lr.cli.so3lr_md import load_state, save_state

    path = tmp_path / "restart.npz"
    save_state(
        _plain_nve_state(), np.array([10.0, 11.0, 12.0]), 7, str(path),
        ensemble="nve", fractional_coordinates=False,
    )

    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            archive[name]  # every field in SO3LR's save schema is non-object

    state, box, cycle = load_state(
        str(path), ensemble="nve", fractional_coordinates=False,
    )
    np.testing.assert_allclose(state.position, _plain_nve_state().position)
    np.testing.assert_allclose(box, [10.0, 11.0, 12.0])
    assert int(cycle) == 7


def test_so3lr_npz_restart_rejects_object_arrays(tmp_path):
    from so3lr.cli.so3lr_md import load_state

    path = tmp_path / "object-restart.npz"
    np.savez(
        path,
        position=np.array([object()], dtype=object),
        momentum=np.zeros((1, 3)),
        force=np.zeros((1, 3)),
        mass=np.ones((1, 1)),
        box=np.ones(3),
        step=1,
        ensemble="nve",
        fractional_coordinates=False,
    )

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_state(str(path), ensemble="nve", fractional_coordinates=False)


def test_params_pickle_loader_emits_trust_warning(tmp_path):
    from so3lr.mlff.utils.checkpoint_utils import load_params_from_workdir

    path = tmp_path / "params.pkl"
    with open(path, "wb") as stream:
        pickle.dump({"parameter": np.array([1.0])}, stream)

    with pytest.warns(UnsafePickleWarning, match="checksum has been verified"):
        params = load_params_from_workdir(tmp_path)
    np.testing.assert_allclose(params["parameter"], [1.0])


def test_installed_bundled_weights_share_the_package_trust_boundary():
    path = PROJECT_ROOT / "so3lr" / "models" / "so3lr-s-2606" / "params.pkl"
    assert _is_bundled_model_path(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = load_pickle_with_warning(path)
    assert isinstance(params, dict)
    assert not any(isinstance(item.message, UnsafePickleWarning) for item in caught)


def test_pysages_restart_loader_emits_trust_warning(monkeypatch, tmp_path):
    pytest.importorskip("pysages")
    from so3lr.cli import so3lr_pysages_interface

    marker = object()
    monkeypatch.setattr(
        so3lr_pysages_interface.pysages.serialization,
        "load",
        lambda path: marker,
    )
    path = tmp_path / "pysages.pkl"
    with pytest.warns(UnsafePickleWarning, match="PySAGES pickle restart"):
        result = so3lr_pysages_interface.load_pysages_state(path)
    assert result is marker
