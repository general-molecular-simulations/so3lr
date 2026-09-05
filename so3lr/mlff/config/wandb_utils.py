"""Privacy-preserving, explicitly opt-in Weights & Biases integration."""

from collections.abc import Mapping
import logging
from pathlib import Path, PureWindowsPath
import re
from typing import Any


logger = logging.getLogger("SO3LR")

REDACTED_CREDENTIAL = "<redacted:credential>"
REDACTED_LOCAL_PATH = "<redacted:absolute-local-path>"

_CREDENTIAL_KEY_PARTS = (
    "apikey",
    "accesstoken",
    "authtoken",
    "authorization",
    "clientsecret",
    "credential",
    "password",
    "passwd",
    "privatekey",
    "secret",
    "token",
)
_PATH_KEY_NAMES = {
    "data",
    "datafile",
    "datafiles",
    "dataset",
    "datasets",
    "filepath",
    "input",
    "inputs",
    "output",
    "outputs",
    "path",
    "workdir",
    "workdirs",
}
_PATH_KEY_SUFFIXES = (
    "dir",
    "directory",
    "file",
    "files",
    "folder",
    "folders",
    "path",
    "paths",
)


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _is_credential_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    return any(part in normalized for part in _CREDENTIAL_KEY_PARTS)


def _is_path_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    return (
        normalized in _PATH_KEY_NAMES
        or normalized.endswith(_PATH_KEY_SUFFIXES)
    )


def _is_absolute_local_path(value: str) -> bool:
    """Recognize POSIX, home-relative, and Windows absolute local paths."""
    if value.startswith("~") or value.lower().startswith("file://"):
        return True
    try:
        expanded = str(Path(value).expanduser())
    except RuntimeError:
        expanded = value
    return Path(expanded).is_absolute() or PureWindowsPath(value).is_absolute()


def redact_config_for_wandb(value: Any, *, key: Any = None) -> Any:
    """Return a telemetry-safe copy of an effective SO3LR configuration.

    Credential-like entries are always removed. Absolute local paths are
    removed when their field name identifies a file, directory, workdir, or
    dataset. Relative paths and non-local URIs remain useful provenance.
    """
    if key is not None and _is_credential_key(key):
        return REDACTED_CREDENTIAL

    if isinstance(value, Mapping):
        return {
            str(child_key): redact_config_for_wandb(
                child_value, key=child_key
            )
            for child_key, child_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [redact_config_for_wandb(item, key=key) for item in value]

    if (
        key is not None
        and _is_path_key(key)
        and isinstance(value, (str, Path))
        and _is_absolute_local_path(str(value))
    ):
        return REDACTED_LOCAL_PATH

    return value


def _plain_dict(value: Any) -> dict:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("W&B configuration must be a mapping-like object.")
    return dict(value)


def _credential_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {
            str(key) for key in value
            if _is_credential_key(key)
        }
        for key, child in value.items():
            if not _is_credential_key(key):
                found.update(_credential_keys(child))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for child in value:
            found.update(_credential_keys(child))
        return found
    return set()


def _remove_credential_entries(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _remove_credential_entries(child)
            for key, child in value.items()
            if not _is_credential_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_remove_credential_entries(child) for child in value]
    return value


def initialize_wandb(config: Any, *, wandb_module: Any = None) -> bool:
    """Initialize W&B only for the literal opt-in value ``use_wandb: true``.

    Returns the boolean passed to the training loop. No W&B import or method
    call occurs for a missing, false, or otherwise non-boolean setting.
    """
    config_dict = _plain_dict(config)
    training = config_dict.get("training", {})
    if not isinstance(training, Mapping):
        raise TypeError("The training configuration must be a mapping.")

    if training.get("use_wandb", False) is not True:
        return False

    init_args = training.get("wandb_init_args", {}) or {}
    init_args = _plain_dict(init_args)

    # Never permit a nested config to replace SO3LR's redacted effective
    # configuration, and do not pass obvious credentials as init kwargs.
    if "config" in init_args:
        logger.warning(
            "Ignoring training.wandb_init_args.config; SO3LR always sends its "
            "redacted effective configuration."
        )
        init_args.pop("config")
    credential_args = _credential_keys(init_args)
    init_args = _remove_credential_entries(init_args)
    if credential_args:
        logger.warning(
            "Ignoring credential-like W&B init argument(s) %s. Configure W&B "
            "authentication outside the SO3LR config.",
            sorted(credential_args),
        )

    telemetry_config = redact_config_for_wandb(config_dict)
    logger.warning(
        "PRIVACY NOTICE: W&B telemetry is enabled. Training metrics and a "
        "redacted copy of the effective SO3LR configuration will be sent to "
        "Weights & Biases. Absolute local data/work paths and obvious "
        "credential-like entries are removed; review the remaining config "
        "before using this opt-in setting."
    )

    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ModuleNotFoundError as exc:
            # Only replace the error when the optional top-level package is
            # genuinely absent. If an installed W&B client fails while
            # importing one of its own dependencies, preserve that exception
            # so users can diagnose the broken environment.
            if exc.name != "wandb":
                raise
            raise ImportError(
                "W&B telemetry was enabled but the optional client is not "
                "installed. Install it with `pip install 'so3lr[wandb]'` "
                "(or `pip install '.[wandb]'` from a checkout), then retry."
            ) from exc
    wandb_module.init(config=telemetry_config, **init_args)
    return True
