"""Unit tests for the model registry resolver (so3lr.model_registry).

These are fast and do NOT load any model parameters/checkpoints; they only
exercise name/path resolution.
"""

import logging
import pathlib

import pytest

from so3lr import model_registry as mr


PKG = pathlib.Path(mr.__file__).parent.resolve()

CANONICAL = (
    "so3lr-tauri-s-2602",
    "so3lr-tauri-m-2602",
    "so3lr-tauri-l-2602",
    "so3lr-tauri-s-2606",
    "so3lr-tauri-m-2606",
    "so3lr-tauri-l-2606",
    "so3lr",
)


def test_defaults():
    assert mr.DEFAULT_MODEL == "so3lr-tauri-m-2606"
    assert mr.DEFAULT_HEAD_INDEX == 5
    # The default must be a canonical name, never a floating alias, so that the
    # no-argument default cannot drift when a new release is registered.
    assert mr.DEFAULT_MODEL in mr.canonical_models()
    # No name at all (bare calculator, CLI without --model) resolves to it.
    assert mr.resolve_model(None) == mr.resolve_model(mr.DEFAULT_MODEL)


def test_canonical_models():
    assert mr.canonical_models() == CANONICAL


def test_available_models_is_canonical_plus_aliases():
    names = mr.available_models()
    assert names[: len(CANONICAL)] == CANONICAL
    assert set(names) == set(CANONICAL) | set(mr.model_aliases())
    assert len(names) == len(set(names))  # no duplicates


@pytest.mark.parametrize("name", CANONICAL)
def test_resolve_canonical(name):
    """Canonical names map 1:1 onto a directory of the same name."""
    spec = mr.resolve_model(name)
    assert spec.dir == PKG / "models" / name
    assert spec.dir.exists()
    # Every bundled model ships a params.pkl export.
    assert (spec.dir / "params.pkl").exists()
    assert spec.from_file is True
    assert spec.head_index == 5
    assert spec.family == ("vanilla" if name == "so3lr" else "v2")


@pytest.mark.parametrize("alias,target", sorted(mr.model_aliases().items()))
def test_aliases_resolve_to_their_canonical_target(alias, target):
    assert target in mr.canonical_models()
    assert mr.resolve_model(alias) == mr.resolve_model(target)


def test_size_aliases_point_at_the_current_release():
    for size in ("s", "m", "l"):
        current = f"so3lr-tauri-{size}-2606"
        assert mr.resolve_model(f"so3lr-tauri-{size}").dir.name == current


def test_registry_logs_on_the_package_logger():
    """The alias line must go to the logger the CLI attaches handlers to.

    ``getLogger(__name__)`` would make this a sibling of "SO3LR" rather than a
    child, so the redirect would be silently dropped on every CLI run while
    still passing a caplog-based test (caplog installs a root handler).
    """
    assert mr.logger.name == "SO3LR" or mr.logger.name.startswith("SO3LR.")


def test_floating_alias_logs_its_target_once(caplog):
    mr._LOGGED_ALIASES.discard("so3lr-tauri-l")
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        mr.resolve_model("so3lr-tauri-l")
        mr.resolve_model("so3lr-tauri-l")
    assert caplog.text.count("so3lr-tauri-l-2606") == 1


def test_canonical_name_does_not_log(caplog):
    with caplog.at_level(logging.INFO, logger="SO3LR"):
        mr.resolve_model("so3lr-tauri-m-2606")
    assert caplog.text == ""


def test_unknown_name_raises():
    with pytest.raises(FileNotFoundError):
        mr.resolve_model("definitely-not-a-real-model-xyz")


def test_existing_path_is_accepted(tmp_path):
    spec = mr.resolve_model(str(tmp_path))
    assert spec.dir == tmp_path.resolve()
    assert spec.from_file is False  # empty dir: no checkpoints/ and no params.pkl
    assert spec.head_index == 5
    assert spec.family == "custom"


def test_spec_unpacks_as_legacy_triple():
    """Old ``dir, from_file, head_index`` call sites keep working."""
    spec = mr.resolve_model("so3lr-tauri-m-2606")
    workdir, from_file, head_index = spec
    assert (workdir, from_file, head_index) == (
        spec.dir, spec.from_file, spec.head_index
    )
    assert spec[0] == spec.dir
