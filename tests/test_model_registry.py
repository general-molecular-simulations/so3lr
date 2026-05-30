"""Unit tests for the model registry resolver (so3lr.model_registry).

These are fast and do NOT load any model parameters/checkpoints; they only
exercise name/path resolution.
"""

import pathlib

import pytest

from so3lr import model_registry as mr


PKG = pathlib.Path(mr.__file__).parent.resolve()


def test_defaults():
    assert mr.DEFAULT_MODEL == "so3lr-m"
    assert mr.DEFAULT_HEAD_INDEX == 5


def test_available_models():
    names = mr.available_models()
    for expected in ("so3lr-s", "so3lr-m", "so3lr-l", "so3lr", "so3lr-og"):
        assert expected in names


@pytest.mark.parametrize("name", ["so3lr-s", "so3lr-m", "so3lr-l"])
def test_resolve_bundled_models(name):
    workdir, from_file, head_index = mr.resolve_model(name)
    assert workdir == PKG / "models" / name
    assert workdir.exists()
    # Bundled v2 models ship as a single params.pkl (no orbax checkpoints/).
    assert (workdir / "params.pkl").exists()
    assert from_file is True
    assert head_index == 5


@pytest.mark.parametrize("name", ["so3lr", "so3lr-og"])
def test_resolve_legacy_model(name):
    workdir, from_file, head_index = mr.resolve_model(name)
    assert workdir == PKG / "params"
    # The legacy dir ships a canonical params.pkl, which the resolver prefers
    # over its (incomplete) orbax checkpoint, so from_file is True.
    assert (workdir / "params.pkl").exists()
    assert from_file is True
    assert head_index == 5


def test_default_resolves_to_so3lr_m():
    assert mr.resolve_model(None) == mr.resolve_model("so3lr-m")


def test_unknown_name_raises():
    with pytest.raises(FileNotFoundError):
        mr.resolve_model("definitely-not-a-real-model-xyz")


def test_existing_path_is_accepted(tmp_path):
    workdir, from_file, head_index = mr.resolve_model(str(tmp_path))
    assert workdir == tmp_path.resolve()
    assert from_file is False  # empty dir: no checkpoints/ and no params.pkl
    assert head_index == 5
