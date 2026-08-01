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
    "so3lr-protos-s-2602",
    "so3lr-protos-m-2602",
    "so3lr-protos-l-2602",
    "so3lr-protos-s-2606",
    "so3lr-protos-m-2606",
    "so3lr-protos-l-2606",
    "so3lr",
)


def test_defaults():
    assert mr.DEFAULT_MODEL == "so3lr-protos-m-2606"
    assert mr.DEFAULT_HEAD_INDEX == 5
    # The default must be a canonical name, never a floating alias, so that the
    # no-argument default cannot drift when a new release is registered.
    assert mr.DEFAULT_MODEL in mr.canonical_models()


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


@pytest.mark.parametrize("name", CANONICAL)
def test_canonical_family(name):
    expected = "vanilla" if name == "so3lr" else "protos"
    assert mr.resolve_model(name).family == expected


@pytest.mark.parametrize("alias,target", sorted(mr.model_aliases().items()))
def test_aliases_resolve_to_their_canonical_target(alias, target):
    assert target in mr.canonical_models()
    assert mr.resolve_model(alias) == mr.resolve_model(target)


def test_size_aliases_point_at_the_current_release():
    for size in ("s", "m", "l"):
        current = f"so3lr-protos-{size}-2606"
        assert mr.resolve_model(f"so3lr-{size}").dir.name == current
        assert mr.resolve_model(f"so3lr-protos-{size}").dir.name == current


def test_dated_names_are_pinned():
    """The 2602 pins must keep resolving to the older release."""
    for size in ("s", "m", "l"):
        name = f"so3lr-protos-{size}-2602"
        assert mr.resolve_model(name).dir.name == name


def test_legacy_vanilla_model():
    for name in ("so3lr", "so3lr-o"):
        spec = mr.resolve_model(name)
        assert spec.dir == PKG / "models" / "so3lr"
        assert spec.family == "vanilla"
        # The legacy dir ships a canonical params.pkl, which the resolver
        # prefers over its (incomplete) orbax checkpoint.
        assert spec.from_file is True


def test_default_resolves_to_the_new_m_model():
    assert mr.resolve_model(None) == mr.resolve_model("so3lr-protos-m-2606")
    assert mr.resolve_model(None) == mr.resolve_model("so3lr-m")


def test_floating_alias_logs_its_target(caplog):
    mr._LOGGED_ALIASES.discard("so3lr-m")
    with caplog.at_level(logging.INFO, logger=mr.__name__):
        mr.resolve_model("so3lr-m")
    assert "so3lr-protos-m-2606" in caplog.text


def test_floating_alias_logs_only_once(caplog):
    mr._LOGGED_ALIASES.discard("so3lr-l")
    with caplog.at_level(logging.INFO, logger=mr.__name__):
        mr.resolve_model("so3lr-l")
        mr.resolve_model("so3lr-l")
    assert caplog.text.count("so3lr-protos-l-2606") == 1


def test_canonical_name_does_not_log(caplog):
    with caplog.at_level(logging.INFO, logger=mr.__name__):
        mr.resolve_model("so3lr-protos-m-2606")
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
    spec = mr.resolve_model("so3lr-protos-m-2606")
    workdir, from_file, head_index = spec
    assert (workdir, from_file, head_index) == (
        spec.dir, spec.from_file, spec.head_index
    )
    assert spec[0] == spec.dir
