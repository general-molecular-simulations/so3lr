import json

from ase import Atoms
import pytest

from so3lr.prepare_dimer_xyz import (
    check_supported_elements,
    supported_elements_for_model,
)


def test_default_dimer_coverage_matches_v2_models():
    supported = supported_elements_for_model()
    assert min(supported) == 1
    assert max(supported) == 85
    assert len(supported) == 85

    is_supported, unsupported = check_supported_elements(
        Atoms(numbers=[1, 11, 85]), supported
    )
    assert is_supported
    assert unsupported == set()


def test_legacy_dimer_coverage_remains_available_by_model_name():
    supported = supported_elements_for_model("so3lr")
    is_supported, unsupported = check_supported_elements(
        Atoms(numbers=[1, 11, 17]), supported
    )
    assert not is_supported
    assert unsupported == {11}


def test_custom_model_coverage_uses_its_hyperparameters(tmp_path):
    (tmp_path / "hyperparameters.json").write_text(json.dumps({
        "model": {
            "legacy_so3lr_bool": True,
            "dispersion_energy_bool": True,
        }
    }))
    assert supported_elements_for_model(tmp_path) == supported_elements_for_model(
        "so3lr"
    )


def test_custom_model_without_metadata_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="hyperparameters.json"):
        supported_elements_for_model(tmp_path)
