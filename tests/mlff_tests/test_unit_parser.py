import math

import pytest
from ase import units
from ml_collections import config_dict

from so3lr.mlff.config import from_config
from so3lr.mlff.config.unit_parser import parse_unit_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("eV", units.eV),
        ("kcal / mol", units.kcal / units.mol),
        ("Bohr", units.Bohr),
        ("e * Angstrom", units.Angstrom),
        ("Angstrom ** 3", units.Angstrom ** 3),
        ("1.0", 1.0),
        (1, 1.0),
    ],
)
def test_parse_supported_unit_expressions(expression, expected):
    assert parse_unit_expression(expression, field_name="data.unit") == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("builtins").sum([20, 22])',
        "open('/tmp/so3lr-unit-parser-proof', 'w')",
        "ase_units.eV",
        "[eV][0]",
        "sqrt(eV)",
        "lambda: 1",
        "True",
    ],
)
def test_parse_rejects_executable_or_non_arithmetic_syntax(expression):
    with pytest.raises(ValueError):
        parse_unit_expression(expression, field_name="data.unit")


@pytest.mark.parametrize("expression", ["", "0", "-1", "1 / 0", "1e309", "eV ** 13"])
def test_parse_rejects_invalid_conversion_factors(expression):
    with pytest.raises(ValueError):
        parse_unit_expression(expression, field_name="data.unit")


def test_parse_result_is_finite_float():
    result = parse_unit_expression("Debye", field_name="data.dipole_vec_unit")
    assert isinstance(result, float)
    assert math.isfinite(result)


def test_config_unit_accessors_use_safe_parser():
    config = config_dict.ConfigDict({
        "data": {
            "energy_unit": "kcal / mol",
            "length_unit": "Bohr",
            "electric_charge_unit": "e",
            "dipole_vec_unit": "e * Angstrom",
        }
    })

    assert from_config.energy_unit_from_config(config) == pytest.approx(units.kcal / units.mol)
    assert from_config.length_unit_from_config(config) == pytest.approx(units.Bohr)
    assert from_config.charge_unit_from_config(config) == pytest.approx(1.0)
    assert from_config.dipole_vec_unit_from_config(config) == pytest.approx(units.Angstrom)


@pytest.mark.parametrize(
    ("accessor", "field"),
    [
        (from_config.energy_unit_from_config, "energy_unit"),
        (from_config.length_unit_from_config, "length_unit"),
        (from_config.charge_unit_from_config, "electric_charge_unit"),
        (from_config.dipole_vec_unit_from_config, "dipole_vec_unit"),
    ],
)
def test_config_unit_accessors_reject_code(accessor, field):
    config = config_dict.ConfigDict({
        "data": {
            "energy_unit": "eV",
            "length_unit": "Angstrom",
            "electric_charge_unit": "e",
            "dipole_vec_unit": "e * Angstrom",
        }
    })
    config.data[field] = '__import__("builtins").sum([20, 22])'

    with pytest.raises(ValueError, match=f"data\\.{field}"):
        accessor(config)
