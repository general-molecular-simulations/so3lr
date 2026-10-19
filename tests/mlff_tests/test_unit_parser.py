import pytest
from ase import units
from ml_collections import config_dict

from so3lr.mlff.config import from_config
from so3lr.mlff.config.unit_parser import parse_unit_expression


@pytest.mark.parametrize(("expression", "expected"), [
    ("eV", units.eV), ("kcal / mol", units.kcal / units.mol), ("Bohr", units.Bohr),
    ("e * Angstrom", units.Angstrom), ("Angstrom ** 3", units.Angstrom ** 3), ("1.0", 1.0), (1, 1.0),
])
def test_parse_supported_unit_expressions(expression, expected):
    assert parse_unit_expression(expression, "data.unit") == pytest.approx(expected)


@pytest.mark.parametrize("expression", [
    '__import__("os").getcwd()', "ase_units.eV", "[eV][0]", "sqrt(eV)", "lambda: 1", "True",
    "eV + 1", "", "0", "-1", "1 / 0", "1e309", "(-1) ** 0.5", "9 ** 9 ** 9 ** 9",
])
def test_parse_rejects_code_and_invalid_factors(expression):
    with pytest.raises(ValueError, match="data.unit"):
        parse_unit_expression(expression, "data.unit")


def test_config_unit_accessors():
    config = config_dict.ConfigDict({"data": {
        "energy_unit": "kcal / mol", "length_unit": "Bohr",
        "electric_charge_unit": "e", "dipole_vec_unit": "e * Angstrom"}})
    assert from_config.energy_unit_from_config(config) == pytest.approx(units.kcal / units.mol)
    assert from_config.length_unit_from_config(config) == pytest.approx(units.Bohr)
    assert from_config.dipole_vec_unit_from_config(config) == pytest.approx(units.Angstrom)
    config.data.energy_unit = '__import__("os").getcwd()'
    with pytest.raises(ValueError, match=r"data\.energy_unit"):
        from_config.energy_unit_from_config(config)
