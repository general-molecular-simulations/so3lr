"""Numerical regression tests for the Ewald tuning error bound."""

import ast
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np


COULOMB_SCALE = 14.399645351950548
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TUNER_SOURCE = REPOSITORY_ROOT / "so3lr" / "cli" / "tune_ewald.py"
BSD_LICENSE = REPOSITORY_ROOT / "LICENSE-BSD-3-CLAUSE"


def test_tuner_attribution_references_bundled_bsd_license():
    module = ast.parse(TUNER_SOURCE.read_text())
    module_docstring = ast.get_docstring(module, clean=False)

    assert module_docstring is not None
    assert "LICENSE-BSD-3-CLAUSE" in module_docstring
    assert "LICENSE-BSD-3-CLAUSE-torch-pme" not in module_docstring
    assert "THIRD_PARTY_NOTICES.md" not in module_docstring

    license_text = BSD_LICENSE.read_text()
    assert license_text.startswith("BSD 3-Clause License\n")
    assert "Copyright (c) 2024, torch-pme developers" in license_text
    for condition in (
        "1. Redistributions of source code must retain",
        "2. Redistributions in binary form must reproduce",
        "3. Neither the name of the copyright holder",
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
    ):
        assert condition in license_text


def _error_bounds(charges, cell):
    from so3lr.cli.tune_ewald import EwaldErrorBounds

    return EwaldErrorBounds(
        charges=jnp.asarray(charges),
        cell=jnp.asarray(cell),
        positions=jnp.zeros((len(charges), 3)),
    )


def test_ewald_kspace_error_has_closed_form_reference_value():
    """Exercise the corrected wavelength dependence with an exact reduction.

    The final torch-pme expression at commit
    1f612664e8e627143b19e8622bac9d2d81e85022 reduces to exp(-2) / pi when
    prefactor=smearing=1 and volume=spacing=pi. This reference value is derived
    independently of the implementation below.
    """
    charge_for_unit_prefactor = math.sqrt(1.0 / (2.0 * COULOMB_SCALE))
    bounds = _error_bounds(
        charges=[[charge_for_unit_prefactor]],
        cell=np.diag([math.pi, 1.0, 1.0]),
    )

    actual = bounds.err_kspace(smearing=1.0, spacing=math.pi)

    np.testing.assert_allclose(
        actual,
        math.exp(-2.0) / math.pi,
        rtol=2e-6,
        atol=0.0,
    )


def test_ewald_kspace_error_is_linear_in_charge_scale():
    """The RMS force error scales with |q|, not q squared.

    Since the internal prefactor scales with the sum of squared charges, the
    corrected square root from torch-pme 1f612664 makes the bound linear in a
    uniform charge rescaling. This specifically guards against ``prefac * 0.5``.
    """
    charges = np.array([[0.5], [-0.25]])
    cell = np.diag([2.0, 3.0, 4.0])
    base = _error_bounds(charges=charges, cell=cell)
    scaled = _error_bounds(charges=3.0 * charges, cell=cell)

    base_error = base.err_kspace(smearing=0.5, spacing=4.0)
    scaled_error = scaled.err_kspace(smearing=0.5, spacing=4.0)

    np.testing.assert_allclose(
        scaled_error,
        3.0 * base_error,
        rtol=2e-6,
        atol=0.0,
    )
