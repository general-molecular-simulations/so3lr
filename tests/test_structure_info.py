"""One charge/spin rule for eval, guided setup and MD input checks."""
import logging

import numpy as np
import pytest
from ase import Atoms

from so3lr.structure_info import charge_and_multiplicity, electron_count_problem, recorded_charge_and_unpaired


def test_eval_rule_prefers_multiplicity_and_reads_omol_spin():
    assert charge_and_multiplicity({'charge': -1, 'multiplicity': 2, 'spin': 4}) == (-1, 2)
    assert charge_and_multiplicity({'spin': 3}) == (None, 3)
    atoms = Atoms('H2O', positions=np.zeros((3, 3)), info={'charge': 1, 'spin': 2})
    assert charge_and_multiplicity(atoms) == (1, 2)
    assert recorded_charge_and_unpaired(atoms) == (1, 1)


def test_dataloader_and_flash_share_the_rule():
    from so3lr.flash import evaluate
    from so3lr.mlff.data import dataloader_sparse_ase
    assert dataloader_sparse_ase.charge_and_multiplicity is charge_and_multiplicity
    assert evaluate.charge_and_multiplicity is charge_and_multiplicity


@pytest.mark.parametrize('settings, file_info, expected', [
    ((0, 0), {}, []),
    ((0, 0), {'charge': 0, 'multiplicity': 1}, []),
    ((0, 0), {'charge': -1, 'multiplicity': 1}, ['records charge=-1, but this run uses total charge 0']),
    ((-1, 0), {'charge': -1, 'spin': 2},  # OH-: 11 electrons cannot all pair up
     ['records multiplicity 2, but this run uses 0 unpaired electrons', 'cannot have 0 unpaired electrons']),
    ((1, 0), {}, ['Charge +1 leaves 9 electrons (an odd number)']),
])
def test_md_input_check_warns_without_changing_settings(caplog, settings, file_info, expected):
    from so3lr.cli.so3lr_md import _warn_about_charge_and_spin
    atoms = Atoms('OH2', positions=np.zeros((3, 3)), info=file_info)  # 10 electrons when neutral
    with caplog.at_level(logging.WARNING, logger='SO3LR'):
        _warn_about_charge_and_spin(atoms, *settings)
    messages = ' | '.join(record.getMessage() for record in caplog.records)
    for fragment in expected:
        assert fragment in messages
    assert len(caplog.records) == len(expected)


def test_electron_count_problem_is_empty_for_valid_states():
    assert electron_count_problem(10, 0, 0) == '' and electron_count_problem(10, 1, 1) == ''
