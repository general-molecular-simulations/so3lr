"""The extxyz spin key: OMol25/OPoly26 store the spin multiplicity (2S+1) under
`spin=`, not `multiplicity=`. Verified against the datasets' own <S^2>:
spin=4 -> s_squared~3.75 (quartet), spin=6 -> ~8.75 (sextet). The ASE loader read
only `multiplicity`, so every such structure was fed to the model as a closed-shell
singlet (num_unpaired_electrons = 0). Proof it mattered: two frames that are the
same geometry differing only in multiplicity got byte-identical predictions.

This pins the fallback: `spin` is used as the multiplicity when `multiplicity`
is absent, and the canonical `multiplicity` still wins when both are present.
"""
import numpy as np
import pytest
from ase import Atoms

from so3lr.mlff.data import AseDataLoaderSparse


def _num_unpaired(tmp_path, info):
    atoms = Atoms('OH', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    atoms.info.update(info)
    path = tmp_path / 'frag.extxyz'
    atoms.write(path)
    loaded_data, _ = AseDataLoaderSparse(str(path)).load(cutoff=5.0)
    graph, _ = loaded_data[0]
    return int(np.asarray(graph.globals['num_unpaired_electrons']).reshape(-1)[0])


@pytest.mark.parametrize('spin, expected_unpaired', [(1, 0), (2, 1), (4, 3)])
def test_spin_key_sets_unpaired_electrons(tmp_path, spin, expected_unpaired):
    assert _num_unpaired(tmp_path, {'spin': spin}) == expected_unpaired


def test_multiplicity_takes_precedence_over_spin(tmp_path):
    assert _num_unpaired(tmp_path, {'multiplicity': 1, 'spin': 4}) == 0


def test_default_is_singlet_when_neither_present(tmp_path):
    assert _num_unpaired(tmp_path, {}) == 0
