"""A v1 model directory without ``legacy_so3lr_bool`` in its config still loads.

The flag selects the v1 heads (attention Hirshfeld head, learnable ZBL, no C6 head). The
bundled ``so3lr`` model and fine-tunes of it carry the flag, but a directory trained
before the flag existed does not; its checkpoint is still v1-shaped.
"""
import json
import pathlib
import re
import shutil

import numpy as np
import pytest
from ase import Atoms

from so3lr import So3lrCalculator
from so3lr import model_registry as mr


def _water():
    atoms = Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
    atoms.info['charge'] = 0.0
    return atoms


def _energy(model):
    atoms = _water()
    atoms.calc = So3lrCalculator(model=str(model), lr_cutoff=12.0, dtype=np.float32)
    return atoms.get_potential_energy()


def _without_flag(source, target):
    shutil.copytree(source, target)
    hyper = target / 'hyperparameters.json'
    config = json.loads(hyper.read_text())
    assert config['model'].pop('legacy_so3lr_bool') is True
    hyper.write_text(json.dumps(config))
    yaml_file = target / 'hyperparameters.yaml'
    if yaml_file.exists():
        yaml_file.write_text(re.sub(r'(?m)^\s*legacy_so3lr_bool:.*\n', '', yaml_file.read_text()))


def test_legacy_checkpoint_without_the_flag_matches_the_flagged_model(tmp_path):
    legacy = pathlib.Path(mr.resolve_model('so3lr').dir)
    stripped = tmp_path / 'legacy_without_flag'
    _without_flag(legacy, stripped)

    np.testing.assert_allclose(_energy(stripped), _energy(legacy), rtol=0, atol=1e-6)
