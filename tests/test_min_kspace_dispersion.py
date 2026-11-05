"""The relaxation (``so3lr opt``, and the one before MD) builds the model with the same
dispersion k-space settings as MD, so it minimises the energy the MD then samples."""
from pathlib import Path

import pytest

from so3lr.cli import so3lr_md

WATER = Path(__file__).parent / 'test_data' / 'water_64.xyz'


class _Built(Exception):
    pass


@pytest.mark.parametrize('kspace_dispersion', [None, 'pme'])
def test_perform_min_passes_kspace_dispersion_to_the_model(tmp_path, monkeypatch, kspace_dispersion):
    seen = {}

    def load_model(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(so3lr_md, 'load_model', load_model)
    settings = dict(input_file=str(WATER), output_file=str(tmp_path / 'opt.xyz'), lr_cutoff=12.0,
                    kspace_dispersion=kspace_dispersion, kspace_dispersion_interp_nodes=4)
    with pytest.raises(_Built):
        so3lr_md.perform_min(settings, try_flash=False)
    assert seen['kspace_dispersion'] == kspace_dispersion
    assert seen['kspace_dispersion_interp_nodes'] == 4
