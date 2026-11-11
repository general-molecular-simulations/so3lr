import subprocess
import sys

import pytest

from so3lr.cli.setup_physics import long_range_settings
from so3lr.model_registry import DEFAULT_MODEL


def test_water64_proposal_fits_cell_and_preserves_both_long_range_terms():
    settings = long_range_settings(True, 12.4, DEFAULT_MODEL)
    assert settings['lr_cutoff'] == 5.5
    assert settings['kspace_electrostatics'] == settings['kspace_dispersion'] == 'pme'
    assert settings['kspace_smearing'] == settings['kspace_dispersion_smearing'] == 2.0
    assert 2 * settings['lr_cutoff'] < 12.4


def test_large_and_isolated_structures_keep_existing_cutoff_policy():
    assert long_range_settings(False, None, DEFAULT_MODEL) == {'lr_cutoff': 100.0}
    assert long_range_settings(True, 999.9, DEFAULT_MODEL) == {'lr_cutoff': 12.0}


def test_invalid_model_cell_and_cutoff_are_rejected_before_export():
    with pytest.raises(ValueError, match='too small'):
        long_range_settings(True, 5.0, DEFAULT_MODEL)
    with pytest.raises(ValueError, match='at most 6.2'):
        long_range_settings(True, 12.4, DEFAULT_MODEL, 12.0)
    with pytest.raises(ValueError, match='short-range cutoff'):
        long_range_settings(True, 12.4, DEFAULT_MODEL, 3.0)
    for invalid in (float('nan'), float('inf'), 0):
        with pytest.raises(ValueError):
            long_range_settings(True, 12.4, DEFAULT_MODEL, invalid)
    with pytest.raises(ValueError, match='at least 10'):
        long_range_settings(False, None, DEFAULT_MODEL, 5)


def test_proposal_reads_model_metadata_without_initializing_jax():
    subprocess.run([sys.executable, '-c',
                    "from so3lr.cli.setup_physics import long_range_settings; "
                    "from so3lr.model_registry import DEFAULT_MODEL; "
                    "long_range_settings(True, 12.4, DEFAULT_MODEL); "
                    "import sys; assert 'jax' not in sys.modules; assert 'flax' not in sys.modules"],
                   check=True)
