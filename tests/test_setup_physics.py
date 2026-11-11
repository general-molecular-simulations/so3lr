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
    assert settings['kspace_spacing'] == 1.0
    assert settings['kspace_dispersion_spacing'] == 3.0
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


@pytest.mark.parametrize('height', [24.0, 24.5, 24.7])
def test_npt_proposal_and_custom_cutoff_leave_backend_contraction_margin(height):
    from so3lr.flash.check import NPT_BOX_MARGIN
    from so3lr.pbc import minimum_image_problem
    settings = long_range_settings(True, height, DEFAULT_MODEL, ensemble='npt')
    contracted = height * (1 - NPT_BOX_MARGIN)
    cell = [[contracted, 0, 0], [0, contracted, 0], [0, 0, contracted]]
    assert minimum_image_problem(cell, settings['lr_cutoff']) is None
    assert settings['kspace_electrostatics'] == settings['kspace_dispersion'] == 'pme'
    with pytest.raises(ValueError, match='3% NPT contraction margin'):
        long_range_settings(True, height, DEFAULT_MODEL, cutoff=12, ensemble='npt')
    for ensemble in ('nvt', 'nve', 'nvt-langevin'):
        assert long_range_settings(True, height, DEFAULT_MODEL, ensemble=ensemble) == {'lr_cutoff': 12.0}


def test_npt_margin_applies_to_short_range_and_half_cell_user_cutoffs():
    from so3lr.flash.check import model_cutoff
    height = 2 * model_cutoff(DEFAULT_MODEL)
    assert long_range_settings(True, height, DEFAULT_MODEL)['lr_cutoff'] == height / 2
    with pytest.raises(ValueError, match='short-range cutoff.*NPT contraction margin'):
        long_range_settings(True, height, DEFAULT_MODEL, ensemble='npt')
    assert long_range_settings(True, 30, DEFAULT_MODEL, cutoff=15) == {'lr_cutoff': 15.0}
    with pytest.raises(ValueError, match='at most 14.55'):
        long_range_settings(True, 30, DEFAULT_MODEL, cutoff=15, ensemble='npt')
    assert long_range_settings(True, 30, DEFAULT_MODEL, ensemble='npt') == {'lr_cutoff': 12.0}


def test_proposed_pme_mesh_reduces_reciprocal_discretization_error(x64):
    """An independent neutral charge system checks the mesh, without model weights.

    This measures reciprocal discretization only, not total SO3LR convergence.
    The finer reference keeps the same real-space split and periodic geometry.
    """
    import jax.numpy as jnp
    from jaxpme.kspace import generate_kvectors, get_reciprocal
    from jaxpme.potentials import potential
    from jaxpme.solvers import pme
    from so3lr.kspace_defaults import DEFAULT_KSPACE_SPACING
    from so3lr.mlff.utils.kspace_utils import setup_kspace_grid

    settings = long_range_settings(True, 12.4, DEFAULT_MODEL)
    cell = jnp.eye(3) * 12.4
    positions = jnp.array([[.2, .7, 1.1], [3.3, 2.5, 1.7],
                          [7.1, 6.4, 5.2], [8.5, 10.2, 9.6]])
    charges = jnp.array([.8, -.8, .5, -.5])
    reciprocal = get_reciprocal(cell)
    volume = jnp.linalg.det(cell)
    solver = pme(potential(), interpolation_nodes=4)

    def energy(spacing):
        grid, smearing = setup_kspace_grid('pme', settings['kspace_smearing'], spacing, cell)
        vectors = generate_kvectors(reciprocal, grid.shape, dtype=jnp.float64, for_ewald=False)
        values = solver.kspace(smearing, charges, reciprocal, grid, vectors,
                               positions, volume, cell, jnp.array([True] * 3))
        return float(jnp.sum(charges * values) * 14.399645351950548)

    coarse = energy(DEFAULT_KSPACE_SPACING)
    proposed = energy(settings['kspace_spacing'])
    reference = energy(settings['kspace_spacing'] / 2)
    assert abs(proposed - reference) < 1e-3  # eV for this fixed illustrative system
    assert abs(proposed - reference) < abs(coarse - reference) / 5


def test_proposal_reads_model_metadata_without_initializing_jax():
    subprocess.run([sys.executable, '-c',
                    "from so3lr.cli.setup_physics import long_range_settings; "
                    "from so3lr.model_registry import DEFAULT_MODEL; "
                    "long_range_settings(True, 12.4, DEFAULT_MODEL, ensemble='npt'); "
                    "import sys; assert 'jax' not in sys.modules; assert 'flax' not in sys.modules"],
                   check=True)
