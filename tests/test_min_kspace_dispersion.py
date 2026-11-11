"""The relaxation (``so3lr opt``, and the one before MD) builds the model with the same
dispersion k-space settings as MD, so it minimises the energy the MD then samples."""
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from ase import Atoms
from ase.io import write

from so3lr.cli import so3lr_md

WATER = Path(__file__).parent / 'test_data' / 'water_64.xyz'


class _Built(Exception):
    pass


@pytest.mark.parametrize('kspace_dispersion', [None, 'pme'])
def test_perform_min_minimises_with_the_dispersion_kspace(tmp_path, monkeypatch, kspace_dispersion):
    """The model gets the dispersion method and nodes, and the energy function its grid."""
    seen = {}
    monkeypatch.setattr(so3lr_md, 'load_model', lambda *args, **kwargs: seen.update(model=kwargs))
    neighbor_fn = SimpleNamespace(allocate=lambda *args, **kwargs: None)

    def energy(*args, **kwargs):
        seen.update(energy=kwargs)
        raise _Built

    monkeypatch.setattr(so3lr_md, 'process_model', lambda **kwargs: (neighbor_fn, neighbor_fn, energy))
    monkeypatch.setattr(so3lr_md, 'nbrs_cutoff', lambda potential: 0.0)
    monkeypatch.setattr(so3lr_md, 'check_box_for_neighbour_lists', lambda *args: None)
    monkeypatch.setattr(so3lr_md, 'setup_kspace_grid',
                        lambda kspace_electrostatics, kspace_spacing, kspace_smearing, box:
                        (None, None) if kspace_electrostatics is None
                        else (f'grid {kspace_electrostatics} {kspace_spacing}', kspace_smearing))
    # the energy function is first called inside the minimiser's setup
    monkeypatch.setattr(so3lr_md, 'create_fire_fn',
                        lambda energy_fn, **kwargs: energy_fn(jnp.zeros((1, 3))))

    settings = dict(input_file=str(WATER), output_file=str(tmp_path / 'opt.xyz'), lr_cutoff=12.0,
                    kspace_dispersion=kspace_dispersion, kspace_dispersion_interp_nodes=4,
                    kspace_dispersion_smearing=2.5, kspace_dispersion_spacing=3.5)
    with pytest.raises(_Built):
        so3lr_md.perform_min(settings, try_flash=False)
    assert seen['model']['kspace_dispersion'] == kspace_dispersion
    assert seen['model']['kspace_dispersion_interp_nodes'] == 4  # the only PME value
    expected = (None, None) if kspace_dispersion is None else ('grid pme 3.5', 2.5)
    assert (seen['energy']['k_grid_disp'], seen['energy']['k_smearing_disp']) == expected


def test_perform_min_refuses_kspace_dispersion_without_a_cell(tmp_path):
    """A gas-phase structure has no box for the dispersion grid."""
    write(tmp_path / 'water.xyz', Atoms('OH2', positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)]))
    settings = dict(input_file=str(tmp_path / 'water.xyz'), output_file=str(tmp_path / 'opt.xyz'),
                    lr_cutoff=12.0, kspace_dispersion='pme')
    with pytest.raises(ValueError, match='kspace_dispersion'):
        so3lr_md.perform_min(settings, try_flash=False)
