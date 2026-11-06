#!/usr/bin/env python
"""Validation for the cylindrical / funnel restraint NVT integration test.

Confirms the feature runs end-to-end with the real model and that the restrained
mass-weighted COM stays confined near the axis (and the run produced no NaNs, i.e.
the on-axis safe-norm path is sound in a real simulation).
"""
import sys

import h5py
import numpy as np
from ase.io import read
from validation_utils import Validator


def validate():
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])
    n_cycles = settings['md_cycles']

    v.check_files_exist(['trajectory.hdf5', 'so3lr.log'])

    # The run succeeded, was NVT, and the restraint actually activated.
    v.check_log_contains(
        'stderr.log',
        required_strings=['completed successfully', 'NVT', 'Cylindrical'],
    )

    v.validate_hdf5(
        'trajectory.hdf5',
        required_datasets=['positions'],
        check_shapes={'positions': (n_cycles, n_atoms, 3)},
    )

    # Physics check: mass-weighted COM stays within the cylinder (+margin), and no NaNs.
    masses = read(settings['input_file']).get_masses()
    idx = np.asarray(settings['cylinder_restraint_atom_indices'], dtype=int)
    w = (masses[idx] / masses[idx].sum())[:, None]
    radius = float(settings['cylinder_restraint_radius'])
    origin = np.asarray(settings['cylinder_restraint_axis_origin'], dtype=float)
    with h5py.File('trajectory.hdf5', 'r') as f:
        pos = np.asarray(f['positions'])  # (n_cycles, n_atoms, 3)
    if not np.all(np.isfinite(pos)):
        v.errors.append('trajectory positions contain non-finite values (NaN/Inf)')
    com = (w * pos[:, idx]).sum(axis=1)                      # (n_cycles, 3)
    r_perp = np.hypot(com[:, 0] - origin[0], com[:, 1] - origin[1])
    margin = 0.5
    if r_perp.max() > radius + margin:
        v.errors.append(
            f'COM perpendicular distance {r_perp.max():.3f} Å exceeded '
            f'radius+margin ({radius + margin:.3f} Å)'
        )
    else:
        print(f'COM r_perp max = {r_perp.max():.4f} Å (radius={radius} Å) OK')

    return v.report()


if __name__ == '__main__':
    sys.exit(0 if validate() else 1)
