#!/usr/bin/env python
"""Validation script for observables test."""
import sys
from validation_utils import Validator


def validate():
    """Validate that observables are saved in trajectory."""
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])
    n_cycles = settings['md_cycles']

    # Check trajectory exists
    v.check_file_exists('trajectory.hdf5')

    # Validate HDF5 structure
    v.validate_hdf5(
        'trajectory.hdf5',
        required_datasets=['positions', 'dipole_vec', 'partial_charges'],
        check_shapes={
            'positions': (n_cycles, n_atoms, 3),
            'dipole_vec': (n_cycles, 1, 3),
            'partial_charges': (n_cycles, n_atoms),
        },
    )

    # Check log mentions observables
    v.check_log_contains(
        'stderr.log',
        required_strings=['completed successfully'],
    )

    # Check all settings are reflected in logs
    v.check_settings_in_logs(settings)

    return v.report()


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
