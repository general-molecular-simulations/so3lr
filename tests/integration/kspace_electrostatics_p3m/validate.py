#!/usr/bin/env python
"""Validation script for k-space electrostatics P3M test."""
import sys
from validation_utils import Validator


def validate():
    """Validate k-space electrostatics P3M simulation output."""
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])
    n_cycles = settings['md_cycles']

    # Check trajectory exists
    v.check_file_exists('trajectory.hdf5')

    # Validate HDF5 structure
    v.validate_hdf5(
        'trajectory.hdf5',
        required_datasets=['positions', 'box'],
        check_shapes={'positions': (n_cycles, n_atoms, 3),
                        'box': (n_cycles, 3)},
    )

    # Validate log contains success message
    v.check_log_contains(
        'stderr.log',
        required_strings=['completed successfully', 'Doing P3M'],
    )

    # Check all settings are reflected in logs
    v.check_settings_in_logs(settings)

    print("K-space electrostatics simulation completed")
    return v.report()


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
