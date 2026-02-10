#!/usr/bin/env python
"""Validation script for NPT ensemble test."""
import sys
from validation_utils import Validator


def validate():
    """Validate NPT simulation output."""
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])
    n_cycles = settings['md_cycles']

    # Check trajectory exists
    v.check_file_exists('trajectory.hdf5')

    # Validate HDF5 has positions and cell info
    v.validate_hdf5(
        'trajectory.hdf5',
        required_datasets=['positions', 'box'],
        check_shapes={
            'positions': (n_cycles, 1, n_atoms, 3),
            'box': (n_cycles, 1, 3),
        },
    )

    # Validate log contains NPT and success message
    v.check_log_contains(
        'stderr.log',
        required_strings=['NPT', 'completed successfully'],
    )

    # Check all settings are reflected in logs
    v.check_settings_in_logs(settings)

    print("NPT simulation completed")
    return v.report()


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
