#!/usr/bin/env python
"""Validation script for restart ensemble change test."""
import sys
from validation_utils import Validator


def validate():
    """Validate restart with ensemble change (NVE -> NVT).
    """
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])
    n_cycles = settings['md_cycles']

    # Check required files exist
    v.check_files_exist([
        'nve_restart.npz',
        'trajectory.hdf5',
    ])

    # Validate HDF5 structure
    v.validate_hdf5(
        'trajectory.hdf5',
        required_datasets=['positions'],
        check_shapes={
            'positions': (n_cycles, 1, n_atoms, 3),
        },
    )

    # Check for success
    v.check_log_contains(
        'stderr.log',
        required_strings=['completed successfully',
                          'Loading state from',
                          # RestartInNewEnsembleError message (reworded in fdf60ff):
                          # "Restart file was saved under ensemble=NVE; loading
                          #  into ensemble=NVT. ..."
                          'Restart file was saved under ensemble=NVE',
                          'loading into ensemble=NVT'],
    )

    # Check all settings are reflected in logs
    v.check_settings_in_logs(settings)

    print("Restart ensemble change test completed")
    return v.report()


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
