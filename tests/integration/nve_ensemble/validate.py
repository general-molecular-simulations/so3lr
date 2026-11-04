#!/usr/bin/env python
"""Validation script for NVE ensemble test."""
import sys
import h5py
import numpy as np
from validation_utils import Validator


def validate():
    """Validate NVE simulation output."""
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])
    n_cycles = settings['md_cycles']

    # Check required files exist
    v.check_files_exist([
        'trajectory.hdf5',
        'nve_restart.npz',
    ])

    # Validate log contains NVE and success message
    v.check_log_contains(
        'stderr.log',
        required_strings=['NVE', 'completed successfully'],
    )

    # Validate HDF5 structure
    v.validate_hdf5(
        'trajectory.hdf5',
        required_datasets=['positions'],
        check_shapes={
            'positions': (n_cycles, n_atoms, 3),
        },
    )

    with h5py.File(v.workdir / 'trajectory.hdf5', 'r') as trajectory:
        expected_dtype = np.dtype(settings['precision'])
        if trajectory['positions'].dtype != expected_dtype:
            v.errors.append(
                f"Trajectory dtype is {trajectory['positions'].dtype}; "
                f"expected configured precision {expected_dtype}"
            )

    # The final restart must be independent of full trajectory buffers.
    # md_cycles=3 and save_buffer=2 leaves a one-frame tail.
    with np.load(v.workdir / 'nve_restart.npz', allow_pickle=False) as restart:
        if int(restart['step']) != n_cycles:
            v.errors.append(
                f"Restart step is stale: expected {n_cycles}, got {restart['step']}"
            )

    # Check all settings are reflected in logs
    v.check_settings_in_logs(settings)

    print("NVE simulation completed with restart file")
    return v.report()


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
