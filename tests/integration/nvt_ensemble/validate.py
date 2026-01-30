#!/usr/bin/env python
"""Validation script for acetone vacuum NVT test."""
import sys
import pathlib

def validate():
    """Validate test output."""
    workdir = pathlib.Path('.')
    
    # Check trajectory file exists
    trajectory = workdir / 'trajectory.hdf5'
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        return False
    
    # Check log file exists
    log_file = workdir / 'so3lr.log'
    if not log_file.exists():
        print(f"ERROR: Log file not found: {log_file}")
        return False
    
    # Validate HDF5 structure
    try:
        import h5py
        with h5py.File(trajectory, 'r') as f:
            # Check for expected datasets
            if 'R' not in f and 'positions' not in f:
                print("ERROR: Missing positions dataset in HDF5")
                return False
            print(f"HDF5 datasets: {list(f.keys())}")
    except ImportError:
        print("WARNING: h5py not available, skipping HDF5 validation")
    except Exception as e:
        print(f"ERROR: Failed to read HDF5: {e}")
        return False
    
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
