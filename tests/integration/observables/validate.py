#!/usr/bin/env python
"""Validation script for observables test."""
import sys
import pathlib

def validate():
    """Validate that observables are saved in trajectory."""
    workdir = pathlib.Path('.')
    
    trajectory = workdir / 'trajectory.hdf5'
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        return False
    
    try:
        import h5py
        with h5py.File(trajectory, 'r') as f:
            datasets = list(f.keys())
            print(f"HDF5 datasets: {datasets}")
            
            # Check for observables
            if 'dipole_vec' not in f and 'dipole' not in f:
                print("ERROR: dipole_vec not found in trajectory")
                return False
            
            if 'partial_charges' not in f and 'charges' not in f:
                print("ERROR: partial_charges not found in trajectory")
                return False
            
            print("Found dipole and partial_charges in output")
            
    except ImportError:
        print("WARNING: h5py not available, skipping HDF5 validation")
    except Exception as e:
        print(f"ERROR: Failed to validate HDF5: {e}")
        return False
    
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
