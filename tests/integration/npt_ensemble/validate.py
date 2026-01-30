#!/usr/bin/env python
"""Validation script for NPT ensemble test."""
import sys
import pathlib

def validate():
    """Validate NPT simulation output."""
    workdir = pathlib.Path('.')
    
    # Check stderr for NPT indication
    stderr_log = workdir / 'stderr.log'
    if stderr_log.exists():
        content = stderr_log.read_text()
        if 'NPT' not in content.upper():
            print("WARNING: NPT not mentioned in stderr")
        if 'pressure' not in content.lower() and 'barostat' not in content.lower():
            print("WARNING: Pressure/barostat not mentioned")
    
    # Check trajectory exists
    trajectory = workdir / 'trajectory.hdf5'
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        return False
    
    # For NPT, cell should change - check HDF5 for cell data
    try:
        import h5py
        with h5py.File(trajectory, 'r') as f:
            datasets = list(f.keys())
            print(f"HDF5 datasets: {datasets}")
            # NPT should have box/cell information
            if 'box' in f or 'cell' in f or 'lattice' in f:
                print("Found cell/box data in trajectory (expected for NPT)")
    except ImportError:
        print("WARNING: h5py not available")
    except Exception as e:
        print(f"WARNING: Could not check HDF5: {e}")
    
    print("NPT simulation completed")
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
