#!/usr/bin/env python
"""Validation script for NVE ensemble test."""
import sys
import pathlib

def validate():
    """Validate NVE simulation output."""
    workdir = pathlib.Path('.')
    
    # Check stderr for NVE indication
    stderr_log = workdir / 'stderr.log'
    if stderr_log.exists():
        content = stderr_log.read_text()
        if 'NVE' not in content.upper():
            print("WARNING: NVE not mentioned in stderr")
    
    # Check trajectory exists
    trajectory = workdir / 'trajectory.hdf5'
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        return False
    
    # Check restart file created
    restart = workdir / 'nve_restart.npz'
    if not restart.exists():
        print(f"ERROR: Restart file not found: {restart}")
        return False
    
    print("NVE simulation completed with restart file")
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
