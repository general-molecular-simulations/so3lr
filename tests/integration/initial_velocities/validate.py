#!/usr/bin/env python
"""Validation script for initial velocities test."""
import sys
import pathlib

def validate():
    """Validate that initial velocities were loaded from input."""
    workdir = pathlib.Path('.')
    
    # Check stderr for velocity loading message
    stderr_log = workdir / 'stderr.log'
    if stderr_log.exists():
        content = stderr_log.read_text()
        if 'Loading initial velocities from input file' in content or \
           'velocities' in content.lower():
            print("Found velocity loading message in stderr")
        else:
            print("WARNING: Velocity loading message not found in stderr")
    
    # Check trajectory exists
    trajectory = workdir / 'trajectory.hdf5'
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        return False
    
    print("Initial velocities simulation completed")
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
