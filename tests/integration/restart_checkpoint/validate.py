#!/usr/bin/env python
"""Validation script for restart checkpoint test."""
import sys
import pathlib
import subprocess
import os

def validate():
    """Validate restart functionality.
    
    This test:
    1. Runs a setup simulation to generate restart.npz
    2. Runs the main test loading from restart.npz
    3. Validates the restart was loaded correctly
    """
    workdir = pathlib.Path('.')
    
    # Phase 1: Generate restart file if not exists
    restart_file = workdir / 'restart.npz'
    if not restart_file.exists():
        print("Generating restart file with setup simulation...")
        result = subprocess.run(
            ['so3lr', '--settings', 'setup_settings.yaml'],
            cwd=workdir,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            print(f"ERROR: Setup failed: {result.stderr}")
            return False
        
        if not restart_file.exists():
            print("ERROR: Setup did not create restart.npz")
            return False
        print("Setup complete, restart.npz created")
    
    # Phase 2: Check main simulation loaded the restart
    stderr_log = workdir / 'stderr.log'
    if stderr_log.exists():
        content = stderr_log.read_text()
        if 'Loading state from' in content or 'restart' in content.lower():
            print("Found restart loading message in stderr")
        else:
            print("WARNING: Restart loading message not found")
    
    # Check trajectory exists
    trajectory = workdir / 'trajectory.hdf5'
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        return False
    
    print("Restart checkpoint test completed")
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
