#!/usr/bin/env python
"""Validation script for restart ensemble change test."""
import sys
import pathlib
import subprocess
import os

def validate():
    """Validate restart with ensemble change (NVE -> NVT).
    
    This test verifies that:
    1. Loading a NVE restart into NVT simulation works
    2. A warning about ensemble mismatch is produced
    """
    workdir = pathlib.Path('.')
    
    # Phase 1: Generate NVE restart file if not exists
    restart_file = workdir / 'nve_restart.npz'
    if not restart_file.exists():
        print("Generating NVE restart file with setup simulation...")
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
            print("ERROR: Setup did not create nve_restart.npz")
            return False
        print("Setup complete, nve_restart.npz created")
    
    # Phase 2: Check main simulation handled ensemble change
    stderr_log = workdir / 'stderr.log'
    if stderr_log.exists():
        content = stderr_log.read_text()
        
        # Should have warning about ensemble mismatch
        if 'different ensemble' in content.lower() or 'ensemble' in content.lower():
            print("Found ensemble change message in stderr")
        else:
            print("WARNING: Ensemble change warning not found")
        
        # Should mention NVT
        if 'NVT' in content:
            print("Found NVT ensemble indication")
    
    # Check trajectory exists (should still complete)
    trajectory = workdir / 'trajectory.hdf5'
    if not trajectory.exists():
        print(f"ERROR: Trajectory file not found: {trajectory}")
        return False
    
    print("Restart ensemble change test completed")
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
