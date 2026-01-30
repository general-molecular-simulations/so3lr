#!/usr/bin/env python
"""Validation script for acetone optimization test."""
import sys
import pathlib

def validate():
    """Validate test output."""
    workdir = pathlib.Path('.')
    
    # Check optimized structure exists
    output_file = workdir / 'optimized.xyz'
    if not output_file.exists():
        print(f"ERROR: Output file not found: {output_file}")
        return False
    
    # Validate XYZ structure
    try:
        from ase.io import read
        atoms = read(output_file)
        if len(atoms) != 10:  # Acetone has 10 atoms
            print(f"ERROR: Expected 10 atoms, got {len(atoms)}")
            return False
        print(f"Optimized structure has {len(atoms)} atoms")
    except ImportError:
        print("WARNING: ASE not available, skipping structure validation")
    except Exception as e:
        print(f"ERROR: Failed to read output: {e}")
        return False
    
    print("Validation PASSED")
    return True


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
