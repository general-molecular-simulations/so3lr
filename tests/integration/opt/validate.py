#!/usr/bin/env python
"""Validation script for acetone optimization test."""
import sys
from validation_utils import Validator


def validate():
    """Validate optimization test output."""
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])

    # Check output file exists
    v.check_file_exists('optimized.xyz')

    # Validate XYZ structure
    v.validate_xyz('optimized.xyz', expected_atoms=n_atoms)

    # Check log for convergence and success
    v.check_log_contains(
        'stderr.log',
        required_strings=['optimization', 'completed successfully'],
    )

    # Check all settings are reflected in logs
    v.check_settings_in_logs(settings)

    return v.report()


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
