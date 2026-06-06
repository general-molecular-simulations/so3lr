# SO3LR Integration Tests

This directory contains integration test cases for the SO3LR CLI, testing end-to-end functionality of the simulation engine with various configurations and features.

## Structure

Each subdirectory represents a test case with specific settings and validation:

```
test_case_name/
├── settings.yaml          # Required: SO3LR configuration for the test
├── setup_settings.yaml    # Optional: Configuration for setup phase (e.g., restart tests)
├── input.xyz              # Input geometry (or input.gro for some tests)
└── validate.py            # Custom validation script using Validator utilities
```

### Shared Utilities

- [`validation_utils.py`](validation_utils.py): Shared `Validator` class providing common validation methods for HDF5 files, NPZ checkpoints, XYZ structures, log parsing, and settings verification

## Test Case Requirements

### settings.yaml

A valid SO3LR YAML configuration. Use minimal cycles/steps for fast testing:

```yaml
input_file: input.xyz
output_file: trajectory.hdf5
precision: float64
ensemble: nvt
md_T: 300
md_dt: 0.5
md_cycles: 2      # Keep small for fast tests
md_steps: 2       # Keep small for fast tests
relax_before_run: false
```

### setup_settings.yaml (Optional)

For two-phase tests (e.g., restart tests), this configuration runs before the main test to generate required files.

### validate.py

Custom validation script using the shared `Validator` class:

```python
#!/usr/bin/env python
"""Validation script for test case."""
import sys
from validation_utils import Validator


def validate():
    """Validate simulation output."""
    v = Validator()
    settings = v.load_settings()
    n_atoms = v.count_atoms(settings['input_file'])
    n_cycles = settings['md_cycles']

    # Check required files exist
    v.check_files_exist(['trajectory.hdf5'])

    # Validate log contains expected strings
    v.check_log_contains(
        'stderr.log',
        required_strings=['NVT', 'completed successfully'],
    )

    # Validate HDF5 structure and shapes
    v.validate_hdf5(
        'trajectory.hdf5',
        required_datasets=['positions', 'forces'],
        check_shapes={
            'positions': (n_cycles, 1, n_atoms, 3),
        },
    )

    # Check all settings appear in logs
    v.check_settings_in_logs(settings)

    return v.report()


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
```

## Validator Utilities

The `Validator` class in [`validation_utils.py`](validation_utils.py) provides:

- **File checking**: `check_file_exists()`, `check_files_exist()`
- **Log validation**: `check_log_contains()` with required/forbidden strings
- **HDF5 validation**: `validate_hdf5()` with dataset and shape checking
- **NPZ validation**: `validate_npz()` for restart files
- **XYZ validation**: `validate_xyz()` for structure files
- **Settings verification**: `check_settings_in_logs()` ensures all settings appear in output

See the validation scripts in each test case for usage examples.

## Running Tests

### Run All Integration Tests

```bash
pytest tests/test_integration.py -v
```

### Run Specific Test Case

```bash
pytest tests/test_integration.py -v -k "nve_ensemble"
```

### Run with Parallel Execution

```bash
pytest tests/test_integration.py -v -n 4
```