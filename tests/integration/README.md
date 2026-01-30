# CLI Integration Test Data

This directory contains test cases for SO3LR CLI integration tests.

## Structure

Each subdirectory represents a test case and should contain:

```
test_case_name/
├── settings.yaml      # Required: SO3LR settings file
├── input.xyz          # Required: Input geometry (or whatever is referenced in settings)
├── validate.py        # Optional: Custom validation script
└── expected/          # Optional: Expected output files for comparison
```

## Test Case Requirements

### settings.yaml
The settings file should be a valid SO3LR YAML configuration. Use minimal cycles/steps
for fast testing:

```yaml
input_file: input.xyz
output_file: trajectory.hdf5
precision: float64
md_T: 300
md_dt: 0.0005
md_cycles: 2      # Keep small for fast tests
md_steps: 2       # Keep small for fast tests
relax_before_run: false
```

### validate.py (Optional)
Custom validation script that exits with code 0 on success:

```python
#!/usr/bin/env python
import sys
import h5py

# Custom validation logic
with h5py.File('trajectory.hdf5', 'r') as f:
    if 'positions' not in f:
        print("Missing positions dataset")
        sys.exit(1)

sys.exit(0)
```

## Running Tests

Run all CLI integration tests:
```bash
pytest tests/test_cli_integration.py -v
```

Run with GPU assignment for parallel execution:
```bash
pytest tests/test_cli_integration.py -v -n 4
```

Run a specific test case:
```bash
pytest tests/test_cli_integration.py -v -k "test_case_name"
```

## Adding New Test Cases

1. Create a new directory under `tests/cli_integration/`
2. Add `settings.yaml` with the configuration to test
3. Add input geometry file(s)
4. Optionally add `validate.py` for custom validation
5. The test will be automatically discovered and run
