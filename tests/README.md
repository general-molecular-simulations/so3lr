# SO3LR Tests

## Running Tests

Run the short (default) test suite:

```bash
uv sync --locked --group dev --extra tfds
uv run --no-sync pytest tests/
```

Without the TFDS extra, its data-loader tests are skipped. Add `--extra pysages`
to exercise the enhanced-sampling tests as well. Focused non-TFDS test files can
use the base locked environment.

Run the full test suite (includes slow tests):

```bash
uv run --no-sync pytest tests/ --full
```

### Short vs Full

By default, tests marked `@pytest.mark.slow` are skipped; `--full` runs them too. See [`conftest.py`](conftest.py) for details.

### Run a specific test file or test case

```bash
uv run --no-sync pytest tests/test_ase.py -v
uv run --no-sync pytest tests/test_integration.py -v -k "nve_ensemble"
```

## Test Files

| File | Description |
|------|-------------|
| `test_potential.py` | A float64 run computes in float64 although the checkpoint parameters are float32 |
| `test_ase.py` | Tests ASE calculator against pre-computed reference predictions |
| `test_jaxmd.py` | Tests JAX-MD integration with SO3LR potential |
| `test_cli.py` | Tests CLI command structure and the settings template |
| `test_cli_eval.py` | Tests the CLI `eval` command and its outputs |
| `test_cli_finetune.py` | Tests fine-tuning via the CLI |
| `test_cli_train.py` | Tests training from scratch via the CLI |
| `test_hessian.py` | Tests Hessian calculation and validates against numerical Hessians |
| `test_integration.py` | End-to-end simulation tests using actual CLI commands |
| `test_observables.py` | Tests extraction of individual observables and their summation to total energy |
| `test_model_invariance.py` | On the shipped so3lr-tauri-s: rotation/translation and permutation invariance of a charged molecule, and per-graph independence of a padded batch (charged + neutral) through the `so3lr eval` batching |
| `test_observable_gradients.py` | Tests observable gradients/Jacobians with finite-difference validation |
| `test_pme.py` | Tests PME/Ewald electrostatics against direct jax-pme calculations |

## Subdirectories

### `integration/`

Integration test cases for the SO3LR CLI, testing end-to-end simulation functionality. Each subdirectory contains a `settings.yaml`, input geometry, and a `validate.py` script. See [`integration/README.md`](integration/README.md) for the full structure and how to add new test cases.

Test cases: `npt_ensemble`, `nve_ensemble`, `nvt_langevin`, `restart_checkpoint`, `restart_ensemble_change`, `initial_velocities`, `kspace_electrostatics_p3m`, `observables`.

### `mlff_tests/`

Unit tests for the underlying MLFF framework (model fitting, loss functions, neighbor lists, sparse embeddings, data loading, etc.).

### `test_data/`

Reference data for tests: molecular geometries (`.xyz`) and pre-computed predictions (`.npz`) for molecules like DHA and water.
