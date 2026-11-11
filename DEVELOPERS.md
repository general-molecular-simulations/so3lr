# SO3LR Developer Guide

## Overview
This document is for developers contributing to the SO3LR project. It covers repository structure, development setup, testing, and contribution guidelines.

## Repository Structure
- `so3lr/` — Main source code for the SO3LR model and CLI
- `tests/` — Unit tests
- `tests/integration/` — end-to-end integration tests
- `so3lr/models/` — Bundled pretrained model directories, one per canonical name (`so3lr`, `so3lr-tauri-{s,m,l}-2602`, `so3lr-tauri-{s,m,l}-2606`); short names like `so3lr-tauri-m` are aliases resolved by `so3lr/model_registry.py`
- `pyproject.toml` — Python dependencies and build system

## Running Tests

Create the locked development environment and run the short suite with:

```sh
uv sync --locked --group dev
uv run pytest tests/
```

For a sufficiently large machine, `uv run pytest tests/ -n 16` runs tests in
parallel. The locked development group includes `pytest-xdist`, which provides
the `-n` option.

## Environments and the lockfile
`uv.lock` is the authoritative environment. Always pass `--locked`, so uv fails
instead of silently resolving newer packages. Optional stacks are extras:

```sh
uv sync --locked --group dev --extra tfds      # also: pysages, wandb, cuda12 (Linux x86-64)
uv sync --locked --no-default-groups --group notebooks   # notebook execution tools
```

To reproduce paper runs, check out the published tag or the exact 40-character
commit, then `uv sync --locked`; the wheel and sdist omit notebooks and input
data. Bundled model files can be checked with
`cd so3lr/models && sha256sum -c SHA256SUMS`.

Never edit `uv.lock` by hand: change `pyproject.toml`, run
`uv lock --upgrade-package PACKAGE` (or `uv lock --upgrade`), and commit the
lockfile diff with the change that needed it. Git dependencies must be pinned to
40-character commits; `[tool.uv].override-dependencies` pins GLP's transitive
`comms` dependency, which GLP itself names by branch.

Security updates should use targeted `uv lock --upgrade-package PACKAGE`
commands to preserve the numerical stack and Git revisions. Dependabot checks
the uv project weekly and groups security updates into one pull request. Run
the dependency compatibility checks with all of their optional tools installed:

```sh
uv sync --locked --group dev --group notebooks --extra tfds --extra wandb
uv run --no-sync pytest tests/test_dependency_compat.py
```

The `tfds` extra uses TensorFlow and TensorBoard 2.21 with patched setuptools.
TensorBoard is explicit because TensorFlow 2.21 no longer installs it;
TensorBoard 2.20 imports `pkg_resources`, which recent setuptools removes.
TensorFlow 2.21 also requires h5py below 3.15, so data-loader checks are part of
validating this stack. After an update, rerun `uv sync --locked` in existing
environments to install the patched packages.

## Adding Features or Fixes
1. Create a new branch for your feature or bugfix.
2. Make your changes with clear commit messages.
3. Add or update tests as needed.
4. Run all tests before submitting a pull request.
5. Open a pull request with a description of your changes.

## Adding Tests
Look at `tests/integration/README.md` on how to add new integration tests. This should be very easy. Please also consider writing unit tests for deeper functionality.

## Documentation
- Update the user documentation at <https://so3lr.github.io/> for user-facing
  changes. `README.md` is a landing page: keep it to installation, a quick
  start, the model table and links into the docs.
- Update this `DEVELOPERS.md` for developer-facing changes.
- Use docstrings for all modules, classes, and functions.

## Contact
For questions or support, open an issue or contact the maintainers.
