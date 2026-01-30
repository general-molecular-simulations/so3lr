# SO3LR Developer Guide

## Overview
This document is for developers contributing to the SO3LR project. It covers repository structure, development setup, testing, and contribution guidelines.

## Repository Structure
- `so3lr/` — Main source code for the SO3LR model and CLI
- `tests/` — Unit tests
- `tests/integration/` — end-to-end integration tests
- `params/` — Pretrained model checkpoint directory
- `pyproject.toml` — Python dependencies and build system

## Running Tests
  ```sh
  pytest -n 16
  ```
Setting the number of workers -n can be useful to speed up tests but might require a big machine.

## Adding Features or Fixes
1. Create a new branch for your feature or bugfix.
2. Make your changes with clear commit messages.
3. Add or update tests as needed.
4. Run all tests before submitting a pull request.
5. Open a pull request with a description of your changes.

## Adding Tests
Look at `tests/integration/README.md` on how to add new integration tests. This should be very easy. Please also consider writing unit tests for deeper functionality.

## Documentation
- Update the user `README.md` for user-facing changes.
- Update this `DEVELOPERS.md` for developer-facing changes.
- Use docstrings for all modules, classes, and functions.

## Contact
For questions or support, open an issue or contact the maintainers.
