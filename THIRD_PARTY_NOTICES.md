# Third-party notices

SO3LR is distributed under the MIT License in `LICENSE`, except for the
third-party portions identified below. Those portions remain available under
their original licenses, reproduced in `LICENSE-APACHE-2.0` and
`LICENSE-BSD-3-CLAUSE`.

These notices cover the identified code portions, not a blanket license for
bundled checkpoints, datasets, or scientific reference data. Their licensing
review remains pending; the project's MIT metadata does not relicense them.

## Google Research robust loss

The following files are a JAX adaptation of the Google Research implementation
of *A General and Adaptive Robust Loss Function*:

- `so3lr/mlff/utils/robust_loss_jax/cubic_spline.py`
- `so3lr/mlff/utils/robust_loss_jax/distribution.py`
- `so3lr/mlff/utils/robust_loss_jax/general.py`
- `so3lr/mlff/utils/robust_loss_jax/data/partition_spline.npz`

Upstream project:
<https://github.com/google-research/google-research/tree/master/robust_loss>

The Python sources retain the notice:

> Copyright 2024 The Google Research Authors.

License: Apache License 2.0. The local files replace the upstream numerical
backend with JAX and therefore are modified copies. The exact upstream commit
used for the original port was not recorded in this repository's import
history; that revision remains unconfirmed rather than guessed here.

Relevant publication: Jonathan T. Barron, “A General and Adaptive Robust Loss
Function,” CVPR 2019, <https://arxiv.org/abs/1701.03077>.

## JAX MD safe masking

`safe_mask` in `so3lr/mlff/masking/mask.py` is derived from JAX MD's
`jax_md.util.safe_mask` at commit
`b4bce7ab9b37b6b9b2d0a5f02c143aeeb4e2a560`, as recorded in the source file:

<https://github.com/google/jax-md/blob/b4bce7ab9b37b6b9b2d0a5f02c143aeeb4e2a560/jax_md/util.py#L67>

Upstream project: <https://github.com/google/jax-md>

License: Apache License 2.0. The surrounding SO3LR masking helpers are local
extensions; this notice applies to the JAX MD-derived portion.

## torch-pme Ewald tuning

Portions of `so3lr/cli/tune_ewald.py` are modified JAX/NumPy adaptations of
the following torch-pme tuning sources:

- `src/torchpme/tuning/tuner.py`
- `src/torchpme/tuning/ewald.py`
- `src/torchpme/tuning/pme.py`

Closest verifiable upstream snapshot:
<https://github.com/lab-cosmo/torch-pme/tree/d762bc8fe531f919ae0f296e8b67957a859a673a/src/torchpme/tuning>

License: BSD 3-Clause. Copyright (c) 2024, torch-pme developers. The license
is reproduced in `LICENSE-BSD-3-CLAUSE`.

The SO3LR port was authored on 2025-06-19. Its import commit,
`2b721aba6bfeb110c47df2340b10a135ffc75f9b`, records only the mutable upstream
`main` URL. Commit `d762bc8fe531f919ae0f296e8b67957a859a673a` was the
upstream `main` HEAD on that date and contains matching source material.
Because the original checkout SHA was not recorded, it is identified here as
an immutable matching reference snapshot rather than as an independently
verified historical checkout. Confirmation from the original port author
remains outstanding.

The local implementation replaces PyTorch with JAX/NumPy and adds
SO3LR-specific short-range corrections, dispersion tuning, model integration,
timing, and command-line behavior.

The Ewald reciprocal-space error estimate has since been replaced by the
Kolafa-Perram bound (as in LAMMPS `Ewald::rms`).
