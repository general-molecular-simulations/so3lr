# Third-party notices

SO3LR is distributed under the MIT License in `LICENSE`, except for the
third-party portions identified below. Those portions remain available under
their original license, reproduced in `LICENSE-APACHE-2.0`.

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
