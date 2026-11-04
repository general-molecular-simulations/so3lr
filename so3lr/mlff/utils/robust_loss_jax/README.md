# Robust loss (Google Research, Apache-2.0)

These files are a JAX adaptation of the Google Research implementation of
*A General and Adaptive Robust Loss Function*, used by SO3LR's training loss:

- `cubic_spline.py`
- `distribution.py`
- `general.py`
- `data/partition_spline.npz`

Unlike the rest of SO3LR (MIT), they remain under the Apache License 2.0,
reproduced in
[`licenses/LICENSE.ROBUST-LOSS-APACHE-2.0`](../../../../licenses/LICENSE.ROBUST-LOSS-APACHE-2.0)
and listed in the repository's `NOTICES.md`. The Python sources keep the
upstream notice, *Copyright 2024 The Google Research Authors*. They replace the
upstream numerical backend with JAX, so they are modified copies; the upstream
commit used for the original port was not recorded.

Upstream: <https://github.com/google-research/google-research/tree/master/robust_loss>

Publication: Jonathan T. Barron, "A General and Adaptive Robust Loss Function,"
CVPR 2019, <https://arxiv.org/abs/1701.03077>.
