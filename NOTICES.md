# NOTICES

SO3LR combines source code and pretrained model weights under different
licenses. Except where otherwise noted below, the code is licensed under the
MIT License in [`LICENSE`](LICENSE). The full texts of the other licenses are in
[`licenses/`](licenses/).

## SO3LR Tauri model weights

The pretrained SO3LR Tauri models in `so3lr/models/so3lr-tauri-*/` (every
size and release date: `so3lr-tauri-{s,m,l}-2602` and
`so3lr-tauri-{s,m,l}-2606`), © 2026 Adil Kabylda, are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0), reproduced in
[`licenses/LICENSE.CC-BY-NC-4.0`](licenses/LICENSE.CC-BY-NC-4.0). They may be
used, shared and adapted for non-commercial purposes with attribution; commercial
use is not permitted. This covers the parameters (`params.pkl`) and the files
that describe them in the same directory. Models fine-tuned from these weights
are adaptations and fall under the same license.

The original SO3LR model in `so3lr/models/so3lr/`, published with
[Kabylda et al., JACS 2025](https://doi.org/10.1021/jacs.5c09558), remains
under the MIT License, as previously released.

## Robust loss

`so3lr/mlff/utils/robust_loss_jax/` is a modified JAX adaptation of the
Google Research implementation of *A General and Adaptive Robust Loss Function*
(<https://github.com/google-research/google-research/tree/master/robust_loss>),
Copyright 2024 The Google Research Authors. It is licensed under the Apache
License 2.0, reproduced in
[`licenses/LICENSE.ROBUST-LOSS-APACHE-2.0`](licenses/LICENSE.ROBUST-LOSS-APACHE-2.0).

## Data

These licenses do not relicense the datasets the models were trained on or the
scientific reference data included with the code; those remain under the terms
set by their original authors.
