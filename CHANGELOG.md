# Changelog

All notable changes to SO3LR will be documented in this file. The project uses
semantic versioning for the Python package; model checkpoints are additionally
identified by immutable dated names such as `so3lr-m-2606`.

## [Unreleased]

### Added

- Reproducible release environments managed by `uv`, with an audited lockfile,
  immutable Git dependency revisions, Python 3.12/3.13 coverage, and archive
  content verification (PR #42).
- Machine-readable software and published SO3LR v1 citation metadata in
  `CITATION.cff`. No SO3LR-2 or software DOI is claimed before one is issued.
- SHA-256 coverage for every bundled model artifact, with an installed-package
  verifier available as `python -m so3lr.verify_model_artifacts`.
- The complete Apache License 2.0 and BSD 3-Clause texts and third-party
  notices for the vendored Google Research robust-loss implementation,
  JAX MD-derived masking code, and torch-pme-derived Ewald tuner.

### Changed

- Weights & Biases telemetry is opt-in and installed through the optional
  `wandb` extra rather than the base runtime environment (PRs #44 and #45).
- TFDS-backed training now uses finite, unit-consistent epochs and preserves
  complete data trajectories across restarts (PR #41).
- Documentation and Colab examples now use the immutable
  `so3lr-{s,m,l}-2606` names for reproducible results and describe the actual
  pinned no-argument default, `so3lr-m-2606`.
- README float64 examples enable JAX x64 before importing SO3LR.
- The v2 observable example uses `a0_ratios`; `hirshfeld_ratios` remains the
  documented legacy-v1 output key.
- Example notebooks have been stripped of execution output and personal paths.
- The production HDF5 notebook documents the frame-major trajectory layout in
  which write buffering does not add a public array dimension.

### Fixed

- Evaluation overwrite replaces old predictions and embeddings instead of
  appending or silently overwriting; individual files are saved only after
  serialization succeeds, without a multi-file transaction guarantee.
- SO3LR's NPZ restart preserves the previous checkpoint on serialization failure.
- Fine-tuning restart fingerprints include the resolved parameter masks;
  older TFDS fine-tuning checkpoints remain usable for inference but cannot
  provide exact training continuation without that metadata.
- Preserve molecular dipoles for three-atom systems, apply atom selections to
  HDF5 embeddings, and include requested embeddings in EXTXYZ output.
- TFDS retains explicit periodic axes and empty periodic neighbor-offset
  arrays, and respects its prepared dataset splits independently of global
  ASE/NPZ split counts. Reciprocal-space training checks every batch for
  unsupported mixed-axis periodicity; explicit vacuum flags suppress its
  reciprocal-space contribution even when a cell is present.
- Mask absent atoms before taking dispersion-charge square roots so padded
  PME/Ewald batches have finite derivatives with respect to predicted C6.
- A rejected training restart leaves checkpoint metadata untouched. ASE/NPZ
  fine-tuning resumes weights and step from its existing target checkpoint
  instead of silently using the parent model; optimizer and data-order state
  still restart fresh on this path.
- Verify the complete installed package payload against its source when
  checking release archives, reject empty model manifests, and run CI for
  stacked pull requests as well as requests targeting the main branch.
- Evaluation keeps reference C6 labels separate from physical model
  predictions, preventing a label/prediction mix-up in C6 metrics (PR #37).
- Configuration unit expressions are parsed by a constrained unit parser
  instead of Python `eval` (PR #38).
- Periodic calculations reject unsupported mixed-axis periodicity, including
  mixed-periodicity batches, and dispersion PME uses a residual cutoff
  consistent with the configured damping split (PRs #39 and #40).
- The Ewald tuner uses torch-pme's corrected reciprocal-space error estimate,
  including the square-root prefactor scaling used by its reference formula.
- MD trajectory output retains partially filled buffers and
  protects existing HDF5 trajectories during restart; flexible-cell NPT rejects
  unsafe configurations while retaining semi-isotropic coupling aliases
  (PRs #43 and #46).
- Loading user-supplied pickle artifacts emits an explicit security warning,
  while checksum-covered artifacts bundled with the installed package remain
  trusted (PR #44).
- Dimer calculations validate the selected model's supported elements before
  evaluation (PR #47).
- Replaced obsolete notebook imports from the removed top-level `mlff` package
  with modules under the installed `so3lr` package.
- Corrected example-notebook thermodynamic units: NVT now converts its
  target temperature to energy units, and both NPT examples convert pressure
  and represent the 3D box with three lengths so the barostat uses its volume.
- The production notebook flushes its final partial HDF5 buffer and asserts
  complete, finite, dtype-preserving, frame-major trajectory output instead of
  silently losing the last frames; it uses the maintained HDF5 writer. NPT
  EXTXYZ output retains each frame's cell and periodicity, and all MD notebooks
  fail loudly rather than accepting neighbor-list overflow.
- The Colab workflow now pins its uv bootstrap, rejects partial-periodic NPT
  inputs, and fails the cell when setup or an SO3LR command fails.
- Qualified the README statement about energy supervision: it applies to the
  legacy v1 checkpoint, whereas the dated v2 checkpoints include an energy
  loss.

[Unreleased]: https://github.com/general-molecular-simulations/so3lr/compare/v0.0.1...HEAD
