"""SO3LR-Flash: SO3LR MD for millions of atoms on one GPU.

``engine.FlashSO3LR`` evaluates so3lr-tauri-s/m/l and the legacy so3lr with Pallas-Triton
kernels that build their own cell grid and keep no per-edge arrays: the same energy and
forces as the neighbour-list potential, faster, in a few KiB per atom. Periodic orthorhombic
cells or none (the gas phase); real-space long range or PME; dE/d(strain) for NPT.
``domains.FlashDomains`` evaluates systems larger than one GPU holds a domain at a time,
exactly (energy and forces; real-space long range). ``loader.load_params`` reads a
registry model; ``md.perform_md_flash`` runs ``so3lr nve|nvt|nvt-langevin|npt``, which
use it by default wherever it applies (``check.flash_problem``); ``verify`` compares it
with the neighbour-list potential. On an sm_90 GPU with nvcc, so3lr-tauri-s/m/l's fast mode
runs CUDA kernels (cuda: the attention per pair on tensor cores, 3x the speed), picked
automatically (flash_kernels); the Pallas kernels serve everything else.
"""
