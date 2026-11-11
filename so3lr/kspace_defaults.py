"""K-space defaults, kept free of JAX so `so3lr --help` can read them cheaply.

The CLI, the MD driver, the eval path and callers of make_so3lr all read these
(some through so3lr.mlff.utils.kspace_utils), so a change here reaches every entry point.
"""
MIN_DIRECT_LR_CUTOFF = 10.0              # A; shorter MD cutoffs need k-space electrostatics
DEFAULT_KSPACE_SMEARING = 4.0             # A
DEFAULT_KSPACE_SPACING = 2.0              # A
DEFAULT_KSPACE_INTERP_NODES = 4
# Interpolation orders jax-pme implements: Lagrange weights for PME, B-splines for P3M
KSPACE_INTERP_NODES = {"pme": (4,), "p3m": (1, 2, 3, 4, 5)}
DEFAULT_KSPACE_DISPERSION_SMEARING = 2.0  # A
DEFAULT_KSPACE_DISPERSION_SPACING = 3.0   # A
DEFAULT_KSPACE_DISPERSION_INTERP_NODES = 4
