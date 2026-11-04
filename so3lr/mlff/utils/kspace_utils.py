import logging
from typing import Tuple, Optional

import jax.numpy as jnp
import jaxpme

logger = logging.getLogger("SO3LR")

from so3lr.kspace_defaults import (  # noqa: F401  (re-exported)
    DEFAULT_KSPACE_DISPERSION_INTERP_NODES, DEFAULT_KSPACE_DISPERSION_SMEARING,
    DEFAULT_KSPACE_DISPERSION_SPACING, DEFAULT_KSPACE_INTERP_NODES,
    DEFAULT_KSPACE_SMEARING, DEFAULT_KSPACE_SPACING)


def kspace_grid_shape(kspace_method, kspace_spacing, box):
    """Mesh shape the method would build for this cell, as a hashable tuple (or None).

    The k-grid arrays themselves are pure shape carriers -- jaxpme returns
    ``jnp.ones(shape)`` and every consumer reads only ``.shape``, regenerating the
    k-vectors from each graph's own cell -- so the shape is exactly what a
    JIT-compiled step depends on, and cells that share it can share a batch.
    """
    if kspace_method is None:
        return None
    cell = jnp.asarray(box)
    if kspace_method == 'ewald':
        return tuple(int(s) for s in jaxpme.kspace.get_kgrid_ewald_shape(
            cell, lr_wavelength=jnp.array([kspace_spacing])))
    return tuple(int(s) for s in jaxpme.kspace.get_kgrid_mesh_shape(
        cell, mesh_spacing=jnp.array([kspace_spacing])))

def setup_kspace_grid(
    kspace_electrostatics: str,
    kspace_smearing: float,
    kspace_spacing: float,
    box: jnp.ndarray,
    k_grid: jnp.ndarray = None,
    dtype=None,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Setup the k-space grid for Ewald summation.
    Args:
        kspace_electrostatics (str): Type of k-space electrostatics (None or 'ewald' or 'pme').
        kspace_smearing (float): Smearing value for the k-space grid.
        kspace_spacing (float): Spacing for the k-space grid.
        box (jnp.ndarray): Box of the system. Shape (3, 3), (3,), or (1,).
        k_grid (jnp.ndarray): Existing k-space grid to reuse if shape matches.
        dtype: Working dtype for the returned arrays. They are built from Python
            floats, which become float64 whenever x64 is on -- even in a float32
            run -- and a float64 smearing then promotes everything downstream.
            None keeps the historical behaviour (no cast).
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: k-space grid and smearing values.
    """

    k_spacing = jnp.array([kspace_spacing])
    k_smearing = jnp.array([kspace_smearing])
    if box.shape == (3, 3):
        cell = box
    elif box.shape == (3, ):
        cell = jnp.diag(box)
    elif box.shape == (1,):
        cell = jnp.diag(jnp.repeat(box, 3))
    else:
        raise ValueError(f"k-space grid setup: Invalid box shape {box.shape}. Expected (3, 3), (3,), or (1,).")

    if kspace_electrostatics == 'ewald':
        if k_grid is None or k_grid.shape != jaxpme.kspace.get_kgrid_ewald_shape(cell, lr_wavelength=k_spacing):
            k_grid = jaxpme.kspace.get_kgrid_ewald(cell, lr_wavelength=k_spacing)
            logger.info(f'Doing Ewald with k-grid shape: {k_grid.shape}')
    elif kspace_electrostatics == 'pme' or kspace_electrostatics == 'p3m':
        if k_grid is None or k_grid.shape != jaxpme.kspace.get_kgrid_mesh_shape(cell, mesh_spacing=k_spacing):
            k_grid = jaxpme.kspace.get_kgrid_mesh(cell, mesh_spacing=k_spacing)
            logger.info(f'Doing {kspace_electrostatics.upper()} with k-grid shape: {k_grid.shape}')
    else:
        k_grid = None
        k_smearing = None
    if dtype is not None:
        k_grid = None if k_grid is None else jnp.asarray(k_grid, dtype=dtype)
        k_smearing = None if k_smearing is None else jnp.asarray(k_smearing, dtype=dtype)
    return k_grid, k_smearing
