import logging
from typing import Tuple, Optional

import jax.numpy as jnp
import jaxpme

logger = logging.getLogger("SO3LR")

def setup_kspace_grid(
    kspace_electrostatics: str,
    kspace_smearing: float,
    kspace_spacing: float,
    box: jnp.ndarray,
    k_grid: jnp.ndarray = None
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Setup the k-space grid for Ewald summation.
    Args:
        kspace_electrostatics (str): Type of k-space electrostatics (None or 'ewald' or 'pme').
        kspace_smearing (float): Smearing value for the k-space grid.
        kspace_spacing (float): Spacing for the k-space grid.
        box (jnp.ndarray): Box of the system. Shape (3, 3), (3,), or (1,).
        k_grid (jnp.ndarray): Existing k-space grid to reuse if shape matches.
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
    return k_grid, k_smearing
