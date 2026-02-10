import numpy as np

from typing import (Any, Sequence)
from scipy.spatial.transform import Rotation
Array = Any


def rotate_by(x: Array,
              euler_axes: str,
              angles: Sequence[int],
              degrees: bool = True) -> Array:
    """
    Rotate points in 3D, given euler axes and angles. Wraps around `scipy.spatial.transform.Rotation` so check
    there for details.

    Args:
        x (Array): Points in 3D, shape: (...,3)
        euler_axes (str): Euler axes, e.g. 'y', 'zx' or 'xyz'
        angles (List): Angles for each euler axes.
        degrees (bool): Angles are in degree.

    Returns: Rotated points.

    """

    m_rot = get_rotation_matrix(euler_axes=euler_axes, angles=angles, degrees=degrees)
    return apply_rotation(x, m_rot)


def get_rotation_matrix(euler_axes: str,
                        angles: Sequence[int],
                        degrees: bool = True) -> Array:
    """
    Get a rotation matrix, given the euler axes and angles. Wraps around `scipy.spatial.transform.Rotation` so check
    there for details.

    Args:
        euler_axes (str): Euler axes, e.g. 'y', 'zx' or 'xyz'
        angles (List): Angles for each euler axes.
        degrees (bool): Angles are in degree.

    Returns: rotation matrix, shape: (3,3)

    """
    m_rot = Rotation.from_euler(euler_axes, angles, degrees=degrees).as_matrix()
    return m_rot


def apply_rotation(x: Array, m_rot: Array) -> Array:
    """
    Apply rotation matrix to points in 3D.

    Args:
        x (Array): Points in 3D, shape: (...,3)
        m_rot (Array): Rotation matrix, shape: (3,3)

    Returns: Rotated points.

    """
    return np.einsum('ij, ...j -> ...i', m_rot, x[None, ...]).squeeze(0)
