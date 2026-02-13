import numpy as np

from typing import (Any, Sequence)
Array = Any


def _elementary_rotation(axis: str, angle: float) -> np.ndarray:
    """Build a 3x3 rotation matrix for a single-axis rotation (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Unknown axis: {axis}")


def rotate_by(x: Array,
              euler_axes: str,
              angles: Sequence[int],
              degrees: bool = True) -> Array:
    """
    Rotate points in 3D, given euler axes and angles. Follows the same convention as
    ``scipy.spatial.transform.Rotation.from_euler``: lowercase axes are extrinsic
    (fixed frame), uppercase axes are intrinsic (body frame).

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
    Get a rotation matrix, given the euler axes and angles. Follows the same convention as
    ``scipy.spatial.transform.Rotation.from_euler``: lowercase axes are extrinsic
    (fixed frame), uppercase axes are intrinsic (body frame).

    Args:
        euler_axes (str): Euler axes, e.g. 'y', 'zx' or 'xyz'
        angles (List): Angles for each euler axes.
        degrees (bool): Angles are in degree.

    Returns: rotation matrix, shape: (3,3)

    """
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    if degrees:
        angles = np.deg2rad(angles)

    intrinsic = euler_axes[0].isupper()
    axes = euler_axes.lower()

    matrices = [_elementary_rotation(ax, ang) for ax, ang in zip(axes, angles)]

    if intrinsic:
        # Intrinsic: R = R_1 @ R_2 @ ... @ R_n
        m_rot = matrices[0]
        for m in matrices[1:]:
            m_rot = m_rot @ m
    else:
        # Extrinsic: R = R_n @ ... @ R_2 @ R_1
        m_rot = matrices[-1]
        for m in reversed(matrices[:-1]):
            m_rot = m_rot @ m

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
