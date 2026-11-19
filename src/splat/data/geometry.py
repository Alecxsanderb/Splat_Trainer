# HUMAN-WRITTEN (core) — see CLAUDE.md. Do not AI-generate content in this file.

import torch
import torch.nn.functional as F

# converts a quaternion to a rotation matrix
def quat_to_rotmat(q):
    """
    Convert a quaternion to a rotation matrix.
    Args:
        q: A tensor of shape (..., 4) representing the quaternion (qw, qx, qy, qz).
    Returns:
        A tensor of shape (..., 3, 3) representing the rotation matrix.
    """
    # Normalize the quaternion
    q = F.normalize(q, dim=-1)

    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute the rotation matrix elements
    # I got the formula for this from https://www.mathworks.com/help/nav/ref/quaternion.rotmat.html
    # specifically, I used the formula for a point rotation matrix, which allows us to 
    # convert world coordinates to camera coordinates, which is what we want for rendering a
    # scene from a camera's perspective.

    # 1st row
    r00 = 1 - 2 * (qy**2 + qz**2)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)

    # 2nd row
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx**2 + qz**2)
    r12 = 2 * (qy * qz - qx * qw)

    # 3rd row
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx**2 + qy**2)

    # Stack the elements into a rotation matrix
    rotmat = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    return rotmat

# computes the camera center from rotation and translation
def camera_center(R, t):
    """
    Compute the camera center from rotation matrix R and translation vector t.
    Args:
        R: A tensor of shape (..., 3, 3) representing the rotation matrix.
        t: A tensor of shape (..., 3) representing the translation vector.
    Returns:
        A 3-vector representing the camera center in world coordinates.
    """
    # Compute the camera center using the formula C = -R^T * t
    C = -torch.matmul(R.transpose(-2, -1), t.unsqueeze(-1)).squeeze(-1)
    return C