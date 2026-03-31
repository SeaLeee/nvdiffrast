"""
Camera utilities for multi-view rendering.
"""

import math
import torch
import numpy as np


class Camera:
    """Perspective camera with view/projection matrices."""

    def __init__(self, position, target, up=None, fov=60.0, aspect=1.0,
                 near=0.1, far=1000.0, device='cuda'):
        self.device = device
        self.position = torch.tensor(position, dtype=torch.float32, device=device)
        self.target = torch.tensor(target, dtype=torch.float32, device=device)
        self.up = torch.tensor(up or [0, 1, 0], dtype=torch.float32, device=device)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def view_matrix(self) -> torch.Tensor:
        """Build look-at view matrix [4, 4]."""
        forward = torch.nn.functional.normalize(self.target - self.position, dim=0)
        right = torch.nn.functional.normalize(torch.cross(forward, self.up), dim=0)
        up = torch.cross(right, forward)

        mat = torch.eye(4, dtype=torch.float32, device=self.device)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -forward
        mat[0, 3] = -torch.dot(right, self.position)
        mat[1, 3] = -torch.dot(up, self.position)
        mat[2, 3] = torch.dot(forward, self.position)
        return mat

    def projection_matrix(self) -> torch.Tensor:
        """Build perspective projection matrix [4, 4] (OpenGL convention)."""
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        n, fa = self.near, self.far

        mat = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        mat[0, 0] = f / self.aspect
        mat[1, 1] = f
        mat[2, 2] = (fa + n) / (n - fa)
        mat[2, 3] = (2 * fa * n) / (n - fa)
        mat[3, 2] = -1.0
        return mat

    def mvp_matrix(self, model: torch.Tensor = None) -> torch.Tensor:
        """Model-View-Projection matrix [4, 4]."""
        vp = self.projection_matrix() @ self.view_matrix()
        if model is not None:
            return vp @ model
        return vp


def create_orbit_cameras(num_views: int, distance: float = 3.0,
                         elevation: float = 20.0, target=None,
                         fov: float = 60.0, device: str = 'cuda') -> list:
    """
    Create cameras orbiting around a target point.

    Args:
        num_views:  Number of views evenly spaced around Y axis
        distance:   Distance from target
        elevation:  Elevation angle in degrees
        target:     Look-at target [3], default [0,0,0]
        fov:        Field of view in degrees

    Returns:
        List of Camera objects
    """
    if target is None:
        target = [0.0, 0.0, 0.0]

    cameras = []
    elev_rad = math.radians(elevation)
    y = distance * math.sin(elev_rad)
    r = distance * math.cos(elev_rad)

    for i in range(num_views):
        angle = 2.0 * math.pi * i / num_views
        x = r * math.cos(angle)
        z = r * math.sin(angle)
        cam = Camera(
            position=[x, y, z],
            target=target,
            fov=fov,
            device=device,
        )
        cameras.append(cam)
    return cameras


def transform_pos(mtx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Apply 4x4 matrix transform to positions.

    Args:
        mtx: [4, 4] transformation matrix
        pos: [..., 3] positions

    Returns:
        [..., 4] clip-space positions
    """
    posw = torch.nn.functional.pad(pos, (0, 1), value=1.0)  # [..., 4]
    return (posw @ mtx.t())
