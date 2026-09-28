"""Tri-perspective-view (TPV) feature planes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TPVPlanes:
    """Three orthogonal feature planes describing a 3D scene (Huang et al., *TPVFormer*, 2023).

    Attributes:
        xy: ``(B, C, X, Y)`` top (BEV) plane.
        xz: ``(B, C, X, Z)`` side plane.
        yz: ``(B, C, Y, Z)`` front plane.
    """

    xy: Tensor
    xz: Tensor
    yz: Tensor

    @property
    def resolution(self) -> tuple[int, int, int]:
        return self.xy.shape[2], self.xy.shape[3], self.xz.shape[3]

    @staticmethod
    def plane_shapes(resolution: tuple[int, int, int]) -> list[tuple[int, int]]:
        nx, ny, nz = resolution
        return [(nx, ny), (nx, nz), (ny, nz)]

    @classmethod
    def from_tokens(cls, tokens: Tensor, resolution: tuple[int, int, int]) -> TPVPlanes:
        """Split ``(B, XY + XZ + YZ, C)`` tokens back into planes."""
        shapes = cls.plane_shapes(resolution)
        parts = tokens.split([h * w for h, w in shapes], dim=1)
        b, _, c = tokens.shape
        planes = [
            p.transpose(1, 2).reshape(b, c, h, w) for p, (h, w) in zip(parts, shapes, strict=True)
        ]
        return cls(*planes)

    def to_tokens(self) -> Tensor:
        return torch.cat([p.flatten(2).transpose(1, 2) for p in (self.xy, self.xz, self.yz)], dim=1)

    def upsample(self, scale: int) -> TPVPlanes:
        if scale == 1:
            return self
        up = [
            F.interpolate(p, scale_factor=scale, mode="bilinear", align_corners=False)
            for p in (self.xy, self.xz, self.yz)
        ]
        return TPVPlanes(*up)

    def to_voxels(self) -> Tensor:
        """Broadcast-sum the planes into a ``(B, C, X, Y, Z)`` volume: ``f = xy + xz + yz``."""
        return self.xy[..., None] + self.xz[:, :, :, None, :] + self.yz[:, :, None, :, :]

    def sample_points(self, points_normalized: Tensor) -> Tensor:
        """Features of arbitrary 3D points.

        Args:
            points_normalized: ``(B, P, 3)`` coordinates in ``[-1, 1]`` (see ``GridSpec.to_normalized``).

        Returns:
            ``(B, C, P)`` sum of the three bilinear plane samples.
        """
        x, y, z = points_normalized.unbind(-1)

        def sample(plane: Tensor, col: Tensor, row: Tensor) -> Tensor:
            grid = torch.stack([col, row], -1)[:, None]
            return F.grid_sample(plane, grid, mode="bilinear", align_corners=False).squeeze(2)

        return sample(self.xy, y, x) + sample(self.xz, z, x) + sample(self.yz, z, y)
