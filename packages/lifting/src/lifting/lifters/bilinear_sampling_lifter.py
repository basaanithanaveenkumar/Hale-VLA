"""Simple-BEV lifting (``simple_bev/nets/segnet.py``): parameter-free bilinear unprojection."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from lifting.geometry import Cameras, GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.ops import masked_camera_mean, sample_multi_camera


@LIFTERS.register("simple_bev")
class BilinearSamplingLifter(BaseLifter):
    """Project every voxel centre into every camera and bilinearly sample the features.

    Voxels seen by several cameras average their samples; the height axis is then
    folded into channels and compressed by a conv (Harley et al., *Simple-BEV*, 2023).

    Args:
        grid, in_channels, out_channels: see :class:`BaseLifter`.
        level: which feature pyramid level to sample.
    """

    def __init__(self, grid: GridSpec, in_channels: int, out_channels: int, level: int = 0) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.level = level
        self.compressor = HeightCompressor(in_channels, grid.resolution[2], out_channels)

    def unproject(self, features: Tensor, cameras: Cameras) -> Tensor:
        """``(B, N, C, H, W)`` image features -> ``(B, C, X, Y, Z)`` voxel features."""
        b, _, c = features.shape[:3]
        nx, ny, nz = self.grid.resolution
        points = self.grid.voxel_centers(features.device).view(1, -1, 3).expand(b, -1, -1)
        uv, _, valid = cameras.project(points)
        sampled = sample_multi_camera(features, cameras.normalize_uv(uv))
        return masked_camera_mean(sampled, valid).view(b, c, nx, ny, nz)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        return self.compressor(self.unproject(features[self.level], cameras))
