"""Depth-weighted backward warping (``simple_bev/nets/liftnet.py``)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.lifters.depth_head import DepthHead
from lifting.ops import masked_camera_mean


@LIFTERS.register("depth_warp")
class DepthWarpLifter(BaseLifter):
    """Lift features into a camera frustum weighted by predicted depth, then *pull* them.

    For each voxel the frustum volume ``(C, D, H, W)`` of every camera is sampled
    trilinearly at the voxel's ``(u, v, depth)``: an inverse (gather) formulation of
    Lift-Splat that needs no scatter and leaves no holes.

    Args:
        grid, in_channels, out_channels: see :class:`BaseLifter`.
        depth_bins: depth discretisation of the frustum.
        context_channels: lifted feature width (defaults to ``in_channels``).
        temperature: depth softmax temperature.
        level: feature pyramid level to lift.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        depth_bins: DepthBins | None = None,
        context_channels: int | None = None,
        temperature: float = 1.0,
        level: int = 0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.depth_bins = depth_bins or DepthBins()
        self.level = level
        ctx = context_channels or in_channels
        self.depth_head = DepthHead(in_channels, self.depth_bins.num_bins, ctx, temperature)
        self.compressor = HeightCompressor(ctx, grid.resolution[2], out_channels)

    def frustum_volume(self, features: Tensor) -> Tensor:
        """``(B, N, C, H, W)`` -> depth-weighted frustum ``(B, N, C', D, H, W)``."""
        b, n = features.shape[:2]
        depth, context = self.depth_head(features.flatten(0, 1))
        volume = context[:, :, None] * depth[:, None]
        return volume.view(b, n, *volume.shape[1:])

    def warp(self, volume: Tensor, cameras: Cameras) -> Tensor:
        """Sample ``(B, N, C, D, H, W)`` frustums at every voxel -> ``(B, C, X, Y, Z)``."""
        b, n, c = volume.shape[:3]
        nx, ny, nz = self.grid.resolution
        points = self.grid.voxel_centers(volume.device).view(1, -1, 3).expand(b, -1, -1)
        uv, depth, valid = cameras.project(points)
        valid = valid & self.depth_bins.contains(depth)
        uv_n = 2.0 * cameras.normalize_uv(uv) - 1.0
        coords = torch.cat([uv_n, self.depth_bins.to_normalized(depth)[..., None]], dim=-1)
        grid = coords.reshape(b * n, 1, 1, -1, 3)
        sampled = F.grid_sample(
            volume.flatten(0, 1), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).view(b, n, c, -1)
        return masked_camera_mean(sampled, valid).view(b, c, nx, ny, nz)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        return self.compressor(self.warp(self.frustum_volume(features[self.level]), cameras))
