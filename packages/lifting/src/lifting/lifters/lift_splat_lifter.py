"""Lift-Splat-Shoot (``simple_bev/nets/liftnet2.py``; Philion & Fidler, 2020)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.lifters.depth_head import DepthHead
from lifting.ops import voxel_pooling


@LIFTERS.register("lift_splat")
class LiftSplatLifter(BaseLifter):
    """Lift: outer product of per-pixel depth distribution and context.
    Splat: scatter-sum every frustum point into its voxel (pure-PyTorch voxel pooling).

    Args:
        grid, in_channels, out_channels: see :class:`BaseLifter`.
        depth_bins: depth discretisation of the frustum.
        context_channels: lifted feature width (defaults to ``in_channels``).
        level: feature pyramid level to lift.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        depth_bins: DepthBins | None = None,
        context_channels: int | None = None,
        level: int = 0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.depth_bins = depth_bins or DepthBins()
        self.level = level
        self.context_channels = context_channels or in_channels
        self.depth_head = DepthHead(in_channels, self.depth_bins.num_bins, self.context_channels)
        self.compressor = HeightCompressor(self.context_channels, grid.resolution[2], out_channels)

    def frustum_points(self, cameras: Cameras, height: int, width: int) -> Tensor:
        """Ego-frame ``(B, N, D*H*W, 3)`` points of the frustum grid (pixel centres x depth bins)."""
        device = cameras.device
        depths = self.depth_bins.centers(device)
        vs = torch.arange(height, device=device) + 0.5
        us = torch.arange(width, device=device) + 0.5
        d, v, u = torch.meshgrid(depths, vs, us, indexing="ij")
        b, n = cameras.batch_size, cameras.num_cameras
        uv = torch.stack([u, v], -1).view(1, 1, -1, 2).expand(b, n, -1, -1)
        return cameras.unproject(uv, d.reshape(1, 1, -1).expand(b, n, -1))

    def splat(self, depth: Tensor, context: Tensor, cameras: Cameras) -> Tensor:
        """Scatter lifted features into the voxel grid.

        Args:
            depth: ``(B, N, D, H, W)`` depth probabilities.
            context: ``(B, N, C, H, W)`` context features.
            cameras: cameras already scaled to ``(H, W)``.

        Returns:
            ``(B, C, X, Y, Z)`` voxel features.
        """
        b, n, c, h, w = context.shape
        lifted = depth[:, :, :, None] * context[:, :, None]  # B N D C H W
        lifted = lifted.permute(0, 1, 2, 4, 5, 3).reshape(b, -1, c)
        points = self.frustum_points(cameras, h, w).reshape(b, -1, 3)
        index, valid = self.grid.flat_voxel_index(points)
        nx, ny, nz = self.grid.resolution
        pooled = voxel_pooling(lifted, index, valid, nx * ny * nz, reduce="sum")
        return pooled.view(b, nx, ny, nz, c).permute(0, 4, 1, 2, 3)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        feats = features[self.level]
        b, n, _, h, w = feats.shape
        depth, context = self.depth_head(feats.flatten(0, 1))
        voxels = self.splat(
            depth.view(b, n, *depth.shape[1:]),
            context.view(b, n, *context.shape[1:]),
            cameras.scaled_to(h, w),
        )
        return self.compressor(voxels)
