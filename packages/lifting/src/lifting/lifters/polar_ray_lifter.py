"""Translating Images into Maps (``simple_bev/nets/tiimnet.py``; Saha et al., ICRA 2022)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.layers import ConvNormAct
from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.ops import masked_camera_mean


@LIFTERS.register("tiim")
class PolarRayLifter(BaseLifter):
    """Every image column is translated into a polar ray of depth cells by a transformer.

    1. *Image -> polar*: a column encoder self-attends over the ``H`` pixels of a column;
       ``D`` learned depth queries cross-attend to it, producing ``(C, D, W)`` polar rays.
    2. *Polar -> Cartesian*: each BEV cell is projected into every camera and the polar
       map is resampled at its ``(column, depth)``.

    Args:
        grid, in_channels, out_channels: see :class:`BaseLifter`.
        depth_bins: radial discretisation of the rays.
        num_layers: transformer layers in the encoder and in the decoder.
        num_heads: attention heads.
        level: feature pyramid level to translate.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        depth_bins: DepthBins | None = None,
        num_layers: int = 1,
        num_heads: int = 4,
        level: int = 0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.depth_bins = depth_bins or DepthBins()
        self.level = level
        c = in_channels
        self.row_embed = nn.Sequential(nn.Linear(1, c), nn.ReLU(), nn.Linear(c, c))
        self.depth_queries = nn.Parameter(0.1 * torch.randn(self.depth_bins.num_bins, c))
        enc = nn.TransformerEncoderLayer(c, num_heads, 2 * c, dropout=0.0, batch_first=True)
        dec = nn.TransformerDecoderLayer(c, num_heads, 2 * c, dropout=0.0, batch_first=True)
        self.column_encoder = nn.TransformerEncoder(enc, num_layers, enable_nested_tensor=False)
        self.ray_decoder = nn.TransformerDecoder(dec, num_layers)
        self.output = nn.Sequential(
            ConvNormAct(c, out_channels, 3), nn.Conv2d(out_channels, out_channels, 1)
        )

    def image_to_polar(self, features: Tensor) -> Tensor:
        """``(B, N, C, H, W)`` -> polar rays ``(B, N, C, D, W)``."""
        b, n, c, h, w = features.shape
        columns = features.permute(0, 1, 4, 3, 2).reshape(b * n * w, h, c)
        rows = ((torch.arange(h, device=features.device) + 0.5) / h)[:, None]
        memory = self.column_encoder(columns + self.row_embed(rows)[None])
        queries = self.depth_queries[None].expand(b * n * w, -1, -1)
        rays = self.ray_decoder(queries, memory)  # (BNW, D, C)
        return rays.view(b, n, w, -1, c).permute(0, 1, 4, 3, 2)

    def polar_to_bev(self, polar: Tensor, cameras: Cameras) -> Tensor:
        """Resample ``(B, N, C, D, W)`` polar rays onto the BEV grid -> ``(B, C, X, Y)``."""
        b, n, c, d, w = polar.shape
        nx, ny, _ = self.grid.resolution
        centers = self.grid.voxel_centers(polar.device)[:, :, 0]  # X Y 3
        centers = centers.clone()
        centers[..., 2] = 0.5 * sum(self.grid.z_bounds)
        uv, depth, _ = cameras.project(centers.view(1, -1, 3).expand(b, -1, -1))
        u_norm = 2.0 * uv[..., 0] / cameras.image_size[1] - 1.0
        d_norm = self.depth_bins.to_normalized(depth)
        valid = (depth > 0) & (u_norm.abs() < 1.0) & self.depth_bins.contains(depth)
        grid = torch.stack([u_norm, d_norm], -1).reshape(b * n, 1, -1, 2)
        sampled = F.grid_sample(
            polar.reshape(b * n, c, d, w),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(b, n, c, -1)
        return masked_camera_mean(sampled, valid).view(b, c, nx, ny)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        polar = self.image_to_polar(features[self.level])
        return self.output(self.polar_to_bev(polar, cameras))
