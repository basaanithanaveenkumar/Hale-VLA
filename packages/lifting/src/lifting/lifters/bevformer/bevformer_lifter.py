"""BEVFormer spatial encoder (``simple_bev/nets/bevformernet.py`` / ``bevformernet2.py``)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.lifters.bevformer.bevformer_layer import BEVFormerLayer
from lifting.lifters.pyramid import flatten_pyramid


@LIFTERS.register("bevformer")
class BEVFormerLifter(BaseLifter):
    """Learned BEV queries refined by deformable attention into the cameras.

    Each BEV query owns ``num_anchors`` 3D points along its pillar; they are projected
    into every camera and serve as reference points of the spatial cross-attention
    (Li et al., *BEVFormer*, 2022 - single frame, no temporal attention, as in Simple-BEV).

    Args:
        grid, in_channels, out_channels: see :class:`BaseLifter`.
        num_layers: encoder depth.
        num_heads, num_points, num_anchors, self_attn_points: attention hyper-parameters.
        num_levels: how many (finest) pyramid levels the cross-attention reads -
            1 reproduces ``bevformernet``, >1 the multi-scale ``bevformernet2`` variant.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        num_levels: int = 1,
        num_points: int = 8,
        num_anchors: int = 4,
        self_attn_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.num_levels = num_levels
        self.num_anchors = num_anchors
        nx, ny = grid.bev_shape
        self.queries = nn.Parameter(0.1 * torch.randn(nx * ny, in_channels))
        self.query_pos = nn.Parameter(0.1 * torch.randn(nx * ny, in_channels))
        self.level_embed = nn.Parameter(torch.zeros(num_levels, in_channels))
        self.layers = nn.ModuleList(
            BEVFormerLayer(
                in_channels,
                num_heads,
                num_levels,
                num_points,
                num_anchors,
                self_attn_points,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1)

    def camera_reference_points(self, cameras: Cameras) -> tuple[Tensor, Tensor]:
        """Project pillar anchors: ``(B, N, X*Y, A, 2)`` in ``[0, 1]`` and visibility ``(B, N, X*Y, A)``."""
        b = cameras.batch_size
        anchors = self.grid.pillar_points(self.num_anchors, cameras.device).view(1, -1, 3)
        uv, _, valid = cameras.project(anchors.expand(b, -1, -1))
        n = cameras.num_cameras
        uv = cameras.normalize_uv(uv).view(b, n, -1, self.num_anchors, 2)
        return uv, valid.view(b, n, -1, self.num_anchors)

    def bev_reference_points(self, device: torch.device) -> Tensor:
        """``(X*Y, 2)`` BEV cell centres as normalised ``(col, row)`` = ``(y, x)``."""
        nx, ny = self.grid.bev_shape
        rows = (torch.arange(nx, device=device) + 0.5) / nx
        cols = (torch.arange(ny, device=device) + 0.5) / ny
        r, c = torch.meshgrid(rows, cols, indexing="ij")
        return torch.stack([c, r], -1).view(-1, 2)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        values, shapes = flatten_pyramid(features[: self.num_levels], self.level_embed)
        b = values.shape[0]
        ref_cam, visible = self.camera_reference_points(cameras)
        ref_bev = self.bev_reference_points(values.device)[None].expand(b, -1, -1)
        query = self.queries[None].expand(b, -1, -1)
        pos = self.query_pos[None].expand(b, -1, -1)
        for layer in self.layers:
            query = layer(
                query, pos, ref_bev, self.grid.bev_shape, values, shapes, ref_cam, visible
            )
        nx, ny = self.grid.bev_shape
        bev = query.transpose(1, 2).reshape(b, -1, nx, ny)
        return self.output_proj(bev)
