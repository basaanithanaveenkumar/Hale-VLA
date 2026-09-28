"""TPVFormer encoder (Huang et al., CVPR 2023) in pure PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.layers import HeightCompressor
from lifting.lifters.base import LIFTERS, BaseLifter
from lifting.lifters.pyramid import flatten_pyramid
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes
from lifting.lifters.tpvformer.tpv_reference_points import TPVReferencePoints
from lifting.lifters.tpvformer.tpvformer_layer import TPVFormerLayer


@LIFTERS.register("tpvformer")
class TPVFormerLifter(BaseLifter):
    """Lifts images to three orthogonal planes with learned plane queries.

    * :meth:`lift_planes` returns :class:`TPVPlanes` - use with :class:`TPVAggregator` for
      3D semantic occupancy (see :class:`lifting.models.TPVFormer`).
    * :meth:`forward` fuses the planes into voxels and compresses height, so TPVFormer can
      be benchmarked as a BEV lifter like every other method.

    Args:
        grid, in_channels, out_channels: see :class:`BaseLifter`.
        num_layers: encoder depth.
        num_heads: attention heads.
        num_levels: pyramid levels read by the image cross-attention.
        num_anchors: anchors per query for the ``xy``, ``xz`` and ``yz`` planes.
        points_per_anchor: deformable sampling points per anchor.
        hybrid_points: sampling points of the cross-view hybrid attention.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        num_levels: int = 1,
        num_anchors: tuple[int, int, int] = (4, 8, 8),
        points_per_anchor: int = 2,
        hybrid_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.num_levels = num_levels
        self.references = TPVReferencePoints(grid, tuple(num_anchors))
        tokens = sum(h * w for h, w in self.references.plane_shapes)
        self.queries = nn.Parameter(0.1 * torch.randn(tokens, in_channels))
        self.query_pos = nn.Parameter(0.1 * torch.randn(tokens, in_channels))
        self.level_embed = nn.Parameter(torch.zeros(num_levels, in_channels))
        self.layers = nn.ModuleList(
            TPVFormerLayer(
                in_channels,
                num_heads,
                num_levels,
                tuple(num_anchors),
                points_per_anchor,
                hybrid_points,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.compressor = HeightCompressor(in_channels, grid.resolution[2], out_channels)

    def lift_planes(self, features: Sequence[Tensor], cameras: Cameras) -> TPVPlanes:
        values, shapes = flatten_pyramid(features[: self.num_levels], self.level_embed)
        b = values.shape[0]
        camera_reference = self.references.camera_reference_points(cameras)
        hybrid_reference = self.references.hybrid_reference_points(values.device)[None].expand(
            b, -1, -1, -1
        )
        query = self.queries[None].expand(b, -1, -1)
        pos = self.query_pos[None].expand(b, -1, -1)
        for layer in self.layers:
            query = layer(
                query,
                pos,
                self.references.plane_shapes,
                hybrid_reference,
                values,
                shapes,
                camera_reference,
            )
        return TPVPlanes.from_tokens(query, self.grid.resolution)

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        return self.compressor(self.lift_planes(features, cameras).to_voxels())
