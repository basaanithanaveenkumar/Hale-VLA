"""One TPVFormer encoder layer: cross-view hybrid attention -> image cross-attention -> FFN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lifting.attention import CrossViewHybridAttention, SpatialCrossAttention
from lifting.layers import FeedForward


class TPVFormerLayer(nn.Module):
    """Post-norm layer over the concatenated tokens of the three TPV planes.

    Each plane has its own image cross-attention (own offsets and anchor count), mirroring
    TPVFormer's per-plane ``sampling_offsets``.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 4,
        num_levels: int = 1,
        num_anchors: tuple[int, int, int] = (4, 8, 8),
        points_per_anchor: int = 2,
        hybrid_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hybrid_attn = CrossViewHybridAttention(embed_dims, num_heads, hybrid_points, dropout)
        self.cross_attn = nn.ModuleList(
            SpatialCrossAttention(
                embed_dims, num_heads, num_levels, a * points_per_anchor, a, dropout
            )
            for a in num_anchors
        )
        self.ffn = FeedForward(embed_dims, 2 * embed_dims, dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dims) for _ in range(3))

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        plane_shapes: Sequence[tuple[int, int]],
        hybrid_reference: Tensor,
        image_values: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
        camera_reference: Sequence[tuple[Tensor, Tensor]],
    ) -> Tensor:
        q = self.norms[0](
            query + self.hybrid_attn(query, query_pos, plane_shapes, hybrid_reference)
        )
        sizes = [h * w for h, w in plane_shapes]
        cross = [
            attn(qp + pp, image_values, spatial_shapes, ref, vis)
            for attn, qp, pp, (ref, vis) in zip(
                self.cross_attn,
                q.split(sizes, 1),
                query_pos.split(sizes, 1),
                camera_reference,
                strict=True,
            )
        ]
        q = self.norms[1](q + torch.cat(cross, dim=1))
        return self.norms[2](q + self.ffn(q))
