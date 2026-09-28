"""One BEVFormer encoder layer: BEV self-attention -> spatial cross-attention -> FFN."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.attention import MSDeformableAttention2D, SpatialCrossAttention
from lifting.layers import FeedForward


class BEVFormerLayer(nn.Module):
    """Post-norm transformer layer operating on ``X * Y`` BEV queries."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 4,
        num_levels: int = 1,
        num_points: int = 8,
        num_anchors: int = 4,
        self_attn_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = MSDeformableAttention2D(embed_dims, num_heads, 1, self_attn_points)
        self.cross_attn = SpatialCrossAttention(
            embed_dims, num_heads, num_levels, num_points, num_anchors, dropout
        )
        self.ffn = FeedForward(embed_dims, 2 * embed_dims, dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dims) for _ in range(3))

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        bev_reference: Tensor,
        bev_shape: tuple[int, int],
        image_values: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
        reference_points_cam: Tensor,
        visible: Tensor,
    ) -> Tensor:
        q = self.norms[0](
            query + self.self_attn(query + query_pos, query, bev_reference, [bev_shape])
        )
        q = self.norms[1](
            q
            + self.cross_attn(
                q + query_pos, image_values, spatial_shapes, reference_points_cam, visible
            )
        )
        return self.norms[2](q + self.ffn(q))
