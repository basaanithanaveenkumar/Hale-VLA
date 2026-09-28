"""Deformable attention with several 3D anchors per query (BEVFormer / TPVFormer).

BEVFormer and TPVFormer call this ``MSDeformableAttention3D``: every BEV/TPV query owns
``A`` 3D anchor points (spread along its pillar) that are projected into an image.
Each anchor receives ``P / A`` learned sampling offsets on every feature level.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.attention.offset_init import init_sampling_offsets
from lifting.ops import multi_scale_deformable_attn_2d


class PillarDeformableAttention(nn.Module):
    """Multi-anchor, multi-scale deformable attention into one camera's feature pyramid.

    Args:
        embed_dims: channel width.
        num_heads: attention heads.
        num_levels: image feature levels.
        num_points: total sampling points per head per level (multiple of ``num_anchors``).
        num_anchors: 3D anchors per query.
    """

    def __init__(
        self,
        embed_dims: int = 128,
        num_heads: int = 4,
        num_levels: int = 1,
        num_points: int = 8,
        num_anchors: int = 4,
    ) -> None:
        super().__init__()
        if num_points % num_anchors:
            raise ValueError("num_points must be a multiple of num_anchors")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_anchors = num_anchors
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_sampling_offsets(
            self.sampling_offsets, self.num_heads, self.num_levels, self.num_points
        )
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            query: ``(B, Q, C)``.
            value: ``(B, S, C)`` one camera's flattened feature pyramid.
            reference_points: ``(B, Q, A, 2)`` projected anchors, normalised to ``[0, 1]``.
            spatial_shapes: ``L`` tuples ``(H_l, W_l)``.

        Returns:
            ``(B, Q, C)`` (no output projection - the caller fuses cameras first).
        """
        batch, queries, _ = query.shape
        heads, levels, points, anchors = (
            self.num_heads,
            self.num_levels,
            self.num_points,
            self.num_anchors,
        )
        v = self.value_proj(value).view(batch, value.shape[1], heads, -1)

        offsets = self.sampling_offsets(query).view(batch, queries, heads, levels, points, 2)
        weights = self.attention_weights(query).view(batch, queries, heads, levels * points)
        weights = weights.softmax(-1).view(batch, queries, heads, levels, points)

        normalizer = query.new_tensor([[w, h] for h, w in spatial_shapes])
        offsets = offsets / normalizer[None, None, None, :, None, :]
        offsets = offsets.view(batch, queries, heads, levels, points // anchors, anchors, 2)
        locations = reference_points[:, :, None, None, None, :, :] + offsets
        locations = locations.view(batch, queries, heads, levels, points, 2)
        return multi_scale_deformable_attn_2d(v, spatial_shapes, locations, weights)
