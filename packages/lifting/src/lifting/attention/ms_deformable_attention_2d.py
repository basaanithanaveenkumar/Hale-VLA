"""Multi-scale deformable attention (Deformable DETR) on 2D feature maps."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.attention.offset_init import init_sampling_offsets
from lifting.ops import multi_scale_deformable_attn_2d


class MSDeformableAttention2D(nn.Module):
    """Each query attends to ``num_points`` learned locations per head and per level.

    Args:
        embed_dims: channel width of queries and values.
        num_heads: attention heads.
        num_levels: number of feature levels in ``value``.
        num_points: sampling points per head per level.
    """

    def __init__(
        self, embed_dims: int = 128, num_heads: int = 4, num_levels: int = 1, num_points: int = 4
    ) -> None:
        super().__init__()
        if embed_dims % num_heads:
            raise ValueError("embed_dims must be divisible by num_heads")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_sampling_offsets(
            self.sampling_offsets, self.num_heads, self.num_levels, self.num_points
        )
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            query: ``(B, Q, C)`` (positional encoding already added).
            value: ``(B, S, C)`` flattened multi-level features.
            reference_points: ``(B, Q, 2)`` or ``(B, Q, L, 2)`` normalised ``(x, y)`` in ``[0, 1]``.
            spatial_shapes: ``L`` tuples ``(H_l, W_l)``.

        Returns:
            ``(B, Q, C)``.
        """
        batch, queries, _ = query.shape
        heads, levels, points = self.num_heads, self.num_levels, self.num_points
        v = self.value_proj(value).view(batch, value.shape[1], heads, -1)

        offsets = self.sampling_offsets(query).view(batch, queries, heads, levels, points, 2)
        weights = self.attention_weights(query).view(batch, queries, heads, levels * points)
        weights = weights.softmax(-1).view(batch, queries, heads, levels, points)

        if reference_points.dim() == 3:
            reference_points = reference_points[:, :, None].expand(-1, -1, levels, -1)
        normalizer = query.new_tensor([[w, h] for h, w in spatial_shapes])
        locations = (
            reference_points[:, :, None, :, None, :]
            + offsets / normalizer[None, None, None, :, None, :]
        )
        out = multi_scale_deformable_attn_2d(v, spatial_shapes, locations, weights)
        return self.output_proj(out)
