"""TPVFormer cross-view hybrid attention: deformable self-attention across the 3 TPV planes."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.attention.ms_deformable_attention_2d import MSDeformableAttention2D


class CrossViewHybridAttention(nn.Module):
    """Treats the ``xy``, ``xz`` and ``yz`` planes as three levels of one deformable attention.

    Every query (on any plane) samples points on all three planes, so information flows
    between the top view and the two side views.
    """

    def __init__(
        self, embed_dims: int = 128, num_heads: int = 4, num_points: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attention = MSDeformableAttention2D(
            embed_dims, num_heads, num_levels=3, num_points=num_points
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        plane_shapes: Sequence[tuple[int, int]],
        reference_points: Tensor,
    ) -> Tensor:
        """
        Args:
            query: ``(B, Q, C)`` concatenated tokens of the three planes.
            query_pos: ``(B, Q, C)`` positional encodings.
            plane_shapes: three ``(H, W)`` plane shapes, in token order.
            reference_points: ``(B, Q, 3, 2)`` per-plane normalised reference points.

        Returns:
            ``(B, Q, C)``.
        """
        out = self.attention(query + query_pos, query, reference_points, plane_shapes)
        return self.dropout(out)
