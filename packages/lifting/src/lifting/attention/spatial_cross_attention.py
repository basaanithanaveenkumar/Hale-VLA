"""Spatial cross-attention from 3D queries into multi-camera image features."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.attention.pillar_deformable_attention import PillarDeformableAttention


class SpatialCrossAttention(nn.Module):
    """BEVFormer / TPVFormer image cross-attention, dense and CUDA-free.

    Every query attends into each camera where at least one of its anchors is visible;
    per-camera results are averaged over the cameras that see the query.
    """

    def __init__(
        self,
        embed_dims: int = 128,
        num_heads: int = 4,
        num_levels: int = 1,
        num_points: int = 8,
        num_anchors: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.deformable_attention = PillarDeformableAttention(
            embed_dims, num_heads, num_levels, num_points, num_anchors
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        image_values: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
        reference_points_cam: Tensor,
        visible: Tensor,
    ) -> Tensor:
        """
        Args:
            query: ``(B, Q, C)`` queries with positional encoding added.
            image_values: ``(B, N, S, C)`` flattened feature pyramid for each camera.
            spatial_shapes: ``L`` tuples ``(H_l, W_l)``.
            reference_points_cam: ``(B, N, Q, A, 2)`` normalised projected anchors.
            visible: ``(B, N, Q, A)`` anchor visibility.

        Returns:
            ``(B, Q, C)``.
        """
        batch, cams = image_values.shape[:2]
        queries, channels = query.shape[1:]
        q = query[:, None].expand(batch, cams, queries, channels).flatten(0, 1)
        out = self.deformable_attention(
            q, image_values.flatten(0, 1), reference_points_cam.flatten(0, 1), spatial_shapes
        ).view(batch, cams, queries, channels)

        cam_mask = visible.any(-1).to(out.dtype)  # (B, N, Q)
        count = cam_mask.sum(1).clamp(min=1.0)
        fused = (out * cam_mask[..., None]).sum(1) / count[..., None]
        return self.dropout(self.output_proj(fused))
