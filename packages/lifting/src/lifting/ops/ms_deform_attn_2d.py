"""Pure-PyTorch multi-scale deformable attention over 2D feature maps.

Drop-in replacement for the CUDA kernel of Deformable DETR / BEVFormer / TPVFormer
(``MultiScaleDeformableAttnFunction``). It is built on ``F.grid_sample`` and runs on
any device, including CPU and MPS.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def multi_scale_deformable_attn_2d(
    value: Tensor,
    spatial_shapes: Sequence[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Aggregate bilinearly sampled values with learned attention weights.

    Args:
        value: ``(B, S, M, D)`` flattened multi-level values where
            ``S = sum(H_l * W_l)``, ``M`` heads and ``D`` channels per head.
        spatial_shapes: ``L`` tuples ``(H_l, W_l)`` describing how ``S`` is split.
        sampling_locations: ``(B, Q, M, L, P, 2)`` normalised ``(x, y)`` in ``[0, 1]``.
        attention_weights: ``(B, Q, M, L, P)``; typically softmax-normalised over ``L*P``.

    Returns:
        ``(B, Q, M * D)`` attended features.
    """
    batch, _, heads, dim = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    if len(spatial_shapes) != levels:
        raise ValueError(f"expected {levels} spatial shapes, got {len(spatial_shapes)}")

    split_sizes = [int(h) * int(w) for h, w in spatial_shapes]
    value_levels = value.split(split_sizes, dim=1)
    grids = 2.0 * sampling_locations - 1.0  # grid_sample expects [-1, 1]

    sampled = []
    for level, (h, w) in enumerate(spatial_shapes):
        # (B, H*W, M, D) -> (B*M, D, H, W)
        v = (
            value_levels[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, dim, int(h), int(w))
        )
        # (B, Q, M, P, 2) -> (B*M, Q, P, 2)
        g = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        )

    # (B*M, D, Q, L*P)
    stacked = torch.stack(sampled, dim=-2).flatten(-2)
    weights = attention_weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    out = (stacked * weights).sum(-1).view(batch, heads * dim, queries)
    return out.transpose(1, 2).contiguous()
