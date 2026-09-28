"""Pure-PyTorch multi-scale deformable attention over 3D (volumetric) feature maps.

The volumetric analogue of :func:`multi_scale_deformable_attn_2d`: each query
samples ``P`` points per head and per level with trilinear interpolation from
multi-scale voxel volumes. Useful for voxel self-attention in occupancy networks.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def multi_scale_deformable_attn_3d(
    value: Tensor,
    volume_shapes: Sequence[tuple[int, int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Aggregate trilinearly sampled values with learned attention weights.

    Args:
        value: ``(B, S, M, D)`` with ``S = sum(D_l * H_l * W_l)``; each level is stored
            in ``(D_l, H_l, W_l)`` row-major order.
        volume_shapes: ``L`` tuples ``(D_l, H_l, W_l)``.
        sampling_locations: ``(B, Q, M, L, P, 3)`` normalised ``(x, y, z)`` in ``[0, 1]``
            addressing ``(W, H, D)`` respectively.
        attention_weights: ``(B, Q, M, L, P)``.

    Returns:
        ``(B, Q, M * D)`` attended features.
    """
    batch, _, heads, dim = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    if len(volume_shapes) != levels:
        raise ValueError(f"expected {levels} volume shapes, got {len(volume_shapes)}")

    split_sizes = [int(d) * int(h) * int(w) for d, h, w in volume_shapes]
    value_levels = value.split(split_sizes, dim=1)
    grids = 2.0 * sampling_locations - 1.0

    sampled = []
    for level, (d, h, w) in enumerate(volume_shapes):
        v = value_levels[level].flatten(2).transpose(1, 2)
        v = v.reshape(batch * heads, dim, int(d), int(h), int(w))
        # (B, Q, M, P, 3) -> (B*M, Q, P, 1, 3) : a 5D grid for grid_sample
        g = grids[:, :, :, level].transpose(1, 2).flatten(0, 1).unsqueeze(-2)
        s = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled.append(s.squeeze(-1))  # (B*M, D, Q, P)

    stacked = torch.stack(sampled, dim=-2).flatten(-2)
    weights = attention_weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    out = (stacked * weights).sum(-1).view(batch, heads * dim, queries)
    return out.transpose(1, 2).contiguous()
