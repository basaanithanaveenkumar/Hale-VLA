"""Bilinear sampling of per-camera feature maps at projected points."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def sample_multi_camera(features: Tensor, uv_normalized: Tensor) -> Tensor:
    """Sample ``(B, N, C, H, W)`` features at ``(B, N, P, 2)`` points in ``[0, 1]``.

    Returns:
        ``(B, N, C, P)``; points outside the image read zeros.
    """
    b, n, c, h, w = features.shape
    points = uv_normalized.shape[2]
    grid = (2.0 * uv_normalized - 1.0).reshape(b * n, 1, points, 2)
    out = F.grid_sample(
        features.reshape(b * n, c, h, w),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return out.view(b, n, c, points)


def masked_camera_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Average ``(B, N, C, P)`` over cameras where ``valid`` ``(B, N, P)`` holds."""
    mask = valid.to(values.dtype)[:, :, None]
    return (values * mask).sum(1) / mask.sum(1).clamp(min=1.0)
