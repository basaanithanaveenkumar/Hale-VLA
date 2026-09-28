"""Scatter ("splat") point features into a flattened grid - pure PyTorch.

Replaces the ``QuickCumsum`` / ``VoxelsSumming`` CUDA-style autograd functions of
Lift-Splat-Shoot with ``Tensor.index_add_``, which is differentiable and runs anywhere.
"""

from __future__ import annotations

import torch
from torch import Tensor


def voxel_pooling(
    features: Tensor,
    cell_index: Tensor,
    valid: Tensor,
    num_cells: int,
    reduce: str = "sum",
) -> Tensor:
    """Pool point features into ``num_cells`` grid cells.

    Args:
        features: ``(B, P, C)`` per-point features.
        cell_index: ``(B, P)`` long tensor with the flat cell of every point.
        valid: ``(B, P)`` bool mask; invalid points are dropped.
        num_cells: size of the flattened output grid.
        reduce: ``"sum"`` or ``"mean"`` (mean over points that landed in the cell).

    Returns:
        ``(B, num_cells, C)`` pooled grid.
    """
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
    batch, _, channels = features.shape
    offsets = (torch.arange(batch, device=cell_index.device) * num_cells)[:, None]
    flat_index = (cell_index + offsets)[valid]
    flat_feats = features[valid]

    out = features.new_zeros(batch * num_cells, channels)
    out.index_add_(0, flat_index, flat_feats)
    if reduce == "mean":
        count = features.new_zeros(batch * num_cells)
        count.index_add_(0, flat_index, features.new_ones(flat_index.shape[0]))
        out = out / count.clamp(min=1.0)[:, None]
    return out.view(batch, num_cells, channels)
