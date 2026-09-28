"""Flatten a multi-camera feature pyramid into attention tokens."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def flatten_pyramid(
    features: Sequence[Tensor], level_embed: Tensor | None = None
) -> tuple[Tensor, list[tuple[int, int]]]:
    """``L`` maps ``(B, N, C, H_l, W_l)`` -> tokens ``(B, N, sum(H_l W_l), C)`` and their shapes."""
    tokens, shapes = [], []
    for level, feat in enumerate(features):
        h, w = feat.shape[-2:]
        t = feat.flatten(3).transpose(2, 3)  # B N HW C
        if level_embed is not None:
            t = t + level_embed[level]
        tokens.append(t)
        shapes.append((h, w))
    return torch.cat(tokens, dim=2), shapes
