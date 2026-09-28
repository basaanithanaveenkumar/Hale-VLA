"""Feature pyramid network neck."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn


class FPN(nn.Module):
    """Top-down pyramid mapping backbone stages to a common channel width."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.smooth = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        )

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        """``features`` are ordered fine -> coarse; the output keeps that order."""
        laterals = [conv(f) for conv, f in zip(self.lateral, features, strict=True)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False
            )
        return [conv(f) for conv, f in zip(self.smooth, laterals, strict=True)]
