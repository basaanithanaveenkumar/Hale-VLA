"""Lightweight from-scratch CNN encoder (no pretrained weights, no extra dependencies)."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.encoders.base import ImageEncoder
from lifting.encoders.fpn import FPN
from lifting.layers import ConvNormAct, ResidualBlock


class SimpleConvEncoder(ImageEncoder):
    """Residual CNN + FPN producing features at strides ``4, 8, 16, ...``.

    Args:
        out_channels: channels of every pyramid level.
        num_levels: number of pyramid levels (1 - 3).
        width: base width of the backbone.
    """

    def __init__(self, out_channels: int = 64, num_levels: int = 2, width: int = 32) -> None:
        super().__init__()
        if not 1 <= num_levels <= 3:
            raise ValueError("num_levels must be in [1, 3]")
        self.out_channels = out_channels
        self.strides = tuple(4 * 2**i for i in range(num_levels))
        self.stem = nn.Sequential(ConvNormAct(3, width, 3, 2), ResidualBlock(width, width, 2))
        stage_channels = [width * 2**i for i in range(num_levels)]
        self.stages = nn.ModuleList()
        prev = width
        for i, ch in enumerate(stage_channels):
            self.stages.append(ResidualBlock(prev, ch, stride=1 if i == 0 else 2))
            prev = ch
        self.neck = FPN(stage_channels, out_channels)

    def extract(self, images: Tensor) -> list[Tensor]:
        x = self.stem(images)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return self.neck(feats)
