"""Collapse the height axis of a voxel volume into BEV channels (Simple-BEV style)."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.layers.conv_norm_act import ConvNormAct


class HeightCompressor(nn.Module):
    """``(B, C, X, Y, Z) -> (B, C * Z, X, Y) -> conv -> (B, out_channels, X, Y)``."""

    def __init__(self, in_channels: int, height: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvNormAct(in_channels * height, out_channels, 3),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, volume: Tensor) -> Tensor:
        b, c, x, y, z = volume.shape
        return self.conv(volume.permute(0, 1, 4, 2, 3).reshape(b, c * z, x, y))
