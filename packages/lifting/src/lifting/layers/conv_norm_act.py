"""Conv -> BatchNorm -> ReLU."""

from __future__ import annotations

from torch import nn


class ConvNormAct(nn.Sequential):
    """A 2D convolution followed by batch norm and ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
