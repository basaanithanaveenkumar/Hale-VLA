"""Per-cell classification head."""

from __future__ import annotations

from torch import nn

from lifting.layers import ConvNormAct


class SegmentationHead(nn.Sequential):
    """``conv3x3 -> BN -> ReLU -> conv1x1`` producing ``num_classes`` logits per BEV cell."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ConvNormAct(in_channels, in_channels, 3), nn.Conv2d(in_channels, num_classes, 1)
        )
