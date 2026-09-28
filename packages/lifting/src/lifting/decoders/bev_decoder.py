"""Small U-Net refining lifted BEV features (Simple-BEV's ``Decoder``, compact)."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from lifting.layers import ConvNormAct, ResidualBlock


class BEVDecoder(nn.Module):
    """Two down-sampling stages and two skip-connected up-sampling stages.

    Args:
        in_channels: lifted BEV channels.
        channels: base width (output channels) of the decoder.
    """

    def __init__(self, in_channels: int, channels: int | None = None) -> None:
        super().__init__()
        c = channels or in_channels
        self.out_channels = c
        self.enc0 = ResidualBlock(in_channels, c)
        self.enc1 = ResidualBlock(c, 2 * c, stride=2)
        self.enc2 = ResidualBlock(2 * c, 4 * c, stride=2)
        self.up1 = ConvNormAct(4 * c, 2 * c, 3)
        self.up0 = ConvNormAct(2 * c, c, 3)
        self.fuse1 = ResidualBlock(2 * c, 2 * c)
        self.fuse0 = ResidualBlock(c, c)

    @staticmethod
    def _upsample_to(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, bev: Tensor) -> Tensor:
        s0 = self.enc0(bev)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        x = self.fuse1(self.up1(self._upsample_to(s2, s1)) + s1)
        return self.fuse0(self.up0(self._upsample_to(x, s0)) + s0)
