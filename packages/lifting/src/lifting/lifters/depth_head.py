"""Per-pixel categorical depth + context head shared by depth-based lifters."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.layers import ConvNormAct


class DepthHead(nn.Module):
    """Predicts a depth distribution over ``num_bins`` and ``context_channels`` features.

    Args:
        in_channels: input feature channels.
        num_bins: number of depth bins.
        context_channels: channels of the context feature that gets lifted.
        temperature: softmax temperature (Simple-BEV's liftnet uses 0.07 for sharp depth).
    """

    def __init__(
        self, in_channels: int, num_bins: int, context_channels: int, temperature: float = 1.0
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        self.net = nn.Sequential(
            ConvNormAct(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, num_bins + context_channels, 1),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """``(B, C, H, W)`` -> depth probabilities ``(B, D, H, W)`` and context ``(B, C', H, W)``."""
        out = self.net(features)
        depth = (out[:, : self.num_bins] / self.temperature).softmax(dim=1)
        return depth, out[:, self.num_bins :]
