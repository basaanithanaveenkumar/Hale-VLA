"""RMSNorm, the normalisation used by Gemma (and therefore by every π model)."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """``x / rms(x) * (1 + w)`` with the Gemma convention of a zero-initialised scale offset.

    Args:
        width: feature size.
        eps: numerical floor.
    """

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        del cond  # same signature as AdaRMSNorm
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * (1 + self.weight)
