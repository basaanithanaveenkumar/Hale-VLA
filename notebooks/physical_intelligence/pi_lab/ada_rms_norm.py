"""Adaptive RMSNorm: the flow-matching timestep modulates the action expert's normalisation."""

from __future__ import annotations

import torch
from torch import nn


class AdaRMSNorm(nn.Module):
    """RMSNorm whose scale and shift are predicted from a conditioning vector (the timestep).

    π0 injected the flow time τ by concatenating its embedding to every noisy action token
    (an MLP "action-time mixer"). π0.5 and later inject it as **adaptive RMSNorm** in every
    action-expert layer, as DiT-style models do. The time signal then reaches every layer
    directly, instead of having to survive through the residual stream.

    ``y = rms_norm(x) * (1 + scale(c)) + shift(c)``; the projection is zero-initialised, so a
    freshly built layer behaves like plain RMSNorm.

    Args:
        width: feature size of ``x``.
        cond_width: size of the conditioning vector.
        eps: numerical floor.
    """

    def __init__(self, width: int, cond_width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_width, 2 * width)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)
        self.eps = eps

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is None:
            raise ValueError("AdaRMSNorm needs a conditioning vector")
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        scale, shift = self.to_scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + scale) + shift
