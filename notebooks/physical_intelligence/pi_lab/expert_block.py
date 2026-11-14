"""One transformer layer's weights for one "expert" (token group) of a mixture of transformers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pi_lab.ada_rms_norm import AdaRMSNorm
from pi_lab.rms_norm import RMSNorm


class ExpertBlock(nn.Module):
    """Pre-norm attention projections + gated-GELU MLP (Gemma style) for one token group.

    The attention itself is *not* computed here: :class:`MixtureOfTransformers` concatenates
    the queries/keys/values of all experts and runs one joint attention, so experts only
    differ in their weights. They can even differ in width (π0: 2048-wide VLM, 1024-wide
    action expert) because all project to the same ``heads × head_dim``.

    Args:
        width: residual width of this expert.
        heads: attention heads (shared across experts).
        head_dim: per-head size (shared across experts).
        mlp_ratio: MLP hidden size / width.
        cond_width: if set, norms are :class:`AdaRMSNorm` conditioned on a vector of this size.
    """

    def __init__(self, width: int, heads: int, head_dim: int, mlp_ratio: int = 4,
                 cond_width: int | None = None) -> None:
        super().__init__()
        norm = (lambda: AdaRMSNorm(width, cond_width)) if cond_width else (lambda: RMSNorm(width))
        self.attn_norm, self.mlp_norm = norm(), norm()
        inner = heads * head_dim
        self.q, self.k, self.v = (nn.Linear(width, inner, bias=False) for _ in range(3))
        self.o = nn.Linear(inner, width, bias=False)
        hidden = width * mlp_ratio
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.heads, self.head_dim = heads, head_dim

    def qkv(self, x: torch.Tensor, cond: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """``(B, L, width)`` → q, k, v each ``(B, heads, L, head_dim)``."""
        h = self.attn_norm(x, cond)
        B, L, _ = h.shape
        return tuple(p(h).view(B, L, self.heads, self.head_dim).transpose(1, 2)
                     for p in (self.q, self.k, self.v))

    def mlp(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """Residual gated-GELU MLP."""
        h = self.mlp_norm(x, cond)
        return x + self.down(F.gelu(self.gate(h), approximate="tanh") * self.up(h))
