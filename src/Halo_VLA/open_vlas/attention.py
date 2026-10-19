"""Attention blocks shared by the SmolVLA action expert and the GR00T diffusion transformer.

One :class:`Attention` module does self-attention (``context=None``) or cross-attention (``context`` given) and can return
its attention weights, which the notebooks plot to show *where the action tokens look*.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias


def init_linear(module: nn.Module) -> None:
    """Truncated-normal (std 0.02) weights and zero biases, the ViT / transformer convention."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Attention(nn.Module):
    """Multi-head attention; self-attention when no ``context`` is given, cross-attention otherwise.

    Args:
        dim: width of the query tokens (and of the output).
        context_dim: width of the key/value tokens for cross-attention (defaults to ``dim``). The key and value projections are
            what adapts the *other* network's features to this one's width ("feature projectors" in SmolVLA).
        num_heads / head_dim: attention geometry.

    Masks are boolean, ``True`` = may attend, broadcastable to (B, heads, Lq, Lk).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, context_dim: int | None = None):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(context_dim or dim, inner, bias=False)
        self.v = nn.Linear(context_dim or dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        self.apply(init_linear)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ):
        ctx = x if context is None else context
        q, k, v = self._heads(self.q(x)), self._heads(self.k(ctx)), self._heads(self.v(ctx))
        if return_weights:
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = scores.softmax(-1)
            out = weights @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.o(out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))
        return (out, weights) if return_weights else out


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learned scale (Gemma / SmolLM convention: ``1 + weight``)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)


class SwiGLU(nn.Module):
    """Gated MLP: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.apply(init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def causal_mask(length: int, device=None) -> torch.Tensor:
    """Lower-triangular boolean mask (L, L): token i may attend tokens j ≤ i."""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()
