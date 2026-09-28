"""Transformer feed-forward block."""

from __future__ import annotations

from torch import Tensor, nn


class FeedForward(nn.Module):
    """``Linear -> GELU -> Dropout -> Linear -> Dropout``."""

    def __init__(
        self, embed_dims: int, hidden_dims: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or 2 * embed_dims
        self.net = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
