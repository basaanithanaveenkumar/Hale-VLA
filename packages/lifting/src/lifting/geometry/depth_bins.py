"""Uniform depth discretisation used by depth-based lifters (LSS, depth-warp, polar rays)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DepthBins:
    """``num_bins`` depth planes whose centres are spread uniformly over ``[min_depth, max_depth]``.

    Bin ``i`` is centred at ``min_depth + (i + 0.5) * step``.
    """

    min_depth: float = 1.0
    max_depth: float = 25.0
    num_bins: int = 24

    @property
    def step(self) -> float:
        return (self.max_depth - self.min_depth) / self.num_bins

    def centers(self, device: torch.device | str | None = None) -> Tensor:
        return self.min_depth + (torch.arange(self.num_bins, device=device) + 0.5) * self.step

    def to_normalized(self, depth: Tensor) -> Tensor:
        """Depth -> ``[-1, 1]`` coordinate for ``grid_sample(align_corners=False)``."""
        return 2.0 * (depth - self.min_depth) / (self.max_depth - self.min_depth) - 1.0

    def contains(self, depth: Tensor) -> Tensor:
        return (depth >= self.min_depth) & (depth < self.max_depth)
