"""Abstract image encoder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageEncoder(nn.Module, ABC):
    """Maps ``(B*N, 3, H, W)`` images in ``[0, 1]`` to a feature pyramid.

    Subclasses implement :meth:`extract` and declare ``out_channels`` and ``strides``.
    Every level of the pyramid has ``out_channels`` channels and is ordered fine -> coarse.
    """

    out_channels: int
    strides: tuple[int, ...]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def forward(self, images: Tensor) -> list[Tensor]:
        return self.extract((images - self.mean) / self.std)

    @abstractmethod
    def extract(self, images: Tensor) -> list[Tensor]:
        """Return ``num_levels`` feature maps for normalised images."""
