"""A set of oriented 3D boxes resting on the ground plane."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SceneObjects:
    """``K`` yaw-rotated boxes standing on ``z = 0``.

    Attributes:
        centers: ``(K, 2)`` ego-frame ``(x, y)`` of the box footprints.
        sizes: ``(K, 3)`` ``(length, width, height)``.
        yaws: ``(K,)`` heading in radians.
        labels: ``(K,)`` class ids (``>= 1``).
        colors: ``(K, 3)`` RGB.
    """

    centers: Tensor
    sizes: Tensor
    yaws: Tensor
    labels: Tensor
    colors: Tensor

    def __len__(self) -> int:
        return self.labels.shape[0]

    def to_local(self, points_xy: Tensor) -> Tensor:
        """Express ``(..., 2)`` ego ``(x, y)`` in every box frame -> ``(..., K, 2)``."""
        delta = points_xy[..., None, :] - self.centers
        cos, sin = self.yaws.cos(), self.yaws.sin()
        x = cos * delta[..., 0] + sin * delta[..., 1]
        y = -sin * delta[..., 0] + cos * delta[..., 1]
        return torch.stack([x, y], -1)

    def footprint_mask(self, points_xy: Tensor) -> Tensor:
        """``(..., K)`` whether each ``(x, y)`` lies inside each footprint."""
        local = self.to_local(points_xy)
        half = self.sizes[:, :2] / 2
        return (local.abs() <= half).all(-1)
