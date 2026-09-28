"""A collated mini-batch of synthetic scenes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from lifting.geometry import Cameras


@dataclass
class SceneBatch:
    """Everything a lifting model and its losses need.

    Attributes:
        images: ``(B, N, 3, H, W)`` in ``[0, 1]``.
        cameras: calibration.
        bev_labels: ``(B, X, Y)`` class ids.
        occupancy: ``(B, X, Y, Z)`` class ids.
        depth: ``(B, N, H, W)`` ground-truth depth.
    """

    images: Tensor
    cameras: Cameras
    bev_labels: Tensor
    occupancy: Tensor
    depth: Tensor

    def to(self, device: torch.device | str) -> SceneBatch:
        return SceneBatch(
            self.images.to(device),
            self.cameras.to(device),
            self.bev_labels.to(device),
            self.occupancy.to(device),
            self.depth.to(device),
        )

    @classmethod
    def collate(cls, samples: Sequence[dict]) -> SceneBatch:
        """``DataLoader`` ``collate_fn`` for :class:`SyntheticSceneDataset` samples."""

        def stack(key: str) -> Tensor:
            return torch.stack([s[key] for s in samples])

        cameras = Cameras(
            stack("intrinsics"), stack("cam_to_ego"), tuple(samples[0]["images"].shape[-2:])
        )
        return cls(
            stack("images"), cameras, stack("bev_labels"), stack("occupancy"), stack("depth")
        )
