"""Deliberately wrong calibration, to prove a model actually relies on geometry."""

from __future__ import annotations

import math

import torch

from lifting.data import SceneBatch


class ExtrinsicCorruption:
    """Rotates every camera about the ego z-axis while images and labels stay unchanged.

    A lifter that truly uses the extrinsics will place objects at the wrong BEV location,
    so its IoU must collapse; a geometry-free model is unaffected.
    """

    def __init__(self, yaw_deg: float = 90.0) -> None:
        self.yaw_deg = yaw_deg

    def transform(self) -> torch.Tensor:
        a = math.radians(self.yaw_deg)
        t = torch.eye(4)
        t[0, 0], t[0, 1], t[1, 0], t[1, 1] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
        return t

    def __call__(self, batch: SceneBatch) -> SceneBatch:
        cams = batch.cameras.transformed(self.transform().to(batch.images.device))
        return SceneBatch(batch.images, cams, batch.bev_labels, batch.occupancy, batch.depth)
