"""3D semantic occupancy task (TPVFormer)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lifting.data import SceneBatch
from lifting.training.task import Task


class OccupancyTask(Task):
    """Weighted cross-entropy on ``(B, K, X, Y, Z)`` voxel logits against ``batch.occupancy``."""

    def __init__(self, num_classes: int = 3, class_weights: Sequence[float] | None = None) -> None:
        self.num_classes = num_classes
        weights = class_weights or [1.0] + [8.0] * (num_classes - 1)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, model: nn.Module, batch: SceneBatch) -> dict:
        return model(batch.images, batch.cameras)

    def loss(self, outputs: dict, batch: SceneBatch) -> Tensor:
        logits = outputs["voxel_logits"]
        return F.cross_entropy(logits, batch.occupancy, weight=self.class_weights.to(logits.device))

    def predict(self, outputs: dict) -> Tensor:
        return outputs["voxel_logits"].argmax(1)

    def target(self, batch: SceneBatch) -> Tensor:
        return batch.occupancy
