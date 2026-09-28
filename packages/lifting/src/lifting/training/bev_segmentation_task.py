"""BEV semantic segmentation task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lifting.data import SceneBatch
from lifting.training.task import Task


class BEVSegmentationTask(Task):
    """Weighted cross-entropy on ``(B, K, X, Y)`` logits against ``batch.bev_labels``.

    Args:
        num_classes: classes including background.
        class_weights: per-class CE weights (objects are rare - up-weight them).
    """

    def __init__(self, num_classes: int = 3, class_weights: Sequence[float] | None = None) -> None:
        self.num_classes = num_classes
        weights = class_weights or [1.0] + [5.0] * (num_classes - 1)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, model: nn.Module, batch: SceneBatch) -> dict:
        return model(batch.images, batch.cameras)

    def loss(self, outputs: dict, batch: SceneBatch) -> Tensor:
        logits = outputs["logits"]
        return F.cross_entropy(
            logits, batch.bev_labels, weight=self.class_weights.to(logits.device)
        )

    def predict(self, outputs: dict) -> Tensor:
        return outputs["logits"].argmax(1)

    def target(self, batch: SceneBatch) -> Tensor:
        return batch.bev_labels
