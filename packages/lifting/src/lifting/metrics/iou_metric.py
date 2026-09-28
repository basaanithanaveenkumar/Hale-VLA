"""Confusion-matrix based intersection-over-union."""

from __future__ import annotations

import torch
from torch import Tensor


class IoUMetric:
    """Accumulates a confusion matrix over batches and reports per-class IoU.

    Args:
        num_classes: number of classes including background (class ``0``).
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def reset(self) -> None:
        self.confusion.zero_()

    def update(self, prediction: Tensor, target: Tensor) -> None:
        """Add integer ``prediction`` / ``target`` tensors of identical shape."""
        k = self.num_classes
        idx = target.flatten().cpu() * k + prediction.flatten().cpu()
        self.confusion += torch.bincount(idx, minlength=k * k).view(k, k)

    def per_class(self) -> Tensor:
        """``(K,)`` IoU of each class (NaN if the class never appears)."""
        tp = self.confusion.diag().float()
        fp = self.confusion.sum(0).float() - tp
        fn = self.confusion.sum(1).float() - tp
        denom = tp + fp + fn
        return torch.where(denom > 0, tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))

    def mean_foreground(self) -> float:
        """Mean IoU over the non-background classes that were present."""
        iou = self.per_class()[1:]
        valid = ~iou.isnan()
        return float(iou[valid].mean()) if valid.any() else 0.0
