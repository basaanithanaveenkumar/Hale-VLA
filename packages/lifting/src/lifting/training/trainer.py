"""Minimal, task-agnostic training loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch
from torch import nn

from lifting.data import SceneBatch
from lifting.metrics import IoUMetric
from lifting.training.task import Task

StepCallback = Callable[[int, float], None]


@dataclass
class TrainingHistory:
    """Loss per optimisation step."""

    losses: list[float] = field(default_factory=list)

    def smoothed(self, window: int = 10) -> list[float]:
        out = []
        for i in range(len(self.losses)):
            chunk = self.losses[max(0, i - window + 1) : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out


class Trainer:
    """Trains any model through a :class:`Task` strategy.

    Args:
        model: network to optimise.
        task: defines forward / loss / predictions.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: max gradient norm (``None`` disables clipping).
        device: where to train.
    """

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        grad_clip: float | None = 5.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.task = task
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = TrainingHistory()

    def train_step(self, batch: SceneBatch) -> float:
        self.model.train()
        batch = batch.to(self.device)
        loss = self.task.loss(self.task.forward(self.model, batch), batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        value = float(loss.detach())
        self.history.losses.append(value)
        return value

    def fit(
        self,
        batches: Iterable[SceneBatch],
        steps: int,
        callbacks: Iterable[StepCallback] = (),
        schedule: bool = True,
    ) -> TrainingHistory:
        """Run ``steps`` optimisation steps, cycling over ``batches`` if needed.

        ``schedule`` enables a cosine learning-rate decay over the ``steps``.
        """
        callbacks = list(callbacks)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=steps)
            if schedule
            else None
        )
        for step, batch in zip(range(steps), _cycle(batches), strict=False):
            loss = self.train_step(batch)
            if scheduler is not None:
                scheduler.step()
            for cb in callbacks:
                cb(step, loss)
        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        batches: Iterable[SceneBatch],
        transform: Callable[[SceneBatch], SceneBatch] | None = None,
    ) -> IoUMetric:
        """Accumulate IoU over ``batches`` (optionally transformed, e.g. corrupted cameras)."""
        self.model.eval()
        metric = IoUMetric(self.task.num_classes)
        for batch in batches:
            batch = batch.to(self.device)
            if transform is not None:
                batch = transform(batch)
            metric.update(
                self.task.predict(self.task.forward(self.model, batch)), self.task.target(batch)
            )
        return metric

    @torch.no_grad()
    def predict(self, batch: SceneBatch) -> torch.Tensor:
        self.model.eval()
        batch = batch.to(self.device)
        return self.task.predict(self.task.forward(self.model, batch))


def _cycle(iterable: Iterable[SceneBatch]) -> Iterator[SceneBatch]:
    while True:
        yielded = False
        for item in iterable:
            yielded = True
            yield item
        if not yielded:
            raise ValueError("cannot train on an empty iterable of batches")
