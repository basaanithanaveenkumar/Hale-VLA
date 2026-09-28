"""A task decides how a model is called, what the loss is and what counts as a prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from lifting.data import SceneBatch


class Task(ABC):
    """Strategy object used by :class:`Trainer` (keeps the trainer model-agnostic)."""

    num_classes: int

    @abstractmethod
    def forward(self, model: nn.Module, batch: SceneBatch) -> dict:
        """Run the model on a batch."""

    @abstractmethod
    def loss(self, outputs: dict, batch: SceneBatch) -> Tensor:
        """Scalar training loss."""

    @abstractmethod
    def predict(self, outputs: dict) -> Tensor:
        """Integer class predictions."""

    @abstractmethod
    def target(self, batch: SceneBatch) -> Tensor:
        """Integer ground truth aligned with :meth:`predict`."""
