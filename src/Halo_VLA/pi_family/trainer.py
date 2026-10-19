"""A minimal training loop shared by the π-family notebooks (AdamW, warm-up + cosine, gradient clipping)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrainHistory:
    losses: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    seconds: float = 0.0

    def smoothed(self, window: int = 25) -> np.ndarray:
        x = np.asarray(self.losses, dtype=float)
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    def series(self, key: str) -> np.ndarray:
        return np.asarray([m.get(key, np.nan) for m in self.metrics], dtype=float)


class PiTrainer:
    """Trains any object with ``compute_loss(batch) -> PiLosses`` on batches from ``dataset.sample``."""

    def __init__(
        self,
        model,
        dataset,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        warmup: int = 50,
        grad_clip: float = 1.0,
        seed: int = 0,
    ):
        self.model, self.dataset = model, dataset
        self.batch_size, self.warmup, self.grad_clip = batch_size, warmup, grad_clip
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.base_lr = lr
        self.rng = np.random.default_rng(seed)
        self.history = TrainHistory()

    def _lr(self, step: int, total: int) -> float:
        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup
        progress = (step - self.warmup) / max(1, total - self.warmup)
        return self.base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    def fit(self, steps: int, log_every: int = 0) -> TrainHistory:
        self.model.train()
        start = time.time()
        for step in range(steps):
            for group in self.optimizer.param_groups:
                group["lr"] = self._lr(step, steps)
            losses = self.model.compute_loss(self.dataset.sample(self.batch_size, self.rng))
            self.optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.history.losses.append(losses.total.item())
            self.history.metrics.append(losses.metrics)
            if log_every and (step + 1) % log_every == 0:
                pretty = ", ".join(f"{k} {v:.3f}" for k, v in losses.metrics.items())
                print(f"step {step + 1:5d}  loss {losses.total.item():.4f}  ({pretty})")
        self.history.seconds += time.time() - start
        self.model.eval()
        return self.history
