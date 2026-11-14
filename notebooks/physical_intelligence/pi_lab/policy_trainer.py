"""A minimal AdamW training loop shared by the notebooks."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn


class PolicyTrainer:
    """Optimises ``loss_fn(batch)`` with AdamW, cosine decay and gradient clipping.

    Args:
        model: the module whose parameters are optimised.
        lr: peak learning rate.
        weight_decay: AdamW weight decay.
        clip: gradient-norm clip.
    """

    def __init__(self, model: nn.Module, lr: float = 3e-3, weight_decay: float = 1e-4,
                 clip: float = 1.0) -> None:
        self.model = model
        self.lr, self.weight_decay, self.clip = lr, weight_decay, clip

    def fit(self, loss_fn: Callable[[dict], torch.Tensor], make_batch: Callable[[], dict],
            steps: int, log_every: int = 0) -> list[float]:
        """Train for ``steps`` steps and return the loss history."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps, pct_start=0.05)
        losses, start = [], time.time()
        self.model.train()
        for step in range(1, steps + 1):
            loss = loss_fn(make_batch())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.clip)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            if log_every and step % log_every == 0:
                print(f"step {step:>5}  loss {sum(losses[-log_every:]) / log_every:.4f}  "
                      f"({time.time() - start:.0f}s)")
        self.model.eval()
        return losses
