"""A 2-D playground for comparing flow matching with DDPM / DDIM on a bimodal distribution.

Both methods are trained on the same data (two Gaussian blobs) with the same small MLP, so differences in the sampled paths come from
the *method*. Flow matching uses the GR00T convention (``t = 0`` noise, ``x_t = (1 − t)·x_0 + t·x_1``, target velocity ``x_1 − x_0``);
DDPM uses ε-prediction on a cosine schedule.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from vla_foundations.diffusion import DDPMSchedule


def bimodal_data(
    n: int, rng: np.random.Generator, separation: float = 2.0, spread: float = 0.25
) -> torch.Tensor:
    """Two Gaussian blobs at ``(±separation, 0)``."""
    centers = np.where(rng.random(n) < 0.5, -separation, separation)
    pts = np.stack([centers, np.zeros(n)], 1) + spread * rng.normal(size=(n, 2))
    return torch.from_numpy(pts.astype(np.float32))


class TimeMLP(nn.Module):
    """``f(x, t) → R²``: a small MLP on ``[x, t]`` used as the velocity / noise network."""

    def __init__(self, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t[:, None]], -1))


def train_flow_2d(data_fn, steps: int = 3000, batch: int = 256, seed: int = 0) -> TimeMLP:
    """Conditional flow matching: ``E‖v(x_t, t) − (x_1 − x_0)‖²`` with ``x_t = (1 − t)·x_0 + t·x_1``."""
    torch.manual_seed(seed)
    net = TimeMLP()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x1 = data_fn(batch)
        x0 = torch.randn_like(x1)
        t = torch.rand(batch)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = (net(xt, t) - (x1 - x0)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def train_ddpm_2d(
    data_fn, schedule: DDPMSchedule, steps: int = 3000, batch: int = 256, seed: int = 0
) -> TimeMLP:
    """DDPM ε-prediction: ``E‖ε_θ(x_t, t) − ε‖²``."""
    torch.manual_seed(seed)
    net = TimeMLP()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0 = data_fn(batch)
        t = torch.randint(0, schedule.steps, (batch,))
        eps = torch.randn_like(x0)
        loss = (
            (net(schedule.q_sample(x0, t, eps), (t.float() + 1) / schedule.steps) - eps)
            .pow(2)
            .mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def sample_flow_2d(net: TimeMLP, n: int, steps: int, seed: int = 0) -> list:
    """Forward Euler from noise: ``x ← x + v(x, t)/K``. Returns the ``steps + 1`` intermediate tensors."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    out = [x]
    for k in range(steps):
        x = x + net(x, torch.full((n,), k / steps)) / steps
        out.append(x)
    return out


@torch.no_grad()
def sample_ddpm_2d(net: TimeMLP, schedule: DDPMSchedule, n: int, seed: int = 0) -> list:
    """Full ancestral DDPM chain (``T`` network calls, fresh noise at every step)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    out = [x]
    for t in range(schedule.steps - 1, -1, -1):
        eps = net(x, torch.full((n,), (t + 1) / schedule.steps))
        mean = (x - schedule.betas[t] / (1 - schedule.alpha_bar[t]).sqrt() * eps) / schedule.alphas[
            t
        ].sqrt()
        x = mean + (
            schedule.posterior_var[t].sqrt() * torch.randn(n, 2, generator=g) if t > 0 else 0.0
        )
        out.append(x)
    return out


@torch.no_grad()
def sample_ddim_2d(
    net: TimeMLP, schedule: DDPMSchedule, n: int, steps: int, seed: int = 0, clip: float = 4.0
) -> list:
    """Deterministic DDIM (``η = 0``) over ``steps`` strided timesteps. The predicted clean sample is clipped to ``±clip``
    (standard practice: at the first step ᾱ ≈ 0 and ``x_0 = (x_t − √(1−ᾱ)·ε)/√ᾱ`` amplifies any error)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    ab = schedule.alpha_bar
    ts = sorted(
        {int(round(v)) for v in torch.linspace(schedule.steps - 1, 0, steps).tolist()}, reverse=True
    )
    out = [x]
    for i, t in enumerate(ts):
        eps = net(x, torch.full((n,), (t + 1) / schedule.steps))
        x0 = ((x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt()).clamp(-clip, clip)
        ab_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        out.append(x)
    return out


def path_straightness(trajectory: list) -> float:
    """Mean ratio ``(path length) / (distance from start to end)`` over the sampled points; 1 = perfectly straight."""
    pts = torch.stack(trajectory, 0)  # (K+1, N, 2)
    length = (pts[1:] - pts[:-1]).norm(dim=-1).sum(0)
    chord = (pts[-1] - pts[0]).norm(dim=-1).clamp(min=1e-6)
    return float((length / chord).mean())
