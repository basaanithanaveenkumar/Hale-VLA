"""Conditional flow matching for action chunks (openpi convention: t = 1 is pure noise)."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


class FlowMatching:
    """Train a velocity field from noise to actions, then integrate it with Euler steps.

    Interpolation (openpi's convention): ``x_t = t · ε + (1 − t) · a``, with ``t = 1`` pure noise
    and ``t = 0`` the clean action chunk. The regression target is the constant velocity
    ``u = ε − a``, the derivative of ``x_t`` with respect to ``t``. Sampling starts from ``ε`` at
    ``t = 1`` and takes ``N`` Euler steps of ``−1/N`` (π0, π0.5: 10 steps; π0.7: 5).

    Training times are drawn from ``0.999 · Beta(1.5, 1) + 0.001``, which puts more weight on
    **noisy** times. That is π0's choice: at low noise the observation barely matters, while the
    hard, observation-dependent decisions (which way around the obstacle) are made early in
    denoising.

    Args:
        beta_a: first Beta parameter.
        beta_b: second Beta parameter.
    """

    def __init__(self, beta_a: float = 1.5, beta_b: float = 1.0) -> None:
        self.beta = torch.distributions.Beta(beta_a, beta_b)

    def sample_time(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        """``n`` training times in ``(0.001, 1)`` on ``device``."""
        return (self.beta.sample((n,)) * 0.999 + 0.001).to(device)

    @staticmethod
    def interpolate(actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x_t = t·ε + (1−t)·a`` with ``t`` broadcast over the chunk."""
        t = t.view(-1, *([1] * (actions.dim() - 1)))
        return t * noise + (1 - t) * actions

    @staticmethod
    def target(actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """The velocity to regress, ``ε − a``."""
        return noise - actions

    @staticmethod
    def time_embedding(t: torch.Tensor, width: int, min_period: float = 4e-3,
                       max_period: float = 4.0) -> torch.Tensor:
        """Sine-cosine embedding of ``t ∈ [0, 1]`` (openpi's ``posemb_sincos`` periods)."""
        fraction = torch.linspace(0, 1, width // 2, device=t.device)
        period = min_period * (max_period / min_period) ** fraction
        angle = t[:, None] * (2 * math.pi / period)[None]
        return torch.cat([angle.sin(), angle.cos()], dim=-1)

    @torch.no_grad()
    def sample(self, velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               noise: torch.Tensor, steps: int = 10, return_path: bool = False):
        """Integrate ``dx/dt = v(x, t)`` from ``t = 1`` (noise) to ``t = 0`` (actions).

        Args:
            velocity: ``v(x_t, t) -> (B, H, D)``; closes over the cached observation.
            noise: ``(B, H, D)`` starting point.
            steps: Euler steps.
            return_path: also return every intermediate ``x_t`` (for animations).
        """
        x, dt = noise, -1.0 / steps
        path = [x.clone()]
        for i in range(steps):
            t = torch.full((x.shape[0],), 1.0 + i * dt, device=x.device)
            x = x + dt * velocity(x, t)
            path.append(x.clone())
        return (x, path) if return_path else x
