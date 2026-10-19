"""Forward-time conditional flow matching, in the convention of the GR00T N1 paper.

GR00T N1 (and FLARE) write the noised chunk as ``A^τ = τ·A + (1 − τ)·ε``: **τ = 0 is pure noise and τ = 1 is data**, the
network regresses the velocity ``A − ε``, and sampling integrates forward from τ = 0 to τ = 1. This is the mirror image of
the openpi / π0 code convention used in :mod:`pi_family.flow_matching` (where τ = 1 is noise); the two are the same
method under ``τ ↔ 1 − τ`` and the notebooks say so wherever it matters.

``p(τ) = Beta((s − τ) / s; 1.5, 1)`` with ``s = 0.999`` puts most training time on the noisy end (small τ).
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class ForwardFlow:
    """Time sampling, interpolation, targets and Euler integration with τ = 0 as noise.

    Args:
        noise_scale: the paper's ``s``; τ is drawn in ``[0, s]`` so it is never exactly 1.
        beta_a: first shape of the Beta distribution of ``u = (s − τ)/s`` (the second is fixed to 1).
    """

    def __init__(self, noise_scale: float = 0.999, beta_a: float = 1.5):
        self.s = noise_scale
        self.beta_a = beta_a

    def sample_time(
        self, n: int, device=None, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """τ = s · (1 − u), ``u ~ Beta(a, 1)``. Beta(a, 1) has CDF ``x^a`` so ``u = U^(1/a)``."""
        u = torch.rand(n, generator=generator).pow(1.0 / self.beta_a)
        t = self.s * (1.0 - u)
        return t.to(device) if device is not None else t

    @staticmethod
    def interpolate(actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``A^τ = τ A + (1 − τ) ε``; ``t`` is (B,) and broadcasts over the trailing dims."""
        while t.dim() < actions.dim():
            t = t.unsqueeze(-1)
        return t * actions + (1.0 - t) * noise

    @staticmethod
    def target_velocity(actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """``dA^τ/dτ = A − ε`` (constant along the straight path)."""
        return actions - noise

    @staticmethod
    def integrate(
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        num_steps: int,
        return_trajectory: bool = False,
    ):
        """Forward Euler from τ = 0 to 1: ``A ← A + (1/K) · V(A, τ)`` (GR00T uses K = 4)."""
        x = noise
        trajectory = [x]
        for k in range(num_steps):
            t = torch.full((noise.shape[0],), k / num_steps, device=noise.device)
            x = x + velocity_fn(x, t) / num_steps
            trajectory.append(x)
        return (x, trajectory) if return_trajectory else x
