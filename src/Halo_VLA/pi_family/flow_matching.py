"""Conditional flow matching for action chunks, in the convention used by π0 / openpi.

Notation (τ is the *flow time*, not the robot time step):

* ``x_τ = τ · ε + (1 - τ) · a``  - a straight line from the data ``a`` (τ = 0) to Gaussian noise ``ε`` (τ = 1)
* ``u = ε - a``                   - the velocity of that line, which the network ``v_θ(x_τ, τ | o)`` regresses
* sampling integrates ``dx/dτ = v_θ`` from τ = 1 back to τ = 0 with ``num_steps`` Euler steps.

The π0 paper writes the same thing with τ flipped (τ = 1 is data); openpi's code uses the convention above.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def sinusoidal_time_embedding(
    t: torch.Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0
) -> torch.Tensor:
    """Embed a flow time τ ∈ [0, 1] with sines/cosines of geometrically spaced periods (openpi's choice).

    ``t`` may have any shape; the output has shape ``t.shape + (dim,)``. The short periods resolve small differences
    in τ near the end of sampling, the long ones give a smooth, coarse signal.
    """
    if dim % 2:
        raise ValueError("time embedding dim must be even")
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    angle = t.float().unsqueeze(-1) * (2.0 * math.pi / period)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class FlowMatching:
    """Time sampling, interpolation, targets and Euler integration for rectified-flow action generation."""

    def __init__(self, beta_a: float = 1.5, beta_b: float = 1.0, time_max: float = 0.999):
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.time_max = time_max

    # ---------------------------------------------------------------------------------------------- training
    def sample_time(
        self, batch_size: int, generator: torch.Generator | None = None, device=None
    ) -> torch.Tensor:
        """τ = time_max · Beta(a, b) + (1 - time_max). Beta(1.5, 1) puts more mass near τ = 1 (high noise).

        Rationale (π0 §IV): with a strong observation the clean end is easy, so training time is better spent on the
        noisy part where the network must decide *which* mode of the action distribution to go to.
        """
        if self.beta_b == 1.0:
            # Beta(a, 1) has CDF x^a, so inverse-CDF sampling is x = u^(1/a); this also honours ``generator``.
            sample = torch.rand(batch_size, generator=generator).pow(1.0 / self.beta_a)
        else:
            beta = torch.distributions.Beta(torch.tensor(self.beta_a), torch.tensor(self.beta_b))
            sample = beta.sample((batch_size,))
        t = self.time_max * sample + (1.0 - self.time_max)
        return t.to(device) if device is not None else t

    @staticmethod
    def interpolate(actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_τ = τ ε + (1 - τ) a. ``t`` broadcasts over trailing dims ((B,), (B, H) or full shape)."""
        while t.dim() < actions.dim():
            t = t.unsqueeze(-1)
        return t * noise + (1.0 - t) * actions

    @staticmethod
    def target_velocity(actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """u = dx_τ/dτ = ε - a (constant along the straight path)."""
        return noise - actions

    # ---------------------------------------------------------------------------------------------- sampling
    @staticmethod
    def integrate(
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        num_steps: int,
        return_trajectory: bool = False,
    ):
        """Euler-integrate from τ = 1 to τ = 0: ``x ← x - (1/N) · v(x, τ)``.

        ``velocity_fn(x, t)`` receives ``t`` with shape (B,). With ``return_trajectory`` the list of all intermediate
        ``x`` (N + 1 tensors, starting with the noise) is returned too, which the notebooks plot.
        """
        x = noise
        dt = 1.0 / num_steps
        trajectory = [x]
        for step in range(num_steps):
            t = torch.full((noise.shape[0],), 1.0 - step * dt, device=noise.device)
            x = x - dt * velocity_fn(x, t)
            trajectory.append(x)
        return (x, trajectory) if return_trajectory else x
