"""Diffusion Policy (Chi et al., 2023): DDPM denoising of action chunks, with DDIM for fast inference.

DDPM defines a *fixed forward noising process* ``x_t = sqrt(ᾱ_t)·x_0 + sqrt(1 − ᾱ_t)·ε`` over ``T`` discrete steps (``T = 100``
in the paper) and trains a network to predict the noise ``ε``. Sampling reverses the chain **stochastically** (ancestral sampling:
every step adds fresh noise) and needs all ``T`` network evaluations. DDIM (Song et al.) shows the same trained network can be run
along a *deterministic* path over a strided subset of steps, which is how Diffusion Policy gets real-time inference (100 training
steps, 10 inference steps, 0.1 s on an RTX 3080).

:class:`DiffusionPolicy` reuses the action expert of :class:`open_vlas.SmolVLAPolicy` (same VLM features, same transformer) and
changes only the training target, the noise schedule and the sampler, so comparisons with flow matching isolate those choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from open_vlas.data import VLABatch
from open_vlas.smolvla import SmolVLAConfig, SmolVLAPolicy
from pi_family.policy import PiLosses


@dataclass
class DDPMSchedule:
    """Noise schedule of a ``T``-step DDPM chain.

    Attributes:
        steps: ``T``.
        kind: ``"cosine"`` (Nichol & Dhariwal) or ``"linear"`` (the original DDPM).
    """

    steps: int = 100
    kind: str = "cosine"

    def __post_init__(self) -> None:
        t = self.steps
        if self.kind == "linear":
            betas = torch.linspace(1e-4, 0.02 * 1000 / t, t)
        elif self.kind == "cosine":
            s = 0.008
            grid = torch.arange(t + 1, dtype=torch.float64) / t
            f = torch.cos((grid + s) / (1 + s) * math.pi / 2) ** 2
            betas = (1 - f[1:] / f[:-1]).clamp(max=0.999).float()
        else:
            raise ValueError(f"unknown schedule {self.kind!r}")
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)
        prev = torch.cat([torch.ones(1), self.alpha_bar[:-1]])
        self.posterior_var = betas * (1 - prev) / (1 - self.alpha_bar)  # β̃_t

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Noise ``x0`` to step ``t`` (0-based index into the schedule): ``sqrt(ᾱ)·x0 + sqrt(1−ᾱ)·ε``."""
        ab = self.alpha_bar.to(x0)[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


class DiffusionPolicy(SmolVLAPolicy):
    """A DDPM/DDIM action policy on top of the SmolVLA expert.

    Actions are scaled by ``1 / action_scale`` so they lie in ``[-1, 1]`` (as in Diffusion Policy) and the network predicts ``ε``.
    ``sample_actions(method="ddpm")`` runs the full ancestral chain (``T`` evaluations); ``method="ddim"`` runs ``num_steps``
    deterministic (``eta = 0``) or partly stochastic (``eta > 0``) steps.
    """

    def __init__(
        self,
        vlm,
        cfg: SmolVLAConfig,
        schedule: DDPMSchedule | None = None,
        action_scale: float = 1.5,
    ):
        super().__init__(vlm, cfg)
        self.schedule = schedule or DDPMSchedule()
        self.action_scale = action_scale

    def _eps(self, ctx, x: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        """ε-prediction; the denoising step is given to the network as the continuous time ``(t + 1) / T``."""
        return self.velocity(ctx, x, (t_index.float() + 1) / self.schedule.steps)

    def compute_loss(self, batch: VLABatch) -> PiLosses:
        """``‖ε_θ(x_t, t, o) − ε‖²`` with ``t`` uniform over the chain."""
        ctx = self.encode(batch)
        x0 = (
            batch.actions[:, :, : self.cfg.action_dim].to(self.action_position.device)
            / self.action_scale
        )
        k = self.cfg.flow_samples_per_example
        if k > 1:
            from open_vlas.smolvla import SmolVLAContext

            ctx = SmolVLAContext(
                [f.repeat_interleave(k, 0) for f in ctx.features],
                ctx.feature_mask.repeat_interleave(k, 0),
                ctx.state.repeat_interleave(k, 0),
            )
            x0 = x0.repeat_interleave(k, 0)
        t = torch.randint(0, self.schedule.steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = self._eps(ctx, self.schedule.q_sample(x0, t, noise), t)
        loss = (pred - noise).pow(2).mean()
        return PiLosses(loss, loss, loss.new_zeros(()), {"flow": loss.item()})

    @torch.no_grad()
    def sample_actions(
        self,
        batch: VLABatch,
        method: str = "ddim",
        num_steps: int = 10,
        eta: float = 0.0,
        noise: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ):
        """Denoise from Gaussian noise. ``num_steps`` is ignored by ``"ddpm"`` (it always uses all ``T`` steps)."""
        sch = self.schedule
        ctx = self.encode(batch)
        b, h, a = batch.size, self.cfg.chunk_size, self.cfg.action_dim
        dev = self.action_position.device
        x = noise.to(dev) if noise is not None else torch.randn(b, h, a, device=dev)
        ab = sch.alpha_bar.to(dev)
        if method == "ddpm":
            steps = list(range(sch.steps - 1, -1, -1))
        elif method == "ddim":
            steps = sorted(
                {int(round(v)) for v in torch.linspace(sch.steps - 1, 0, num_steps).tolist()},
                reverse=True,
            )
        else:
            raise ValueError("method must be 'ddpm' or 'ddim'")
        traj = [x * self.action_scale]
        for i, t in enumerate(steps):
            tt = torch.full((b,), t, device=dev)
            eps = self._eps(ctx, x, tt)
            if method == "ddpm":
                coef = sch.betas.to(dev)[t] / (1 - ab[t]).sqrt()
                mean = (x - coef * eps) / sch.alphas.to(dev)[t].sqrt()
                x = mean + (
                    sch.posterior_var.to(dev)[t].sqrt() * torch.randn_like(x) if t > 0 else 0.0
                )
            else:
                x0 = ((x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt()).clamp(-1.0, 1.0)
                ab_prev = ab[steps[i + 1]] if i + 1 < len(steps) else torch.tensor(1.0, device=dev)
                sigma = (
                    eta * ((1 - ab_prev) / (1 - ab[t]) * (1 - ab[t] / ab_prev)).clamp(min=0).sqrt()
                )
                dir_xt = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps
                x = ab_prev.sqrt() * x0 + dir_xt + (sigma * torch.randn_like(x) if eta > 0 else 0.0)
            traj.append(x * self.action_scale)
        out = self._pad(x * self.action_scale)
        return (out, [self._pad(v) for v in traj]) if return_trajectory else out

    def nfe(self, method: str, num_steps: int = 10) -> int:
        """Network evaluations per chunk."""
        return self.schedule.steps if method == "ddpm" else num_steps
