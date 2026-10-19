"""Latent actions: GR00T N1's way to learn from video that has no action labels.

Human egocentric video and generated "neural trajectories" come without robot actions. GR00T N1 trains a **VQ-VAE
inverse-dynamics model** on pairs of frames ``(x_t, x_{t+H})``:

* the **encoder** sees both frames and emits a small continuous vector ``z_t``;
* ``z_t`` is **quantised** to the nearest entry of a small codebook (the bottleneck that forces it to be a compact "what happened");
* the **decoder** sees only ``x_t`` and the quantised code and must reconstruct ``x_{t+H}``.

Because the future cannot be predicted from ``x_t`` alone (the robot could go to any object), the code has to carry what the motion
was. After training, the encoder is an inverse-dynamics model: the *pre-quantisation* embedding ``z_t`` is used as the "action" label
(a pseudo-embodiment called LAPA) with the usual flow-matching loss. Trained on every dataset together, the latent space is
shared across embodiments, which is what lets human video teach a robot.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from open_vlas.embodiments import LATENT_DIM


class LatentActionModel(nn.Module):
    """VQ-VAE over frame pairs. ``latent(x_t, x_tH)`` returns the standardised pre-quantisation embedding used as the action label.

    Args:
        image_size: side of the (square) frames.
        hidden: channel width.
        latent_dim: size of ``z`` (= the LAPA embodiment's action dimension).
        codebook_size: number of discrete codes.
        commitment: weight of the commitment loss.
    """

    def __init__(
        self,
        image_size: int = 32,
        hidden: int = 32,
        latent_dim: int = LATENT_DIM,
        codebook_size: int = 8,
        commitment: float = 0.25,
    ):
        super().__init__()
        self.latent_dim, self.commitment = latent_dim, commitment
        side = image_size // 4
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(hidden * side * side, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0, 1.0)
        self.decoder = nn.Sequential(
            nn.Conv2d(3 + latent_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))

    def encode(self, x_t: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        """Continuous pre-quantisation latent ``z_e`` (B, latent_dim)."""
        return self.encoder(torch.cat([x_t, x_future], 1))

    def quantize(self, z: torch.Tensor):
        """Nearest codebook entry with a straight-through gradient. Returns ``(z_q, code index)``."""
        d = torch.cdist(z, self.codebook.weight)
        idx = d.argmin(-1)
        zq = self.codebook(idx)
        return z + (zq - z).detach(), idx, zq

    def decode(self, x_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict the future frame from the current frame and the (quantised) latent: a residual on ``x_t``."""
        zmap = z[:, :, None, None].expand(-1, -1, x_t.shape[2], x_t.shape[3])
        return x_t + self.decoder(torch.cat([x_t, zmap], 1))

    def forward(self, x_t: torch.Tensor, x_future: torch.Tensor):
        """VQ-VAE loss: reconstruction + codebook + commitment. Returns ``(loss, metrics)``."""
        ze = self.encode(x_t, x_future)
        zq_st, idx, zq = self.quantize(ze)
        recon = self.decode(x_t, zq_st)
        # almost every pixel is unchanged between the two frames, so a plain MSE is dominated by the static background and the
        # code learns nothing; normalise by how much the frame really changed (1.0 = "predict no change at all")
        change = F.mse_loss(x_t, x_future).clamp(min=1e-5)
        rec = F.mse_loss(recon, x_future) / change
        codebook = F.mse_loss(zq, ze.detach())
        commit = F.mse_loss(ze, zq.detach())
        loss = rec + codebook + self.commitment * commit
        self._last_encodings = ze.detach()
        return loss, dict(recon=rec.item(), codes_used=int(idx.unique().numel()), idx=idx.detach())

    @torch.no_grad()
    def reset_dead_codes(self, usage: torch.Tensor) -> int:
        """Re-initialise codebook entries that were never selected to random recent encoder outputs (prevents codebook collapse)."""
        dead = (usage == 0).nonzero().flatten()
        if len(dead) == 0 or not hasattr(self, "_last_encodings"):
            return 0
        pick = torch.randint(0, self._last_encodings.shape[0], (len(dead),))
        self.codebook.weight[dead] = self._last_encodings[pick]
        return len(dead)

    @torch.no_grad()
    def latent(self, x_t: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        """The latent-action label: standardised pre-quantisation embedding (≈ zero mean, unit variance)."""
        return (self.encode(x_t, x_future) - self.mean) / self.std

    @torch.no_grad()
    def calibrate(self, x_t: torch.Tensor, x_future: torch.Tensor) -> None:
        """Fix the standardisation statistics from a batch of frame pairs."""
        ze = self.encode(x_t, x_future)
        self.mean.copy_(ze.mean(0))
        self.std.copy_(ze.std(0).clamp(min=1e-3))


@dataclass
class LatentTrainHistory:
    losses: list = field(default_factory=list)
    recon: list = field(default_factory=list)
    codes_used: list = field(default_factory=list)
    seconds: float = 0.0


def frame_pair_sampler(source, horizon: int) -> Callable:
    """``sample(n, rng) -> (x_t, x_{t+H}, extras)`` over an :class:`EmbodiedSource`'s episodes (no actions are read).

    ``extras`` carries the *true* robot displacement over the window (columns 0-1) and its end position (columns 2-3), kept only to check afterwards what the latent learned;
    the model itself never sees it.
    """

    def sample(n: int, rng: np.random.Generator):
        xs, fs, disp = [], [], []
        for _ in range(n):
            ep = source.episodes[rng.integers(len(source.episodes))]
            t = int(rng.integers(0, max(ep.active_steps, 1)))
            xs.append(source.frame(ep, t))
            fs.append(source.frame(ep, t + horizon))
            end = min(t + horizon, len(ep.robot) - 1)
            disp.append(
                np.concatenate([ep.robot[end] - ep.robot[t], ep.robot[end]])
            )  # displacement, end position
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(fs)), np.stack(disp)

    return sample


def train_latent_action_model(
    model: LatentActionModel,
    sampler: Callable,
    steps: int,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 0,
    log_every: int = 0,
) -> LatentTrainHistory:
    """Train the VQ-VAE on frame pairs, then calibrate the latent's scale."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist, start = LatentTrainHistory(), time.time()
    model.train()
    usage = torch.zeros(model.codebook.num_embeddings)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 50)
        x, f, _ = sampler(batch_size, rng)
        loss, info = model(x, f)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        usage += torch.bincount(info["idx"], minlength=usage.numel()).float()
        if (step + 1) % 50 == 0:
            model.reset_dead_codes(usage)
            usage.zero_()
        hist.losses.append(loss.item())
        hist.recon.append(info["recon"])
        hist.codes_used.append(info["codes_used"])
        if log_every and (step + 1) % log_every == 0:
            print(
                f"latent-action model step {step + 1:5d}  recon {info['recon']:.4f}  codes used {info['codes_used']}"
            )
    model.eval()
    x, f, _ = sampler(512, rng)
    model.calibrate(x, f)
    hist.seconds = time.time() - start
    return hist
