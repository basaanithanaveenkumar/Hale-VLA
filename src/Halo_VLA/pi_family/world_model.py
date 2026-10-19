"""π0.7's lightweight world model g_ψ: (current image, subtask text) → subgoal image, trained with flow matching.

The report initialises it from BAGEL (a 14B image generation / editing model) and trains with
``max_ψ E[L_CFM(g*_t, g_ψ(o_t, ℓ̂_t, m))]`` where g*_t is the frame at the end of the current segment. Its job is to
bring knowledge the robot data lacks (web, human video) into the policy *through pictures*.

Here it is a small DiT: patches of ``[noisy goal ‖ current image]`` plus the subtask words are processed jointly
by transformer layers whose RMSNorms are modulated by the flow time (adaRMSNorm, as in the action expert).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from pi_family.flow_matching import FlowMatching, sinusoidal_time_embedding
from pi_family.layers import ExpertLayer, RMSNorm, _init_linear
from pi_family.tokenizers import PiVocabulary


class SubgoalWorldModel(nn.Module):
    """Conditional rectified-flow image generator for 3×S×S subgoal images."""

    def __init__(
        self,
        vocab: PiVocabulary | None = None,
        image_size: int = 32,
        patch_size: int = 4,
        width: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        max_words: int = 8,
    ):
        super().__init__()
        self.vocab = vocab or PiVocabulary()
        self.image_size, self.patch_size, self.width = image_size, patch_size, width
        self.grid = image_size // patch_size
        patch_dim = 6 * patch_size * patch_size  # noisy goal (3) + observation (3)
        self.patch_in = nn.Linear(patch_dim, width)
        self.patch_position = nn.Parameter(torch.zeros(1, self.grid**2, width))
        self.word_embedding = nn.Embedding(len(self.vocab.words), width)
        self.word_position = nn.Parameter(torch.zeros(1, max_words, width))
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.layers = nn.ModuleList(
            ExpertLayer(width, num_heads, width // num_heads, 2.0, cond_dim=width)
            for _ in range(depth)
        )
        self.norm = RMSNorm(width)
        self.patch_out = nn.Linear(width, 3 * patch_size * patch_size)
        self.flow = FlowMatching()
        for p in (self.patch_position, self.word_position, self.word_embedding.weight):
            nn.init.trunc_normal_(p, std=0.02)
        for m in (self.patch_in, self.patch_out, *self.time_mlp):
            _init_linear(m)

    # ------------------------------------------------------------------------------------------ helpers
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, ch = x.shape[:2]
        p, g = self.patch_size, self.grid
        return x.view(b, ch, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(b, g * g, ch * p * p)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        p, g = self.patch_size, self.grid
        return tokens.view(b, g, g, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, g * p, g * p)

    def _words(self, texts: list, device) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.word_position.shape[1]
        ids = torch.zeros(len(texts), n, dtype=torch.long, device=device)
        valid = torch.zeros(len(texts), n, dtype=torch.bool, device=device)
        for i, t in enumerate(texts):
            enc = self.vocab.encode(t or "")[:n]
            ids[i, : len(enc)] = torch.tensor(enc, device=device)
            valid[i, : len(enc)] = True
        return self.word_embedding(ids) + self.word_position, valid

    def velocity(
        self, noisy_goal: torch.Tensor, observation: torch.Tensor, texts: list, t: torch.Tensor
    ) -> torch.Tensor:
        """v_ψ(g_τ, τ | o, ℓ̂) with images in [-1, 1] space."""
        x = (
            self.patch_in(self._patchify(torch.cat([noisy_goal, observation], 1)))
            + self.patch_position
        )
        words, word_valid = self._words(texts, x.device)
        seq = torch.cat([words, x], 1)
        valid = torch.cat(
            [word_valid, torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)], 1
        )
        mask = valid[:, None, None, :].expand(-1, 1, seq.shape[1], -1)
        cond = self.time_mlp(sinusoidal_time_embedding(t, self.width))
        for layer in self.layers:
            q, k, v, gate = layer.project(seq, cond)
            seq = layer.finish(
                seq, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), gate, cond
            )
        return self._unpatchify(self.patch_out(self.norm(seq)[0][:, words.shape[1] :]))

    # ------------------------------------------------------------------------------------------ API
    def compute_loss(
        self, observation: torch.Tensor, texts: list, goal: torch.Tensor
    ) -> torch.Tensor:
        """L_CFM on images mapped from [0, 1] to [-1, 1]."""
        obs, target = observation * 2 - 1, goal * 2 - 1
        noise = torch.randn_like(target)
        t = self.flow.sample_time(target.shape[0], device=target.device)
        pred = self.velocity(self.flow.interpolate(target, noise, t), obs, texts, t)
        return (pred - self.flow.target_velocity(target, noise)).pow(2).mean()

    @torch.no_grad()
    def sample(self, observation: torch.Tensor, texts: list, num_steps: int = 10) -> torch.Tensor:
        obs = observation * 2 - 1
        x = self.flow.integrate(
            lambda x, t: self.velocity(x, obs, texts, t), torch.randn_like(obs), num_steps
        )
        return ((x + 1) / 2).clamp(0, 1)
