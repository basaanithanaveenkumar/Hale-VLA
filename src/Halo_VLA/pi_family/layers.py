"""Transformer building blocks shared by the π-family policies.

The key idea is the **mixture-of-transformers** (π0 §IV, "action expert"): two sets of weights, one for the
backbone ("VLM") tokens and one for the action tokens, that meet only inside attention. Each token is projected to
queries / keys / values *by the weights of its own expert*, then all tokens attend each other jointly (subject to
the block-causal mask). Both experts use the same number of heads and head size, so their keys and values live in
the same space even though their residual widths differ.

Because no backbone token ever attends an action token, the backbone stack can be run once per observation and its
per-layer keys / values cached; every denoising step then only runs the (small) action expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias


def _init_linear(module: nn.Module) -> None:
    """ViT-style init: truncated normal (std 0.02) for weights, zeros for biases."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RMSNorm(nn.Module):
    """Root-mean-square norm (Gemma). ``forward`` returns ``(normed, gate)`` with ``gate = 1`` for API parity."""

    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(width))  # Gemma convention: scale = 1 + weight

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        return self.normalize(x), 1.0


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm (π0.5 / π0.7 action expert): ``norm(x) · (1 + scale(c)) + shift(c)`` plus a residual gate.

    ``cond`` is the flow-time embedding, either one per example (B, C) or one per token (B, L, C); per-token
    conditioning is what lets π0.7's training-time RTC give the already-committed actions τ = 0 while the rest are
    noisy. The modulation is zero-initialised (adaLN-Zero, as in DiT): at init every layer is the identity.
    """

    def __init__(self, width: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(cond_dim, 3 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        if cond is None:
            raise ValueError("AdaRMSNorm needs a conditioning vector")
        if cond.dim() == 2:
            cond = cond[:, None]
        scale, shift, gate = self.modulation(cond).chunk(3, dim=-1)
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * (1.0 + scale) + shift, gate


class IdentityNorm(nn.Module):
    """No-op with the norm API. Used as the action expert's *output* norm at toy widths (see ``Pi0Config``)."""

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        return x, 1.0


class SwiGLU(nn.Module):
    """Gated MLP used by Gemma: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ExpertLayer(nn.Module):
    """One layer of one expert: pre-norm attention projections + gated MLP, with its own weights.

    The layer is split in two halves so the caller can join the attention of several experts:
    :meth:`project` (norm → q, k, v) and :meth:`finish` (output projection, residual, MLP).
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        cond_dim: int | None = None,
    ):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        inner = num_heads * head_dim
        make_norm = (lambda: AdaRMSNorm(width, cond_dim)) if cond_dim else (lambda: RMSNorm(width))
        self.attn_norm, self.mlp_norm = make_norm(), make_norm()
        self.q = nn.Linear(width, inner, bias=False)
        self.k = nn.Linear(width, inner, bias=False)
        self.v = nn.Linear(width, inner, bias=False)
        self.o = nn.Linear(inner, width, bias=False)
        self.mlp = SwiGLU(width, int(width * mlp_ratio))
        for proj in (self.q, self.k, self.v, self.o):
            _init_linear(proj)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, heads, L, head_dim)

    def project(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        h, gate = self.attn_norm(x, cond)
        return self._heads(self.q(h)), self._heads(self.k(h)), self._heads(self.v(h)), gate

    def finish(
        self, x: torch.Tensor, attended: torch.Tensor, gate, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, length, _ = attended.shape
        x = x + gate * self.o(attended.transpose(1, 2).reshape(b, length, -1))
        h, gate2 = self.mlp_norm(x, cond)
        return x + gate2 * self.mlp(h)


@dataclass
class PrefixCache:
    """Per-layer backbone keys / values, computed once per observation and reused by every denoising step."""

    keys: list
    values: list

    def repeat(self, k: int) -> PrefixCache:
        return PrefixCache(
            [x.repeat_interleave(k, 0) for x in self.keys],
            [x.repeat_interleave(k, 0) for x in self.values],
        )

    def detached(self) -> PrefixCache:
        return PrefixCache([k.detach() for k in self.keys], [v.detach() for v in self.values])


class MixtureOfTransformers(nn.Module):
    """Backbone expert + action expert sharing attention (π0's architecture, used by all three generations).

    * :meth:`forward_prefix` runs the backbone tokens alone (they never see action tokens) and returns the final
      hidden states and a :class:`PrefixCache`.
    * :meth:`forward_suffix` runs the action-expert tokens; at layer ``l`` their queries attend
      ``[cached backbone keys of layer l, their own keys]``.

    ``insulate=True`` is π0.5-KI's *knowledge insulation*: the action expert reads the backbone's keys / values
    through ``detach()``, so the flow-matching loss produces no gradient in the backbone.
    """

    def __init__(
        self,
        width: int,
        expert_width: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        expert_cond_dim: int | None = None,
        expert_output_norm: bool = True,
    ):
        super().__init__()
        self.backbone = nn.ModuleList(
            ExpertLayer(width, num_heads, head_dim, mlp_ratio) for _ in range(depth)
        )
        self.expert = nn.ModuleList(
            ExpertLayer(expert_width, num_heads, head_dim, mlp_ratio, expert_cond_dim)
            for _ in range(depth)
        )
        self.backbone_norm = RMSNorm(width)
        if not expert_output_norm:
            self.expert_norm = IdentityNorm()
        elif expert_cond_dim:
            self.expert_norm = AdaRMSNorm(expert_width, expert_cond_dim)
        else:
            self.expert_norm = RMSNorm(expert_width)

    def forward_prefix(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, PrefixCache]:
        """``x`` (B, Lp, width), ``mask`` (B, Lp, Lp) bool → final hidden (B, Lp, width) and the KV cache."""
        keys, values = [], []
        for layer in self.backbone:
            q, k, v, gate = layer.project(x)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
            x = layer.finish(x, attended, gate)
            keys.append(k)
            values.append(v)
        return self.backbone_norm(x)[0], PrefixCache(keys, values)

    def forward_suffix(
        self,
        y: torch.Tensor,
        cache: PrefixCache,
        mask: torch.Tensor,
        cond: torch.Tensor | None = None,
        insulate: bool = False,
    ) -> torch.Tensor:
        """``y`` (B, Ls, expert_width), ``mask`` (B, Ls, Lp + Ls) → final expert hidden (B, Ls, expert_width)."""
        if insulate:
            cache = cache.detached()
        for layer, k_prefix, v_prefix in zip(self.expert, cache.keys, cache.values):
            q, k, v, gate = layer.project(y, cond)
            attended = F.scaled_dot_product_attention(
                q,
                torch.cat([k_prefix, k], dim=2),
                torch.cat([v_prefix, v], dim=2),
                attn_mask=mask[:, None],
            )
            y = layer.finish(y, attended, gate, cond)
        return self.expert_norm(y, cond)[0]

    def forward_joint(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        prefix_mask: torch.Tensor,
        suffix_mask: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference implementation: both streams layer by layer in one pass (no cache).

        Used by the notebooks and tests to check that the cached two-pass computation is exact.
        """
        for back, exp in zip(self.backbone, self.expert):
            qx, kx, vx, gx = back.project(x)
            qy, ky, vy, gy = exp.project(y, cond)
            q, k, v = torch.cat([qx, qy], 2), torch.cat([kx, ky], 2), torch.cat([vx, vy], 2)
            lp, ls = x.shape[1], y.shape[1]
            top = torch.cat(
                [prefix_mask, torch.zeros(x.shape[0], lp, ls, dtype=torch.bool, device=x.device)], 2
            )
            full = torch.cat([top, suffix_mask], 1)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=full[:, None])
            x = back.finish(x, attended[:, :, :lp], gx)
            y = exp.finish(y, attended[:, :, lp:], gy, cond)
        return self.backbone_norm(x)[0], self.expert_norm(y, cond)[0]
