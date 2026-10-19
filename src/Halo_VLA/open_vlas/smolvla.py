"""SmolVLA (Shukor et al., 2025, arXiv:2506.01844) built from first principles.

SmolVLA = a small pretrained VLM that *perceives* + a small action expert that *acts*, trained with flow matching, with four
design choices that make it cheap:

1. **fewer visual tokens**: no image tiling, pixel shuffle → 64 tokens per frame (here the same idea: 16);
2. **layer skipping**: the action expert reads the VLM's features from layers ``1..N`` only, ``N = L/2``; the top half of the
   VLM is deleted;
3. **a thin, interleaved action expert**: hidden size ``0.75 x`` the VLM's, alternating **self-attention** blocks (causal,
   over the action tokens) and **cross-attention** blocks (to the VLM features of the matching layer);
4. **a frozen VLM**: only the expert, the state projection and the feature projectors are trained.

The state is projected to **one token inside the VLM's prefix** (a paper ablation: this beats giving it to the expert).
Time is mixed into each action token by the π0-style MLP ``W3·swish(W2·[W1·a^τ, φ(τ)])``. Convention: τ = 1 is noise
(the LeRobot / openpi code); the paper's equation writes ``A^τ = τA + (1 − τ)ε`` but its implementation follows π0.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from pi_family.flow_matching import FlowMatching
from pi_family.pi0 import ActionTimeMLP
from pi_family.policy import PiLosses

from open_vlas.attention import Attention, RMSNorm, SwiGLU, causal_mask, init_linear
from open_vlas.data import VLABatch
from open_vlas.flow import ForwardFlow
from open_vlas.vlm import ToyVLM, VLMConfig


@dataclass
class SmolVLAConfig:
    """SmolVLA hyper-parameters; published values in the comments."""

    vlm: VLMConfig = field(default_factory=VLMConfig)
    skip_to: int | None = (
        None  # N: use VLM layers 1..N (paper: N = L/2; the main model keeps 16 of 32)
    )
    expert_ratio: float = 0.75  # expert hidden size / VLM hidden size (paper: 0.75)
    layout: str = "interleaved"  # "interleaved" (SA, CA, SA, CA ...) | "cross" (all CA) | "joint" (all π0-style joint SA)
    self_attention: str = (
        "causal"  # "causal" (paper) | "bidirectional" (worse in the paper's ablation)
    )
    state_in: str = (
        "prefix"  # "prefix": a token in the VLM (paper) | "suffix": a token in the expert
    )
    objective: str = "flow"  # "flow" (paper) | "regression" (L1, the paper's ablation) | "mse" (L2 regression: predicts the mean)
    state_dim: int = 2
    action_dim: int = 2
    chunk_size: int = 16  # n (paper: 50)
    num_denoise_steps: int = 10  # paper: 10
    flow_samples_per_example: int = 4  # (ε, τ) draws that share one VLM pass
    freeze_vlm: bool = True  # SmolVLA freezes the VLM; False fine-tunes it (used by the language-backbone experiments)
    flow_convention: str = "pi0"  # "pi0": τ = 1 is noise (LeRobot / openpi) | "forward": t = 0 is noise, x_t = (1-t)·noise + t·action (GR00T)
    expert_heads: int = 4
    mlp_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.skip_to is None:
            self.skip_to = self.vlm.depth // 2
        if not 1 <= self.skip_to <= self.vlm.depth:
            raise ValueError("skip_to must be in [1, vlm.depth]")
        for name, allowed in (
            ("layout", ("interleaved", "cross", "joint")),
            ("self_attention", ("causal", "bidirectional")),
            ("state_in", ("prefix", "suffix")),
            ("objective", ("flow", "regression", "mse")),
            ("flow_convention", ("pi0", "forward")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")

    @property
    def expert_width(self) -> int:
        heads = self.expert_heads
        return max(heads, int(round(self.vlm.width * self.expert_ratio / heads)) * heads)

    def block_kinds(self) -> list:
        """Kind of each expert layer: ``"self"``, ``"cross"`` or ``"joint"`` (LeRobot: even layers self-attend, odd layers cross-attend)."""
        n = self.skip_to
        if self.layout == "cross":
            return ["cross"] * n
        if self.layout == "joint":
            return ["joint"] * n
        return ["self" if i % 2 == 0 else "cross" for i in range(n)]


class ExpertBlock(nn.Module):
    """One action-expert layer: pre-norm attention + gated MLP, both residual. ``kind`` selects the attention:

    * ``"cross"``: queries from the action tokens, keys / values from the VLM's hidden state *of the matching layer*; the ``k`` / ``v``
      projections are the "feature projectors" that map the VLM's width to the expert's;
    * ``"self"``: the action tokens attend each other only (causal in SmolVLA) and never see the VLM in this block;
    * ``"joint"``: π0-style - the action tokens attend the concatenation ``[VLM features ; action tokens]`` (the paper's "SA" baseline
      against which interleaving is compared).
    """

    def __init__(self, kind: str, width: int, vlm_width: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.kind = kind
        self.attn_norm, self.mlp_norm = RMSNorm(width), RMSNorm(width)
        self.attn = Attention(
            width, heads, width // heads, context_dim=vlm_width if kind == "cross" else None
        )
        self.feature_proj = nn.Linear(vlm_width, width) if kind == "joint" else None
        if self.feature_proj is not None:
            init_linear(self.feature_proj)
        self.mlp = SwiGLU(width, int(width * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        features: torch.Tensor | None,
        feature_mask: torch.Tensor | None,
        self_mask: torch.Tensor | None,
        return_weights: bool = False,
    ):
        h = self.attn_norm(x)
        if self.kind == "cross":
            res = self.attn(h, features, feature_mask, return_weights)
        elif self.kind == "joint":
            ctx = torch.cat([self.feature_proj(features), h], 1)
            b, lq = h.shape[0], h.shape[1]
            fmask = feature_mask.expand(b, 1, lq, -1)
            amask = (
                self_mask.expand(b, 1, lq, lq)
                if self_mask is not None
                else torch.ones(b, 1, lq, lq, dtype=torch.bool, device=h.device)
            )
            res = self.attn(h, ctx, torch.cat([fmask, amask], -1), return_weights)
        else:
            res = self.attn(h, None, self_mask, return_weights)
        out, weights = res if return_weights else (res, None)
        x = x + out
        x = x + self.mlp(self.mlp_norm(x))
        return (x, weights) if return_weights else x


@dataclass
class SmolVLAContext:
    """Everything the expert needs from one VLM pass (computed once per observation, reused by every denoising step)."""

    features: list  # per kept layer: (B, L, vlm_width)
    feature_mask: torch.Tensor  # (B, 1, 1, L) bool: which VLM tokens are real
    state: torch.Tensor  # (B, state_dim), kept for the "suffix" variant
    segments: list = field(default_factory=list)
    num_image_tokens: int = 0


class SmolVLAPolicy(nn.Module):
    """SmolVLA at toy scale: frozen, truncated VLM + interleaved cross/self-attention flow-matching action expert.

    Args:
        vlm: a *pre-trained* :class:`ToyVLM`; it is deep-copied, truncated to ``cfg.skip_to`` layers and frozen.
        cfg: model configuration.
    """

    def __init__(self, vlm: ToyVLM, cfg: SmolVLAConfig):
        super().__init__()
        self.cfg = cfg
        self.vlm = copy.deepcopy(vlm).truncate(cfg.skip_to)
        if cfg.freeze_vlm:
            self.vlm.freeze(vision=True, language=True)
        w, ew = cfg.vlm.width, cfg.expert_width
        self.state_proj = nn.Linear(cfg.state_dim, w if cfg.state_in == "prefix" else ew)
        self.action_time = ActionTimeMLP(cfg.action_dim, ew)
        self.action_position = nn.Parameter(torch.zeros(1, cfg.chunk_size, ew))
        self.blocks = nn.ModuleList(
            ExpertBlock(kind, ew, w, cfg.expert_heads, cfg.mlp_ratio) for kind in cfg.block_kinds()
        )
        self.out_norm = RMSNorm(ew)
        self.action_out = nn.Linear(ew, cfg.action_dim)
        nn.init.trunc_normal_(self.action_position, std=0.02)
        init_linear(self.state_proj)
        init_linear(self.action_out)
        self.flow = FlowMatching()
        self.forward_flow = ForwardFlow()

    # ---------------------------------------------------------------------------------------- VLM side
    def encode(self, batch: VLABatch) -> SmolVLAContext:
        """One pass of the (frozen, truncated) VLM over ``[image | prompt | state token]``; keeps every layer's features."""
        device = self.action_position.device
        state = batch.state[:, : self.cfg.state_dim].to(device)
        extra = self.state_proj(state)[:, None] if self.cfg.state_in == "prefix" else None
        inputs = self.vlm.build_inputs(batch.images.to(device), batch.task, extra=extra)
        feats = self.vlm.forward_layers(inputs)
        return SmolVLAContext(
            feats, inputs.valid[:, None, None, :], state, inputs.segments, inputs.num_image_tokens
        )

    # ---------------------------------------------------------------------------------------- expert side
    def velocity(
        self,
        ctx: SmolVLAContext,
        noisy: torch.Tensor,
        time: torch.Tensor,
        return_weights: bool = False,
    ):
        """v_θ(A^τ, τ | VLM features): the action expert. ``noisy`` (B, H, A), ``time`` (B,) → (B, H, A)."""
        h = noisy.shape[1]
        t = time[:, None].expand(-1, h)
        x = self.action_time(noisy, t) + self.action_position
        n_extra = 0
        if self.cfg.state_in == "suffix":
            x = torch.cat([self.state_proj(ctx.state)[:, None], x], 1)
            n_extra = 1
        self_mask = None
        if self.cfg.self_attention == "causal":
            self_mask = causal_mask(x.shape[1], x.device)[None, None]
        weights = []
        for block, feats in zip(self.blocks, ctx.features):
            res = block(x, feats, ctx.feature_mask, self_mask, return_weights)
            if return_weights:
                x, wts = res
                weights.append((block.kind, wts))
            else:
                x = res
        out = self.action_out(self.out_norm(x))[:, n_extra:]
        return (out, weights) if return_weights else out

    # ---------------------------------------------------------------------------------------- training
    def compute_loss(self, batch: VLABatch) -> PiLosses:
        """Flow matching ``‖v_θ(A^τ, τ) − (ε − A)‖²`` (or the L1 regression ablation)."""
        ctx = self.encode(batch)
        actions = batch.actions[:, :, : self.cfg.action_dim].to(self.action_position.device)
        if self.cfg.objective in ("regression", "mse"):
            zeros = torch.zeros_like(actions)
            pred = self.velocity(ctx, zeros, torch.ones(actions.shape[0], device=actions.device))
            loss = (pred - actions).abs().mean() if self.cfg.objective == "regression" else (pred - actions).pow(2).mean()
        else:
            k = self.cfg.flow_samples_per_example
            if k > 1:  # several (ε, τ) draws share one VLM pass
                ctx = SmolVLAContext(
                    [f.repeat_interleave(k, 0) for f in ctx.features],
                    ctx.feature_mask.repeat_interleave(k, 0),
                    ctx.state.repeat_interleave(k, 0),
                )
                actions = actions.repeat_interleave(k, 0)
            noise = torch.randn_like(actions)
            flow = self.forward_flow if self.cfg.flow_convention == "forward" else self.flow
            tau = flow.sample_time(actions.shape[0], device=actions.device)
            noisy = flow.interpolate(actions, noise, tau)
            pred = self.velocity(ctx, noisy, tau)
            loss = (pred - flow.target_velocity(actions, noise)).pow(2).mean()
        return PiLosses(loss, loss, loss.new_zeros(()), {"flow": loss.item()})

    # ---------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def sample_actions(
        self,
        batch: VLABatch,
        num_steps: int | None = None,
        noise: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ):
        """Euler-integrate the learned vector field from noise (τ = 1) to an action chunk (τ = 0); the VLM runs once."""
        ctx = self.encode(batch)
        b, h, a = batch.size, self.cfg.chunk_size, self.cfg.action_dim
        device = self.action_position.device
        if self.cfg.objective in ("regression", "mse"):
            out = self.velocity(
                ctx, torch.zeros(b, h, a, device=device), torch.ones(b, device=device)
            )
            return self._pad(out)
        x0 = noise.to(device) if noise is not None else torch.randn(b, h, a, device=device)
        flow = self.forward_flow if self.cfg.flow_convention == "forward" else self.flow
        res = flow.integrate(
            lambda x, t: self.velocity(ctx, x, t),
            x0,
            num_steps or self.cfg.num_denoise_steps,
            return_trajectory,
        )
        if return_trajectory:
            return self._pad(res[0]), [self._pad(x) for x in res[1]]
        return self._pad(res)

    def _pad(self, actions: torch.Tensor) -> torch.Tensor:
        """Zero-pad to MAX_ACTION_DIM so the shared evaluator can consume it."""
        from open_vlas.embodiments import MAX_ACTION_DIM

        return F.pad(actions, (0, MAX_ACTION_DIM - actions.shape[-1]))

    def parameter_counts(self) -> dict:
        def count(m, trainable_only=False):
            return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)

        return dict(
            total=count(self),
            trainable=count(self, True),
            vlm=count(self.vlm),
            expert=count(self.blocks) + count(self.action_time) + count(self.action_out),
        )


__all__ = ["ExpertBlock", "SmolVLAConfig", "SmolVLAContext", "SmolVLAPolicy"]
