"""GR00T N1 and N1.5 (NVIDIA, 2025) built from first principles.

**GR00T N1** (Bjorck et al., arXiv:2503.14734) is a *dual-system* model:

* **System 2** - a pretrained VLM (Eagle-2) reads the images and the instruction; its hidden states from a **middle layer**
  (the 12th of 24) are the conditioning, because they are both faster and *better* than the last layer's;
* **System 1** - a **diffusion transformer (DiT)** with *alternating cross-attention and self-attention blocks* denoises an action
  chunk. Self-attention runs over ``[state token, noised action tokens]``; cross-attention reads the VLM features; the denoising
  step τ enters through **adaptive layer norm**;
* **embodiment-specific MLPs** encode the state, encode (noised action, τ) and decode the DiT output, one set of weights per robot
  body, so one network can be trained on arms, humanoids and even *latent actions* extracted from video.

Training is flow matching with ``A^τ = τA + (1 − τ)ε`` (τ = 0 is noise), ``p(τ) = Beta((s − τ)/s; 1.5, 1)`` and K = 4 Euler steps.

**GR00T N1.5** keeps this and changes four things (NVIDIA GEAR release notes):

1. the **VLM is frozen** during pre-training *and* fine-tuning (Eagle 2.5, tuned for grounding);
2. the adapter between the VLM and the DiT gets **layer normalisation** on the visual and text tokens;
3. a **FLARE** loss (Zheng et al., arXiv:2505.15659) is added with weight 0.2: a few *future tokens* in the DiT are trained, at a
   middle layer, to match the embedding of the future observation (cosine similarity), which also lets the model learn from
   action-less human video;
4. new data (DreamGen neural trajectories).

This module implements 1-3.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from pi_family.flow_matching import sinusoidal_time_embedding
from pi_family.policy import PiLosses

from open_vlas.attention import Attention, init_linear
from open_vlas.data import VLABatch
from open_vlas.embodiments import MAX_ACTION_DIM, MAX_STATE_DIM, NUM_EMBODIMENTS
from open_vlas.flow import ForwardFlow
from open_vlas.vlm import ToyVLM, VLMConfig


@dataclass
class GR00TConfig:
    """GR00T hyper-parameters; published values in the comments."""

    vlm: VLMConfig = field(default_factory=VLMConfig)
    vlm_layer: int | None = (
        None  # use the features after this many LLM layers (paper: 12 of 24 = half)
    )
    freeze: str = (
        "language"  # N1: freeze the language model, train the vision encoder | N1.5: "all" | "none"
    )
    vl_layernorm: bool = False  # N1.5: LayerNorm on the VLM tokens before the DiT reads them
    dit_width: int = 64  # N1: 1536 for the 2B model
    dit_depth: int = 4  # alternating cross / self blocks
    dit_heads: int = 4
    mlp_ratio: float = 2.0
    embodiment_hidden: int = 64  # width of the per-embodiment MLPs
    chunk_size: int = 16  # H = 16 in the paper
    num_denoise_steps: int = 4  # K = 4 in the paper
    flow_samples_per_example: int = 4
    flare_weight: float = 0.0  # N1.5: 0.2
    num_future_tokens: int = 4  # M (FLARE uses 32 with a Q-former-compressed target)
    flare_layer: int | None = (
        None  # DiT block whose future-token output is aligned (default: the middle one)
    )

    def __post_init__(self) -> None:
        if self.vlm_layer is None:
            self.vlm_layer = self.vlm.depth // 2
        if self.flare_layer is None:
            self.flare_layer = self.dit_depth // 2
        if self.freeze not in ("language", "all", "none"):
            raise ValueError("freeze must be 'language', 'all' or 'none'")
        grid = int(math.isqrt(self.vlm.num_image_tokens))
        cells = int(math.isqrt(self.num_future_tokens))
        if self.flare_weight > 0 and (
            grid * grid != self.vlm.num_image_tokens
            or cells * cells != self.num_future_tokens
            or grid % cells
        ):
            raise ValueError(
                "FLARE pools the image-token grid into num_future_tokens square cells: sizes must divide"
            )

    @classmethod
    def n1(cls, **kw) -> GR00TConfig:
        """GR00T N1: language model frozen, vision encoder trained, no FLARE."""
        return cls(**kw)

    @classmethod
    def n1_5(cls, **kw) -> GR00TConfig:
        """GR00T N1.5: whole VLM frozen, layer-normalised adapter, FLARE with weight 0.2."""
        kw.setdefault("freeze", "all")
        kw.setdefault("vl_layernorm", True)
        kw.setdefault("flare_weight", 0.2)
        return cls(**kw)


class CategorySpecificLinear(nn.Module):
    """A linear layer with a separate weight matrix per embodiment (GR00T's ``CategorySpecificLinear``).

    ``forward(x, cat)`` applies ``W[cat[b]]`` to sample ``b``: one batch can mix robots with different action spaces and each
    sample is routed through its own robot's weights. Padded input dimensions simply meet weights that learn to ignore them.
    """

    def __init__(self, num_categories: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_categories, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_categories, out_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        w, b = self.weight[cat], self.bias[cat]  # (B, in, out), (B, out)
        if x.dim() == 2:
            return torch.einsum("bi,bio->bo", x, w) + b
        return torch.einsum("bti,bio->bto", x, w) + b[:, None]


class CategorySpecificMLP(nn.Module):
    """``Linear → ReLU → Linear`` with embodiment-specific weights (state encoder, action decoder)."""

    def __init__(self, num_categories: int, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, in_dim, hidden)
        self.layer2 = CategorySpecificLinear(num_categories, hidden, out_dim)

    def forward(self, x: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x, cat)), cat)


class EmbodimentActionEncoder(nn.Module):
    """Encodes the noised action together with the denoising step: ``W3 · swish(W2 · [W1·a^τ , φ(τ)])`` per embodiment."""

    def __init__(self, num_categories: int, action_dim: int, width: int):
        super().__init__()
        self.width = width
        self.w1 = CategorySpecificLinear(num_categories, action_dim, width)
        self.w2 = CategorySpecificLinear(num_categories, 2 * width, width)
        self.w3 = CategorySpecificLinear(num_categories, width, width)

    def forward(self, noisy: torch.Tensor, tau: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        t = tau[:, None].expand(-1, noisy.shape[1])
        emb = sinusoidal_time_embedding(t, self.width).to(noisy)
        h = self.w1(noisy, cat)
        return self.w3(F.silu(self.w2(torch.cat([h, emb], -1), cat)), cat)


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale and shift are predicted from the denoising-step embedding (DiT's timestep conditioning).

    Zero-initialised: at the start every block sees a plain LayerNorm.
    """

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.proj = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(F.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """One DiT block: adaptive-norm attention (self *or* cross) followed by a plain-norm GELU MLP, both residual."""

    def __init__(self, kind: str, width: int, vlm_width: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.kind = kind
        self.norm1 = AdaLayerNorm(width)
        self.attn = Attention(
            width, heads, width // heads, context_dim=vlm_width if kind == "cross" else None
        )
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(width * mlp_ratio), width),
        )
        self.ff.apply(init_linear)

    def forward(self, x, temb, features, feature_mask, return_weights: bool = False):
        h = self.norm1(x, temb)
        res = (
            self.attn(h, features, feature_mask, return_weights)
            if self.kind == "cross"
            else self.attn(h, return_weights=return_weights)
        )
        out, weights = res if return_weights else (res, None)
        x = x + out
        x = x + self.ff(self.norm2(x))
        return (x, weights) if return_weights else x


@dataclass
class GR00TContext:
    """One VLM pass: the (optionally layer-normalised) System-2 features the DiT cross-attends to."""

    features: torch.Tensor  # (B, L, vlm_width)
    mask: torch.Tensor  # (B, 1, 1, L)
    num_image_tokens: int = 0


class GR00TPolicy(nn.Module):
    """GR00T N1 / N1.5 at toy scale: VLM (System 2) + embodiment-aware DiT (System 1).

    Args:
        vlm: a pre-trained :class:`ToyVLM`; deep-copied, truncated to ``cfg.vlm_layer`` layers and frozen according to ``cfg.freeze``.
        cfg: :class:`GR00TConfig` (use ``GR00TConfig.n1()`` / ``GR00TConfig.n1_5()``).
    """

    def __init__(self, vlm: ToyVLM, cfg: GR00TConfig):
        super().__init__()
        self.cfg = cfg
        self.target_vision = None
        if cfg.flare_weight > 0:  # a frozen copy of the vision encoder defines FLARE's targets
            self.target_vision = copy.deepcopy(vlm.vision).requires_grad_(False)
        self.vlm = copy.deepcopy(vlm).truncate(cfg.vlm_layer)
        self.vlm.freeze(vision=cfg.freeze == "all", language=cfg.freeze != "none")
        w, vw = cfg.dit_width, cfg.vlm.width
        self.vl_norm = nn.LayerNorm(vw) if cfg.vl_layernorm else nn.Identity()
        self.state_encoder = CategorySpecificMLP(
            NUM_EMBODIMENTS, MAX_STATE_DIM, cfg.embodiment_hidden, w
        )
        self.action_encoder = EmbodimentActionEncoder(NUM_EMBODIMENTS, MAX_ACTION_DIM, w)
        self.action_decoder = CategorySpecificMLP(
            NUM_EMBODIMENTS, w, cfg.embodiment_hidden, MAX_ACTION_DIM
        )
        self.action_position = nn.Parameter(torch.zeros(1, cfg.chunk_size, w))
        self.future_tokens = (
            nn.Parameter(torch.zeros(1, cfg.num_future_tokens, w)) if cfg.flare_weight > 0 else None
        )
        self.time_mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList(
            DiTBlock("cross" if i % 2 == 0 else "self", w, vw, cfg.dit_heads, cfg.mlp_ratio)
            for i in range(cfg.dit_depth)
        )
        self.out_norm = nn.LayerNorm(w)
        self.flare_head = (
            nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, vw))
            if cfg.flare_weight > 0
            else None
        )
        for p in (self.action_position,) + (
            (self.future_tokens,) if self.future_tokens is not None else ()
        ):
            nn.init.trunc_normal_(p, std=0.02)
        self.time_mlp.apply(init_linear)
        if self.flare_head is not None:
            self.flare_head.apply(init_linear)
        self.flow = ForwardFlow()

    # ---------------------------------------------------------------------------------------- System 2
    def encode(self, batch: VLABatch) -> GR00TContext:
        """Run the VLM on ``[image | prompt]`` and return the features of the chosen middle layer."""
        device = self.action_position.device
        trains_vlm = any(p.requires_grad for p in self.vlm.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and trains_vlm):
            inputs = self.vlm.build_inputs(batch.images.to(device), batch.task)
            hidden = self.vlm.forward_layers(inputs)[-1]
        return GR00TContext(
            self.vl_norm(hidden), inputs.valid[:, None, None, :], inputs.num_image_tokens
        )

    # ---------------------------------------------------------------------------------------- System 1
    def velocity(
        self,
        ctx: GR00TContext,
        noisy: torch.Tensor,
        tau: torch.Tensor,
        state: torch.Tensor,
        emb: torch.Tensor,
        return_weights: bool = False,
        return_future: bool = False,
    ):
        """V_θ(φ_t, A^τ, q_t): the DiT. Returns the action velocity (B, H, A) [and attention weights / future-token activations]."""
        b, h, _ = noisy.shape
        temb = self.time_mlp(sinusoidal_time_embedding(tau, self.cfg.dit_width).to(noisy))
        tokens = [
            self.state_encoder(state, emb)[:, None],
            self.action_encoder(noisy, tau, emb) + self.action_position,
        ]
        if self.future_tokens is not None:
            tokens.append(self.future_tokens.expand(b, -1, -1))
        x = torch.cat(tokens, 1)
        weights, future = [], None
        for i, block in enumerate(self.blocks):
            res = block(x, temb, ctx.features, ctx.mask, return_weights)
            if return_weights:
                x, wts = res
                weights.append((block.kind, wts))
            else:
                x = res
            if return_future and i + 1 == self.cfg.flare_layer and self.future_tokens is not None:
                future = x[:, -self.cfg.num_future_tokens :]
        action_tokens = self.out_norm(x)[:, 1 : 1 + h]
        out = self.action_decoder(action_tokens, emb)
        extras = []
        if return_weights:
            extras.append(weights)
        if return_future:
            extras.append(future)
        return (out, *extras) if extras else out

    # ---------------------------------------------------------------------------------------- FLARE
    @torch.no_grad()
    def future_targets(self, future_images: torch.Tensor) -> torch.Tensor:
        """FLARE's targets: the frozen vision encoder's tokens of the **future** frame, average-pooled into M cells.

        The paper aligns with a compact, action-aware embedding of the future observation (a Q-former with 32 tokens); here the
        target is the pooled token grid of a frozen copy of the pre-trained vision encoder: ``(B, M, vlm_width)``.
        """
        device = self.action_position.device
        tokens = self.target_vision(future_images.to(device))  # (B, N, W)
        b, n, wdt = tokens.shape
        g = int(math.isqrt(n))
        cells = int(math.isqrt(self.cfg.num_future_tokens))
        grid = tokens.transpose(1, 2).reshape(b, wdt, g, g)
        pooled = F.adaptive_avg_pool2d(grid, cells)  # (B, W, cells, cells)
        return pooled.flatten(2).transpose(1, 2)

    # ---------------------------------------------------------------------------------------- training
    def compute_loss(self, batch: VLABatch) -> PiLosses:
        """``L = L_fm + λ·L_align``: flow matching on action chunks (+ FLARE's future-latent cosine alignment)."""
        device = self.action_position.device
        ctx = self.encode(batch)
        actions = batch.actions.to(device)
        mask = batch.action_mask.to(
            device
        )  # (B, A) 1 for the dimensions this embodiment really has
        state, emb = batch.state.to(device), batch.embodiment.to(device)
        weight = (
            batch.flow_weight if batch.flow_weight is not None else torch.ones(batch.size)
        ).to(device)
        k = self.cfg.flow_samples_per_example
        if k > 1:  # several (ε, τ) draws per observation share one VLM pass
            ctx = GR00TContext(
                ctx.features.repeat_interleave(k, 0),
                ctx.mask.repeat_interleave(k, 0),
                ctx.num_image_tokens,
            )
            actions, mask, state, emb, weight = (
                x.repeat_interleave(k, 0) for x in (actions, mask, state, emb, weight)
            )
        noise = torch.randn_like(actions) * mask[:, None, :]  # padded dimensions carry no noise
        tau = self.flow.sample_time(actions.shape[0], device=device)
        noisy = self.flow.interpolate(actions, noise, tau)
        want_future = self.cfg.flare_weight > 0 and batch.future_images is not None
        res = self.velocity(ctx, noisy, tau, state, emb, return_future=want_future)
        pred, future = res if want_future else (res, None)
        err = (pred - self.flow.target_velocity(actions, noise)).pow(2) * mask[:, None, :]
        per_sample = err.sum((1, 2)) / (mask.sum(1) * actions.shape[1]).clamp(min=1)
        flow = (per_sample * weight).sum() / weight.sum().clamp(min=1e-6)
        total, metrics = flow, {"flow": flow.item()}
        align = flow.new_zeros(())
        if want_future:
            target = self.future_targets(batch.future_images).repeat_interleave(k, 0)
            align = 1.0 - F.cosine_similarity(self.flare_head(future), target, dim=-1).mean()
            total = total + self.cfg.flare_weight * align
            metrics["flare"] = align.item()
        return PiLosses(total, flow, align, metrics)

    # ---------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def sample_actions(
        self,
        batch: VLABatch,
        num_steps: int | None = None,
        noise: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ):
        """K Euler steps from noise (τ = 0) to an action chunk (τ = 1); the VLM runs once."""
        device = self.action_position.device
        ctx = self.encode(batch)
        state, emb = batch.state.to(device), batch.embodiment.to(device)
        mask = batch.action_mask.to(device)
        b, h = batch.size, self.cfg.chunk_size
        x0 = (
            noise.to(device)
            if noise is not None
            else torch.randn(b, h, MAX_ACTION_DIM, device=device)
        ) * mask[:, None, :]
        res = self.flow.integrate(
            lambda x, t: self.velocity(ctx, x, t, state, emb) * mask[:, None, :],
            x0,
            num_steps or self.cfg.num_denoise_steps,
            return_trajectory,
        )
        return res

    def parameter_counts(self) -> dict:
        def count(m, trainable_only=False):
            return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)

        embodiment_modules = [self.state_encoder, self.action_encoder, self.action_decoder]
        return dict(
            total=count(self),
            trainable=count(self, True),
            vlm=count(self.vlm),
            dit=count(self.blocks) + count(self.time_mlp) + count(self.out_norm),
            embodiment_specific=sum(count(m) for m in embodiment_modules),
        )


__all__ = [
    "AdaLayerNorm",
    "CategorySpecificLinear",
    "CategorySpecificMLP",
    "DiTBlock",
    "EmbodimentActionEncoder",
    "GR00TConfig",
    "GR00TContext",
    "GR00TPolicy",
]
