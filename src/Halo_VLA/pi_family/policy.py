"""Shared machinery of the π-family policies.

Every generation is "a prefix for the backbone + a suffix for the action expert":

* :meth:`PiPolicy.build_prefix` (generation specific) turns a :class:`PiBatch` into image blocks and a token stream
  per example; :meth:`PiPolicy.assemble_prefix` (shared) pads and embeds them into a :class:`PrefixSequence`.
* :meth:`PiPolicy.build_suffix` (generation specific) embeds the noisy action chunk (+ π0's state token).
* this base class does the rest: block-causal masks, the KV-cached prefix pass, the flow-matching loss, the
  text (subtask / FAST) loss, Euler sampling with optional committed actions and classifier-free guidance, and
  greedy text generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from pi_family.config import Pi0Config
from pi_family.flow_matching import FlowMatching
from pi_family.layers import MixtureOfTransformers, PrefixCache, _init_linear
from pi_family.masks import make_attention_mask, split_prefix_suffix_mask
from pi_family.tokenizers import PiVocabulary
from pi_family.vision import PatchImageEncoder

CONTINUOUS = (
    -1
)  # placeholder id in a TokenStream: "take the next continuous embedding" (π0.7 state tokens)


@dataclass
class TokenStream:
    """The text part of one example's prefix, token by token.

    ``ar``      - starts a new attention block (True for causal tokens, see :mod:`pi_family.masks`)
    ``target``  - this token is predicted by the previous position (subtask words, FAST tokens)
    ``visible`` - the action expert may attend this token (False for FAST tokens)
    """

    ids: list = field(default_factory=list)
    ar: list = field(default_factory=list)
    target: list = field(default_factory=list)
    visible: list = field(default_factory=list)
    continuous: list = field(default_factory=list)  # tensors for CONTINUOUS placeholders, in order

    def add(
        self, ids: Sequence[int], *, causal: bool, target: bool = False, visible: bool = True
    ) -> TokenStream:
        for i in ids:
            self.ids.append(int(i))
            self.ar.append(causal)
            self.target.append(target)
            self.visible.append(visible)
        return self

    def add_continuous(
        self, vectors: torch.Tensor, *, causal: bool, visible: bool = True
    ) -> TokenStream:
        for v in vectors:
            self.add([CONTINUOUS], causal=causal, visible=visible)
            self.continuous.append(v)
        return self

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class ImageBlock:
    """Image tokens (B, P, D) that form one bidirectional attention block; ``present`` (B,) masks absent images."""

    tokens: torch.Tensor
    present: torch.Tensor | None = None
    name: str = "image"


@dataclass
class PrefixSequence:
    """The padded backbone input. ``segments`` records (name, start, end) for the notebooks' mask plots."""

    embeds: torch.Tensor  # (B, L, D)
    ids: torch.Tensor  # (B, L) token ids (-2 for image tokens, -1 continuous, 0 pad)
    ar: torch.Tensor  # (B, L) bool
    valid: torch.Tensor  # (B, L) bool
    action_visible: torch.Tensor  # (B, L) bool
    targets: torch.Tensor  # (B, L) next-token targets, -100 = none
    flow_weight: torch.Tensor  # (B,) weight of each example in the flow-matching loss
    segments: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.embeds.shape[1]

    def repeat(self, k: int) -> PrefixSequence:
        """Each example repeated ``k`` times along the batch (for several flow samples per observation)."""

        def rep(x):
            return x.repeat_interleave(k, 0)

        return PrefixSequence(
            rep(self.embeds),
            rep(self.ids),
            rep(self.ar),
            rep(self.valid),
            rep(self.action_visible),
            rep(self.targets),
            rep(self.flow_weight),
            self.segments,
        )

    def last_valid_index(self) -> torch.Tensor:
        positions = torch.arange(self.length, device=self.valid.device)[None]
        return (positions * self.valid).argmax(dim=1)


@dataclass
class SuffixSequence:
    embeds: torch.Tensor  # (B, Ls, De)
    ar: torch.Tensor  # (Ls,) bool, first entry True
    cond: torch.Tensor | None  # (B, Ls, C) adaRMS conditioning or None
    action_start: int  # index of the first action token in the suffix


@dataclass
class PiLosses:
    total: torch.Tensor
    flow: torch.Tensor
    text: torch.Tensor
    metrics: dict = field(default_factory=dict)


class PiPolicy(nn.Module, ABC):
    """Base class: backbone + action expert with flow matching (π0 §IV), specialised by π0 / π0.5 / π0.7."""

    def __init__(self, config: Pi0Config, vocab: PiVocabulary, expert_cond_dim: int | None = None):
        super().__init__()
        self.config, self.vocab = config, vocab
        c = config
        self.image_encoder = PatchImageEncoder(
            c.image_size, c.patch_size, c.width, c.vision_stem_channels
        )
        self.token_embedding = nn.Embedding(vocab.size, c.width)
        self.prefix_position = nn.Embedding(c.max_prefix_len, c.width)
        self.transformer = MixtureOfTransformers(
            c.width,
            c.expert_width,
            c.depth,
            c.num_heads,
            c.head_dim,
            c.mlp_ratio,
            expert_cond_dim,
            c.expert_output_norm,
        )
        self.lm_head = nn.Linear(c.width, vocab.size, bias=False)
        self.action_position = nn.Parameter(torch.zeros(1, c.chunk_size, c.expert_width))
        self.action_out = nn.Linear(c.expert_width, c.action_dim)
        self.flow = FlowMatching(c.time_beta_a, c.time_beta_b, c.time_max)
        for p in (self.token_embedding.weight, self.prefix_position.weight, self.action_position):
            nn.init.trunc_normal_(p, std=0.02)
        _init_linear(self.lm_head)
        _init_linear(self.action_out)

    # ================================================================================== generation-specific
    @abstractmethod
    def build_prefix(
        self, batch, *, train: bool, mode: str = "act", generated: list | None = None
    ) -> PrefixSequence:
        """``mode``: "act" (prompt for acting), or a text-generation mode the subclass supports.
        ``generated``: tokens produced so far by :meth:`generate` (appended where generation happens)."""

    @abstractmethod
    def build_suffix(
        self, batch, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> SuffixSequence:
        """``noisy_actions`` (B, H, A) and per-token flow time ``time`` (B, H)."""

    @property
    def insulate(self) -> bool:
        """Knowledge insulation (π0.5-KI): stop the action expert's gradient into the backbone."""
        return False

    def prepare_flow_inputs(self, actions: torch.Tensor, time: torch.Tensor):
        """Hook for π0.7's training-time RTC. Returns ``(time (B, H), loss_mask (B, H))``."""
        return time, torch.ones_like(time)

    def encode_observation(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) current frames → observation tokens. π0.7 overrides this to go through its history encoder, so
        VQA pre-training (which calls this) trains the *same* path the policy later reads the scene through."""
        return self.image_encoder(images)

    # ================================================================================== prefix assembly
    def assemble_prefix(
        self, blocks: list, streams: list, flow_weight: torch.Tensor | None = None
    ) -> PrefixSequence:
        """Concatenate image blocks and per-example token streams into one padded, embedded prefix."""
        device = self.prefix_position.weight.device
        b = len(streams)
        parts_embed, parts_ids, parts_ar, parts_valid, parts_vis, segments = [], [], [], [], [], []
        offset = 0
        for block in blocks:
            p = block.tokens.shape[1]
            present = (
                block.present
                if block.present is not None
                else torch.ones(b, dtype=torch.bool, device=device)
            )
            ar = torch.zeros(b, p, dtype=torch.bool, device=device)
            ar[:, 0] = True
            parts_embed.append(block.tokens)
            parts_ids.append(torch.full((b, p), -2, dtype=torch.long, device=device))
            parts_ar.append(ar)
            parts_valid.append(present[:, None].expand(b, p))
            parts_vis.append(present[:, None].expand(b, p))
            segments.append((block.name, offset, offset + p))
            offset += p
        lt = max(len(s) for s in streams) if streams else 0
        ids = torch.zeros(b, lt, dtype=torch.long, device=device)
        ar = torch.zeros(b, lt, dtype=torch.bool, device=device)
        valid = torch.zeros(b, lt, dtype=torch.bool, device=device)
        vis = torch.zeros(b, lt, dtype=torch.bool, device=device)
        is_target = torch.zeros(b, lt, dtype=torch.bool, device=device)
        for i, s in enumerate(streams):
            n = len(s)
            ids[i, :n] = torch.tensor(s.ids, device=device)
            ar[i, :n] = torch.tensor(s.ar, device=device)
            vis[i, :n] = torch.tensor(s.visible, device=device)
            is_target[i, :n] = torch.tensor(s.target, device=device)
            valid[i, :n] = True
        text_embed = self.token_embedding(ids.clamp(min=0))
        for i, s in enumerate(streams):  # drop in continuous embeddings (π0.7 state tokens)
            if s.continuous:
                where = (ids[i] == CONTINUOUS).nonzero().flatten()
                text_embed[i, where] = torch.stack(s.continuous).to(text_embed)
        parts_embed.append(text_embed)
        parts_ids.append(ids)
        parts_ar.append(ar)
        parts_valid.append(valid)
        parts_vis.append(vis & valid)
        segments.append(("text", offset, offset + lt))
        embeds = torch.cat(parts_embed, 1)
        if embeds.shape[1] > self.config.max_prefix_len:
            raise ValueError(
                f"prefix of {embeds.shape[1]} tokens exceeds max_prefix_len {self.config.max_prefix_len}"
            )
        all_ids = torch.cat(parts_ids, 1)
        targets = torch.full_like(all_ids, -100)
        tgt_cols = torch.cat(
            [torch.zeros(b, offset, dtype=torch.bool, device=device), is_target], 1
        )
        # the logits at position j predict the token at j + 1
        targets[:, :-1] = torch.where(
            tgt_cols[:, 1:], all_ids[:, 1:], torch.full_like(all_ids[:, 1:], -100)
        )
        return PrefixSequence(
            embeds=embeds,
            ids=all_ids,
            ar=torch.cat(parts_ar, 1),
            valid=torch.cat(parts_valid, 1),
            action_visible=torch.cat(parts_vis, 1),
            targets=targets,
            flow_weight=flow_weight if flow_weight is not None else torch.ones(b, device=device),
            segments=segments,
        )

    # ================================================================================== forward passes
    def encode_prefix(self, prefix: PrefixSequence) -> tuple[torch.Tensor, PrefixCache]:
        """Run the backbone once over the prefix; returns final hidden states and the per-layer KV cache."""
        positions = torch.arange(prefix.length, device=prefix.embeds.device)
        x = prefix.embeds + self.prefix_position(positions)[None]
        return self.transformer.forward_prefix(x, make_attention_mask(prefix.ar, prefix.valid))

    def suffix_mask(self, prefix: PrefixSequence, suffix: SuffixSequence) -> torch.Tensor:
        return split_prefix_suffix_mask(prefix.ar, prefix.valid, prefix.action_visible, suffix.ar)[
            1
        ]

    def velocity(
        self,
        batch,
        prefix: PrefixSequence,
        cache: PrefixCache,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """v_θ(x_τ, τ | prefix): run the action expert against the cached prefix. Returns (B, H, A)."""
        suffix = self.build_suffix(batch, noisy_actions, time)
        mask = mask if mask is not None else self.suffix_mask(prefix, suffix)
        hidden = self.transformer.forward_suffix(
            suffix.embeds, cache, mask, suffix.cond, insulate=self.insulate
        )
        return self.action_out(hidden[:, suffix.action_start :])

    # ================================================================================== training
    def compute_loss(self, batch) -> PiLosses:
        """``text_w · CE(text targets) + flow_w · ||v_θ(x_τ, τ) - (ε - a)||²`` (regression: ``||f(o) - a||²``)."""
        c = self.config
        prefix = self.build_prefix(batch, train=True)
        hidden, cache = self.encode_prefix(prefix)
        zero = hidden.new_zeros(())
        text_loss = zero
        metrics = {}
        if (prefix.targets != -100).any():
            logits = self.lm_head(hidden)
            text_loss = F.cross_entropy(
                logits.flatten(0, 1), prefix.targets.flatten(), ignore_index=-100
            )
            metrics["text_acc"] = float(
                (logits.argmax(-1) == prefix.targets)[prefix.targets != -100].float().mean()
            )
        actions = batch.actions.to(hidden.device)
        flow_weight = prefix.flow_weight
        if c.action_objective == "regression":
            b, h, _ = actions.shape
            time = torch.ones(b, h, device=hidden.device)
            pred = self.velocity(batch, prefix, cache, torch.zeros_like(actions), time)
            per_token = (pred - actions).pow(2).mean(-1)
            mask = torch.ones_like(per_token)
        else:
            k = c.flow_samples_per_example
            if k > 1:
                # several (ε, τ) draws per observation share one backbone pass through the KV cache
                prefix, cache, batch = (
                    prefix.repeat(k),
                    cache.repeat(k),
                    batch.replace(state=batch.state.repeat_interleave(k, 0)),
                )
                actions, flow_weight = (
                    actions.repeat_interleave(k, 0),
                    flow_weight.repeat_interleave(k, 0),
                )
            b, h, _ = actions.shape
            noise = torch.randn_like(actions)
            time = self.flow.sample_time(b, device=hidden.device)[:, None].expand(b, h).clone()
            time, mask = self.prepare_flow_inputs(actions, time)
            noisy = self.flow.interpolate(actions, noise, time)
            pred = self.velocity(batch, prefix, cache, noisy, time)
            per_token = (pred - self.flow.target_velocity(actions, noise)).pow(2).mean(-1)
        weight = mask * flow_weight[:, None]
        flow_loss = (per_token * weight).sum() / weight.sum().clamp(min=1.0)
        total = c.flow_loss_weight * flow_loss + c.text_loss_weight * text_loss
        metrics.update(flow=flow_loss.item(), text=text_loss.item())
        return PiLosses(total, flow_loss, text_loss, metrics)

    # ================================================================================== inference
    @torch.no_grad()
    def sample_actions(
        self,
        batch,
        num_steps: int | None = None,
        noise: torch.Tensor | None = None,
        committed: torch.Tensor | None = None,
        cfg_scale: float = 0.0,
        uncond_batch=None,
        return_trajectory: bool = False,
    ):
        """Generate an action chunk (B, H, A) by Euler integration from noise.

        * ``committed`` (B, d, A): the first ``d`` actions are already being executed (real-time chunking); they
          are clamped at τ = 0 and the rest of the chunk is generated consistently with them.
        * ``cfg_scale = β > 0`` with ``uncond_batch``: classifier-free guidance ``v = v_c + β (v_c - v_u)``.
        * the prefix is encoded **once**; each step only runs the action expert (KV cache).
        """
        c = self.config
        steps = num_steps or c.num_denoise_steps
        prefix = self.build_prefix(batch, train=False)
        _, cache = self.encode_prefix(prefix)
        b = prefix.embeds.shape[0]
        shape = (b, c.chunk_size, c.action_dim)
        device = prefix.embeds.device
        if c.action_objective == "regression":
            out = self.velocity(
                batch,
                prefix,
                cache,
                torch.zeros(shape, device=device),
                torch.ones(shape[:2], device=device),
            )
            return (out, [out]) if return_trajectory else out
        guided = cfg_scale > 0 and uncond_batch is not None
        if guided:
            u_prefix = self.build_prefix(uncond_batch, train=False)
            _, u_cache = self.encode_prefix(u_prefix)
        d = 0 if committed is None else committed.shape[1]
        committed = committed.to(device).float() if committed is not None else None

        def velocity_fn(x, t):
            time = t[:, None].expand(b, c.chunk_size).clone()
            if d:
                x = x.clone()
                x[:, :d] = committed
                time[:, :d] = 0.0
            v = self.velocity(batch, prefix, cache, x, time)
            if guided:
                v_u = self.velocity(uncond_batch, u_prefix, u_cache, x, time)
                v = v + cfg_scale * (v - v_u)
            if d:
                v[:, :d] = 0.0
            return v

        x0 = noise.to(device) if noise is not None else torch.randn(shape, device=device)
        if d:
            x0 = x0.clone()
            x0[:, :d] = committed
        result = self.flow.integrate(velocity_fn, x0, steps, return_trajectory)
        return result

    @torch.no_grad()
    def generate(
        self,
        batch,
        mode: str,
        max_new_tokens: int = 8,
        stop_token: str = "<eos>",
        allowed_ids: Sequence[int] | None = None,
    ) -> list:
        """Greedy decoding in a subclass text mode (π0.5 "subtask" / "fast"). Returns new token ids per example."""
        b = batch.size
        generated = [[] for _ in range(b)]
        done = [False] * b
        stop = self.vocab[stop_token] if stop_token in self.vocab.word_to_id else None
        allowed = None
        if allowed_ids is not None:
            allowed = torch.full((self.vocab.size,), float("-inf"))
            allowed[list(allowed_ids)] = 0.0
        for _ in range(max_new_tokens):
            prefix = self.build_prefix(batch, train=False, mode=mode, generated=generated)
            hidden, _ = self.encode_prefix(prefix)
            last = hidden[torch.arange(b), prefix.last_valid_index()]
            logits = self.lm_head(last)
            if allowed is not None:
                logits = logits + allowed.to(logits.device)
            nxt = logits.argmax(-1).tolist()
            for i, tok in enumerate(nxt):
                if done[i]:
                    continue
                if tok == stop:
                    done[i] = True
                else:
                    generated[i].append(tok)
            if all(done):
                break
        return generated

    # ================================================================================== introspection
    def parameter_counts(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        backbone = count(self.transformer.backbone) + count(self.transformer.backbone_norm)
        expert = count(self.transformer.expert) + count(self.transformer.expert_norm)
        return dict(
            total=count(self),
            vision=count(self.image_encoder),
            backbone=backbone,
            action_expert=expert,
        )
