"""π0.7: a steerable VLA conditioned on metadata, subgoal images and memory (PI, 2026)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pi_lab.block_attention_mask import BlockAttentionMask
from pi_lab.flow_matching import FlowMatching
from pi_lab.mask_block import MaskBlock
from pi_lab.mixture_of_transformers import KVCache, MixtureOfTransformers
from pi_lab.patch_encoder import PatchEncoder


class Pi07Policy(nn.Module):
    """Toy-scale π0.7: the prompt describes *how* to act, not just *what* to do.

    The context (VLM prefix) holds:

    * the current image and a **memory** of past frames, compressed MEM-style: each past frame
      is patch-encoded, then the frames are averaged per patch, so the whole history costs the
      token budget of **one** frame;
    * the current and past **states**, each a linear projection to one token (π0.7 switched back
      from π0.5's text-token states to continuous projections);
    * an optional **subgoal image** (what the scene should look like a little later), produced
      at test time by a world model;
    * the **task** text and **episode metadata**: ``speed:``, ``quality:``, ``mistake:``, and the
      control ``mode:``.

    Training uses **prompt dropout** (metadata, subgoal and history are randomly blanked) so the
    model works with any subset of the context, and blanking the metadata yields an
    unconditional model for **classifier-free guidance** at test time.

    **Knowledge insulation**: the VLM is trained only by next-token cross-entropy on FAST
    tokens; the action expert reads the VLM's keys/values through a stop-gradient, so the flow
    loss cannot overwrite the VLM's representations.

    **Training-time real-time chunking**: a random prefix of ``d`` actions is given clean (the
    part of the previous chunk that will still execute while the new chunk is computed), marked
    with a learned embedding and excluded from the loss. At test time the policy then continues
    smoothly from what the robot is already doing.

    Args:
        vocab_size: vocabulary size (words + FAST tokens).
        task_len: task tokens.
        meta_len: metadata + control-mode tokens.
        history: number of past frames / states.
        fast_len: padded FAST length.
        horizon: actions per chunk.
        action_dim: action size.
        state_dim: state size.
        vlm_width: VLM width.
        expert_width: action expert width.
        depth: layers.
        image_size: camera resolution.
        patch: patch size.
        max_delay: largest simulated inference delay (actions) for RTC training.
    """

    def __init__(self, vocab_size: int, task_len: int = 3, meta_len: int = 8, history: int = 3,
                 fast_len: int = 16, horizon: int = 10, action_dim: int = 2, state_dim: int = 2,
                 vlm_width: int = 64, expert_width: int = 64, depth: int = 3, image_size: int = 32,
                 patch: int = 8, max_delay: int = 3) -> None:
        super().__init__()
        self.horizon, self.action_dim, self.max_delay = horizon, action_dim, max_delay
        self.image = PatchEncoder(image_size, patch, vlm_width)
        n_patch = (image_size // patch) ** 2
        self.slot = nn.Parameter(torch.randn(3, 1, 1, vlm_width) * 0.02)  # current/memory/subgoal type
        self.empty_image = nn.Parameter(torch.zeros(1, n_patch, vlm_width))  # dropped memory/subgoal
        self.state_in = nn.Linear(state_dim, vlm_width)
        self.state_pos = nn.Parameter(torch.randn(1, history + 1, vlm_width) * 0.02)
        self.embed = nn.Embedding(vocab_size, vlm_width)
        self.text_pos = nn.Parameter(torch.randn(1, task_len + meta_len + fast_len + 1, vlm_width) * 0.02)
        self.lm_head = nn.Linear(vlm_width, vocab_size, bias=False)
        self.action_in = nn.Linear(action_dim, expert_width)
        self.given = nn.Parameter(torch.zeros(1, 1, expert_width))     # RTC: "this action is fixed"
        self.action_pos = nn.Parameter(torch.randn(1, horizon, expert_width) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(expert_width, expert_width), nn.SiLU(),
                                      nn.Linear(expert_width, expert_width))
        self.action_out = nn.Linear(expert_width, action_dim)
        self.backbone = MixtureOfTransformers([vlm_width, expert_width], depth,
                                              cond_widths=[None, expert_width])
        self.n_context = 3 * n_patch + history + 1 + task_len + meta_len
        self.layout = BlockAttentionMask([
            MaskBlock("context", self.n_context),
            MaskBlock("fast", fast_len + 1, causal_inside=True),
            MaskBlock("actions", horizon, sees=("context",)),
        ])
        self.mask = self.layout.build()
        self.spans = self.layout.spans()
        self.flow = FlowMatching(1.0, 1.0)
        self.expert_width = expert_width

    # ------------------------------------------------------------------ context
    def context_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Current image, compressed memory, subgoal, states, task and metadata tokens."""
        B = batch["image"].shape[0]
        current = self.image(batch["image"]) + self.slot[0]
        past = self.image(batch["history"].flatten(0, 1)).unflatten(0, (B, -1)).mean(1)  # MEM: T frames → 1
        memory = torch.where(batch["has_history"][:, None, None], past, self.empty_image) + self.slot[1]
        goal = torch.where(batch["has_subgoal"][:, None, None], self.image(batch["subgoal"]),
                           self.empty_image) + self.slot[2]
        states = self.state_in(batch["states"]) + self.state_pos
        text = self.embed(torch.cat([batch["task"], batch["meta"]], 1))
        text = text + self.text_pos[:, : text.shape[1]]
        return torch.cat([current, memory, goal, states, text], 1)

    def time_condition(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(FlowMatching.time_embedding(t, self.expert_width))

    def action_tokens(self, x: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        """Noisy (or, for the first ``delay`` steps, clean and fixed) action tokens."""
        fixed = (torch.arange(self.horizon)[None] < delay[:, None]).unsqueeze(-1)
        return self.action_in(x) + self.action_pos + fixed * self.given

    # ------------------------------------------------------------------ training
    def loss(self, batch: dict[str, torch.Tensor], fast_marker: int,
             insulate: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        """FAST cross-entropy (trains the VLM) + flow matching (trains the action expert).

        Args:
            batch: a :class:`SteerableDemoDataset` minibatch.
            fast_marker: id of the token that opens the FAST block.
            insulate: stop flow-matching gradients from reaching the VLM (knowledge insulation).
        """
        fast_in = torch.cat([torch.full_like(batch["fast"][:, :1], fast_marker), batch["fast"]], 1)
        ctx = self.context_tokens(batch)
        fast = self.embed(fast_in) + self.text_pos[:, -fast_in.shape[1]:]
        vlm = torch.cat([ctx, fast], 1)
        L = vlm.shape[1]
        (vlm_out,), kv = self.backbone([vlm], [0], self.mask[:L, :L], return_kv=True)
        fst = self.spans["fast"]
        targets = batch["fast"].masked_fill(~batch["fast_mask"], -100)
        ce = F.cross_entropy(self.lm_head(vlm_out[:, fst.start: fst.stop - 1]).flatten(0, 1),
                             targets.flatten(), ignore_index=-100)
        # action expert: attends to the context keys/values only (not FAST), through a stop-grad
        n = self.n_context
        ctx_kv = [(k[:, :, :n], v[:, :, :n]) for k, v in kv]
        actions = batch["actions"]
        noise = torch.randn_like(actions)
        t = self.flow.sample_time(actions.shape[0])
        noisy = self.flow.interpolate(actions, noise, t)
        delay = torch.randint(0, self.max_delay + 1, (actions.shape[0],))
        fixed = (torch.arange(self.horizon)[None] < delay[:, None]).unsqueeze(-1)
        noisy = torch.where(fixed, actions, noisy)
        act = self.spans["actions"]
        mask = self.mask[act][:, list(range(n)) + list(range(act.start, act.stop))]
        (out,), _ = self.backbone([self.action_tokens(noisy, delay)], [1], mask,
                                  conds=[self.time_condition(t)], past_kv=ctx_kv, detach_past=insulate)
        err = (self.action_out(out) - self.flow.target(actions, noise)).pow(2).mean(-1)
        flow = (err * (~fixed[..., 0])).sum() / (~fixed[..., 0]).sum().clamp(min=1)
        return ce + flow, {"ce_fast": float(ce), "flow": float(flow)}

    # ------------------------------------------------------------------ inference
    def context_kv(self, batch: dict[str, torch.Tensor]) -> KVCache:
        """Keys/values of the context (computed once per control step)."""
        n = self.n_context
        _, kv = self.backbone([self.context_tokens(batch)], [0], self.mask[:n, :n], return_kv=True)
        return kv

    @torch.no_grad()
    def sample(self, batch: dict[str, torch.Tensor], uncond: dict[str, torch.Tensor] | None = None,
               guidance: float = 1.0, steps: int = 5, prefix: torch.Tensor | None = None,
               noise: torch.Tensor | None = None, return_path: bool = False):
        """Sample a chunk; optionally with classifier-free guidance and an RTC action prefix.

        Args:
            batch: context with the metadata to condition on.
            uncond: the same context with metadata blanked (needed when ``guidance != 1``).
            guidance: CFG weight β: ``v = v_uncond + β (v_cond − v_uncond)``.
            steps: Euler steps (π0.7: 5).
            prefix: ``(B, d, D)`` actions already committed from the previous chunk (RTC).
            noise: starting noise.
            return_path: return intermediate states too.
        """
        B = batch["image"].shape[0]
        n = self.n_context
        act = self.spans["actions"]
        mask = self.mask[act][:, list(range(n)) + list(range(act.start, act.stop))]
        kv_c = self.context_kv(batch)
        kv_u = self.context_kv(uncond) if uncond is not None and guidance != 1.0 else None
        d = 0 if prefix is None else prefix.shape[1]
        delay = torch.full((B,), d)

        def run(kv, x, t):
            (out,), _ = self.backbone([self.action_tokens(x, delay)], [1], mask,
                                      conds=[self.time_condition(t)], past_kv=kv)
            return self.action_out(out)

        def velocity(x, t):
            if d:
                x = torch.cat([prefix, x[:, d:]], 1)
            v = run(kv_c, x, t)
            if kv_u is not None:
                v_u = run(kv_u, x, t)
                v = v_u + guidance * (v - v_u)
            if d:
                v = torch.cat([torch.zeros_like(v[:, :d]), v[:, d:]], 1)  # committed actions stay put
            return v

        noise = torch.randn(B, self.horizon, self.action_dim) if noise is None else noise
        if d:
            noise = torch.cat([prefix, noise[:, d:]], 1)
        return self.flow.sample(velocity, noise, steps, return_path)
