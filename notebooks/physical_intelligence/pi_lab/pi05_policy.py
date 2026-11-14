"""π0.5: one model that predicts subtasks (text), FAST tokens and flow-matched actions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pi_lab.block_attention_mask import BlockAttentionMask
from pi_lab.flow_matching import FlowMatching
from pi_lab.mask_block import MaskBlock
from pi_lab.mixture_of_transformers import KVCache, MixtureOfTransformers
from pi_lab.patch_encoder import PatchEncoder


class Pi05Policy(nn.Module):
    """Toy-scale π0.5 (Physical Intelligence, 2025).

    What changes from :class:`Pi0Policy`:

    * **discrete state**: the proprioceptive state enters the *prompt* as text tokens
      (one bin per dimension) instead of a continuous token in the action expert;
    * **hierarchical outputs from one network**: after the prompt, the VLM decodes the next
      **subtask** as text ("go to red"), then the action expert produces actions conditioned on
      it. High-level and low-level inference reuse the same weights;
    * **FAST tokens**: the VLM also learns to emit the chunk as discrete FAST tokens with plain
      next-token cross-entropy (π0-FAST style), which trains fast and keeps the VLM "language-like";
    * **AdaRMS time**: the flow time modulates every action-expert norm (:class:`AdaRMSNorm`)
      instead of being concatenated into the action tokens;
    * **mask**: FAST tokens and continuous action tokens never attend to each other, so at test
      time the FAST tokens can be skipped entirely.

    Training follows the paper's two stages: ``alpha = 0`` (pre-training: text + FAST only,
    action expert unused), then ``alpha = 10`` (post-training: add flow matching).

    Args:
        vocab_size: vocabulary size (words + state bins + FAST tokens).
        instruction_len: overall-instruction tokens.
        state_len: discrete state tokens.
        subtask_len: subtask tokens.
        fast_len: (padded) FAST tokens.
        horizon: actions per chunk.
        action_dim: action size.
        vlm_width: VLM width.
        expert_width: action expert width.
        depth: layers.
        image_size: camera resolution.
        patch: patch size.
    """

    def __init__(self, vocab_size: int, instruction_len: int = 4, state_len: int = 2,
                 subtask_len: int = 3, fast_len: int = 24, horizon: int = 10, action_dim: int = 2,
                 vlm_width: int = 64, expert_width: int = 32, depth: int = 3, image_size: int = 32,
                 patch: int = 8) -> None:
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.image = PatchEncoder(image_size, patch, vlm_width)
        self.embed = nn.Embedding(vocab_size, vlm_width)
        self.lm_head = nn.Linear(vlm_width, vocab_size, bias=False)
        n_patch = (image_size // patch) ** 2
        self.n_prompt = n_patch + instruction_len + state_len
        self.lens = {"subtask": subtask_len + 1, "fast": fast_len + 1}  # +1: a leading marker token
        self.text_pos = nn.Parameter(torch.randn(1, instruction_len + state_len + sum(self.lens.values()),
                                                 vlm_width) * 0.02)
        self.action_in = nn.Linear(action_dim, expert_width)
        self.action_pos = nn.Parameter(torch.randn(1, horizon, expert_width) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(expert_width, expert_width), nn.SiLU(),
                                      nn.Linear(expert_width, expert_width))
        self.action_out = nn.Linear(expert_width, action_dim)
        self.backbone = MixtureOfTransformers([vlm_width, expert_width], depth,
                                              cond_widths=[None, expert_width])
        prompt = ("image+instruction+state",)
        self.layout = BlockAttentionMask([
            MaskBlock("image+instruction+state", self.n_prompt),
            MaskBlock("subtask", self.lens["subtask"], causal_inside=True),
            MaskBlock("fast", self.lens["fast"], causal_inside=True, sees=(*prompt, "subtask")),
            MaskBlock("actions", horizon, sees=(*prompt, "subtask")),
        ])
        self.mask = self.layout.build()
        self.spans = self.layout.spans()
        self.flow = FlowMatching()
        self.expert_width = expert_width

    # ------------------------------------------------------------------ tokens
    def text_tokens(self, instruction, state_tokens, subtask_in, fast_in):
        """Embedded prompt text + subtask + FAST blocks (teacher-forced inputs)."""
        ids = torch.cat([instruction, state_tokens, subtask_in, fast_in], dim=1)
        return self.embed(ids) + self.text_pos[:, : ids.shape[1]]

    def time_condition(self, t: torch.Tensor) -> torch.Tensor:
        """Flow time → AdaRMS conditioning vector."""
        return self.time_mlp(FlowMatching.time_embedding(t, self.expert_width))

    @staticmethod
    def shift_right(tokens: torch.Tensor, marker: int) -> torch.Tensor:
        """Teacher-forcing inputs: ``[marker, t0, …, t_{n-1}]`` (length ``n + 1``)."""
        return torch.cat([torch.full_like(tokens[:, :1], marker), tokens], dim=1)

    # ------------------------------------------------------------------ training
    def loss(self, batch: dict[str, torch.Tensor], alpha: float, markers: dict[str, int],
             pad_id: int) -> tuple[torch.Tensor, dict[str, float]]:
        """``CE(subtask) + CE(FAST) + alpha · flow-matching MSE`` (π0.5 Eq. 1).

        Args:
            batch: a :class:`MultiStageDemoDataset` minibatch.
            alpha: flow-matching weight (0 in pre-training, 10 in post-training).
            markers: token ids that start the subtask and FAST blocks.
            pad_id: FAST padding id (ignored by the loss).
        """
        subtask_in = self.shift_right(batch["subtask"], markers["subtask"])
        fast_in = self.shift_right(batch["fast"], markers["fast"])
        text = self.text_tokens(batch["instruction"], batch["state_tokens"], subtask_in, fast_in)
        vlm = torch.cat([self.image(batch["images"]), text], dim=1)
        actions = batch["actions"]
        noise = torch.randn_like(actions)
        t = self.flow.sample_time(actions.shape[0])
        expert = self.action_in(self.flow.interpolate(actions, noise, t)) + self.action_pos
        (vlm_out, act_out), _ = self.backbone([vlm, expert], [0, 1], self.mask,
                                              conds=[None, self.time_condition(t)])
        logits = self.lm_head(vlm_out)
        sub, fst = self.spans["subtask"], self.spans["fast"]
        # position i predicts token i+1 inside each causal block
        sub_logits = logits[:, sub.start: sub.stop - 1]
        ce_sub = F.cross_entropy(sub_logits.flatten(0, 1), batch["subtask"].flatten())
        fast_logits = logits[:, fst.start: fst.stop - 1]
        fast_targets = batch["fast"].masked_fill(~batch["fast_mask"], -100)
        ce_fast = F.cross_entropy(fast_logits.flatten(0, 1), fast_targets.flatten(), ignore_index=-100)
        loss = ce_sub + ce_fast
        flow = torch.zeros(())
        if alpha > 0:
            flow = F.mse_loss(self.action_out(act_out), self.flow.target(actions, noise))
            loss = loss + alpha * flow
        return loss, {"ce_subtask": float(ce_sub), "ce_fast": float(ce_fast), "flow": float(flow)}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_subtask(self, images, instruction, state_tokens, marker: int) -> torch.Tensor:
        """High-level inference: greedily decode the subtask tokens."""
        n_sub = self.lens["subtask"] - 1
        sub_in = torch.full((images.shape[0], 1), marker)
        for _ in range(n_sub):
            L = self.n_prompt + sub_in.shape[1]
            ids = torch.cat([instruction, state_tokens, sub_in], 1)
            vlm = torch.cat([self.image(images), self.embed(ids) + self.text_pos[:, : ids.shape[1]]], 1)
            (out,), _ = self.backbone([vlm], [0], self.mask[:L, :L])
            nxt = self.lm_head(out[:, -1]).argmax(-1, keepdim=True)
            sub_in = torch.cat([sub_in, nxt], 1)
        return sub_in[:, 1:]

    def _context(self, images, instruction, state_tokens, subtask, marker: int) -> KVCache:
        """Keys/values of prompt + subtask (what the action expert attends to)."""
        sub_in = self.shift_right(subtask, marker)
        ids = torch.cat([instruction, state_tokens, sub_in], 1)
        vlm = torch.cat([self.image(images), self.embed(ids) + self.text_pos[:, : ids.shape[1]]], 1)
        L = vlm.shape[1]
        _, kv = self.backbone([vlm], [0], self.mask[:L, :L], return_kv=True)
        return kv

    @torch.no_grad()
    def sample_actions(self, images, instruction, state_tokens, subtask, marker: int,
                       steps: int = 10, noise: torch.Tensor | None = None, return_path: bool = False):
        """Low-level inference: flow-matching actions conditioned on prompt + subtask (no FAST)."""
        kv = self._context(images, instruction, state_tokens, subtask, marker)
        act = self.spans["actions"]
        ctx = self.n_prompt + self.lens["subtask"]
        keep = list(range(ctx)) + list(range(act.start, act.stop))
        mask = self.mask[act][:, keep]

        def velocity(x, t):
            tokens = self.action_in(x) + self.action_pos
            (out,), _ = self.backbone([tokens], [1], mask, conds=[self.time_condition(t)], past_kv=kv)
            return self.action_out(out)

        B = images.shape[0]
        noise = torch.randn(B, self.horizon, self.action_dim) if noise is None else noise
        return self.flow.sample(velocity, noise, steps, return_path)

    @torch.no_grad()
    def decode_fast(self, images, instruction, state_tokens, subtask, markers: dict[str, int],
                    max_tokens: int) -> torch.Tensor:
        """π0-FAST-style inference: greedily decode FAST tokens after the subtask (slow path)."""
        sub_in = self.shift_right(subtask, markers["subtask"])
        fast_in = torch.full((images.shape[0], 1), markers["fast"])
        for _ in range(max_tokens):
            ids = torch.cat([instruction, state_tokens, sub_in, fast_in], 1)
            vlm = torch.cat([self.image(images), self.embed(ids) + self.text_pos[:, : ids.shape[1]]], 1)
            L = vlm.shape[1]
            (out,), _ = self.backbone([vlm], [0], self.mask[:L, :L])
            fast_in = torch.cat([fast_in, self.lm_head(out[:, -1]).argmax(-1, keepdim=True)], 1)
        return fast_in[:, 1:]
