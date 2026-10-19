"""RT-2 / OpenVLA-style policy: a VLM that emits *actions as text tokens*.

The sequence is ``[image | prompt | a_1^1 a_1^2 … a_H^D]`` where every action dimension is one of ``num_bins`` bin tokens
(:class:`ActionBinner`). Training is the VLM's ordinary next-token cross-entropy (teacher forcing); inference decodes the action
tokens **one at a time**, each conditioned on the previous ones, so a chunk of ``H`` actions with ``D`` dimensions costs ``H·D``
sequential forward passes. RT-2 uses ``H = 1`` and 8 tokens per step.

*Co-fine-tuning* (RT-2's recipe) trains the same model on robot action tokens **and** on the original vision-language task (here
VQA), so the language knowledge is not forgotten while the actions are learned.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from open_vlas.data import VLABatch
from open_vlas.embodiments import MAX_ACTION_DIM
from open_vlas.vlm import ToyVLM
from pi_family.policy import PiLosses

from vla_foundations.discretization import ActionBinner


@dataclass
class ARConfig:
    """Settings of the autoregressive action policy."""

    num_bins: int = 256
    chunk_size: int = 1  # H (RT-2 predicts one step at a time)
    action_dim: int = 2
    low: float = -1.5
    high: float = 1.5

    @property
    def tokens_per_chunk(self) -> int:
        return self.chunk_size * self.action_dim


@dataclass
class CoBatch:
    """A robot batch plus a vision-language (VQA) batch for co-fine-tuning."""

    robot: VLABatch
    vqa_images: torch.Tensor
    vqa_questions: list
    vqa_answers: list


class AutoregressiveActionPolicy(torch.nn.Module):
    """A VLM fine-tuned to emit action tokens (RT-2).

    Args:
        vlm: the (pre-trained or random) :class:`ToyVLM`; deep-copied and trained in full. Its vocabulary must have at least
            ``cfg.num_bins`` ``fast`` slots, which serve as the bin tokens.
        cfg: :class:`ARConfig`.
        binner: how actions map to bins (default: uniform over ``[cfg.low, cfg.high]``).
    """

    def __init__(self, vlm: ToyVLM, cfg: ARConfig, binner: ActionBinner | None = None):
        super().__init__()
        if vlm.vocab.fast_size < cfg.num_bins:
            raise ValueError(
                "the VLM vocabulary needs at least num_bins action-token slots (PiVocabulary(fast_size=...))"
            )
        self.cfg = cfg
        self.vlm = copy.deepcopy(vlm)
        self.vocab = vlm.vocab
        self.binner = binner or ActionBinner(cfg.num_bins, cfg.low, cfg.high)
        self.base = self.vocab.fast_offset  # id of bin 0

    # ---------------------------------------------------------------------------------------- tokens
    def action_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, H, D) continuous → (B, H·D) token ids, time-major: ``a_1^1 a_1^2 … a_2^1 …``."""
        b = actions.shape[0]
        bins = self.binner.encode(actions[:, : self.cfg.chunk_size, : self.cfg.action_dim])
        return bins.reshape(b, -1) + self.base

    def tokens_to_actions(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, H·D) token ids → (B, H, MAX_ACTION_DIM) bin-centre actions (zero-padded for the shared evaluator)."""
        b = ids.shape[0]
        a = self.binner.decode(ids - self.base).reshape(b, self.cfg.chunk_size, self.cfg.action_dim)
        return F.pad(a, (0, MAX_ACTION_DIM - self.cfg.action_dim))

    # ---------------------------------------------------------------------------------------- training
    def _ce(self, images, prompts, answers):
        inputs = self.vlm.build_inputs(images, prompts, answers=answers)
        hidden = self.vlm.forward_layers(inputs)[-1]
        logits = self.vlm.logits(hidden)
        targets = torch.full(inputs.valid.shape, -100, dtype=torch.long, device=hidden.device)
        for i, ids in enumerate(answers):
            start = int(inputs.valid[i].sum()) - len(ids)
            targets[i, start - 1 : start - 1 + len(ids)] = torch.as_tensor(
                list(ids), device=hidden.device
            )
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        hit = (logits.argmax(-1) == targets)[targets != -100].float().mean()
        return loss, float(hit)

    def compute_loss(self, batch) -> PiLosses:
        """Next-token cross-entropy on the action tokens (+ on VQA answers when given a :class:`CoBatch`)."""
        robot = batch.robot if isinstance(batch, CoBatch) else batch
        ce, acc = self._ce(robot.images, robot.task, self.target_sequences(robot.actions))
        metrics = {"flow": ce.item(), "token_acc": acc}
        total = ce
        if isinstance(batch, CoBatch):
            v = self.vocab
            answers = [v.encode(a) + [v["<eos>"]] for a in batch.vqa_answers]
            vqa, vacc = self._ce(batch.vqa_images, batch.vqa_questions, answers)
            total = ce + vqa
            metrics.update(vqa=vqa.item(), vqa_acc=vacc)
        return PiLosses(total, ce, ce.new_zeros(()), metrics)

    def target_sequences(self, actions: torch.Tensor) -> list[list[int]]:
        """The token sequence the model is trained to emit for each example (fixed length ``H·D`` for bins)."""
        return [row.tolist() for row in self.action_tokens(actions.to(self.base_device))]

    @property
    def max_new_tokens(self) -> int:
        """Upper bound on the number of tokens decoded per chunk."""
        return self.cfg.tokens_per_chunk

    def _all_done(self, ids: list) -> bool:
        """True when decoding may stop early (never for fixed-length bin tokens)."""
        return False

    def finish(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) decoded token ids → (B, H, MAX_ACTION_DIM) actions."""
        return self.tokens_to_actions(ids)

    @property
    def base_device(self):
        return self.vlm.tok_emb.weight.device

    # ---------------------------------------------------------------------------------------- inference
    def _pick(self, logits: torch.Tensor, temperature: float, step: int = 0) -> torch.Tensor:
        """Choose a bin token: argmax, or a sample at ``temperature``; only the bin tokens are allowed."""
        bins = logits[:, self.base : self.base + self.cfg.num_bins]
        if temperature <= 0:
            return bins.argmax(-1) + self.base
        return torch.multinomial(F.softmax(bins / temperature, -1), 1)[:, 0] + self.base

    @torch.no_grad()
    def sample_actions(
        self, batch: VLABatch, temperature: float = 0.0, cached: bool = True
    ) -> torch.Tensor:
        """Decode the action tokens one by one. ``cached`` reuses the prefix keys / values (needs equal-length prompts)."""
        return (
            self.generate_cached(batch, temperature)
            if cached
            else self.generate_naive(batch, temperature)
        )

    @torch.no_grad()
    def generate_naive(self, batch: VLABatch, temperature: float = 0.0) -> torch.Tensor:
        """Re-run the whole sequence for every new token: ``H·D`` full forward passes."""
        out = [[] for _ in range(batch.size)]
        for _ in range(self.max_new_tokens):
            inputs = self.vlm.build_inputs(batch.images, batch.task, answers=out)
            hidden = self.vlm.forward_layers(inputs)[-1]
            last = inputs.valid.long().cumsum(1).argmax(1)
            tok = self._pick(
                self.vlm.logits(hidden[torch.arange(batch.size), last]), temperature, len(out[0])
            )
            for o, t in zip(out, tok.tolist()):
                o.append(t)
            if self._all_done(out):
                break
        return self.finish(torch.tensor(out))

    @torch.no_grad()
    def generate_cached(self, batch: VLABatch, temperature: float = 0.0) -> torch.Tensor:
        """Exact KV-cached decoding: one full prefix pass, then ``H·D`` single-token passes (still *sequential*)."""
        lengths = {len(self.vlm.prompt_ids(t)) for t in batch.task}
        if len(lengths) != 1:
            return self.generate_naive(batch, temperature)
        vlm = self.vlm
        inputs = vlm.build_inputs(batch.images, batch.task)
        x, mask = inputs.embeds, inputs.mask
        caches = []
        for layer in vlm.layers:
            q, k, v, gate = layer.project(x)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
            x = layer.finish(x, att, gate)
            caches.append((k, v))
        length = x.shape[1]
        tok = self._pick(vlm.logits(x[:, -1]), temperature, 0)
        ids = [tok]
        for step in range(self.max_new_tokens - 1):
            if self._all_done(ids):
                break
            h = (vlm.tok_emb(tok) + vlm.pos_emb(torch.tensor(length + step, device=x.device)))[
                :, None
            ]
            for li, layer in enumerate(vlm.layers):
                q, k, v, gate = layer.project(h)
                kk, vv = torch.cat([caches[li][0], k], 2), torch.cat([caches[li][1], v], 2)
                caches[li] = (kk, vv)
                h = layer.finish(h, F.scaled_dot_product_attention(q, kk, vv), gate)
            tok = self._pick(vlm.logits(h[:, 0]), temperature, step + 1)
            ids.append(tok)
        return self.finish(torch.stack(ids, 1))

    def forward_passes(self) -> int:
        """Sequential model evaluations needed for one action chunk (the cost the paper's Hz numbers reflect)."""
        return self.cfg.tokens_per_chunk

    def parameter_counts(self) -> dict:
        return dict(total=sum(p.numel() for p in self.vlm.parameters()))


def make_cofine_tune_data(robot_data, vqa_sampler, vqa_ratio: float = 0.5):
    """A dataset whose ``sample`` mixes a robot batch with a VQA batch (RT-2's co-fine-tuning); ``vqa_sampler(n)`` → (images, questions, answers)."""

    class _Data:
        def sample(self, batch_size: int, rng: np.random.Generator) -> CoBatch:
            images, questions, answers = vqa_sampler(max(1, int(batch_size * vqa_ratio)))
            return CoBatch(robot_data.sample(batch_size, rng), images, questions, answers)

    return _Data()
