"""Embodied thinking: write a short reasoning trace in words *before* the action tokens (Gemini Robotics 1.5, embodied chain-of-thought).

The trace lists where each object is (two number words per object, the vocabulary of the VLM's own VQA pre-training) and then names the chosen
target, ``red 20 40 green 60 30 blue 80 70 go to green``, followed by the usual action tokens. Training is ordinary next-token
cross-entropy on ``[thought | actions]``; at inference the model first generates the thought (any word), then the action bins.
The thought costs extra decoding steps but turns an *implicit* computation (which object is leftmost?) into an *explicit* one that a
language model is good at.
"""

from __future__ import annotations

import torch
from open_vlas.data import VLABatch
from pi_family.toy.world import COLORS

from vla_foundations.autoregressive import ARConfig, AutoregressiveActionPolicy


def thought_text(objects: dict, target: str) -> str:
    """``red 20 40 green 60 30 blue 80 70 go to green`` (positions in tens of percent, fixed colour order)."""
    missing = [c for c in COLORS if c not in objects]
    if missing:
        raise ValueError(
            f"the reasoning trace needs all three colours in the scene (missing {missing})"
        )
    parts = []
    for colour in COLORS:
        x, y = (int(10 * round(10 * float(v))) for v in objects[colour])
        parts += [colour, str(x), str(y)]
    return " ".join(parts) + f" go to {target}"


class ThinkingPolicy(AutoregressiveActionPolicy):
    """Autoregressive policy that decodes a thought, then the action tokens.

    Args:
        vlm / cfg: as for :class:`AutoregressiveActionPolicy`.
        think: if ``False`` the trace is dropped and this is the plain direct policy (used as the baseline with the same code path).
    """

    THOUGHT_LEN = 3 * 3 + 3  # three (colour, x, y) triples + "go to <colour>"

    def __init__(self, vlm, cfg: ARConfig, think: bool = True):
        super().__init__(vlm, cfg)
        self.think = think
        self.last_thoughts: list[str] = []
        self.forced_thoughts: list[list[int]] | None = (
            None  # teacher-forced thoughts (oracle reasoning) at inference
        )

    @property
    def thought_len(self) -> int:
        return self.THOUGHT_LEN if self.think else 0

    # ------------------------------------------------------------------------------------ training
    def target_sequences(
        self, actions: torch.Tensor, batch: VLABatch | None = None
    ) -> list[list[int]]:
        bins = super().target_sequences(actions)
        if not self.think:
            return bins
        if batch is None or batch.meta is None:
            raise ValueError(
                "a thinking policy needs batch.meta (build batches with EmbodiedSource)"
            )
        out = []
        for row, meta in zip(bins, batch.meta):
            thought = self.vocab.encode(thought_text(meta["objects"], meta["targets"][0]))
            out.append(thought + row)
        return out

    def compute_loss(self, batch):
        robot = batch.robot if hasattr(batch, "robot") else batch
        ce, acc = self._ce(
            robot.images,
            robot.task,
            self.target_sequences(robot.actions.to(self.base_device), robot),
        )
        from pi_family.policy import PiLosses

        return PiLosses(ce, ce, ce.new_zeros(()), {"flow": ce.item(), "token_acc": acc})

    # ------------------------------------------------------------------------------------ inference
    @property
    def max_new_tokens(self) -> int:
        return self.thought_len + self.cfg.tokens_per_chunk

    def _pick(self, logits: torch.Tensor, temperature: float, step: int = 0) -> torch.Tensor:
        if step < self.thought_len:  # reasoning phase: any word of the vocabulary
            words = logits[:, : self.vocab.state_offset]
            if self.forced_thoughts is not None:
                return torch.tensor([row[step] for row in self.forced_thoughts])
            return words.argmax(-1)
        return super()._pick(logits, temperature, step)

    def finish(self, ids: torch.Tensor) -> torch.Tensor:
        t = self.thought_len
        self.last_thoughts = [self.vocab.decode(row[:t]) for row in ids.tolist()] if t else []
        return self.tokens_to_actions(ids[:, t:])

    @torch.no_grad()
    def sample_actions(self, batch: VLABatch, temperature: float = 0.0, cached: bool = False):
        # the thought tokens are not bins, so use the exact (non-cached) decoder, which re-embeds every token
        return super().sample_actions(batch, temperature, cached=False)
