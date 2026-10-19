"""π0-FAST-style policy: a VLM that emits a *compressed* action chunk (:class:`FastActionTokenizer`) as text tokens.

Where :class:`AutoregressiveActionPolicy` writes ``H·D`` bin tokens per chunk, this policy writes the DCT + BPE tokens of the
whole chunk followed by ``<eos>``: a smooth chunk takes far fewer tokens, so far fewer sequential decoder passes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from open_vlas.embodiments import MAX_ACTION_DIM
from open_vlas.vlm import ToyVLM
from pi_family.tokenizers import FastActionTokenizer

from vla_foundations.autoregressive import ARConfig, AutoregressiveActionPolicy


class FastActionPolicy(AutoregressiveActionPolicy):
    """Autoregressive policy over FAST tokens.

    Args:
        vlm: the (pre-trained or random) VLM; its vocabulary needs ``fast_size >= tokenizer.vocab_size``.
        tokenizer: a *fitted* :class:`FastActionTokenizer` (``chunk_size`` and ``action_dim`` define the chunk).
        max_tokens: hard cap on the decoded length (a safety net when the model never emits ``<eos>``).
    """

    def __init__(self, vlm: ToyVLM, tokenizer: FastActionTokenizer, max_tokens: int = 64):
        cfg = ARConfig(
            num_bins=tokenizer.vocab_size,
            chunk_size=tokenizer.chunk_size,
            action_dim=tokenizer.action_dim,
        )
        super().__init__(vlm, cfg)
        self.tokenizer, self.max_tokens = tokenizer, max_tokens
        self.eos = self.vocab["<eos>"]
        self.last_tokens: list[list[int]] = []

    # ----------------------------------------------------------------------------------- tokens
    def target_sequences(self, actions: torch.Tensor) -> list[list[int]]:
        """FAST ids of each chunk (shifted into the vocabulary's action slots) followed by ``<eos>``."""
        chunks = actions[:, : self.cfg.chunk_size, : self.cfg.action_dim].float().cpu()
        return [[t + self.base for t in self.tokenizer.encode(c)] + [self.eos] for c in chunks]

    @property
    def max_new_tokens(self) -> int:
        return self.max_tokens

    def _pick(self, logits: torch.Tensor, temperature: float, step: int = 0) -> torch.Tensor:
        """Only FAST tokens and ``<eos>`` are allowed."""
        allowed = torch.full_like(logits, float("-inf"))
        allowed[:, self.base : self.base + self.tokenizer.vocab_size] = 0
        allowed[:, self.eos] = 0
        logits = logits + allowed
        if temperature <= 0:
            return logits.argmax(-1)
        return torch.multinomial(F.softmax(logits / temperature, -1), 1)[:, 0]

    def _all_done(self, ids: list) -> bool:
        rows = (
            torch.stack(ids, 1).tolist() if isinstance(ids[0], torch.Tensor) else ids
        )  # cached decoding stores one tensor per step, naive one list per example
        return all(self.eos in row for row in rows)

    def finish(self, ids: torch.Tensor) -> torch.Tensor:
        """Cut each row at ``<eos>``, undo the BPE and the DCT; wrong lengths are padded or truncated by the tokenizer."""
        out, self.last_tokens = [], []
        for row in ids.tolist():
            row = row[: row.index(self.eos)] if self.eos in row else row
            self.last_tokens.append(
                [t - self.base for t in row]
            )  # local FAST ids of the last decoded chunks
            out.append(self.tokenizer.decode(self.last_tokens[-1]))
        return F.pad(torch.stack(out), (0, MAX_ACTION_DIM - self.cfg.action_dim))

    def mean_tokens(self, actions: torch.Tensor) -> float:
        """Average number of tokens (incl. ``<eos>``) per chunk: the sequential decoder passes it costs."""
        seqs = self.target_sequences(actions)
        return sum(len(s) for s in seqs) / len(seqs)
