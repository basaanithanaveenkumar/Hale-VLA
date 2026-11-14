"""Block-wise attention masks: bidirectional inside a block, causal between blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pi_lab.mask_block import MaskBlock


class BlockAttentionMask:
    """Builds the ``(L, L)`` boolean "query may attend to key" matrix from :class:`MaskBlock` s.

    π0 uses three blocks: ``[image + text]`` (bidirectional, like PaliGemma's prefix),
    ``[state]`` and ``[noisy actions]``. Each block sees itself and every earlier block, never
    a later one. Two consequences matter for speed: the prefix never looks at the state or the
    actions, so its keys/values can be **cached once** per observation; and the state block
    can be cached too, so each denoising step only recomputes the action tokens.

    π0.5 adds causal text blocks (subtask, FAST tokens) and uses ``sees`` to stop FAST tokens
    and continuous action tokens from attending to each other.

    Args:
        blocks: the blocks in sequence order.
    """

    def __init__(self, blocks: Sequence[MaskBlock]) -> None:
        self.blocks = list(blocks)

    @property
    def length(self) -> int:
        return sum(b.length for b in self.blocks)

    def spans(self) -> dict[str, slice]:
        """Name → token slice."""
        out, start = {}, 0
        for b in self.blocks:
            out[b.name] = slice(start, start + b.length)
            start += b.length
        return out

    def build(self) -> torch.Tensor:
        """The boolean mask, ``True`` = allowed."""
        L = self.length
        mask = torch.zeros(L, L, dtype=torch.bool)
        spans = self.spans()
        for i, q in enumerate(self.blocks):
            qs = spans[q.name]
            for k in self.blocks[: i + 1]:
                if k is not q and q.sees is not None and k.name not in q.sees:
                    continue
                ks = spans[k.name]
                if k is q and q.causal_inside:
                    mask[qs, ks] = torch.ones(q.length, q.length, dtype=torch.bool).tril()
                else:
                    mask[qs, ks] = True
        return mask
