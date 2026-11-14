"""One contiguous run of tokens in a block-wise attention mask."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskBlock:
    """A contiguous run of tokens.

    Attributes:
        name: label for plots (``"image+text"``, ``"state"``, ``"actions"`` …).
        length: number of tokens.
        causal_inside: tokens attend causally within the block (text generation).
        sees: names of the earlier blocks this block may attend to (``None`` = all earlier).
    """

    name: str
    length: int
    causal_inside: bool = False
    sees: tuple[str, ...] | None = None
