"""Tokenisers for the π family: words, discretised state (π0.5), FAST action tokens (π0.5 / π0.7).

All of them write into **one shared id space** (:class:`PiVocabulary`), the way PaliGemma / Gemma re-purpose the last
entries of their text vocabulary for state bins and action tokens:

``[ special + words | state bins | FAST tokens ]``
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sep>", "<boa>", "<eoa>")

# Words of the toy point-world (see ``pi_family.toy``). Synonyms such as "crimson" exist in the vocabulary but are
# never used in robot data: only the world model's "web" data mentions them (π0.7 notebook).
DEFAULT_WORDS = (
    (
        "go to red green blue any collect then the marked object crimson emerald azure "
        "task subtask speed quality mistake true false control mode joint ee state where is left right top bottom"
    ).split()
    + [str(n) for n in range(0, 101, 10)]
    + [str(n) for n in range(1, 6)]
)


class PiVocabulary:
    """Shared id space. ``word ids`` first, then ``state_bins`` state ids, then ``fast_size`` FAST ids."""

    def __init__(
        self, words: Sequence[str] = DEFAULT_WORDS, state_bins: int = 0, fast_size: int = 0
    ):
        unique = list(dict.fromkeys(list(SPECIAL_TOKENS) + list(words)))
        self.words = unique
        self.word_to_id = {w: i for i, w in enumerate(unique)}
        self.state_bins = state_bins
        self.fast_size = fast_size
        self.state_offset = len(unique)
        self.fast_offset = self.state_offset + state_bins

    @property
    def size(self) -> int:
        return self.fast_offset + self.fast_size

    def __getitem__(self, word: str) -> int:
        return self.word_to_id[word]

    # ----------------------------------------------------------------------------------------- words
    def encode(self, text: str) -> list[int]:
        """Lower-case, split on whitespace and punctuation. Unknown words raise (the toy vocabulary is closed)."""
        tokens = [t for t in re.split(r"[\s.,:;]+", text.lower()) if t]
        missing = [t for t in tokens if t not in self.word_to_id]
        if missing:
            raise KeyError(f"words not in vocabulary: {missing}")
        return [self.word_to_id[t] for t in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i < self.state_offset:
                out.append(self.words[i])
            elif i < self.fast_offset:
                out.append(f"<s{i - self.state_offset}>")
            else:
                out.append(f"<f{i - self.fast_offset}>")
        return " ".join(out)

    # ----------------------------------------------------------------------------------------- other ranges
    def state_token(self, bins: torch.Tensor) -> torch.Tensor:
        return bins + self.state_offset

    def fast_token(self, local: Sequence[int]) -> list[int]:
        return [self.fast_offset + int(t) for t in local]

    def is_fast(self, ids: torch.Tensor) -> torch.Tensor:
        return ids >= self.fast_offset

    def word_ids(self) -> range:
        return range(self.state_offset)


class StateDiscretizer:
    """π0.5 feeds proprioception to the VLM as text: each normalised dimension is cut into ``num_bins`` bins."""

    def __init__(self, num_bins: int, low: float = 0.0, high: float = 1.0):
        self.num_bins, self.low, self.high = num_bins, low, high

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        unit = ((state - self.low) / (self.high - self.low)).clamp(0.0, 1.0 - 1e-6)
        return (unit * self.num_bins).long()

    def decode(self, bins: torch.Tensor) -> torch.Tensor:
        return self.low + (bins.float() + 0.5) / self.num_bins * (self.high - self.low)


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``D`` (n, n): ``coefficients = D @ signal``, ``signal = D.T @ coefficients``."""
    k = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(n, dtype=torch.float64)[None]
    d = torch.cos(math.pi / n * (i + 0.5) * k) * math.sqrt(2.0 / n)
    d[0] /= math.sqrt(2.0)
    return d.float()


class FastActionTokenizer:
    """FAST (Pertsch et al., 2025, arXiv:2501.09747): Frequency-space Action Sequence Tokenisation.

    1. DCT along time, per action dimension: smooth chunks put almost all energy in the first coefficients.
    2. Quantise ``round(scale · coefficient)`` and clip to ``[-K, K]``: most high-frequency values become 0.
    3. Flatten **frequency-major** (all dims of coefficient 0, then of coefficient 1, ...), so the many trailing zeros
       form long runs.
    4. Byte-pair-encode the integer sequence: frequent pairs (e.g. ``0 0``) are merged into single tokens.

    Unlike the original we fit a small BPE ourselves and skip the byte-level fallback; decoding is the exact inverse
    of steps 3-4 and the inverse DCT, so the only loss is the quantisation in step 2.
    """

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        scale: float = 6.0,
        max_coefficient: int = 12,
        num_merges: int = 48,
        order: str = "frequency",
    ):
        if order not in ("frequency", "dimension"):
            raise ValueError("order must be 'frequency' (all dims of coefficient 0, then of 1, ...) or 'dimension'")
        self.order = order
        self.chunk_size, self.action_dim = chunk_size, action_dim
        self.scale, self.max_coefficient, self.num_merges = scale, max_coefficient, num_merges
        self.dct = dct_matrix(chunk_size)
        self.base_size = 2 * max_coefficient + 1
        self.merges: list[tuple[int, int]] = []  # merge r creates token base_size + r

    @property
    def vocab_size(self) -> int:
        return self.base_size + self.num_merges

    # ----------------------------------------------------------------------------------------- steps 1-3
    def quantize(self, chunk: torch.Tensor) -> torch.Tensor:
        """(H, A) actions → (H, A) integer DCT coefficients."""
        coeff = self.dct @ chunk.float()
        return (
            torch.round(coeff * self.scale)
            .clamp(-self.max_coefficient, self.max_coefficient)
            .long()
        )

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return self.dct.T @ (q.float() / self.scale)

    def _symbols(self, q: torch.Tensor) -> list[int]:
        flat = q.reshape(-1) if self.order == "frequency" else q.T.reshape(-1)  # (H, A) row-major == frequency-major
        return (flat + self.max_coefficient).tolist()

    # ----------------------------------------------------------------------------------------- step 4 (BPE)
    @staticmethod
    def _merge(seq: list[int], pair: tuple[int, int], new: int) -> list[int]:
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                out.append(new)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def fit(self, chunks: torch.Tensor) -> FastActionTokenizer:
        """Learn ``num_merges`` BPE merges from a set of action chunks (N, H, A)."""
        sequences = [self._symbols(self.quantize(c)) for c in chunks]
        self.merges = []
        for r in range(self.num_merges):
            pairs = Counter()
            for seq in sequences:
                pairs.update(zip(seq, seq[1:]))
            if not pairs:
                break
            pair = pairs.most_common(1)[0][0]
            new = self.base_size + r
            self.merges.append(pair)
            sequences = [self._merge(seq, pair, new) for seq in sequences]
        return self

    def encode(self, chunk: torch.Tensor) -> list[int]:
        """(H, A) actions → list of local FAST ids in ``[0, vocab_size)``.

        BPE merges are applied in learned order. Each symbol is mapped to one unicode character so every merge is a
        ``str.replace`` - identical to the left-to-right, non-overlapping merge of :meth:`_merge`, at C speed.
        """
        text = "".join(chr(0x100 + t) for t in self._symbols(self.quantize(chunk)))
        for r, (a, b) in enumerate(self.merges):
            text = text.replace(chr(0x100 + a) + chr(0x100 + b), chr(0x100 + self.base_size + r))
        return [ord(ch) - 0x100 for ch in text]

    def expanded_length(self, tokens: Sequence[int]) -> int:
        """Number of quantised coefficients the tokens expand to (``chunk_size · action_dim`` for a valid sequence)."""
        seq = [int(t) for t in tokens if 0 <= int(t) < self.vocab_size]
        for r in reversed(range(len(self.merges))):
            new = self.base_size + r
            seq = [x for t in seq for x in ((self.merges[r][0], self.merges[r][1]) if t == new else (t,))]
        return len(seq)

    def is_valid(self, tokens: Sequence[int]) -> bool:
        """True if the tokens decode to exactly one full chunk (autoregressive sampling can produce too few or too many)."""
        return self.expanded_length(tokens) == self.chunk_size * self.action_dim

    def decode(self, tokens: Sequence[int]) -> torch.Tensor:
        """Local FAST ids → (H, A) actions. Robust to wrong lengths (pads with zeros / truncates), which matters
        when the ids come from autoregressive decoding."""
        seq = [int(t) for t in tokens if 0 <= int(t) < self.vocab_size]
        for r in reversed(range(len(self.merges))):
            new, (a, b) = self.base_size + r, self.merges[r]
            expanded = []
            for t in seq:
                expanded.extend((a, b) if t == new else (t,))
            seq = expanded
        n = self.chunk_size * self.action_dim
        seq = (seq + [self.max_coefficient] * n)[:n]  # symbol K is the integer 0
        q = torch.tensor(seq)
        q = (q.view(self.chunk_size, self.action_dim) if self.order == "frequency" else q.view(self.action_dim, self.chunk_size).T) - self.max_coefficient
        return self.dequantize(q)


@dataclass
class EpisodeMetadata:
    """π0.7 episode metadata. ``None`` fields are omitted from the prompt (that is how dropout is expressed)."""

    speed: int | None = None  # episode length in steps, binned (π0.7: bins of 500 steps at 50 Hz)
    quality: int | None = None  # 1..5, 5 = best
    mistake: bool | None = None  # did the robot make a mistake in this segment?

    def prompt(self) -> str:
        parts = []
        if self.speed is not None:
            parts.append(f"speed {self.speed}")
        if self.quality is not None:
            parts.append(f"quality {self.quality}")
        if self.mistake is not None:
            parts.append(f"mistake {'true' if self.mistake else 'false'}")
        return " ; ".join(parts)

    def is_empty(self) -> bool:
        return self.speed is None and self.quality is None and self.mistake is None


def bin_speed(num_steps: int, bin_size: int = 10, max_value: int = 100) -> int:
    """Round an episode length to the nearest ``bin_size`` (π0.7 bins 1750-2250 to "2000")."""
    return int(min(max_value, max(bin_size, bin_size * round(num_steps / bin_size))))
