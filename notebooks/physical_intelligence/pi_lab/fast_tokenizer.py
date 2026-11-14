"""FAST: frequency-space action tokenization (DCT → quantise → byte-pair encoding)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np


class FASTTokenizer:
    """Compress an action chunk into a short sequence of discrete tokens (Pertsch et al., 2025).

    Naive per-step binning makes one token per dimension per time step, and neighbouring
    tokens are almost perfectly correlated, so next-token prediction learns little. FAST:

    1. **DCT** each action dimension over time (smooth chunks → energy in few low frequencies);
    2. **scale and round** the coefficients (``round(γ · c)``): most high-frequency terms become 0;
    3. **flatten frequency-major** (all dimensions' DC terms, then the first harmonics, …);
    4. **BPE** merges frequent integer pairs (especially runs of zeros) into single tokens.

    Decoding inverts each step; rounding is the only loss.

    Args:
        scale: quantisation scale γ (higher = finer = more tokens).
        clip: coefficients are clipped to ``[-clip, clip]`` before BPE (bounded base alphabet).
    """

    def __init__(self, scale: float = 10.0, clip: int = 30) -> None:
        self.scale, self.clip = scale, clip
        self.merges: list[tuple[int, int]] = []
        self.horizon = self.dims = 0

    # ------------------------------------------------------------------ DCT
    @staticmethod
    def dct_matrix(n: int) -> np.ndarray:
        """Orthonormal DCT-II matrix (its transpose is the inverse)."""
        k, i = np.arange(n)[:, None], np.arange(n)[None]
        m = np.cos(np.pi * (i + 0.5) * k / n) * np.sqrt(2 / n)
        m[0] /= np.sqrt(2)
        return m

    def coefficients(self, chunk: np.ndarray) -> np.ndarray:
        """``(H, D)`` chunk → quantised integer DCT coefficients ``(H, D)``."""
        c = self.dct_matrix(len(chunk)) @ np.asarray(chunk, dtype=np.float64)
        return np.clip(np.round(c * self.scale), -self.clip, self.clip).astype(int)

    # ------------------------------------------------------------------ symbols
    @property
    def base_size(self) -> int:
        """Number of base symbols (one per integer in ``[-clip, clip]``)."""
        return 2 * self.clip + 1

    @property
    def vocab_size(self) -> int:
        """Base symbols + learned merges."""
        return self.base_size + len(self.merges)

    def symbols(self, chunk: np.ndarray) -> list[int]:
        """Frequency-major flattened coefficients as base-symbol ids."""
        return [int(v) + self.clip for v in self.coefficients(chunk).reshape(-1)]

    # ------------------------------------------------------------------ BPE
    def fit(self, chunks: Iterable[np.ndarray], merges: int = 200) -> FASTTokenizer:
        """Learn ``merges`` BPE merges on a corpus of chunks (all of one shape)."""
        corpus = [self.symbols(c) for c in chunks]
        self.merges = []
        for _ in range(merges):
            pairs = Counter(p for seq in corpus for p in zip(seq, seq[1:], strict=False))
            if not pairs:
                break
            (a, b), count = pairs.most_common(1)[0]
            if count < 2:
                break
            new = self.base_size + len(self.merges)
            self.merges.append((a, b))
            corpus = [self._merge(seq, a, b, new) for seq in corpus]
        return self

    @staticmethod
    def _merge(seq: list[int], a: int, b: int, new: int) -> list[int]:
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(new)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def encode(self, chunk: np.ndarray) -> list[int]:
        """Chunk → FAST token ids (``0 … vocab_size-1``)."""
        self.horizon, self.dims = np.asarray(chunk).shape
        seq = self.symbols(chunk)
        for k, (a, b) in enumerate(self.merges):
            seq = self._merge(seq, a, b, self.base_size + k)
        return seq

    def decode(self, tokens: Sequence[int], horizon: int, dims: int) -> np.ndarray:
        """FAST token ids → ``(horizon, dims)`` chunk (invalid lengths are padded/truncated)."""
        seq = list(tokens)
        for k in range(len(self.merges) - 1, -1, -1):
            a, b = self.merges[k]
            new = self.base_size + k
            expanded = []
            for t in seq:
                expanded.extend((a, b) if t == new else (t,))
            seq = expanded
        values = np.array([s - self.clip for s in seq if s < self.base_size], dtype=np.float64)
        values = np.resize(np.pad(values, (0, max(0, horizon * dims - len(values)))), horizon * dims)
        coeffs = values.reshape(horizon, dims) / self.scale
        return self.dct_matrix(horizon).T @ coeffs
