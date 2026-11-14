"""A tiny word-level vocabulary shared by prompts, subtasks, metadata and FAST tokens."""

from __future__ import annotations


class ToyVocabulary:
    """Fixed word list + optional discrete-state bins + a FAST token range.

    Real π models use the Gemma tokenizer (256k entries); the toy needs only a few dozen words.
    Discrete-state tokens (π0.5) and FAST action tokens live in reserved id ranges after the
    words, exactly as PaliGemma's last vocabulary entries are re-purposed for FAST in π0-FAST.

    Args:
        state_bins: bins per state dimension for discretised-state tokens (0 = none).
        state_dims: number of state dimensions.
        fast_vocab: number of FAST (BPE) action tokens (0 = none).
    """

    WORDS = ["<pad>", "<bos>", "<eos>", "go", "to", "red", "blue", "green", "then", "visit",
             "done", "speed:", "slow", "normal", "fast", "quality:", "1", "2", "3", "4", "5",
             "mistake:", "yes", "no", "subtask:", "state:", "action:", "mode:", "ee", "joint"]

    def __init__(self, state_bins: int = 0, state_dims: int = 2, fast_vocab: int = 0) -> None:
        self.stoi = {w: i for i, w in enumerate(self.WORDS)}
        self.state_bins = state_bins
        self.state_offset = len(self.WORDS)
        self.fast_offset = self.state_offset + state_bins * state_dims
        self.size = self.fast_offset + fast_vocab

    def encode(self, text: str) -> list[int]:
        """Word ids of a space-separated string."""
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        """Readable string (state/FAST ids shown as ``<s..>`` / ``<f..>``)."""
        out = []
        for i in ids:
            if i < self.state_offset:
                out.append(self.WORDS[i])
            elif i < self.fast_offset:
                out.append(f"<s{i - self.state_offset}>")
            else:
                out.append(f"<f{i - self.fast_offset}>")
        return " ".join(out)

    def state_tokens(self, state: list[float] | tuple[float, ...]) -> list[int]:
        """π0.5-style discretised state: each coordinate in ``[0, 1]`` → one bin token."""
        if self.state_bins == 0:
            raise ValueError("vocabulary built without state bins")
        tokens = []
        for d, value in enumerate(state):
            b = min(self.state_bins - 1, max(0, int(value * self.state_bins)))
            tokens.append(self.state_offset + d * self.state_bins + b)
        return tokens
