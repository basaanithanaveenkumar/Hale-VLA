"""Demonstrations for "visit A then B", labelled with the current subtask (π0.5-style data)."""

from __future__ import annotations

import numpy as np
import torch

from pi_lab.fast_tokenizer import FASTTokenizer
from pi_lab.toy_reach_env import ToyReachEnv
from pi_lab.toy_vocabulary import ToyVocabulary


class MultiStageDemoDataset:
    """Two-leg episodes with high-level labels, FAST targets and discretised state.

    Every example carries what π0.5 trains on at once: the image, the overall instruction
    ("visit red then blue"), the **discretised state** as text tokens, the **subtask** label
    ("go to red", the high-level target), the continuous action chunk (flow-matching target)
    and the same chunk as **FAST tokens** (cross-entropy target). A target disappears from the
    image once visited, so the image alone tells which leg the robot is on.

    Args:
        env: toy environment.
        vocab: vocabulary with ``state_bins`` and ``fast_vocab`` set.
        fast: a fitted :class:`FASTTokenizer`.
        scenes: episodes to generate.
        per_leg: examples drawn per leg.
        fast_len: FAST sequences are right-padded (``<pad>``) to this length.
    """

    def __init__(self, env: ToyReachEnv, vocab: ToyVocabulary, fast: FASTTokenizer,
                 scenes: int = 1500, per_leg: int = 3, fast_len: int = 24) -> None:
        rows = {k: [] for k in ("images", "instruction", "state_tokens", "state", "subtask",
                                "actions", "fast", "fast_mask")}
        for _ in range(scenes):
            scene = env.sample_scene(obstacle_prob=0.3)
            order = ["red", "blue"] if env.rng.random() < 0.5 else ["blue", "red"]
            for leg, colour in enumerate(order):
                if leg == 1:
                    scene.visited.add(order[0])
                path = env.expert_path(scene, colour)
                starts = np.concatenate([scene.robot[None], path[:-1]], axis=0)
                for i in env.rng.choice(len(starts), size=min(per_leg, len(starts)), replace=False):
                    s = scene.copy()
                    s.robot = starts[i]
                    chunk = env.chunk_from_path(starts[i], path[i:])
                    tokens = [vocab.fast_offset + t for t in fast.encode(chunk)][:fast_len]
                    rows["images"].append(env.render(s))
                    rows["instruction"].append(vocab.encode(f"visit {order[0]} then {order[1]}"))
                    rows["state_tokens"].append(vocab.state_tokens(starts[i]))
                    rows["state"].append(starts[i])
                    rows["subtask"].append(vocab.encode(f"go to {colour}"))
                    rows["actions"].append(chunk)
                    rows["fast"].append(tokens + [vocab.stoi["<pad>"]] * (fast_len - len(tokens)))
                    rows["fast_mask"].append([1] * len(tokens) + [0] * (fast_len - len(tokens)))
                scene.robot = path[-1]
        self.data = {k: torch.tensor(np.array(v)) for k, v in rows.items()}
        self.data["state"] = self.data["state"].float()
        self.data["fast_mask"] = self.data["fast_mask"].bool()

    def __len__(self) -> int:
        return len(self.data["actions"])

    def batch(self, size: int) -> dict[str, torch.Tensor]:
        """A random minibatch."""
        idx = torch.randint(len(self), (size,))
        return {k: v[idx] for k, v in self.data.items()}
