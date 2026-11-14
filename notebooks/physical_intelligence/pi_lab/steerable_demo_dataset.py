"""Mixed-quality demonstrations annotated with metadata, memory and subgoals (π0.7-style data)."""

from __future__ import annotations

import numpy as np
import torch

from pi_lab.fast_tokenizer import FASTTokenizer
from pi_lab.toy_reach_env import ToyReachEnv
from pi_lab.toy_scene import ToyScene
from pi_lab.toy_vocabulary import ToyVocabulary

SPEEDS = {"slow": 0.6, "normal": 1.0, "fast": 1.6}


class SteerableDemoDataset:
    """Demonstrations of every quality, kept and labelled instead of filtered out.

    π0.7 trains on demonstrations, failed autonomous rollouts, interventions and human
    video, i.e. data of very uneven quality. Instead of discarding the bad parts, every episode
    is described by **metadata** (speed, quality 1–5, whether a mistake happened). At test time
    the prompt asks for ``speed: fast quality: 5 mistake: no``, and classifier-free guidance
    pushes harder in that direction. Here:

    * **speed**: step-size multiplier (slow / normal / fast);
    * **quality**: Gaussian jitter on every action, larger for lower quality (5 = clean);
    * **mistake**: the episode starts with a short excursion away from the target.

    Each example also has a **memory** (3 earlier frames and states, 2 steps apart), an
    optional **subgoal image** (the scene re-rendered with the robot 6 steps ahead on the
    *executed* trajectory) and **prompt dropout** masks.

    Args:
        env: toy environment.
        vocab: vocabulary with a FAST range.
        fast: fitted :class:`FASTTokenizer`.
        scenes: episodes to generate.
        per_scene: examples per episode.
        fast_len: FAST padding length.
        seed: dropout RNG seed.
    """

    HISTORY, STRIDE, SUBGOAL_AHEAD = 3, 2, 6

    def __init__(self, env: ToyReachEnv, vocab: ToyVocabulary, fast: FASTTokenizer,
                 scenes: int = 2000, per_scene: int = 4, fast_len: int = 16, seed: int = 0) -> None:
        self.env, self.vocab = env, vocab
        rng = np.random.default_rng(seed)
        keys = ("image", "history", "states", "subgoal", "task", "meta", "actions", "fast",
                "fast_mask", "has_history", "has_subgoal")
        rows = {k: [] for k in keys}
        self.labels: list[dict[str, str]] = []
        for _ in range(scenes):
            colour = "red" if rng.random() < 0.5 else "blue"
            scene = env.sample_scene(blocked=colour)
            label = {"speed": rng.choice(list(SPEEDS)), "quality": str(rng.integers(1, 6)),
                     "mistake": "yes" if rng.random() < 0.25 else "no"}
            traj = self.trajectory(scene, colour, label, rng)
            for i in rng.choice(len(traj) - 1, size=min(per_scene, len(traj) - 1), replace=False):
                ex = self.example(scene, colour, traj, int(i), label, fast, fast_len)
                drop_meta = rng.random() < 0.15
                meta = self.meta_tokens(label, drop=drop_meta,
                                        drop_fields={f for f in label if rng.random() < 0.05})
                ex.update(meta=meta, has_history=rng.random() >= 0.3, has_subgoal=rng.random() < 0.25)
                for k in keys:
                    rows[k].append(ex[k])
                self.labels.append(label)
        self.data = {k: torch.tensor(np.array(v)) for k, v in rows.items()}
        for k in ("states", "actions"):
            self.data[k] = self.data[k].float()

    # ------------------------------------------------------------------ episode construction
    def trajectory(self, scene: ToyScene, colour: str, label: dict[str, str],
                   rng: np.random.Generator) -> np.ndarray:
        """Executed positions (start included) for one labelled episode."""
        start = scene.robot.copy()
        points = [start]
        if label["mistake"] == "yes":  # wander away from the goal for a few steps first
            away = start - scene.targets[colour]
            away = away / (np.linalg.norm(away) + 1e-9)
            for _ in range(3):
                points.append(np.clip(points[-1] + away * self.env.step, 0.03, 0.97))
        s = scene.copy()
        s.robot = points[-1]
        path = self.env.expert_path(s, colour, speed=SPEEDS[label["speed"]])
        sigma = (5 - int(label["quality"])) * 0.012
        jitter = rng.normal(0, sigma, size=path.shape) if sigma else 0.0
        return np.concatenate([np.array(points), path + jitter], axis=0)

    def example(self, scene: ToyScene, colour: str, traj: np.ndarray, i: int,
                label: dict[str, str], fast: FASTTokenizer, fast_len: int) -> dict:
        """Observation, memory, subgoal and targets at trajectory index ``i``."""
        env = self.env
        past = [traj[max(0, i - self.STRIDE * k)] for k in range(1, self.HISTORY + 1)]
        s = scene.copy()
        s.robot = traj[i]
        deltas = np.diff(traj[i: i + env.horizon + 1], axis=0)
        deltas = np.concatenate([deltas, np.zeros((env.horizon - len(deltas), 2))]) / env.ACTION_SCALE
        chunk = deltas.astype(np.float32)
        tokens = [self.vocab.fast_offset + t for t in fast.encode(chunk)][:fast_len]
        pad = self.vocab.stoi["<pad>"]
        return {
            "image": env.render(s),
            "history": np.stack([env.render(s, robot_at=p) for p in past]),
            "states": np.stack([traj[i], *past]),
            "subgoal": env.render(s, robot_at=traj[min(len(traj) - 1, i + self.SUBGOAL_AHEAD)]),
            "task": self.vocab.encode(f"go to {colour}"),
            "actions": chunk,
            "fast": tokens + [pad] * (fast_len - len(tokens)),
            "fast_mask": [True] * len(tokens) + [False] * (fast_len - len(tokens)),
        }

    def meta_tokens(self, label: dict[str, str], drop: bool = False,
                    drop_fields: set[str] | None = None) -> list[int]:
        """``speed: X quality: Y mistake: Z mode: ee`` (8 tokens); dropped parts become ``<pad>``."""
        pad = self.vocab.stoi["<pad>"]
        out = []
        for field in ("speed", "quality", "mistake"):
            words = [f"{field}:", label[field]]
            gone = drop or (drop_fields is not None and field in drop_fields)
            out += [pad, pad] if gone else self.vocab.encode(" ".join(words))
        return out + self.vocab.encode("mode: ee")  # control mode is never dropped (as in π0.7)

    def __len__(self) -> int:
        return len(self.data["actions"])

    def batch(self, size: int) -> dict[str, torch.Tensor]:
        """A random minibatch."""
        idx = torch.randint(len(self), (size,))
        return {k: v[idx] for k, v in self.data.items()}
