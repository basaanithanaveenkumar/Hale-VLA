"""Batches of (observation, prompt, action chunk) training examples cut out of demonstrations."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch

from pi_family.tokenizers import EpisodeMetadata
from pi_family.toy.episodes import Episode
from pi_family.toy.world import PointWorld


@dataclass
class PiBatch:
    """Everything any π generation may read. Each policy uses the fields it understands and ignores the rest.

    Shapes: ``images`` (B, 3, S, S) current view; ``state`` (B, 2); ``actions`` (B, H, 2) in each example's control
    mode; ``history`` (B, T, 3, S, S) with frame 0 = current; ``subgoal`` (B, 3, S, S) with ``subgoal_valid`` (B,).
    """

    images: torch.Tensor
    state: torch.Tensor
    task: list
    subtask: list
    actions: torch.Tensor | None = None
    history: torch.Tensor | None = None
    history_valid: torch.Tensor | None = None
    state_history: torch.Tensor | None = None
    subgoal: torch.Tensor | None = None
    subgoal_valid: torch.Tensor | None = None
    metadata: list = field(default_factory=list)
    control_mode: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.images.shape[0]

    def replace(self, **changes) -> PiBatch:
        return dataclasses.replace(self, **changes)

    def without_metadata(self) -> PiBatch:
        """The "unconditional" context used by π0.7's classifier-free guidance on metadata."""
        return self.replace(metadata=[EpisodeMetadata() for _ in range(self.size)])


class ChunkDataset:
    """Samples random time steps of random episodes and renders what the robot saw then.

    ``subgoal`` follows π0.7 §VI-C: with probability 0.25 the end-of-segment frame, otherwise a frame uniformly
    0..``subgoal_horizon`` steps in the future.
    """

    def __init__(
        self,
        episodes: list,
        world: PointWorld,
        chunk_size: int,
        history_frames: int = 1,
        history_stride: int = 2,
        subgoal_horizon: int = 8,
        idle_fraction: float = 0.1,
    ):
        self.episodes, self.world = episodes, world
        self.chunk_size = chunk_size
        self.history_frames, self.history_stride = history_frames, history_stride
        self.subgoal_horizon = subgoal_horizon
        self.idle_fraction = idle_fraction  # share of samples taken after the task is done

    def __len__(self) -> int:
        return len(self.episodes)

    # ------------------------------------------------------------------------------------------ rendering
    def frame(self, ep: Episode, t: int) -> np.ndarray:
        t = int(np.clip(t, 0, len(ep.robot) - 1))
        cue = ep.scene.marked if t < self.world.cue_steps else None
        return self.world.render(ep.robot[t], ep.scene.objects, ep.collected(t), cue)

    def chunk(self, ep: Episode, t: int) -> np.ndarray:
        """Actions t .. t+H-1 in the episode's control mode, zero-padded past the end."""
        a = ep.actions[t : t + self.chunk_size]
        a = np.concatenate([a, np.zeros((self.chunk_size - len(a), 2), np.float32)])
        return self.world.from_ee(a, ep.control_mode)

    def example(self, ep: Episode, t: int, rng: np.random.Generator) -> dict:
        ages = [k * self.history_stride for k in range(self.history_frames)]
        future = (
            int(ep.segment_end[t])
            if rng.random() < 0.25
            else t + int(rng.integers(0, self.subgoal_horizon + 1))
        )
        window = ep.mistake[t : t + self.chunk_size]
        meta = dataclasses.replace(
            ep.metadata, mistake=bool(window.any()) if len(window) else False
        )
        return dict(
            image=self.frame(ep, t),
            state=ep.robot[t],
            actions=self.chunk(ep, t),
            history=[self.frame(ep, t - a) for a in ages],
            history_valid=[t - a >= 0 for a in ages],
            state_history=[ep.robot[max(t - a, 0)] for a in ages],
            subgoal=self.frame(ep, min(future, len(ep.robot) - 1)),
            task=ep.task,
            subtask=ep.subtask[t],
            metadata=meta,
            control_mode=ep.control_mode,
        )

    def sample_world_model(self, batch_size: int, rng: np.random.Generator):
        """(observation, subtask text, subgoal) triples for :class:`pi_family.SubgoalWorldModel`: the goal is the frame
        at the end of the current subtask segment (π0.7 §V-B), like ``g*_t = o_{t_end}``."""
        images, texts, goals = [], [], []
        for _ in range(batch_size):
            ep = self.episodes[rng.integers(len(self.episodes))]
            t = int(rng.integers(0, max(ep.active_steps, 1)))
            images.append(self.frame(ep, t))
            texts.append(ep.subtask[t])
            goals.append(self.frame(ep, int(ep.segment_end[t])))
        return torch.from_numpy(np.stack(images)), texts, torch.from_numpy(np.stack(goals))

    # ------------------------------------------------------------------------------------------ batching
    def sample(self, batch_size: int, rng: np.random.Generator) -> PiBatch:
        items = []
        for _ in range(batch_size):
            ep = self.episodes[rng.integers(len(self.episodes))]
            if rng.random() < self.idle_fraction:
                t = int(rng.integers(ep.active_steps, ep.length))
            else:
                t = int(rng.integers(0, max(ep.active_steps, 1)))
            items.append(self.example(ep, t, rng))
        return collate(items)


def collate(items: list) -> PiBatch:
    def t(key):
        return torch.from_numpy(np.stack([np.asarray(i[key], dtype=np.float32) for i in items]))

    return PiBatch(
        images=t("image"),
        state=t("state"),
        actions=t("actions"),
        task=[i["task"] for i in items],
        subtask=[i["subtask"] for i in items],
        history=t("history"),
        history_valid=torch.tensor([i["history_valid"] for i in items]),
        state_history=t("state_history"),
        subgoal=t("subgoal"),
        subgoal_valid=torch.ones(len(items), dtype=torch.bool),
        metadata=[i["metadata"] for i in items],
        control_mode=[i["control_mode"] for i in items],
    )
