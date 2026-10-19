"""Batches, per-embodiment data sources and a mixture sampler for the cross-embodiment experiments.

A :class:`VLABatch` is what both :class:`SmolVLAPolicy` and :class:`GR00TPolicy` consume. Besides the usual image / state / action
chunk it carries the pieces GR00T N1 / N1.5 need:

* ``embodiment`` - which per-embodiment MLPs to use for each sample;
* ``action_mask`` - which of the zero-padded action dimensions are real for that embodiment;
* ``flow_weight`` - 0 for samples whose actions must not be regressed (action-less video trained with FLARE only);
* ``future_images`` - the frame ``H`` steps ahead, the target of FLARE's future-latent alignment and the input of the latent-action model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pi_family.toy import DemoGenerator, PointWorld
from pi_family.toy.episodes import Episode
from pi_family.toy.world import COLORS, Scene

from open_vlas.embodiments import (
    EMBODIMENTS,
    LATENT_DIM,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    Embodiment,
    embodied_arrays,
)


@dataclass
class VLABatch:
    """One training / inference batch. Shapes: ``images`` (B, 3, S, S), ``state`` (B, MAX_STATE_DIM), ``actions`` (B, H, MAX_ACTION_DIM)."""

    images: torch.Tensor
    state: torch.Tensor
    task: list
    embodiment: torch.Tensor
    actions: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    flow_weight: torch.Tensor | None = None
    future_images: torch.Tensor | None = None
    meta: list | None = None  # per-example scene information (objects, targets), used to build reasoning traces

    @property
    def size(self) -> int:
        return self.images.shape[0]

    def replace(self, **changes) -> VLABatch:
        return dataclasses.replace(self, **changes)


def reach_task(
    world: PointWorld,
    rng: np.random.Generator,
    target: str | None = None,
    colors: Sequence[str] | None = None,
    num_objects: int = 2,
):
    """A ``go to <colour>`` task whose target colour can be forced.

    Returns ``(scene, text, [target], [target])`` in the format :meth:`DemoGenerator.demo` accepts. With ``target=None`` the
    target is drawn from ``colors`` (default: all three).
    """
    pool = list(colors) if colors is not None else list(COLORS)
    target = target or str(rng.choice(pool))
    others = [c for c in COLORS if c != target]
    chosen = [target] + [str(c) for c in rng.permutation(others)[: num_objects - 1]]
    scene = world.sample_scene(rng, colors=chosen)
    return scene, f"go to {target}", [target], [target]


def make_reach_demos(
    generator: DemoGenerator,
    n: int,
    rng: np.random.Generator,
    target_colors: Sequence[str] = tuple(COLORS),
    style: str = "expert",
) -> list:
    """``n`` expert ``go to <colour>`` demonstrations with targets restricted to ``target_colors``."""
    return [
        generator.demo(
            "reach",
            rng,
            style,
            task=reach_task(generator.world, rng, str(rng.choice(list(target_colors)))),
        )
        for _ in range(n)
    ]


class EmbodiedSource:
    """Samples (observation, instruction, action chunk) examples of one embodiment from demonstrations.

    Args:
        world: renders the observations.
        episodes: expert demonstrations (their end-effector actions are converted to ``embodiment``'s action space).
        embodiment: the robot body the data "came from".
        chunk_size: ``H``; also the horizon of the future frame.
        latent_fn: for the ``lapa`` pseudo-embodiment: maps ``(frames_t, frames_t+H)`` batches to latent actions; the latent is
            repeated over the chunk and used as the flow-matching target.
        flow_weight: 0.0 for action-less video that only trains FLARE.
        idle_fraction: share of samples taken after the task is done (so chunks learn to stop).
    """

    def __init__(
        self,
        world: PointWorld,
        episodes: list,
        embodiment: Embodiment,
        chunk_size: int = 16,
        latent_fn: Callable | None = None,
        flow_weight: float = 1.0,
        idle_fraction: float = 0.1,
    ):
        self.world, self.episodes, self.embodiment = world, episodes, embodiment
        self.chunk_size, self.latent_fn = chunk_size, latent_fn
        self.flow_weight, self.idle_fraction = flow_weight, idle_fraction
        self._arrays = (
            [embodied_arrays(ep, embodiment) for ep in episodes] if embodiment.is_robot else None
        )

    def frame(self, ep: Episode, t: int) -> np.ndarray:
        t = int(np.clip(t, 0, len(ep.robot) - 1))
        cue = ep.scene.marked if t < self.world.cue_steps else None
        return self.world.render(ep.robot[t], ep.scene.objects, ep.collected(t), cue)

    def sample_items(self, n: int, rng: np.random.Generator) -> list:
        items = []
        for _ in range(n):
            i = int(rng.integers(len(self.episodes)))
            ep = self.episodes[i]
            if rng.random() < self.idle_fraction:
                t = int(rng.integers(ep.active_steps, ep.length))
            else:
                t = int(rng.integers(0, max(ep.active_steps, 1)))
            items.append(self._example(i, ep, t))
        return items

    def _example(self, i: int, ep: Episode, t: int) -> dict:
        h = self.chunk_size
        item = dict(
            image=self.frame(ep, t),
            future=self.frame(ep, t + h),
            task=ep.task,
            embodiment=self.embodiment.id,
            flow_weight=self.flow_weight,
            meta=dict(objects=ep.scene.objects, targets=ep.targets),
        )
        if self.embodiment.is_robot:
            states, actions = self._arrays[i]
            chunk = actions[t : t + h]
            if (
                len(chunk) < h
            ):  # past the end of the episode: hold still (zero motion, gripper keeps its last value)
                pad = np.zeros((h - len(chunk), actions.shape[1]), np.float32)
                if actions.shape[1] > 2:
                    pad[:, 2:] = actions[-1, 2:]
                chunk = np.concatenate([chunk, pad])
            item.update(state=states[t], actions=chunk)
        return item


def collate(items: list, latent_fn: Callable | None = None) -> VLABatch:
    """Stack examples, zero-pad states / actions to the maximum dimensions and attach masks.

    Items of the ``lapa`` embodiment get their action chunk from ``latent_fn`` (the latent action of ``(image, future)``).
    """
    n = len(items)
    h = next((i["actions"].shape[0] for i in items if "actions" in i), 16)
    state = np.zeros((n, MAX_STATE_DIM), np.float32)
    actions = np.zeros((n, h, MAX_ACTION_DIM), np.float32)
    mask = np.zeros((n, MAX_ACTION_DIM), np.float32)
    for k, it in enumerate(items):
        emb = next(e for e in EMBODIMENTS.values() if e.id == it["embodiment"])
        mask[k, : emb.action_dim] = 1.0
        if "state" in it:
            state[k, : emb.state_dim] = it["state"]
            actions[k, :, : emb.action_dim] = it["actions"]
    images = torch.from_numpy(np.stack([i["image"] for i in items]))
    future = torch.from_numpy(np.stack([i["future"] for i in items]))
    batch = VLABatch(
        images=images,
        state=torch.from_numpy(state),
        task=[i["task"] for i in items],
        embodiment=torch.tensor([i["embodiment"] for i in items]),
        actions=torch.from_numpy(actions),
        action_mask=torch.from_numpy(mask),
        flow_weight=torch.tensor([i["flow_weight"] for i in items], dtype=torch.float32),
        future_images=future,
        meta=[i.get("meta") for i in items],
    )
    # samples that only train FLARE (flow_weight 0) need no action label, so no latent action is computed for them
    lapa = (batch.embodiment == EMBODIMENTS["lapa"].id) & (batch.flow_weight > 0)
    if lapa.any():
        if latent_fn is None:
            raise ValueError("a batch with 'lapa' samples needs a latent_fn")
        with torch.no_grad():
            z = latent_fn(images[lapa], future[lapa])  # (n_lapa, LATENT_DIM)
        batch.actions[lapa] = z[:, None, :].expand(-1, h, -1).to(batch.actions)
    return batch


class EmbodiedMixture:
    """A weighted mixture of :class:`EmbodiedSource` objects: GR00T's training "data pyramid", in miniature.

    ``sample`` has the signature :class:`pi_family.PiTrainer` expects (``sample(batch_size, rng)``).
    """

    def __init__(
        self,
        sources: Sequence[EmbodiedSource],
        weights: Sequence[float] | None = None,
        latent_fn: Callable | None = None,
    ):
        self.sources = list(sources)
        w = np.asarray(weights if weights is not None else np.ones(len(sources)), dtype=float)
        self.weights = w / w.sum()
        self.latent_fn = latent_fn

    def sample(self, batch_size: int, rng: np.random.Generator) -> VLABatch:
        counts = rng.multinomial(batch_size, self.weights)
        items = []
        for src, c in zip(self.sources, counts):
            if c:
                items.extend(src.sample_items(int(c), rng))
        order = rng.permutation(len(items))
        return collate([items[i] for i in order], self.latent_fn)


__all__ = [
    "EmbodiedMixture",
    "EmbodiedSource",
    "LATENT_DIM",
    "Scene",
    "VLABatch",
    "collate",
    "make_reach_demos",
    "reach_task",
]
