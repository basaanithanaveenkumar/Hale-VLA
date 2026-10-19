"""Closed-loop evaluation: run a policy in the point world, chunk by chunk, possibly with inference delay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from pi_family.tokenizers import EpisodeMetadata
from pi_family.toy.dataset import PiBatch
from pi_family.toy.world import PointWorld, Scene


@dataclass
class Scenario:
    """One evaluation episode: the initial scene, the task text and how success is judged."""

    scene: Scene
    task: str
    targets: list  # visiting order
    alternatives: list  # any of these counts as the first target ("go to any")
    kind: str
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)
    control_mode: str = "ee"


@dataclass
class RolloutResult:
    """Per-scenario outcomes plus the executed trajectories (for plotting)."""

    success: np.ndarray  # (N,) bool
    steps: np.ndarray  # (N,) steps until success (max_steps if failed)
    wrong_visits: np.ndarray  # (N,) touched a non-target object first
    path_length: np.ndarray  # (N,) travelled distance
    trajectories: np.ndarray  # (N, steps + 1, 2)
    actions: np.ndarray  # (N, steps, 2) executed actions (in each scenario's control mode)
    replan_steps: list  # time steps at which a new chunk took over
    reached: list  # first object each scenario touched (or None)

    @property
    def success_rate(self) -> float:
        return float(self.success.mean())

    def summary(self) -> dict:
        ok = self.success
        return dict(
            success=round(float(ok.mean()), 3),
            wrong_first=round(float(self.wrong_visits.mean()), 3),
            steps=round(float(self.steps[ok].mean()) if ok.any() else float("nan"), 1),
            path=round(float(self.path_length[ok].mean()) if ok.any() else float("nan"), 3),
        )


PolicyFn = Callable[[PiBatch, torch.Tensor | None], torch.Tensor]


class ClosedLoopEvaluator:
    """Runs a batch of scenarios in lock-step.

    Every ``exec_horizon`` steps the policy is queried with the current observation. With ``delay = d`` the new
    chunk only arrives ``d`` steps later (simulated inference latency): meanwhile the *old* chunk keeps executing,
    and the first ``d`` actions of the new chunk are skipped because their time has passed. When
    ``pass_committed`` is set, the policy receives those ``d`` already-committed actions (real-time chunking).
    """

    def __init__(
        self,
        world: PointWorld,
        chunk_size: int,
        exec_horizon: int = 8,
        max_steps: int = 60,
        history_frames: int = 1,
        history_stride: int = 2,
    ):
        self.world, self.chunk_size, self.exec_horizon = world, chunk_size, exec_horizon
        self.max_steps, self.history_frames, self.history_stride = (
            max_steps,
            history_frames,
            history_stride,
        )

    def _observe(self, scenarios, robots, collected, frames, t, subtasks) -> PiBatch:
        n = len(scenarios)
        ages = [k * self.history_stride for k in range(self.history_frames)]
        hist = [[frames[i][max(t - a, 0)] for a in ages] for i in range(n)]
        return PiBatch(
            images=torch.from_numpy(np.stack([frames[i][t] for i in range(n)])),
            state=torch.from_numpy(np.stack(robots[:, t])).float(),
            task=[s.task for s in scenarios],
            subtask=list(subtasks),
            history=torch.from_numpy(np.stack([np.stack(h) for h in hist])),
            history_valid=torch.tensor([[t - a >= 0 for a in ages]] * n),
            state_history=torch.from_numpy(
                np.stack([[robots[i, max(t - a, 0)] for a in ages] for i in range(n)])
            ).float(),
            subgoal=None,
            subgoal_valid=torch.zeros(n, dtype=torch.bool),
            metadata=[s.metadata for s in scenarios],
            control_mode=[s.control_mode for s in scenarios],
        )

    def run(
        self, policy: PolicyFn, scenarios: list, delay: int = 0, pass_committed: bool = False
    ) -> RolloutResult:
        n, w = len(scenarios), self.world
        if delay + self.exec_horizon > self.chunk_size:
            raise ValueError("exec_horizon + delay must fit in one chunk")
        robots = np.zeros((n, self.max_steps + 1, 2), np.float32)
        robots[:, 0] = np.stack([s.scene.robot for s in scenarios])
        executed = np.zeros((n, self.max_steps, 2), np.float32)
        collected = [set() for _ in range(n)]
        progress = np.zeros(n, int)  # index of the next target to visit
        success = np.zeros(n, bool)
        steps = np.full(n, self.max_steps)
        wrong = np.zeros(n, bool)
        first_touch = [None] * n
        frames = [[] for _ in range(n)]
        replans = []

        def render(t):
            for i, s in enumerate(scenarios):
                cue = s.scene.marked if t < w.cue_steps else None
                frames[i].append(w.render(robots[i, t], s.scene.objects, collected[i], cue))

        def subtasks():
            return [
                f"go to {s.targets[min(progress[i], len(s.targets) - 1)]}"
                for i, s in enumerate(scenarios)
            ]

        def query(t, committed):
            batch = self._observe(scenarios, robots, collected, frames, t, subtasks())
            with torch.no_grad():
                return policy(batch, committed).detach().cpu().numpy()

        def advance(t, actions):
            for i, s in enumerate(scenarios):
                if success[i]:
                    robots[i, t + 1] = robots[i, t]
                    continue
                executed[i, t] = actions[i]
                robots[i, t + 1] = w.step(robots[i, t], actions[i], s.control_mode)
                for color, xy in s.scene.objects.items():
                    if (
                        color in collected[i]
                        or np.linalg.norm(robots[i, t + 1] - xy) > w.success_radius
                    ):
                        continue
                    if first_touch[i] is None:
                        first_touch[i] = color
                    ok_first = progress[i] == 0 and color in s.alternatives
                    ok_next = progress[i] > 0 and color == s.targets[progress[i]]
                    if ok_first or ok_next:
                        if s.kind == "collect":
                            collected[i].add(color)
                        progress[i] += 1
                        if progress[i] >= len(s.targets):
                            success[i], steps[i] = True, t + 1
                    elif progress[i] == 0:
                        wrong[i] = True  # touched a non-target before the first target
            render(t + 1)

        render(0)
        chunk, idx, t = query(0, None), 0, 0
        while t < self.max_steps and not success.all():
            if idx >= self.exec_horizon:
                committed = chunk[:, idx : idx + delay]
                new = query(t, torch.from_numpy(committed) if (pass_committed and delay) else None)
                for k in range(delay):  # the old chunk keeps running during inference
                    if t >= self.max_steps:
                        break
                    advance(t, committed[:, k])
                    t += 1
                chunk, idx = new, delay
                replans.append(t)
                if t >= self.max_steps:
                    break
            advance(t, chunk[:, idx])
            idx += 1
            t += 1
        robots[:, t + 1 :] = robots[:, t : t + 1]  # finished early: hold the last position
        path = np.linalg.norm(np.diff(robots, axis=1), axis=-1).sum(1)
        return RolloutResult(success, steps, wrong, path, robots, executed, replans, first_touch)


def make_scenarios(
    generator,
    kind: str,
    n: int,
    seed: int,
    metadata: EpisodeMetadata | None = None,
    control_mode: str = "ee",
    task_override: str | None = None,
    num_objects=None,
) -> list:
    """``n`` reproducible evaluation scenarios of ``kind`` (``generator`` is a :class:`DemoGenerator`)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        scene, task, targets, alts = generator.sample_task(kind, rng, num_objects)
        out.append(
            Scenario(
                scene,
                task_override.format(*targets) if task_override else task,
                targets,
                alts,
                kind,
                metadata or EpisodeMetadata(),
                control_mode,
            )
        )
    return out
