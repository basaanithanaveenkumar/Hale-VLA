"""Closed-loop evaluation of chunked policies across embodiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from pi_family.toy import DemoGenerator, PointWorld
from pi_family.toy.world import Scene

from open_vlas.data import VLABatch, reach_task
from open_vlas.embodiments import EMBODIMENTS, MAX_ACTION_DIM, MAX_STATE_DIM


@dataclass
class EmbodiedScenario:
    """One evaluation episode: scene, instruction and which embodiment executes it."""

    scene: Scene
    task: str
    target: str
    embodiment: str
    alternatives: Sequence[str] | None = None  # colours that all count as success (the multimodal "go to any" task)


@dataclass
class EmbodiedResult:
    """Outcomes of a batch of scenarios; ``trajectories`` has shape (N, max_steps + 1, 2)."""

    success: np.ndarray
    steps: np.ndarray
    path_length: np.ndarray
    trajectories: np.ndarray
    wrong_first: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def success_rate(self) -> float:
        return float(self.success.mean())

    def summary(self) -> dict:
        ok = self.success
        return dict(
            success=round(float(ok.mean()), 3),
            steps=round(float(self.steps[ok].mean()), 1) if ok.any() else float("nan"),
            path=round(float(self.path_length[ok].mean()), 3) if ok.any() else float("nan"),
            wrong_first=round(float(self.wrong_first.mean()), 3) if self.wrong_first.size else 0.0,
        )


def make_embodied_scenarios(
    generator: DemoGenerator,
    embodiment: str,
    n: int,
    seed: int,
    target: str | None = None,
    colors: Sequence[str] | None = None,
) -> list:
    """``n`` reproducible ``go to <colour>`` scenarios; force the target colour with ``target``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        scene, task, targets, _ = reach_task(generator.world, rng, target, colors)
        out.append(EmbodiedScenario(scene, task, targets[0], embodiment))
    return out


def make_any_scenarios(generator: DemoGenerator, embodiment: str, n: int, seed: int) -> list:
    """``n`` scenes of the multimodal task ``"go to any"``: two objects, reaching *either* is a success."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        scene, task, targets, alternatives = generator.sample_task("any", rng)
        out.append(EmbodiedScenario(scene, task, targets[0], embodiment, alternatives=list(alternatives)))
    return out


PolicyFn = Callable[[VLABatch], torch.Tensor]


class EmbodiedEvaluator:
    """Runs scenarios in lock-step: every ``exec_horizon`` steps the policy sees the current image + state and returns a chunk.

    The policy returns ``(B, H, MAX_ACTION_DIM)`` actions in each scenario's own action space; the evaluator converts them
    to world motion with the scenario's embodiment and tracks the gripper state for embodiments that have one.
    """

    def __init__(
        self, world: PointWorld, chunk_size: int = 16, exec_horizon: int = 8, max_steps: int = 50
    ):
        self.world, self.chunk_size = world, chunk_size
        self.exec_horizon, self.max_steps = exec_horizon, max_steps

    def observe(self, scenarios, robots, grips, t) -> VLABatch:
        n = len(scenarios)
        images = np.stack(
            [
                self.world.render(robots[i, t], s.scene.objects, set(), None)
                for i, s in enumerate(scenarios)
            ]
        )
        state = np.zeros((n, MAX_STATE_DIM), np.float32)
        for i, s in enumerate(scenarios):
            emb = EMBODIMENTS[s.embodiment]
            state[i, : emb.state_dim] = emb.state(robots[i, t], grips[i])
        return VLABatch(
            images=torch.from_numpy(images),
            state=torch.from_numpy(state),
            task=[s.task for s in scenarios],
            embodiment=torch.tensor([EMBODIMENTS[s.embodiment].id for s in scenarios]),
            action_mask=torch.tensor(
                [
                    [1.0] * EMBODIMENTS[s.embodiment].action_dim
                    + [0.0] * (MAX_ACTION_DIM - EMBODIMENTS[s.embodiment].action_dim)
                    for s in scenarios
                ]
            ),
        )

    def run(self, policy: PolicyFn, scenarios: list) -> EmbodiedResult:
        n, w = len(scenarios), self.world
        robots = np.zeros((n, self.max_steps + 1, 2), np.float32)
        robots[:, 0] = np.stack([s.scene.robot for s in scenarios])
        grips = np.full(n, -1.0)
        success = np.zeros(n, bool)
        steps = np.full(n, self.max_steps)
        wrong = np.zeros(n, bool)
        t = 0
        chunk, idx = None, self.exec_horizon
        while t < self.max_steps and not success.all():
            if idx >= self.exec_horizon:
                with torch.no_grad():
                    chunk = policy(self.observe(scenarios, robots, grips, t)).cpu().numpy()
                idx = 0
            for i, s in enumerate(scenarios):
                if success[i]:
                    robots[i, t + 1] = robots[i, t]
                    continue
                emb = EMBODIMENTS[s.embodiment]
                a = chunk[i, idx]
                robots[i, t + 1] = w.step(robots[i, t], emb.ee_from_action(a), "ee")
                grips[i] = emb.grip_of(a)
                for color, xy in s.scene.objects.items():
                    if np.linalg.norm(robots[i, t + 1] - xy) <= w.success_radius:
                        # an embodiment with a gripper must also close it on the object
                        closed = grips[i] > 0 or emb.state_dim < 3
                        ok_color = color == s.target or (s.alternatives is not None and color in s.alternatives)
                        if ok_color and closed:
                            success[i], steps[i] = True, t + 1
                        else:
                            wrong[i] = True
            idx += 1
            t += 1
        robots[:, t + 1 :] = robots[:, t : t + 1]
        path = np.linalg.norm(np.diff(robots, axis=1), axis=-1).sum(1)
        return EmbodiedResult(success, steps, path, robots, wrong)
