"""Closed-loop execution of chunked policies: receding horizon, temporal ensembling, execution noise.

A chunked policy returns ``H`` actions per query. What the robot *does* with them is a separate design choice (Diffusion Policy's
*receding horizon*, ACT's *temporal ensembling*):

* ``exec_horizon = k``: execute the first ``k`` actions of each chunk, then query again. ``k = H`` is open-loop for ``H`` steps;
  ``k = 1`` is fully reactive but queries the model at every step.
* ``ensemble_m``: query at **every** step and execute, for each time step, the weighted average of what all earlier chunks
  predicted for it, with weights ``exp(−m·i)`` (``i = 0`` is the oldest prediction), as in ACT. It smooths the trajectory.

The runner also injects **action noise** (the executed action differs from the commanded one, the source of compounding error) and
random **pushes**, and reports the metrics the notebook plots: success, steps, trajectory *jerk* (mean second difference of the
path) and the number of model calls per episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from open_vlas.embodiments import EMBODIMENTS
from open_vlas.evaluation import EmbodiedEvaluator
from pi_family.toy import PointWorld


@dataclass
class ChunkRunResult:
    """Per-episode outcomes of :meth:`ChunkedRunner.run` (arrays of length N)."""

    success: np.ndarray
    steps: np.ndarray
    jerk: np.ndarray
    calls: np.ndarray
    path_length: np.ndarray
    trajectories: np.ndarray

    def summary(self) -> dict:
        ok = self.success
        return dict(
            success=round(float(ok.mean()), 3),
            steps=round(float(self.steps[ok].mean()), 1) if ok.any() else float("nan"),
            jerk=round(float(self.jerk.mean()), 4),
            calls=round(float(self.calls.mean()), 1),
        )


class ChunkedRunner:
    """Executes a chunked policy in the point world with a chosen execution scheme.

    Args:
        world: the environment.
        chunk_size: ``H``, the policy's chunk length.
        exec_horizon: actions executed per query (ignored when ``ensemble_m`` is set).
        ensemble_m: ACT temporal-ensembling decay (``None`` = off; 0 = uniform average).
        action_noise: std of Gaussian noise added to every executed action (in action units).
        push_prob / push_size: random displacement of the robot (external disturbance).
        absolute: interpret each chunk as **absolute position targets** (waypoints anchored at the robot position when the chunk was
            predicted) and command ``(waypoint − current position) / max_step`` at every step, like ACT's joint-position targets. The
            default executes the predicted *relative* displacements as they are.
        max_steps: episode length limit.
    """

    def __init__(
        self,
        world: PointWorld,
        chunk_size: int,
        exec_horizon: int,
        ensemble_m: float | None = None,
        action_noise: float = 0.0,
        push_prob: float = 0.0,
        push_size: float = 0.05,
        max_steps: int = 60,
        absolute: bool = False,
    ):
        if exec_horizon > chunk_size:
            raise ValueError("exec_horizon cannot exceed the chunk size")
        self.world, self.chunk_size, self.exec_horizon = world, chunk_size, exec_horizon
        self.ensemble_m, self.action_noise = ensemble_m, action_noise
        self.push_prob, self.push_size, self.max_steps = push_prob, push_size, max_steps
        self.absolute = absolute
        self._obs = EmbodiedEvaluator(world, chunk_size, exec_horizon, max_steps)

    def run(self, policy, scenarios: list, seed: int = 0) -> ChunkRunResult:
        """``policy(VLABatch) -> (B, H, MAX_ACTION_DIM)``. Scenarios are ``EmbodiedScenario`` objects (``point2d`` embodiment)."""
        rng = np.random.default_rng(seed)
        w, n = self.world, len(scenarios)
        emb = EMBODIMENTS["point2d"]
        robots = np.zeros((n, self.max_steps + 1, 2), np.float32)
        robots[:, 0] = np.stack([s.scene.robot for s in scenarios])
        grips = np.full(n, -1.0)
        success, steps = np.zeros(n, bool), np.full(n, self.max_steps)
        calls = np.zeros(n)
        history: list = []  # (t0, chunk (n, H, A)) for temporal ensembling
        chunk, idx, t = None, self.exec_horizon, 0
        while t < self.max_steps and not success.all():
            active = ~success
            if self.ensemble_m is not None or idx >= self.exec_horizon:
                with torch.no_grad():
                    new = policy(self._obs.observe(scenarios, robots, grips, t)).cpu().numpy()
                calls[active] += 1
                if self.absolute:  # anchor the predicted displacements at the current position → absolute waypoints
                    new = robots[:, t][:, None, :] + w.max_step * np.cumsum(new[:, :, :2], axis=1)
                chunk, idx = new, 0
                history.append((t, new))
            if self.ensemble_m is not None:
                # ACT temporal ensembling: every earlier chunk that predicted step t contributes; i = 0 is the oldest prediction
                preds = [c[:, t - t0] for t0, c in history if 0 <= t - t0 < self.chunk_size]
                weights = np.exp(-self.ensemble_m * np.arange(len(preds)))
                action = sum(wt * p for wt, p in zip(weights, preds)) / weights.sum()
            else:
                action = chunk[:, idx]
            if (
                self.absolute
            ):  # the "action" is a target position: command the displacement that closes the gap
                action = np.clip((action[:, :2] - robots[:, t]) / w.max_step, -1.5, 1.5)
            for i, s in enumerate(scenarios):
                if success[i]:
                    robots[i, t + 1] = robots[i, t]
                    continue
                a = np.array(action[i][:2], dtype=np.float32)
                if self.action_noise:
                    a = a + rng.normal(0, self.action_noise, 2).astype(np.float32)
                robots[i, t + 1] = w.step(robots[i, t], emb.ee_from_action(a), "ee")
                if self.push_prob and rng.random() < self.push_prob:
                    robots[i, t + 1] = np.clip(
                        robots[i, t + 1] + rng.uniform(-self.push_size, self.push_size, 2), 0, 1
                    )
                for color, xy in s.scene.objects.items():
                    ok = color == s.target or (
                        s.alternatives is not None and color in s.alternatives
                    )
                    if ok and np.linalg.norm(robots[i, t + 1] - xy) <= w.success_radius:
                        success[i], steps[i] = True, t + 1
            idx += 1
            t += 1
        robots[:, t + 1 :] = robots[:, t : t + 1]
        d1 = np.diff(robots, axis=1)
        valid = np.arange(self.max_steps)[None, :] < steps[:, None]
        jerk = np.array(
            [
                np.linalg.norm(np.diff(d1[i, : max(steps[i], 3)], axis=0), axis=-1).mean()
                for i in range(n)
            ]
        )
        path = (np.linalg.norm(d1, axis=-1) * valid).sum(1)
        return ChunkRunResult(success, steps, jerk, calls, path, robots)


def effective_control_hz(
    model_ms_per_call: float, exec_horizon: int, control_hz: float = 50.0
) -> float:
    """Control rate a policy can sustain when each call costs ``model_ms_per_call`` and yields ``exec_horizon`` actions.

    Calls are assumed sequential (the paper-era synchronous case): the robot needs ``exec_horizon`` actions per call, so the sustainable
    action rate is ``exec_horizon / (call time)``, capped by the robot's own control rate.
    """
    return min(control_hz, exec_horizon / (model_ms_per_call / 1000.0))
