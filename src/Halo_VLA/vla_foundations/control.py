"""Where the VLA sits in a control stack: an inertial plant with a classical tracker, and a dual-rate System 2 / System 1 policy.

* :class:`PlantRunner` - the robot is not a kinematic point but a **lightly damped servo** (spring-damper): it tracks a setpoint with
  ringing. A VLA's chunk becomes absolute waypoints, sent either **directly** to the servo or through a **classical outer loop** running
  several times per tick that injects damping (critically damped for the known plant). The VLA decides *where*, classical control
  decides *how*.
* :class:`DualRateRunner` - GR00T N1's System 2 / System 1 split: the VLM (**System 2**, slow) is refreshed every ``s2_period`` ticks,
  the flow-matching action network (**System 1**, fast) runs at **every** tick on the latest VLM features and the *fresh* proprioception.
  The target object is moved mid-episode, so stale System-2 features send the robot to where the object *was*.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from open_vlas.embodiments import EMBODIMENTS, MAX_ACTION_DIM
from open_vlas.evaluation import EmbodiedEvaluator
from pi_family.toy import PointWorld


@dataclass
class PlantResult:
    """Per-episode outcomes of :meth:`PlantRunner.run` (arrays of length N)."""

    success: (
        np.ndarray
    )  # within the success radius at the END of the episode (reached *and* stayed)
    final_error: np.ndarray
    peak_overshoot: np.ndarray  # largest distance travelled past the target along the approach
    jitter: np.ndarray  # mean speed over the last 10 ticks (should be ~0 once settled)
    trajectories: np.ndarray

    def summary(self) -> dict:
        return dict(
            success=round(float(self.success.mean()), 3),
            final_error=round(float(self.final_error.mean()), 3),
            overshoot=round(float(self.peak_overshoot.mean()), 3),
            jitter=round(float(self.jitter.mean()), 4),
        )


class PlantRunner:
    """Closed loop on a **lightly damped servo**: the arm tracks a position setpoint ``c`` like a spring-damper,
    ``p'' = ω² (c − p) − 2 ζ ω p'`` (``ω`` in rad/tick, ``ζ`` the damping ratio; ``ζ < 1`` rings).

    The policy's chunk is turned into **absolute waypoints** ``w_i = p_0 + max_step · cumsum(a)_i`` (anchored where the arm is when the chunk
    is predicted). Two ways to send them to the servo:

    * ``"direct"``: the setpoint is the current waypoint, ``c = w_i`` (what a VLA that outputs joint-position targets does);
    * ``"tracked"``: a classical outer loop running ``substeps`` times per tick adds **damping injection**, ``c = w_i − k_d p'`` with
      ``k_d = 2 (1 − ζ) / ω``, which makes the closed loop critically damped (``ζ_eff = 1``) for the known plant.

    Args:
        world: geometry and ``max_step`` (the distance of one unit action per tick).
        omega / zeta: natural frequency (rad per tick) and damping ratio of the servo.
        mode: ``"direct"`` or ``"tracked"``.
        chunk_size / exec_horizon: the policy's chunk and how many of its actions are executed per query.
        substeps: micro-steps per tick (the plant and the tracker run at this finer rate).
        max_steps: episode length in ticks.
        kick: std of a random velocity impulse (in units of ``max_step``) added at every tick (a disturbance the servo has to reject).
        tolerance: success radius (default: the world's).
        seed: seed of the disturbance.
    """

    def __init__(
        self,
        world: PointWorld,
        omega: float,
        zeta: float,
        mode: str,
        chunk_size: int = 8,
        exec_horizon: int = 4,
        substeps: int = 20,
        max_steps: int = 60,
        kick: float = 0.0,
        tolerance: float | None = None,
        seed: int = 0,
    ):
        if mode not in ("direct", "tracked"):
            raise ValueError("mode must be 'direct' or 'tracked'")
        self.world, self.omega, self.zeta, self.mode = world, omega, zeta, mode
        self.chunk_size, self.exec_horizon = chunk_size, exec_horizon
        self.substeps, self.max_steps = substeps, max_steps
        self.kick, self.tolerance, self.seed = kick, tolerance, seed
        self._obs = EmbodiedEvaluator(world, chunk_size, exec_horizon, max_steps)

    def run(self, policy, scenarios: list) -> PlantResult:
        rng = np.random.default_rng(self.seed)
        w, n, dt = self.world, len(scenarios), 1.0 / self.substeps
        p = np.stack([s.scene.robot for s in scenarios]).astype(np.float64)
        v = np.zeros_like(p)
        goal = np.stack([s.scene.objects[s.target] for s in scenarios]).astype(np.float64)
        traj = np.zeros((n, self.max_steps + 1, 2))
        traj[:, 0] = p
        kd = 2.0 * (1.0 - self.zeta) / self.omega if self.mode == "tracked" else 0.0
        waypoints, idx = None, self.exec_horizon
        for t in range(self.max_steps):
            if idx >= self.exec_horizon:
                with torch.no_grad():
                    batch = self._obs.observe(
                        scenarios, p[:, None, :].astype(np.float32), np.full(n, -1.0), 0
                    )
                    chunk = policy(batch).cpu().numpy()[:, :, :2].astype(np.float64)
                waypoints = p[:, None, :] + w.max_step * np.cumsum(
                    np.clip(chunk, -1.5, 1.5), axis=1
                )
                idx = 0
            v = v + rng.normal(
                0.0, self.kick * w.max_step, v.shape
            )  # a disturbance (impulse) at the start of every tick
            for _ in range(self.substeps):
                setpoint = waypoints[:, idx] - kd * v
                acc = self.omega**2 * (setpoint - p) - 2.0 * self.zeta * self.omega * v
                v = v + acc * dt
                p = np.clip(p + v * dt, 0.0, 1.0)
            idx += 1
            traj[:, t + 1] = p
        d = np.linalg.norm(traj - goal[:, None], axis=-1)
        approach = (goal - traj[:, 0]) / np.maximum(d[:, 0], 1e-8)[:, None]
        beyond = ((traj - goal[:, None]) * approach[:, None]).sum(
            -1
        )  # > 0 once the arm is past the target
        speed = np.linalg.norm(np.diff(traj[:, -11:], axis=1), axis=-1).mean(1)
        return PlantResult(
            d[:, -1] <= (self.tolerance or w.success_radius),
            d[:, -1],
            np.maximum(beyond.max(1), 0.0),
            speed,
            traj,
        )


@dataclass
class DualRateResult:
    success: np.ndarray
    steps: np.ndarray
    s2_calls: np.ndarray
    s1_calls: np.ndarray

    def summary(self) -> dict:
        ok = self.success
        return dict(
            success=round(float(ok.mean()), 3),
            steps=round(float(self.steps[ok].mean()), 1) if ok.any() else float("nan"),
            s2_calls=round(float(self.s2_calls.mean()), 1),
            s1_calls=round(float(self.s1_calls.mean()), 1),
        )


class DualRateRunner:
    """System 2 (VLM features) refreshed every ``s2_period`` ticks; System 1 (flow-matching DiT) re-run at every tick.

    ``policy`` is a :class:`GR00TPolicy`. The target object jumps to a new place at tick ``jump_step`` (before the robot can reach it),
    so the robot only learns about the change when System 2 next looks at the image.
    """

    def __init__(
        self,
        world: PointWorld,
        policy,
        s2_period: int,
        jump_step: int = 4,
        max_steps: int = 50,
        num_steps: int = 4,
    ):
        self.world, self.policy, self.s2_period = world, policy, s2_period
        self.jump_step, self.max_steps, self.num_steps = jump_step, max_steps, num_steps
        self._obs = EmbodiedEvaluator(world, policy.cfg.chunk_size, 1, max_steps)

    @staticmethod
    def relocate(scenarios: list, seed: int, min_move: float = 0.3) -> list:
        """The new position of every scenario's target (far from its old place and from the other objects)."""
        rng = np.random.default_rng(seed)
        out = []
        for s in scenarios:
            old = s.scene.objects[s.target]
            others = [xy for c, xy in s.scene.objects.items() if c != s.target]
            for _ in range(1000):
                new = rng.uniform(0.12, 0.88, 2).astype(np.float32)
                if (
                    np.linalg.norm(new - old) >= min_move
                    and all(np.linalg.norm(new - o) >= 0.2 for o in others)
                    and np.linalg.norm(new - s.scene.robot) >= 0.3
                ):
                    break
            out.append(new)
        return out

    @torch.no_grad()
    def run(self, scenarios: list, new_positions: list, seed: int = 0) -> DualRateResult:
        w, n, pol = self.world, len(scenarios), self.policy
        scenarios = [copy.deepcopy(s) for s in scenarios]
        emb = EMBODIMENTS["point2d"]
        robots = np.zeros((n, self.max_steps + 1, 2), np.float32)
        robots[:, 0] = np.stack([s.scene.robot for s in scenarios])
        success, steps = np.zeros(n, bool), np.full(n, self.max_steps)
        s2_calls, s1_calls = np.zeros(n), np.zeros(n)
        ctx, device = None, pol.action_position.device
        gen = torch.Generator().manual_seed(seed)
        for t in range(self.max_steps):
            if t == self.jump_step:
                for s, new in zip(scenarios, new_positions):
                    s.scene.objects[s.target] = new
            batch = self._obs.observe(scenarios, robots, np.full(n, -1.0), t)
            if t % self.s2_period == 0:  # System 2: look at the world again
                ctx = pol.encode(batch)
                s2_calls[~success] += 1
            # System 1: a new action chunk from the (possibly stale) context and the *current* state, every tick
            state, e = batch.state.to(device), batch.embodiment.to(device)
            mask = batch.action_mask.to(device)
            x0 = (
                torch.randn(n, pol.cfg.chunk_size, MAX_ACTION_DIM, generator=gen).to(device)
                * mask[:, None, :]
            )
            chunk = pol.flow.integrate(
                lambda x, tau: pol.velocity(ctx, x, tau, state, e) * mask[:, None, :],
                x0,
                self.num_steps,
            )
            s1_calls[~success] += 1
            for i, s in enumerate(scenarios):
                if success[i]:
                    robots[i, t + 1] = robots[i, t]
                    continue
                robots[i, t + 1] = w.step(
                    robots[i, t], emb.ee_from_action(chunk[i, 0].cpu().numpy()), "ee"
                )
                if np.linalg.norm(robots[i, t + 1] - s.scene.objects[s.target]) <= w.success_radius:
                    success[i], steps[i] = True, t + 1
        return DualRateResult(success, steps, s2_calls, s1_calls)


def timed(fn, reps: int = 10) -> float:
    """Mean wall time of ``fn()`` in milliseconds."""
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1000 * (time.perf_counter() - t0) / reps
