"""SmolVLA's asynchronous inference stack (Algorithm 1 of the paper), simulated step by step.

A chunked policy returns ``n`` actions per query. Executing them *synchronously* (use the whole chunk, then observe, then wait
for the model) leaves the robot **idle** while the next chunk is computed and **open-loop** for ``n`` steps. SmolVLA decouples
the two:

* the **RobotClient** keeps an action queue and pops one action per control step;
* when the queue drops below a fraction ``g`` of the chunk (``|A_t| / n < g``) it captures a new observation and sends it to the
  **PolicyServer**, *while it keeps executing the old queue*;
* the new chunk arrives ``ℓ`` steps later (inference + network latency); the part of it whose time has already passed is dropped
  and the rest is **aggregated** into the queue;
* an **observation-similarity filter** skips queries whose joint-space state is within ``ε`` of the last one sent, unless the queue
  is empty.

``g = 0`` is the synchronous limit (idle ``ℓ`` steps per chunk), ``g = 1`` queries at every step (maximally reactive, one model
call per control tick), and ``g ∈ (0, 1)`` trades reactivity against compute. Queue exhaustion is avoided for
``g ≥ ℓ / n`` (the paper's ``E[ℓ_S] / (Δt · n)``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from pi_family.toy import PointWorld

from open_vlas.embodiments import EMBODIMENTS, MAX_STATE_DIM

PolicyCall = Callable[[np.ndarray, np.ndarray, str], np.ndarray]


@dataclass
class AsyncConfig:
    """Settings of one run.

    Attributes:
        chunk_size: ``n``, actions per chunk.
        threshold: ``g``; 0 = synchronous (query only when the queue is empty).
        latency: ``ℓ``, control steps between sending an observation and receiving its chunk.
        similarity_eps: ``ε``, joint-space distance under which a new query is skipped (0 = never skip).
        aggregate: how overlapping old and new actions are combined: ``"replace"`` (new wins), ``"average"``, ``"keep_old"``.
        push_prob / push_size: random external pushes on the robot (to make reactivity matter).
        max_steps: episode length limit.
    """

    chunk_size: int = 16
    threshold: float = 0.0
    latency: int = 4
    similarity_eps: float = 0.0
    aggregate: str = "replace"
    push_prob: float = 0.0
    push_size: float = 0.05
    max_steps: int = 80


@dataclass
class AsyncTrace:
    """What happened in one run, for plots and metrics."""

    success: bool = False
    steps: int = 0
    idle_steps: int = 0
    inferences: int = 0
    queue_sizes: list = field(default_factory=list)
    send_times: list = field(default_factory=list)
    arrival_times: list = field(default_factory=list)
    idle_flags: list = field(default_factory=list)
    positions: list = field(default_factory=list)


class ActionQueue:
    """FIFO queue of actions with overlap aggregation (the paper's ``f(A_t, Ã_{t+1})``)."""

    def __init__(self):
        self._q: deque = deque()

    def __len__(self) -> int:
        return len(self._q)

    def pop(self) -> np.ndarray:
        return self._q.popleft()

    def merge(self, tail: np.ndarray, mode: str) -> None:
        """Combine the remaining old actions with the new chunk's still-valid tail (both start at the current step)."""
        old = list(self._q)
        n_overlap = min(len(old), len(tail))
        if mode == "replace":
            merged = list(tail)
        elif mode == "keep_old":
            merged = old[:n_overlap] + list(tail[n_overlap:])
        elif mode == "average":
            merged = [(old[i] + tail[i]) / 2 for i in range(n_overlap)] + list(tail[n_overlap:])
        else:
            raise ValueError(f"unknown aggregate mode {mode!r}")
        self._q = deque(merged)


class AsyncInferenceRunner:
    """Simulates a RobotClient + PolicyServer pair on one point-world episode.

    Args:
        world: the environment (renders observations, applies actions).
        policy: ``policy(image (3,S,S), state (MAX_STATE_DIM,), task) -> chunk (n, MAX_ACTION_DIM)`` in the embodiment's action space.
        cfg: :class:`AsyncConfig`.
        embodiment: name of the robot body (default ``point2d``).
        rng: randomness for pushes.
    """

    def __init__(
        self,
        world: PointWorld,
        policy: PolicyCall,
        cfg: AsyncConfig,
        embodiment: str = "point2d",
        rng: np.random.Generator | None = None,
    ):
        self.world, self.policy, self.cfg = world, policy, cfg
        self.emb = EMBODIMENTS[embodiment]
        self.rng = rng or np.random.default_rng(0)

    def _state(self, robot: np.ndarray, grip: float) -> np.ndarray:
        s = np.zeros(MAX_STATE_DIM, np.float32)
        s[: self.emb.state_dim] = self.emb.state(robot, grip)
        return s

    def run(self, scene, task: str, target: str) -> AsyncTrace:
        cfg, w = self.cfg, self.world
        robot, grip = scene.robot.copy(), -1.0
        queue = ActionQueue()
        pending = None  # (arrive_at, chunk, sent_at)
        last_sent_state = None
        trace = AsyncTrace()
        for t in range(cfg.max_steps):
            state = self._state(robot, grip)
            self._deliver(queue, pending, t, trace)
            if pending is not None and t >= pending[0]:
                pending = None
            # --- trigger a query when the queue is short (or empty in synchronous mode)
            low = (
                (len(queue) / cfg.chunk_size < cfg.threshold)
                if cfg.threshold > 0
                else len(queue) == 0
            )
            similar = (
                last_sent_state is not None
                and np.linalg.norm(state - last_sent_state) < cfg.similarity_eps
                and len(queue) > 0
            )
            if pending is None and low and not similar:
                image = w.render(robot, scene.objects)
                chunk = self.policy(
                    image, state, task
                )  # the "server" computes while the client keeps acting
                pending = (t + cfg.latency, chunk, t)
                last_sent_state = state
                trace.inferences += 1
                trace.send_times.append(t)
                self._deliver(queue, pending, t, trace)
                if t >= pending[0]:
                    pending = None
            # --- act
            trace.queue_sizes.append(len(queue))
            if len(queue):
                action = queue.pop()
                trace.idle_flags.append(False)
            else:
                action = np.zeros(self.emb.action_dim, np.float32)
                trace.idle_flags.append(True)
                trace.idle_steps += 1
            robot = w.step(robot, self.emb.ee_from_action(np.asarray(action)), "ee")
            grip = self.emb.grip_of(np.asarray(action)) if len(action) > 2 else grip
            if cfg.push_prob and self.rng.random() < cfg.push_prob:
                robot = np.clip(
                    robot + self.rng.uniform(-cfg.push_size, cfg.push_size, 2), 0, 1
                ).astype(np.float32)
            trace.positions.append(robot.copy())
            trace.steps = t + 1
            if np.linalg.norm(robot - scene.objects[target]) <= w.success_radius:
                trace.success = True
                break
        return trace

    def _deliver(self, queue: ActionQueue, pending, t: int, trace: AsyncTrace) -> None:
        """If the in-flight chunk has arrived at step ``t``, drop its already-elapsed prefix and merge the rest."""
        if pending is None or t < pending[0]:
            return
        _, chunk, sent_at = pending
        # While the model was thinking the robot either executed old actions (the new chunk's matching actions are then
        # stale and dropped) or sat idle (its state did not change, so nothing is stale yet).
        idle = sum(trace.idle_flags[sent_at:t])
        stale = (t - sent_at) - idle
        queue.merge(np.asarray(chunk[stale:]), self.cfg.aggregate)
        trace.arrival_times.append(t)


def make_policy_call(model, embodiment: str = "point2d") -> PolicyCall:
    """Wrap a policy with ``sample_actions(VLABatch)`` into the single-observation callable the runner uses."""
    import torch

    from open_vlas.data import VLABatch

    emb = EMBODIMENTS[embodiment]

    def call(image: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        batch = VLABatch(
            images=torch.from_numpy(image[None]),
            state=torch.from_numpy(state[None]),
            task=[task],
            embodiment=torch.tensor([emb.id]),
        )
        with torch.no_grad():
            return model.sample_actions(batch)[0, :, : emb.action_dim].cpu().numpy()

    return call


def evaluate_async(
    world: PointWorld,
    policy: PolicyCall,
    cfg: AsyncConfig,
    scenarios: list,
    seed: int = 0,
    embodiment: str = "point2d",
) -> dict:
    """Run every scenario with the same settings and aggregate: success, time to success, idle steps, model calls."""
    rng = np.random.default_rng(seed)
    traces = [
        AsyncInferenceRunner(world, policy, cfg, embodiment, rng).run(s.scene, s.task, s.target)
        for s in scenarios
    ]
    ok = [tr for tr in traces if tr.success]
    return dict(
        success=round(float(np.mean([tr.success for tr in traces])), 3),
        steps=round(float(np.mean([tr.steps for tr in ok])), 1) if ok else float("nan"),
        idle=round(float(np.mean([tr.idle_steps for tr in traces])), 1),
        calls=round(float(np.mean([tr.inferences for tr in traces])), 1),
        traces=traces,
    )
