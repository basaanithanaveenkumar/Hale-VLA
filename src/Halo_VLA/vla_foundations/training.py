"""Behaviour cloning, covariate shift and its fixes, in the point world.

* :func:`narrow_start_demos` - expert demonstrations whose start positions are confined to a small disc, so a cloned policy has never
  seen the states it will visit when it drifts (the setting of *covariate shift*).
* :func:`expert_episode_from` - the scripted expert's demonstration **starting from any state**: the *label oracle* DAgger needs.
* :func:`dagger_episodes` - Ross et al. (2011): run the **learner**, record the states *it* visits, relabel them with the expert.
* :func:`perturbed_expert_episodes` - noise-injected demonstrations (DART, Laskey et al. 2017): run the **expert** under random pushes
  and relabel the states it was pushed into, so the data contains recoveries.
* :func:`state_action_error` - offline ``‖π(s) − π*(s)‖`` on a set of states: small on the demonstrated states, large on the learner's own.
"""

from __future__ import annotations

import numpy as np
from open_vlas.embodiments import EMBODIMENTS
from open_vlas.evaluation import EmbodiedEvaluator, EmbodiedScenario
from pi_family.toy import DemoGenerator
from pi_family.toy.episodes import Episode
from pi_family.toy.world import Scene

from vla_foundations.chunking import ChunkedRunner


def narrow_start_demos(
    gen: DemoGenerator,
    n: int,
    rng: np.random.Generator,
    radius: float = 0.1,
    centre=(0.5, 0.5),
    style: str = "expert",
) -> list[Episode]:
    """``n`` ``go to <colour>`` demonstrations whose robot starts within ``radius`` of ``centre`` (objects are anywhere)."""
    out = []
    for _ in range(n):
        scene, text, targets, alts = gen.sample_task("reach", rng, num_objects=3)
        angle, r = rng.uniform(0, 2 * np.pi), radius * np.sqrt(rng.uniform())
        start = (np.asarray(centre) + r * np.array([np.cos(angle), np.sin(angle)])).astype(
            np.float32
        )
        scene = Scene(objects=scene.objects, robot=start)
        out.append(gen.demo("reach", rng, style, task=(scene, text, targets, alts)))
    return out


def expert_episode_from(
    gen: DemoGenerator,
    scenario: EmbodiedScenario,
    position: np.ndarray,
    rng: np.random.Generator,
    style: str = "expert",
) -> Episode:
    """The expert's demonstration of ``scenario``'s task when the robot *starts* at ``position`` (the DAgger label oracle)."""
    scene = Scene(objects=scenario.scene.objects, robot=np.asarray(position, dtype=np.float32))
    return gen.demo(
        "reach", rng, style, task=(scene, scenario.task, [scenario.target], [scenario.target])
    )


def relabel_states(
    gen: DemoGenerator,
    scenario: EmbodiedScenario,
    positions: np.ndarray,
    rng: np.random.Generator,
    every: int = 2,
) -> list[Episode]:
    """One expert demonstration from every ``every``-th visited position (each is a 'what would the expert do *here*?' label set)."""
    return [
        expert_episode_from(gen, scenario, p, rng)
        for p in positions[::every]
        if all(
            np.linalg.norm(p - xy) > gen.world.success_radius
            for xy in [scenario.scene.objects[scenario.target]]
        )
    ]


def dagger_episodes(
    gen: DemoGenerator,
    policy,
    scenarios: list,
    rng: np.random.Generator,
    chunk_size: int,
    exec_horizon: int,
    action_noise: float = 0.0,
    push_prob: float = 0.0,
    max_steps: int = 40,
    every: int = 2,
) -> list[Episode]:
    """Run the **learner** ``policy(batch) → chunk`` on ``scenarios`` and return expert-relabelled episodes from the states it visited."""
    runner = ChunkedRunner(
        gen.world,
        chunk_size,
        exec_horizon,
        action_noise=action_noise,
        push_prob=push_prob,
        max_steps=max_steps,
    )
    result = runner.run(policy, scenarios, seed=int(rng.integers(1 << 30)))
    episodes = []
    for s, traj, steps in zip(scenarios, result.trajectories, result.steps):
        episodes += relabel_states(gen, s, traj[: int(steps) + 1], rng, every)
    return episodes


def perturbed_expert_episodes(
    gen: DemoGenerator,
    scenarios: list,
    rng: np.random.Generator,
    push_prob: float = 0.3,
    push_size: float = 0.15,
    max_steps: int = 40,
    every: int = 2,
) -> list[Episode]:
    """Run the **expert** under random pushes and relabel the (off-distribution) states it was pushed into (DART-style)."""
    world, episodes = gen.world, []
    for s in scenarios:
        goal = s.scene.objects[s.target]
        p, states = s.scene.robot.copy(), []
        for _ in range(max_steps):
            states.append(p.copy())
            if np.linalg.norm(goal - p) <= world.success_radius:
                break
            direction = (goal - p) / max(np.linalg.norm(goal - p), 1e-8)
            p = world.step(
                p,
                direction * min(0.9, np.linalg.norm(goal - p) / world.max_step),
            )
            if rng.random() < push_prob:
                p = np.clip(p + rng.uniform(-push_size, push_size, 2), 0, 1).astype(np.float32)
        episodes += relabel_states(gen, s, np.stack(states), rng, every)
    return episodes


def state_action_error(
    gen: DemoGenerator,
    policy,
    scenarios: list,
    positions: list,
    rng: np.random.Generator,
) -> float:
    """Mean ``‖π(s)_0 − π*(s)_0‖`` (first action of the chunk vs the expert's) over one visited state per scenario."""
    world = gen.world
    ev = EmbodiedEvaluator(world, 1, 1, 1)
    robots = np.stack(positions)[:, None, :].astype(np.float32)
    batch = ev.observe(scenarios, robots, np.full(len(scenarios), -1.0), 0)
    pred = policy(batch)[:, 0, :2].cpu().numpy()
    expert = np.stack(
        [expert_episode_from(gen, s, p, rng).actions[0] for s, p in zip(scenarios, positions)]
    )
    emb = EMBODIMENTS["point2d"]
    del emb  # point2d actions are already end-effector displacements
    return float(np.linalg.norm(pred - expert, axis=1).mean())


def concat_batches(a, b):
    """Concatenate two :class:`VLABatch` objects field by field (tensors along the batch axis, lists joined, ``None`` kept)."""
    import dataclasses

    import torch

    values = {}
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, torch.Tensor):
            values[f.name] = torch.cat([x, y])
        elif isinstance(x, list):
            values[f.name] = x + y
        else:
            values[f.name] = x
    return type(a)(**values)


class ReplayMixture:
    """A dataset that mixes ``new`` examples with ``replay`` examples of the old task (``replay_fraction`` of every batch).

    Replaying a little of the data the model was trained on is the standard remedy for catastrophic forgetting during adaptation.
    """

    def __init__(self, new, replay, replay_fraction: float = 0.5):
        if not 0.0 < replay_fraction < 1.0:
            raise ValueError("replay_fraction must be in (0, 1)")
        self.new, self.replay, self.fraction = new, replay, replay_fraction

    def sample(self, batch_size: int, rng: np.random.Generator):
        n_old = max(1, int(round(batch_size * self.fraction)))
        return concat_batches(
            self.new.sample(batch_size - n_old, rng), self.replay.sample(n_old, rng)
        )
