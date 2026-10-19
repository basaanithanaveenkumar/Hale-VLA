"""Scripted demonstrations for the point world, with the annotations each π generation consumes.

Task kinds (``task`` text in quotes):

* ``reach``   - "go to red": one target.
* ``any``     - "go to any": two equally valid targets, the demonstrator picks one at random → **multimodal**
  action distribution (π0's flow matching vs regression).
* ``collect`` - "collect red then blue": two *subtasks* ("go to red", "go to blue"); an object disappears once
  reached (π0.5's hierarchical inference).
* ``marked``  - "go to the marked object": the target is outlined only for the first few steps (π0.7's memory).

Demonstration *styles* differ in speed and quality, and are labelled with π0.7's episode metadata:

=========  =====  ========  ========================================
style      speed  quality   behaviour
=========  =====  ========  ========================================
expert     fast   5         straight line, 90 % of the max speed
careful    slow   4         straight line, 50 % speed
sloppy     slow   2-3       wobbles sideways
mistake    slow   1-2       first heads for a wrong object, then corrects (those steps carry ``mistake = true``)
=========  =====  ========  ========================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pi_family.tokenizers import EpisodeMetadata, bin_speed
from pi_family.toy.world import COLORS, PointWorld, Scene


@dataclass(frozen=True)
class EpisodeStyle:
    name: str
    speed: float  # fraction of ``max_step`` per step
    wobble: float  # sideways sinusoidal motion, in units of ``max_step``
    noise: float  # Gaussian noise, in units of ``max_step``
    quality: tuple  # (low, high) inclusive quality score
    detour: bool = False  # head for a wrong object first


STYLES = {
    "expert": EpisodeStyle("expert", 0.9, 0.0, 0.03, (5, 5)),
    "careful": EpisodeStyle("careful", 0.5, 0.0, 0.03, (4, 4)),
    "sloppy": EpisodeStyle("sloppy", 0.55, 0.9, 0.12, (2, 3)),
    "mistake": EpisodeStyle("mistake", 0.6, 0.0, 0.05, (1, 2), detour=True),
}


@dataclass
class Episode:
    """One demonstration. Arrays are indexed by time step ``t`` (``robot`` has one more entry than ``actions``)."""

    kind: str
    task: str
    scene: Scene
    targets: list  # colours in visiting order
    robot: np.ndarray  # (T + 1, 2)
    actions: np.ndarray  # (T, 2) normalised end-effector actions
    subtask: list  # (T + 1,) subtask text active at each step
    segment_end: np.ndarray  # (T + 1,) step at which the current subtask ends
    mistake: np.ndarray  # (T,) bool
    collected_at: dict  # colour → step after which it is gone
    style: EpisodeStyle
    metadata: EpisodeMetadata
    control_mode: str = "ee"
    active_steps: int = 0  # steps until the last target was reached (the rest is idling)

    @property
    def length(self) -> int:
        return len(self.actions)

    def collected(self, t: int) -> set:
        return {c for c, step in self.collected_at.items() if t > step}


class DemoGenerator:
    """Builds tasks (scene + text + targets) and scripted demonstrations of them."""

    def __init__(self, world: PointWorld, idle_steps: int = 16, max_steps: int = 120):
        self.world = world
        self.idle_steps = idle_steps  # stay still after finishing so chunk targets learn to stop
        self.max_steps = max_steps

    # ------------------------------------------------------------------------------------------ tasks
    def sample_task(self, kind: str, rng: np.random.Generator, num_objects: int | None = None):
        """Return ``(scene, task_text, targets, alternatives)``; ``alternatives`` lists every valid first target."""
        if kind == "reach":
            scene = self.world.sample_scene(rng, num_objects or 2)
            target = str(rng.choice(list(scene.objects)))
            return scene, f"go to {target}", [target], [target]
        if kind == "any":
            scene = self.world.sample_scene(rng, num_objects or 2)
            target = str(rng.choice(list(scene.objects)))
            return scene, "go to any", [target], list(scene.objects)
        if kind == "collect":
            scene = self.world.sample_scene(rng, num_objects or 2)
            first, second = (str(c) for c in rng.permutation(list(scene.objects))[:2])
            return scene, f"collect {first} then {second}", [first, second], [first]
        if kind == "marked":
            scene = self.world.sample_scene(rng, num_objects or 3)
            target = str(rng.choice(list(scene.objects)))
            scene.marked = target
            return scene, "go to the marked object", [target], [target]
        raise ValueError(f"unknown task kind {kind!r}")

    # ------------------------------------------------------------------------------------------ demos
    def _move(
        self, p: np.ndarray, goal: np.ndarray, style: EpisodeStyle, rng, phase: float
    ) -> np.ndarray:
        """One scripted step toward ``goal`` (normalised action, before clipping)."""
        offset = goal - p
        dist = float(np.linalg.norm(offset))
        direction = offset / max(dist, 1e-8)
        along = min(style.speed, dist / self.world.max_step)  # do not overshoot
        normal = np.array([-direction[1], direction[0]])
        fade = min(1.0, dist / (3 * self.world.max_step))  # calm down near the goal
        action = along * direction + fade * (
            style.wobble * np.sin(phase) * normal * style.speed + style.noise * rng.normal(size=2)
        )
        return action.astype(np.float32)

    def demo(
        self,
        kind: str,
        rng: np.random.Generator,
        style: str = "expert",
        control_mode: str = "ee",
        task=None,
    ) -> Episode:
        """Script one demonstration of ``kind`` in ``style``. ``task`` reuses an existing ``sample_task`` result."""
        sty = STYLES[style]
        scene, text, targets, _ = task if task is not None else self.sample_task(kind, rng)
        world = self.world
        p = scene.robot.copy()
        robot, actions, subtasks, mistakes, seg_end = [p.copy()], [], [], [], []
        collected_at: dict = {}
        phase = rng.uniform(0, 2 * np.pi)
        plan = []  # (colour or wrong colour, is_mistake)
        if sty.detour:
            wrong = [c for c in scene.objects if c not in targets]
            if wrong:
                plan.append((str(rng.choice(wrong)), True))
        plan += [(c, False) for c in targets]
        segment_starts = []
        for goal_color, is_mistake in plan:
            goal = scene.objects[goal_color]
            # a detour stops 35-60 % of the way to the wrong object, then the real target is pursued
            stop_frac = rng.uniform(0.35, 0.6) if is_mistake else 0.0
            start_dist = float(np.linalg.norm(goal - p))
            true_target = goal_color if not is_mistake else targets[0]
            segment_starts.append(len(actions))
            while len(actions) < self.max_steps:
                dist = float(np.linalg.norm(goal - p))
                if is_mistake and dist <= (1 - stop_frac) * start_dist:
                    break
                if not is_mistake and dist < 1e-4:
                    break
                a = self._move(p, goal, sty, rng, phase)
                phase += 0.9
                actions.append(a)
                subtasks.append(f"go to {true_target}")
                mistakes.append(is_mistake)
                p = world.step(p, a)
                if not is_mistake and np.linalg.norm(goal - p) < 0.25 * world.max_step:
                    p = goal.copy()  # snap onto the object
                robot.append(p.copy())
            if not is_mistake:
                collected_at[goal_color] = len(actions) if kind == "collect" else 10**9
        active = len(actions)
        for _ in range(self.idle_steps):
            actions.append(np.zeros(2, dtype=np.float32))
            subtasks.append(subtasks[-1] if subtasks else f"go to {targets[-1]}")
            mistakes.append(False)
            robot.append(p.copy())
        subtasks.append(subtasks[-1])
        # end of the segment each step belongs to = the step its true subtask target is reached
        ends = sorted(v for v in collected_at.values() if v < 10**9) or [active]
        seg_end = np.array([next((e for e in ends if e >= t), active) for t in range(len(robot))])
        quality = int(rng.integers(sty.quality[0], sty.quality[1] + 1))
        metadata = EpisodeMetadata(
            speed=bin_speed(active), quality=quality, mistake=bool(any(mistakes))
        )
        return Episode(
            kind=kind,
            task=text,
            scene=scene,
            targets=targets,
            robot=np.stack(robot),
            actions=np.stack(actions),
            subtask=subtasks,
            segment_end=seg_end,
            mistake=np.array(mistakes),
            collected_at=collected_at,
            style=sty,
            metadata=metadata,
            control_mode=control_mode,
            active_steps=active,
        )

    def dataset(
        self,
        kinds,
        num_episodes: int,
        rng: np.random.Generator,
        styles=("expert",),
        style_weights=None,
        joint_fraction: float = 0.0,
    ) -> list:
        """A list of demonstrations, kinds and styles drawn uniformly (or with ``style_weights``)."""
        kinds, styles = list(kinds), list(styles)
        episodes = []
        for _ in range(num_episodes):
            kind = kinds[rng.integers(len(kinds))]
            style = styles[rng.choice(len(styles), p=style_weights)]
            mode = "joint" if rng.random() < joint_fraction else "ee"
            episodes.append(self.demo(kind, rng, style, mode))
        return episodes


__all__ = ["COLORS", "DemoGenerator", "Episode", "EpisodeStyle", "STYLES"]
