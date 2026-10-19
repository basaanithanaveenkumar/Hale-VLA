"""A tiny 2D "point robot" world that every π-family notebook trains on.

The world is the unit square. A few coloured squares (objects) and a white robot are rendered into a 32×32 RGB
image. An action is a 2D motion command; in ``ee`` ("end-effector") mode it is the displacement itself, in
``joint`` mode it is mapped through a fixed linear "Jacobian" first (a stand-in for joint-space control, so π0.7's
control-mode prompt has something to switch between). Actions are normalised so ``|a| ≈ 1`` is one full step.

Coordinates: ``x`` grows to the right (image column), ``y`` grows downwards (image row).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

COLORS = {"red": (0.92, 0.18, 0.15), "green": (0.18, 0.78, 0.25), "blue": (0.22, 0.42, 1.0)}
SYNONYMS = {
    "crimson": "red",
    "emerald": "green",
    "azure": "blue",
}  # only the world model's "web" data uses these
JOINT_MAP = np.array(
    [[1.2, 0.5], [-0.4, 1.0]], dtype=np.float32
)  # ee displacement = JOINT_MAP @ joint action
JOINT_MAP_INV = np.linalg.inv(JOINT_MAP).astype(np.float32)


@dataclass
class Scene:
    """Initial layout of one episode: object positions (colour → xy), robot start and an optional marked object."""

    objects: dict
    robot: np.ndarray
    marked: str | None = None
    collected: set = field(default_factory=set)


class PointWorld:
    """Rendering and dynamics of the point world (pure numpy, deterministic)."""

    def __init__(
        self,
        image_size: int = 32,
        max_step: float = 0.05,
        success_radius: float = 0.05,
        object_half: int = 2,
        cue_steps: int = 2,
    ):
        self.image_size = image_size
        self.max_step = max_step  # displacement of a unit action
        self.success_radius = success_radius
        self.object_half = object_half  # objects are (2·half + 1)² pixels
        self.cue_steps = cue_steps  # the marked object is outlined for this many steps only

    # ------------------------------------------------------------------------------------------ scenes
    def sample_scene(
        self,
        rng: np.random.Generator,
        num_objects: int = 2,
        colors: list | None = None,
        min_separation: float = 0.3,
        marked: str | None = None,
    ) -> Scene:
        colors = (
            list(colors)
            if colors is not None
            else list(rng.permutation(list(COLORS))[:num_objects])
        )
        for _ in range(1000):
            points = rng.uniform(0.12, 0.88, size=(len(colors) + 1, 2)).astype(np.float32)
            d = np.linalg.norm(points[:, None] - points[None], axis=-1) + np.eye(len(points))
            if d.min() >= min_separation:
                break
        objects = {c: points[i] for i, c in enumerate(colors)}
        return Scene(objects=objects, robot=points[-1], marked=marked)

    # ------------------------------------------------------------------------------------------ rendering
    def _pixel(self, xy: np.ndarray) -> tuple[int, int]:
        col, row = np.clip(
            np.round(np.asarray(xy) * (self.image_size - 1)), 0, self.image_size - 1
        ).astype(int)
        return row, col

    def render(
        self, robot: np.ndarray, objects: dict, collected: set = frozenset(), cue: str | None = None
    ) -> np.ndarray:
        """(3, S, S) float32 image in [0, 1]."""
        s, h = self.image_size, self.object_half
        img = np.full((s, s, 3), 0.08, dtype=np.float32)
        for color, xy in objects.items():
            if color in collected:
                continue
            r, c = self._pixel(xy)
            if cue == color:  # white outline ring = "marked"
                img[max(r - h - 2, 0) : r + h + 3, max(c - h - 2, 0) : c + h + 3] = 1.0
                img[max(r - h - 1, 0) : r + h + 2, max(c - h - 1, 0) : c + h + 2] = 0.08
            img[max(r - h, 0) : r + h + 1, max(c - h, 0) : c + h + 1] = COLORS[color]
        r, c = self._pixel(robot)
        img[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2] = (1.0, 1.0, 0.85)
        img[r, c] = (0.25, 0.25, 0.25)  # dark centre: robot ≠ "white object"
        return img.transpose(2, 0, 1).copy()

    # ------------------------------------------------------------------------------------------ dynamics
    @staticmethod
    def to_ee(action: np.ndarray, mode: str) -> np.ndarray:
        """Normalised action in ``mode`` → normalised end-effector displacement."""
        return action @ JOINT_MAP.T if mode == "joint" else action

    @staticmethod
    def from_ee(action: np.ndarray, mode: str) -> np.ndarray:
        """Normalised end-effector displacement → action expressed in ``mode``."""
        return action @ JOINT_MAP_INV.T if mode == "joint" else action

    def step(self, robot: np.ndarray, action: np.ndarray, mode: str = "ee") -> np.ndarray:
        delta = (
            np.clip(self.to_ee(np.asarray(action, dtype=np.float32), mode), -1.5, 1.5)
            * self.max_step
        )
        return np.clip(robot + delta, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def as_tensor(images) -> torch.Tensor:
        return torch.from_numpy(np.stack(images)).float()
