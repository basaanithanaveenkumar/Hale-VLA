"""A tiny, fully observable 2-D "robot" world for training VLA models on a laptop CPU."""

from __future__ import annotations

import numpy as np

from pi_lab.toy_scene import ToyScene

COLOURS = {"red": (0.95, 0.25, 0.25), "blue": (0.25, 0.5, 1.0), "green": (0.3, 0.85, 0.4)}


class ToyReachEnv:
    """Point robot, coloured targets and an obstacle, rendered as a small RGB camera image.

    Everything a VLA needs, at toy scale: an image observation, a proprioceptive state (the
    robot position), a language instruction ("go to red"), and continuous actions (2-D
    displacements). The expert goes **around** an obstacle on a randomly chosen side, so the
    action distribution is genuinely **multimodal**: averaging the two detours drives straight
    into the obstacle. This is the property flow matching is built to handle.

    Args:
        image_size: camera resolution (square).
        horizon: actions per chunk ``H``.
        step: nominal distance moved per action (workspace units).
        seed: RNG seed.
    """

    ACTION_SCALE = 0.1  # actions are stored divided by this, so they are O(1) for the networks
    OBSTACLE_RADIUS = 0.11

    def __init__(self, image_size: int = 32, horizon: int = 10, step: float = 0.05,
                 seed: int = 0) -> None:
        self.image_size = image_size
        self.horizon = horizon
        self.step = step
        self.rng = np.random.default_rng(seed)
        grid = (np.arange(image_size) + 0.5) / image_size
        self._xx, self._yy = np.meshgrid(grid, grid)

    # ------------------------------------------------------------------ scenes
    def sample_scene(self, colours: tuple[str, ...] = ("red", "blue"),
                     obstacle_prob: float = 0.0, blocked: str | None = None) -> ToyScene:
        """Random robot, targets at least 0.35 apart, and (optionally) an obstacle in the way.

        Target keys may repeat a colour with a ``#`` suffix (``"red#1"``, ``"red#2"``): both
        are drawn red, which makes "go to red" ambiguous, a two-mode action distribution.

        Args:
            colours: target keys to place.
            obstacle_prob: chance of an obstacle.
            blocked: target key whose straight path the obstacle blocks (default: the first).
        """
        while True:
            pts = self.rng.uniform(0.12, 0.88, size=(len(colours) + 1, 2))
            dists = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(len(pts))
            if dists.min() > 0.35:
                break
        scene = ToyScene(pts[0], dict(zip(colours, pts[1:], strict=True)))
        if self.rng.random() < obstacle_prob:
            goal = scene.targets[blocked or colours[0]]
            centre = scene.robot + (goal - scene.robot) * self.rng.uniform(0.4, 0.6)
            scene.obstacle = (centre, self.OBSTACLE_RADIUS)
        return scene

    def render(self, scene: ToyScene, robot_at: np.ndarray | None = None) -> np.ndarray:
        """``(3, S, S)`` image: dark floor, grey obstacle, coloured targets, white robot.

        Args:
            scene: what to draw.
            robot_at: draw the robot here instead (memory frames, subgoal images).
        """
        img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32) + 0.08
        if scene.obstacle is not None:
            c, r = scene.obstacle
            img[:, (self._xx - c[0]) ** 2 + (self._yy - c[1]) ** 2 < r**2] = 0.45
        for name, pos in scene.targets.items():
            if name in scene.visited:
                continue
            disc = (self._xx - pos[0]) ** 2 + (self._yy - pos[1]) ** 2 < 0.055**2
            img[:, disc] = np.array(COLOURS[name.split('#')[0]], dtype=np.float32)[:, None]
        robot = robot_at if robot_at is not None else scene.robot
        img[:, (self._xx - robot[0]) ** 2 + (self._yy - robot[1]) ** 2 < 0.04**2] = 1.0
        return img

    # ------------------------------------------------------------------ expert
    def targets_of(self, scene: ToyScene, colour: str) -> list[str]:
        """Keys of the unvisited targets drawn in ``colour``."""
        return [k for k in scene.targets if k.split("#")[0] == colour and k not in scene.visited]

    def reached(self, scene: ToyScene, colour: str, tol: float = 0.06) -> bool:
        """Whether the robot is on any target of ``colour``."""
        return any(np.linalg.norm(scene.robot - scene.targets[k]) < tol
                   for k in self.targets_of(scene, colour))

    def expert_path(self, scene: ToyScene, colour: str, speed: float = 1.0,
                    side: int | None = None) -> np.ndarray:
        """Expert positions from the robot to a target (around the obstacle), one per action.

        Args:
            scene: current scene.
            colour: target key, or a colour; with several targets of that colour the expert
                picks one of them uniformly at random (the source of multimodality).
            speed: step-size multiplier (``1`` = nominal).
            side: ``+1`` / ``-1`` detour side; random when ``None``.

        Returns:
            ``(T, 2)`` positions after each action (the last one is on the target).
        """
        keys = [colour] if colour in scene.targets else self.targets_of(scene, colour)
        goal_key = keys[self.rng.integers(len(keys))]
        start, goal = scene.robot, scene.targets[goal_key]
        waypoints = [goal]
        if scene.obstacle is not None:
            c, r = scene.obstacle
            seg = goal - start
            t = np.clip(np.dot(c - start, seg) / (np.dot(seg, seg) + 1e-9), 0, 1)
            if np.linalg.norm(start + t * seg - c) < r + 0.05:
                normal = np.array([-seg[1], seg[0]]) / (np.linalg.norm(seg) + 1e-9)
                side = side if side is not None else (1 if self.rng.random() < 0.5 else -1)
                waypoints = [c + side * normal * (r + 0.1), goal]
        path, pos = [], start.copy()
        for wp in waypoints:
            while np.linalg.norm(wp - pos) > 1e-6:
                delta = wp - pos
                dist = np.linalg.norm(delta)
                move = delta if dist <= self.step * speed else delta / dist * self.step * speed
                pos = pos + move
                path.append(pos.copy())
        return np.array(path)

    def chunk_from_path(self, start: np.ndarray, path: np.ndarray) -> np.ndarray:
        """First ``H`` scaled displacements along ``path`` (padded with zeros at the goal)."""
        positions = np.concatenate([start[None], path], axis=0)[: self.horizon + 1]
        deltas = np.diff(positions, axis=0)
        if len(deltas) < self.horizon:
            deltas = np.concatenate([deltas, np.zeros((self.horizon - len(deltas), 2))])
        return (deltas / self.ACTION_SCALE).astype(np.float32)

    @classmethod
    def positions_from_chunk(cls, start: np.ndarray, chunk: np.ndarray) -> np.ndarray:
        """Robot positions visited when executing a (scaled) action chunk from ``start``."""
        return start[None] + np.cumsum(np.asarray(chunk) * cls.ACTION_SCALE, axis=0)

    @staticmethod
    def collides(scene: ToyScene, positions: np.ndarray) -> bool:
        """Whether any position enters the obstacle."""
        if scene.obstacle is None:
            return False
        c, r = scene.obstacle
        return bool((np.linalg.norm(positions - c, axis=-1) < r).any())
