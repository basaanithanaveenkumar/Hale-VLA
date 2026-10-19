"""A tiny 3-D tabletop for the perception-VLA interface: pinhole cameras, a ray-cast renderer, depth and *pointmaps*.

Frames: the **robot frame** has its origin at the table corner, ``x, y ∈ [0, 1]`` along the table and ``z`` up. A camera looks at the table
centre from a pose (azimuth, elevation, distance); the **camera frame** is OpenCV-style (``x`` right, ``y`` down, ``z`` forward).

The renderer ray-casts axis-aligned boxes (coloured cubes, plus optional grey *occluder* pillars) and the table plane, and returns RGB, a
depth map (distance along the camera's ``z``) and two *pointmaps* - the 3-D position of what each pixel sees, expressed in the camera frame
or in the robot frame. A pointmap in the robot frame is what a "robot-centric" 3-D VLA feeds its geometry encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COLORS3D = {"red": (0.9, 0.15, 0.15), "green": (0.15, 0.8, 0.2), "blue": (0.2, 0.35, 0.95)}
TABLE_RGB = (0.28, 0.28, 0.3)
OCCLUDER_RGB = (0.55, 0.55, 0.55)
CUBE_HALF = 0.05
CENTRE = np.array([0.5, 0.5, 0.1])


@dataclass
class Camera:
    """A pinhole camera. ``rotation`` rows are the camera axes (right, down, forward) in the robot frame."""

    position: np.ndarray
    rotation: np.ndarray
    size: int = 32
    fov_deg: float = 55.0

    @classmethod
    def look_at(cls, position, target=CENTRE, size: int = 32, fov_deg: float = 55.0) -> Camera:
        position, target = np.asarray(position, float), np.asarray(target, float)
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return cls(position, np.stack([right, down, forward]), size, fov_deg)

    @classmethod
    def from_angles(
        cls, azimuth_deg: float, elevation_deg: float, distance: float, size: int = 32
    ) -> Camera:
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        offset = distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return cls.look_at(CENTRE + offset, size=size)

    @property
    def focal(self) -> float:
        return (self.size / 2) / np.tan(np.radians(self.fov_deg) / 2)

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """Robot-frame points (N, 3) → camera frame."""
        return (np.asarray(points) - self.position) @ self.rotation.T

    def to_robot(self, points_cam: np.ndarray) -> np.ndarray:
        return np.asarray(points_cam) @ self.rotation + self.position

    def project(self, points: np.ndarray) -> np.ndarray:
        """Robot-frame points (N, 3) → ``(u, v, depth)`` with ``u`` = column, ``v`` = row (pixel centres at ``i + 0.5``)."""
        p = self.to_camera(points)
        u = self.focal * p[:, 0] / p[:, 2] + self.size / 2
        v = self.focal * p[:, 1] / p[:, 2] + self.size / 2
        return np.stack([u, v, p[:, 2]], 1)

    def unproject(self, u, v, depth) -> np.ndarray:
        """Pixel coordinates and depth → robot-frame points (the inverse of :meth:`project`)."""
        x = (np.asarray(u) - self.size / 2) * np.asarray(depth) / self.focal
        y = (np.asarray(v) - self.size / 2) * np.asarray(depth) / self.focal
        return self.to_robot(np.stack([x, y, np.asarray(depth)], -1))

    def rays(self) -> np.ndarray:
        """(S·S, 3) unit ray directions in the robot frame, one per pixel centre (row-major)."""
        s = self.size
        v, u = np.meshgrid(np.arange(s) + 0.5, np.arange(s) + 0.5, indexing="ij")
        d_cam = np.stack(
            [(u - s / 2) / self.focal, (v - s / 2) / self.focal, np.ones_like(u)], -1
        ).reshape(-1, 3)
        d = d_cam @ self.rotation
        return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass
class Scene3D:
    """Coloured cubes (colour → centre) and grey occluder pillars ``(x, y, height)`` standing on the table."""

    objects: dict
    occluders: list = field(default_factory=list)


@dataclass
class Observation3D:
    rgb: np.ndarray  # (3, S, S)
    depth: np.ndarray  # (S, S) distance along the camera z axis
    points_cam: np.ndarray  # (3, S, S) in the camera frame
    points_robot: np.ndarray  # (3, S, S) in the robot frame
    camera: Camera


def _boxes(scene: Scene3D):
    """(lo, hi, rgb) arrays of every box in the scene."""
    lo, hi, col = [], [], []
    for name, c in scene.objects.items():
        c = np.asarray(c, float)
        lo.append(c - CUBE_HALF)
        hi.append(c + CUBE_HALF)
        col.append(COLORS3D[name])
    for x, y, h in scene.occluders:
        lo.append([x - 0.06, y - 0.06, 0.0])
        hi.append([x + 0.06, y + 0.06, h])
        col.append(OCCLUDER_RGB)
    return np.array(lo), np.array(hi), np.array(col)


def render(scene: Scene3D, camera: Camera) -> Observation3D:
    """Ray-cast the scene: nearest box hit, else the table plane ``z = 0``, else a far background."""
    o = camera.position
    d = camera.rays()  # (N, 3)
    n = len(d)
    t_best = np.full(n, np.inf)
    rgb = np.tile(np.array(TABLE_RGB), (n, 1))
    # table plane
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = np.where(d[:, 2] < -1e-9, -o[2] / d[:, 2], np.inf)
    hit = np.isfinite(t_plane) & (t_plane > 0)
    t_best[hit] = t_plane[hit]
    lo, hi, col = _boxes(scene)
    for k in range(len(lo)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo[k] - o) / d, (hi[k] - o) / d
        t_near, t_far = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
        ok = (t_near <= t_far) & (t_far > 0) & (t_near < t_best)
        t_best[ok] = np.maximum(t_near[ok], 0)
        rgb[ok] = col[k]
    far = ~np.isfinite(t_best)
    t_best[far] = 3.0
    rgb[far] = 0.05
    pts = o + d * t_best[:, None]
    s = camera.size
    pc = camera.to_camera(pts)
    return Observation3D(
        rgb=rgb.reshape(s, s, 3).transpose(2, 0, 1).astype(np.float32),
        depth=pc[:, 2].reshape(s, s).astype(np.float32),
        points_cam=pc.reshape(s, s, 3).transpose(2, 0, 1).astype(np.float32),
        points_robot=pts.reshape(s, s, 3).transpose(2, 0, 1).astype(np.float32),
        camera=camera,
    )


def sample_scene(
    rng: np.random.Generator,
    heights=(0.05, 0.15, 0.25),
    occluder: bool = False,
    target: str | None = None,
) -> Scene3D:
    """Three cubes (one per colour) at random table positions and heights; with ``occluder`` a tall pillar stands in front of ``target``
    *as seen from the default camera direction* (azimuth −90°, i.e. between the camera and the cube along ``-y``)."""
    for _ in range(1000):
        xy = rng.uniform(0.15, 0.85, size=(3, 2))
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(3)
        if d.min() >= 0.22:
            break
    z = rng.choice(heights, size=3)
    objects = {c: np.array([*xy[i], z[i]]) for i, c in enumerate(COLORS3D)}
    occluders = []
    if occluder and target is not None:
        t = objects[target]
        # the pillar stands in front of the target with a random offset, and a decoy pillar elsewhere, so that the pillar alone
        # does not reveal where the target is
        occluders.append(
            (
                float(np.clip(t[0] + rng.uniform(-0.04, 0.04), 0.1, 0.9)),
                float(np.clip(t[1] - rng.uniform(0.10, 0.2), 0.05, 0.95)),
                0.42,
            )
        )
        occluders.append((float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)), 0.42))
    return Scene3D(objects, occluders)


def random_camera(
    rng: np.random.Generator, fixed: bool = False, size: int = 32, azimuth: float | None = None
) -> Camera:
    """A fixed viewpoint (azimuth −90°, elevation 50°) or a random one (azimuth ±60° around it, elevation 35-65°, distance 1.0-1.4)."""
    if fixed:
        return Camera.from_angles(-90.0, 50.0, 1.2, size)
    az = azimuth if azimuth is not None else -90.0 + rng.uniform(-60, 60)
    return Camera.from_angles(az, rng.uniform(35, 65), rng.uniform(1.0, 1.4), size)


def expert_action(ee: np.ndarray, goal: np.ndarray, max_step: float = 0.05) -> np.ndarray:
    """Normalised 3-D displacement toward ``goal`` (unit length = one ``max_step``), shrinking near the goal so the arm does not overshoot."""
    delta = goal - ee
    dist = float(np.linalg.norm(delta))
    return (delta / max(dist, 1e-8) * min(1.0, dist / max_step)).astype(np.float32)
