"""A compact policy that reads *image-like* inputs with a variable number of channels, for the 3-D perception-interface experiments.

The same transformer is fed one of several **input representations** of the same scene (:func:`channels`): RGB; RGB + depth; RGB + a
camera-frame pointmap; RGB + a robot-frame pointmap; or a robot-frame **bird's-eye-view raster** built from the pointmaps of one or more
cameras (:func:`pointmaps_to_bev`). Only the input differs, so any gap in success is a statement about the *representation*.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from vla_foundations.world3d import (
    COLORS3D,
    Camera,
    Observation3D,
    expert_action,
    random_camera,
    render,
    sample_scene,
)

MODES = ("rgb", "rgbd", "rgb+pcam", "rgb+probot", "bev")
CHANNELS = {"rgb": 3, "rgbd": 4, "rgb+pcam": 6, "rgb+probot": 6, "bev": 4}


def pointmaps_to_bev(observations: list[Observation3D], size: int = 32) -> np.ndarray:
    """Rasterise the robot-frame pointmaps of one or more cameras into a view-independent top-down grid ``(4, S, S)``.

    Each cell holds the colour (3) and height (1) of the highest point that falls in it; table points (``z ≈ 0``) are left empty.
    """
    out = np.zeros((4, size, size), np.float32)
    pts, cols = [], []
    for ob in observations:
        pts.append(ob.points_robot.reshape(3, -1).T)
        cols.append(ob.rgb.reshape(3, -1).T)
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    keep = (
        (pts[:, 2] > 0.015)
        & (pts[:, 0] >= 0)
        & (pts[:, 0] < 1)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < 1)
    )
    pts, cols = pts[keep], cols[keep]
    order = np.argsort(
        pts[:, 2]
    )  # ascending height: later (higher) points overwrite lower ones in the same cell
    ix, iy = (pts[order, 0] * size).astype(int), (pts[order, 1] * size).astype(int)
    out[:3, iy, ix] = cols[order].T
    out[3, iy, ix] = pts[order, 2]
    return out


def channels(observations: list[Observation3D], mode: str) -> np.ndarray:
    """The network input for ``mode`` (first camera for the per-camera modes; all cameras for ``bev``)."""
    ob = observations[0]
    if mode == "rgb":
        return ob.rgb
    if mode == "rgbd":
        return np.concatenate([ob.rgb, ob.depth[None] / 1.5])
    if mode == "rgb+pcam":
        return np.concatenate([ob.rgb, ob.points_cam / 1.5])
    if mode == "rgb+probot":
        return np.concatenate([ob.rgb, ob.points_robot])
    if mode == "bev":
        return pointmaps_to_bev(observations, ob.rgb.shape[-1])
    raise ValueError(f"mode must be one of {MODES}")


class Policy3D(nn.Module):
    """Patch tokens of a ``C``-channel image + one query token (colour word and end-effector position) → a 3-D action."""

    def __init__(
        self,
        in_channels: int,
        width: int = 64,
        depth: int = 3,
        heads: int = 4,
        size: int = 32,
        patch: int = 4,
    ):
        super().__init__()
        self.patchify = nn.Conv2d(in_channels, width, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, (size // patch) ** 2, width))
        self.colour = nn.Embedding(len(COLORS3D), width)
        self.state = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, 0.0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, ee: torch.Tensor, colour: torch.Tensor) -> torch.Tensor:
        tokens = self.patchify(x).flatten(2).transpose(1, 2) + self.pos
        query = (self.colour(colour) + self.state(ee))[:, None]
        return self.head(self.encoder(torch.cat([query, tokens], 1))[:, 0])


class Dataset3D:
    """Pre-rendered scenes; every sample is ``(input, end-effector position, colour) → expert action`` with i.i.d. end-effector positions.

    Args:
        n_scenes: number of rendered (scene, camera set) pairs.
        mode: input representation (:data:`MODES`).
        fixed_camera: one fixed viewpoint, or a random one per scene.
        occluder: a tall pillar hides the target when seen from the *fixed* viewpoint.
        second_camera: ``bev`` only: also render a second, opposite view and merge both pointmaps.
    """

    def __init__(
        self,
        n_scenes: int,
        mode: str,
        fixed_camera: bool,
        seed: int,
        occluder: bool = False,
        second_camera: bool = False,
        size: int = 32,
    ):
        rng = np.random.default_rng(seed)
        self.mode, self.scenes, self.inputs, self.targets = mode, [], [], []
        for _ in range(n_scenes):
            target = str(rng.choice(list(COLORS3D)))
            scene = sample_scene(rng, occluder=occluder, target=target)
            cams = [random_camera(rng, fixed_camera, size)]
            if second_camera:
                cams.append(
                    Camera.from_angles(
                        float(
                            np.degrees(np.arctan2(*(cams[0].position - [0.5, 0.5, 0.1])[[1, 0]]))
                            + 180.0
                        ),
                        50.0,
                        1.2,
                        size,
                    )
                )
            obs = [render(scene, c) for c in cams]
            self.scenes.append(scene)
            self.inputs.append(channels(obs, mode))
        self.inputs = torch.from_numpy(np.stack(self.inputs))

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.scenes), batch)
        colour = rng.integers(0, len(COLORS3D), batch)
        ee = np.stack(
            [
                rng.uniform(0.05, 0.95, batch),
                rng.uniform(0.05, 0.95, batch),
                rng.uniform(0.05, 0.3, batch),
            ],
            1,
        ).astype(np.float32)
        goal = np.stack([self.scenes[i].objects[list(COLORS3D)[c]] for i, c in zip(idx, colour)])
        act = np.stack([expert_action(e, g) for e, g in zip(ee, goal)])
        return (
            self.inputs[idx],
            torch.from_numpy(ee),
            torch.from_numpy(colour),
            torch.from_numpy(act),
        )


def train3d(
    model: Policy3D,
    data: Dataset3D,
    steps: int = 3000,
    batch: int = 128,
    lr: float = 2e-3,
    seed: int = 0,
) -> list[float]:
    """Behaviour cloning (MSE to the expert's action). Returns the loss curve."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(i / steps, 1.0)))
    )
    losses = []
    for _ in range(steps):
        x, ee, colour, act = data.sample(batch, rng)
        loss = (model(x, ee, colour) - act).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def rollout3d(
    model: Policy3D,
    data: Dataset3D,
    n: int,
    seed: int,
    max_steps: int = 60,
    radius: float = 0.06,
    action_noise: float = 0.0,
) -> dict:
    """Closed loop on the first ``n`` scenes of ``data`` (random target colour and start per scene). Returns success rate and final error."""
    rng = np.random.default_rng(seed)
    n = min(n, len(data.scenes))
    colour = rng.integers(0, len(COLORS3D), n)
    ee = np.stack(
        [rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.3, n)], 1
    ).astype(np.float32)
    goal = np.stack([data.scenes[i].objects[list(COLORS3D)[c]] for i, c in enumerate(colour)])
    done, steps = np.zeros(n, bool), np.full(n, max_steps)
    x, c_t = data.inputs[:n], torch.from_numpy(colour)
    for t in range(max_steps):
        a = model(x, torch.from_numpy(ee), c_t).numpy()
        if action_noise:
            a = a + rng.normal(0, action_noise, a.shape)
        ee = np.where(
            done[:, None], ee, np.clip(ee + 0.05 * np.clip(a, -1.5, 1.5), [0, 0, 0], [1, 1, 0.5])
        ).astype(np.float32)
        newly = (~done) & (np.linalg.norm(ee - goal, axis=1) <= radius)
        steps[newly], done = t + 1, done | newly
    return dict(
        success=float(done.mean()),
        final_error=float(np.linalg.norm(ee - goal, axis=1).mean()),
        steps=float(steps[done].mean()) if done.any() else float("nan"),
    )
