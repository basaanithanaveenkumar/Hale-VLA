"""Measuring multimodality: does a policy *cover* the modes of the demonstrations, or average them?

For the task ``"go to any"`` two objects are equally good targets. For one fixed observation we draw many chunks and ask which object
each is aimed at (net displacement within ``tolerance_deg`` of the direction to that object). A policy that models the
distribution splits its samples between the two objects; one that regresses the mean aims at **neither** (the middle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from open_vlas.data import VLABatch
from open_vlas.embodiments import EMBODIMENTS
from open_vlas.evaluation import make_any_scenarios
from pi_family.toy import DemoGenerator, PointWorld


@dataclass
class ModeCoverage:
    """Fractions of sampled chunks aimed at the first / second object, or at neither."""

    first: float
    second: float
    neither: float

    @property
    def aimed(self) -> float:
        return self.first + self.second

    @property
    def balance(self) -> float:
        """1 when both modes are sampled equally, 0 when one mode is never produced."""
        return 2 * min(self.first, self.second) / max(self.aimed, 1e-9) if self.aimed > 0 else 0.0


def separated_any_scenarios(
    world: PointWorld, n: int, seed: int, min_angle_deg: float = 60.0
) -> list:
    """ "Go to any" scenes whose two objects lie at least ``min_angle_deg`` apart *as seen from the robot* (so that the average of the two
    modes cannot accidentally point at either of them), and at least 0.25 away from it."""
    gen = DemoGenerator(world)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        sc = make_any_scenarios(gen, "point2d", 1, int(rng.integers(1 << 30)))[0]
        robot = np.asarray(sc.scene.robot)
        d = [np.asarray(xy) - robot for xy in sc.scene.objects.values()]
        if min(np.linalg.norm(v) for v in d) < 0.25:
            continue
        angle = np.degrees(
            np.arccos(np.clip(d[0] @ d[1] / (np.linalg.norm(d[0]) * np.linalg.norm(d[1])), -1, 1))
        )
        if angle >= min_angle_deg:
            out.append(sc)
    return out


def mode_coverage(
    policy,
    world: PointWorld,
    n_scenes: int = 40,
    samples: int = 32,
    tolerance_deg: float = 20.0,
    seed: int = 0,
) -> ModeCoverage:
    """``policy(VLABatch) -> (B, H, A)``: for each of ``n_scenes`` well-separated "go to any" scenes draw ``samples`` chunks and classify them."""
    scenes = separated_any_scenarios(world, n_scenes, seed)
    counts = np.zeros(3)
    for sc in scenes:
        objs = [np.asarray(xy) for xy in sc.scene.objects.values()]
        img = torch.from_numpy(world.render(sc.scene.robot, sc.scene.objects))[None].repeat(
            samples, 1, 1, 1
        )
        state = torch.zeros(samples, 3)
        state[:, :2] = torch.from_numpy(np.asarray(sc.scene.robot))
        batch = VLABatch(
            img, state, [sc.task] * samples, torch.full((samples,), EMBODIMENTS["point2d"].id)
        )
        with torch.no_grad():
            chunks = policy(batch).cpu().numpy()
        net = chunks[:, :, :2].sum(1)
        for v in net:
            norm = np.linalg.norm(v)
            hit = 2
            if norm > 1e-6:
                angles = [
                    np.degrees(
                        np.arccos(
                            np.clip(
                                v
                                @ (o - sc.scene.robot)
                                / (norm * np.linalg.norm(o - sc.scene.robot) + 1e-9),
                                -1,
                                1,
                            )
                        )
                    )
                    for o in objs
                ]
                best = int(np.argmin(angles))
                hit = best if angles[best] <= tolerance_deg else 2
            counts[hit] += 1
    counts /= counts.sum()
    return ModeCoverage(float(counts[0]), float(counts[1]), float(counts[2]))
