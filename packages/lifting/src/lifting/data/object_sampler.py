"""Random, non-overlapping placement of objects around the ego vehicle."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from lifting.data.object_class import DEFAULT_CLASSES, ObjectClass
from lifting.data.scene_objects import SceneObjects


class ObjectSampler:
    """Samples a random scene layout.

    Args:
        extent: objects are placed in ``[-extent, extent]^2`` (metres).
        num_objects: inclusive ``(min, max)`` number of objects.
        classes: categories to draw from (uniformly).
        min_ego_distance: keep-out radius around the ego origin.
    """

    def __init__(
        self,
        extent: float = 14.0,
        num_objects: tuple[int, int] = (4, 10),
        classes: Sequence[ObjectClass] = DEFAULT_CLASSES,
        min_ego_distance: float = 3.0,
    ) -> None:
        self.extent = extent
        self.num_objects = num_objects
        self.classes = tuple(classes)
        self.min_ego_distance = min_ego_distance

    @staticmethod
    def _uniform(rng: tuple[float, float], g: torch.Generator) -> float:
        return rng[0] + (rng[1] - rng[0]) * torch.rand((), generator=g).item()

    def sample(self, generator: torch.Generator) -> SceneObjects:
        lo, hi = self.num_objects
        target = int(torch.randint(lo, hi + 1, (), generator=generator))
        centers, radii, sizes, yaws, labels, colors = [], [], [], [], [], []
        attempts = 0
        while len(centers) < target and attempts < 50 * target:
            attempts += 1
            cls = self.classes[int(torch.randint(len(self.classes), (), generator=generator))]
            size = [self._uniform(r, generator) for r in (cls.length, cls.width, cls.height)]
            xy = (torch.rand(2, generator=generator) * 2 - 1) * self.extent
            radius = 0.5 * (size[0] ** 2 + size[1] ** 2) ** 0.5
            if xy.norm() < self.min_ego_distance + radius:
                continue
            if any((xy - c).norm() < radius + r for c, r in zip(centers, radii, strict=True)):
                continue
            shade = 0.8 + 0.4 * torch.rand((), generator=generator).item()
            centers.append(xy)
            radii.append(radius)
            sizes.append(size)
            yaws.append(float(torch.rand((), generator=generator) * torch.pi))
            labels.append(cls.label)
            colors.append([min(1.0, c * shade) for c in cls.color])
        return SceneObjects(
            centers=torch.stack(centers) if centers else torch.zeros(0, 2),
            sizes=torch.tensor(sizes, dtype=torch.float32).view(-1, 3),
            yaws=torch.tensor(yaws, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.long),
            colors=torch.tensor(colors, dtype=torch.float32).view(-1, 3),
        )
