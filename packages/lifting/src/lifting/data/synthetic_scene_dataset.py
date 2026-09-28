"""Deterministic, procedurally generated multi-camera driving scenes."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from lifting.data.camera_rig import CameraRig
from lifting.data.label_rasterizer import LabelRasterizer
from lifting.data.object_sampler import ObjectSampler
from lifting.data.ray_cast_renderer import RayCastRenderer
from lifting.geometry import GridSpec


class SyntheticSceneDataset(Dataset):
    """Sample ``i`` is always the same scene (seeded by ``seed + i``).

    Each item is a dict with ``images (N, 3, H, W)``, ``intrinsics (N, 3, 3)``,
    ``cam_to_ego (N, 4, 4)``, ``depth (N, H, W)``, ``bev_labels (X, Y)`` and
    ``occupancy (X, Y, Z)``. Collate with :meth:`SceneBatch.collate`.

    All collaborators are injected, so rigs, object sets and renderers can be swapped.
    """

    def __init__(
        self,
        length: int,
        grid: GridSpec | None = None,
        rig: CameraRig | None = None,
        sampler: ObjectSampler | None = None,
        renderer: RayCastRenderer | None = None,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.grid = grid or GridSpec()
        self.rig = rig or CameraRig()
        self.sampler = sampler or ObjectSampler(
            extent=0.9 * min(self.grid.x_bounds[1], self.grid.y_bounds[1])
        )
        self.renderer = renderer or RayCastRenderer()
        self.rasterizer = LabelRasterizer(self.grid)
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.length:
            raise IndexError(index)
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        objects = self.sampler.sample(g)
        intrinsics = self.rig.intrinsics()
        cam_to_ego = self.rig.extrinsics(g)
        view = self.renderer.render(objects, intrinsics, cam_to_ego, self.rig.image_size)
        return {
            "images": view.images,
            "depth": view.depth,
            "intrinsics": intrinsics,
            "cam_to_ego": cam_to_ego,
            "bev_labels": self.rasterizer.bev(objects),
            "occupancy": self.rasterizer.occupancy(objects),
        }
