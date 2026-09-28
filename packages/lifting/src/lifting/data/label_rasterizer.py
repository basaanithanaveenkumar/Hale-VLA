"""Ground-truth BEV and voxel labels for a box scene."""

from __future__ import annotations

import torch
from torch import Tensor

from lifting.data.scene_objects import SceneObjects
from lifting.geometry import GridSpec


class LabelRasterizer:
    """Rasterises boxes onto the BEV grid and the voxel grid.

    A BEV cell is labelled with an object class if any of its ``supersample**2``
    sub-points lies in the object's footprint.

    Args:
        grid: target grid.
        supersample: sub-samples per cell side.
    """

    def __init__(self, grid: GridSpec, supersample: int = 3) -> None:
        self.grid = grid
        self.supersample = supersample

    def _subpoints(self) -> Tensor:
        """``(X, Y, S*S, 2)`` metric sub-sample positions of every BEV cell."""
        s = self.supersample
        dx, dy, _ = self.grid.voxel_size
        offs = (torch.arange(s) + 0.5) / s - 0.5
        ox, oy = torch.meshgrid(offs * dx, offs * dy, indexing="ij")
        xs, ys = self.grid.axis_centers(0), self.grid.axis_centers(1)
        cx, cy = torch.meshgrid(xs, ys, indexing="ij")
        px = cx[..., None] + ox.reshape(-1)
        py = cy[..., None] + oy.reshape(-1)
        return torch.stack([px, py], -1)

    def _footprint_labels(self, objects: SceneObjects) -> tuple[Tensor, Tensor]:
        """Per BEV cell: class label ``(X, Y)`` and index of the covering object (``-1`` if none)."""
        nx, ny = self.grid.bev_shape
        if len(objects) == 0:
            return torch.zeros(nx, ny, dtype=torch.long), torch.full((nx, ny), -1)
        inside = objects.footprint_mask(self._subpoints()).any(2)  # (X, Y, K)
        covered = inside.any(-1)
        obj = inside.float().argmax(-1)
        labels = torch.where(covered, objects.labels[obj], torch.zeros_like(obj))
        return labels, torch.where(covered, obj, torch.full_like(obj, -1))

    def bev(self, objects: SceneObjects) -> Tensor:
        """``(X, Y)`` long tensor of class ids."""
        return self._footprint_labels(objects)[0]

    def occupancy(self, objects: SceneObjects) -> Tensor:
        """``(X, Y, Z)`` long tensor: a voxel is occupied if its footprint cell is covered
        and its centre height lies within ``[0, object height]``."""
        labels, obj = self._footprint_labels(objects)
        zs = self.grid.axis_centers(2)
        if len(objects) == 0:
            return torch.zeros(*labels.shape, zs.shape[0], dtype=torch.long)
        heights = objects.sizes[obj.clamp(min=0), 2]  # (X, Y)
        within = (zs >= 0) & (zs[None, None] <= heights[..., None])
        occ = within & (obj >= 0)[..., None]
        return torch.where(
            occ, labels[..., None].expand_as(occ), torch.zeros_like(occ, dtype=torch.long)
        )
