"""Axis-aligned metric voxel grid shared by every lifter.

Tensor layout conventions used across the library:

* BEV feature maps: ``(B, C, X, Y)`` - rows follow ego x (forward), columns ego y (left).
* Voxel volumes:    ``(B, C, X, Y, Z)``.
* TPV planes:       ``xy -> (B, C, X, Y)``, ``xz -> (B, C, X, Z)``, ``yz -> (B, C, Y, Z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GridSpec:
    """Metric bounds and resolution of the 3D region of interest.

    Attributes:
        x_bounds, y_bounds, z_bounds: ``(min, max)`` in metres (ego frame).
        resolution: number of cells ``(X, Y, Z)`` along each axis.
    """

    x_bounds: tuple[float, float] = (-16.0, 16.0)
    y_bounds: tuple[float, float] = (-16.0, 16.0)
    z_bounds: tuple[float, float] = (-1.0, 3.0)
    resolution: tuple[int, int, int] = (32, 32, 4)

    # ------------------------------------------------------------------ properties
    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return (self.x_bounds, self.y_bounds, self.z_bounds)

    @property
    def mins(self) -> tuple[float, float, float]:
        return tuple(b[0] for b in self.bounds)  # type: ignore[return-value]

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(b[1] - b[0] for b in self.bounds)  # type: ignore[return-value]

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        return tuple(e / n for e, n in zip(self.extents, self.resolution, strict=True))  # type: ignore[return-value]

    @property
    def bev_shape(self) -> tuple[int, int]:
        return self.resolution[0], self.resolution[1]

    @property
    def num_cells_bev(self) -> int:
        return self.resolution[0] * self.resolution[1]

    # ------------------------------------------------------------------ coordinates
    def axis_centers(self, axis: int, device: torch.device | str | None = None) -> Tensor:
        """Metric cell centres along ``axis`` (0 = x, 1 = y, 2 = z)."""
        lo, _ = self.bounds[axis]
        n = self.resolution[axis]
        size = self.voxel_size[axis]
        return lo + (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * size

    def voxel_centers(self, device: torch.device | str | None = None) -> Tensor:
        """``(X, Y, Z, 3)`` metric centres of every voxel."""
        xs, ys, zs = (self.axis_centers(a, device) for a in range(3))
        return torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)

    def axis_samples(self, axis: int, num: int, device: torch.device | str | None = None) -> Tensor:
        """``num`` points spread uniformly over the full extent of ``axis`` (bin centres)."""
        lo, hi = self.bounds[axis]
        return lo + (torch.arange(num, device=device, dtype=torch.float32) + 0.5) * (hi - lo) / num

    def pillar_points(self, num_anchors: int, device: torch.device | str | None = None) -> Tensor:
        """``(X, Y, A, 3)`` points spread uniformly along the height of each BEV pillar."""
        return self.line_points(axis=2, num_anchors=num_anchors, device=device)

    def line_points(
        self, axis: int, num_anchors: int, device: torch.device | str | None = None
    ) -> Tensor:
        """Anchors along ``axis`` for every cell of the plane spanned by the two other axes.

        Returns ``(R_a, R_b, A, 3)`` where ``a < b`` are the remaining axes, e.g. ``axis=2``
        gives BEV pillars ``(X, Y, A, 3)`` and ``axis=1`` gives ``(X, Z, A, 3)``.
        """
        a, b = (i for i in range(3) if i != axis)
        coords: list[Tensor] = [torch.empty(0)] * 3
        coords[a] = self.axis_centers(a, device)
        coords[b] = self.axis_centers(b, device)
        coords[axis] = self.axis_samples(axis, num_anchors, device)
        mesh = torch.stack(
            torch.meshgrid(*coords, indexing="ij"), dim=-1
        )  # X Y Z 3 (with A on axis)
        order = [a, b, axis]
        return mesh.permute(*order, 3).contiguous()

    def to_index(self, points: Tensor) -> Tensor:
        """Continuous cell coordinates; cell ``i`` covers ``[i, i + 1)``."""
        mins = points.new_tensor(self.mins)
        size = points.new_tensor(self.voxel_size)
        return (points - mins) / size

    def to_normalized(self, points: Tensor) -> Tensor:
        """Map metric points to ``[-1, 1]`` per axis (``align_corners=False`` convention)."""
        mins = points.new_tensor(self.mins)
        ext = points.new_tensor(self.extents)
        return 2.0 * (points - mins) / ext - 1.0

    def contains(self, points: Tensor) -> Tensor:
        """Boolean mask of points inside the grid bounds."""
        idx = self.to_index(points)
        res = points.new_tensor(self.resolution)
        return ((idx >= 0) & (idx < res)).all(dim=-1)

    def flat_bev_index(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Flattened BEV cell index ``x * Y + y`` and an in-bounds mask."""
        idx = self.to_index(points).floor().long()
        valid = self.contains(points)
        flat = idx[..., 0] * self.resolution[1] + idx[..., 1]
        return flat.clamp(0, self.num_cells_bev - 1), valid

    def flat_voxel_index(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Flattened voxel index ``(x * Y + y) * Z + z`` and an in-bounds mask."""
        idx = self.to_index(points).floor().long()
        valid = self.contains(points)
        _, ny, nz = self.resolution
        flat = (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]
        return flat.clamp(0, ny * nz * self.resolution[0] - 1), valid

    def with_resolution(self, resolution: tuple[int, int, int]) -> GridSpec:
        return GridSpec(self.x_bounds, self.y_bounds, self.z_bounds, resolution)
