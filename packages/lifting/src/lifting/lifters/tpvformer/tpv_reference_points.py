"""Reference points for TPV image cross-attention and cross-view hybrid attention."""

from __future__ import annotations

import torch
from torch import Tensor

from lifting.geometry import Cameras, GridSpec
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes

#: The axis orthogonal to each plane, in ``xy, xz, yz`` order.
_NORMAL_AXIS = (2, 1, 0)


class TPVReferencePoints:
    """Builds geometric reference points for the three TPV planes.

    Args:
        grid: 3D region; plane resolutions follow ``grid.resolution``.
        num_anchors: 3D anchors per query along the plane normal, per plane
            (``xy`` pillars along z, ``xz`` lines along y, ``yz`` lines along x).
    """

    def __init__(self, grid: GridSpec, num_anchors: tuple[int, int, int] = (4, 8, 8)) -> None:
        self.grid = grid
        self.num_anchors = num_anchors

    @property
    def plane_shapes(self) -> list[tuple[int, int]]:
        return TPVPlanes.plane_shapes(self.grid.resolution)

    def anchor_points(self, plane: int, device: torch.device | str | None = None) -> Tensor:
        """``(Q_plane, A, 3)`` metric anchors of every query on ``plane`` (0 = xy, 1 = xz, 2 = yz)."""
        pts = self.grid.line_points(_NORMAL_AXIS[plane], self.num_anchors[plane], device)
        return pts.reshape(-1, self.num_anchors[plane], 3)

    def camera_reference_points(self, cameras: Cameras) -> list[tuple[Tensor, Tensor]]:
        """Per plane: normalised projections ``(B, N, Q, A, 2)`` and visibility ``(B, N, Q, A)``."""
        b, n = cameras.batch_size, cameras.num_cameras
        out = []
        for plane in range(3):
            anchors = self.anchor_points(plane, cameras.device)
            q, a, _ = anchors.shape
            uv, _, valid = cameras.project(anchors.view(1, -1, 3).expand(b, -1, -1))
            out.append((cameras.normalize_uv(uv).view(b, n, q, a, 2), valid.view(b, n, q, a)))
        return out

    def hybrid_reference_points(self, device: torch.device | str | None = None) -> Tensor:
        """``(Q_total, 3, 2)`` where each query looks at the *same 3D location* on every plane.

        A query on one plane fixes two coordinates; the missing one is set to the centre
        of its axis. The location is then expressed in each plane's ``(col, row)`` frame.
        """
        nx, ny, nz = self.grid.resolution

        def centers(n: int) -> Tensor:
            return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n

        half = torch.tensor(0.5, device=device)
        xyz_per_plane = []
        for (h, w), normal in zip(self.plane_shapes, _NORMAL_AXIS, strict=True):
            r, c = torch.meshgrid(centers(h), centers(w), indexing="ij")
            r, c = r.reshape(-1), c.reshape(-1)
            axes = [i for i in range(3) if i != normal]
            coords = [half.expand_as(r)] * 3
            coords[axes[0]], coords[axes[1]] = r, c
            xyz_per_plane.append(torch.stack(coords, -1))
        xyz = torch.cat(xyz_per_plane, 0)  # (Q_total, 3) in [0, 1]
        x, y, z = xyz.unbind(-1)
        # (col, row) on planes xy (X rows, Y cols), xz (X rows, Z cols), yz (Y rows, Z cols)
        return torch.stack(
            [torch.stack([y, x], -1), torch.stack([z, x], -1), torch.stack([z, y], -1)], dim=1
        )
