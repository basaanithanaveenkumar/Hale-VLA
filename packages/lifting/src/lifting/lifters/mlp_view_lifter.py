"""Geometry-free MLP view transform (VPN-style) - the control baseline for verification."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.layers import ConvNormAct
from lifting.lifters.base import LIFTERS, BaseLifter


@LIFTERS.register("mlp_view")
class MLPViewLifter(BaseLifter):
    """Learns a fixed per-camera mapping from image cells to BEV cells, ignoring calibration.

    This is the View-Parsing-Network idea (Pan et al., 2020). It cannot adapt to camera
    poses it was not trained on, which makes it the ideal *negative control* for the
    geometric lifters in :mod:`lifting.verification`.

    Args:
        grid, in_channels, out_channels: see :class:`BaseLifter`.
        feature_size: ``(h, w)`` of the pyramid level used.
        num_cameras: number of cameras (one MLP each).
        level: pyramid level (default: coarsest).
    """

    uses_geometry = False

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        out_channels: int,
        feature_size: tuple[int, int] = (3, 6),
        num_cameras: int = 4,
        level: int = -1,
    ) -> None:
        super().__init__(grid, in_channels, out_channels)
        self.level = level
        cells = feature_size[0] * feature_size[1]
        self.view_mlps = nn.ModuleList(
            nn.Linear(cells, grid.num_cells_bev) for _ in range(num_cameras)
        )
        self.output = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 3), nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        feats = features[self.level]  # B N C h w
        b, n, c = feats.shape[:3]
        if n != len(self.view_mlps):
            raise ValueError(f"MLPViewLifter was built for {len(self.view_mlps)} cameras, got {n}")
        flat = feats.flatten(3)
        bev = sum(mlp(flat[:, i]) for i, mlp in enumerate(self.view_mlps))
        nx, ny = self.grid.bev_shape
        return self.output(bev.view(b, c, nx, ny))
