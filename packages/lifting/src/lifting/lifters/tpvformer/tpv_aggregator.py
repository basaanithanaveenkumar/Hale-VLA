"""TPV aggregator: decode voxel / point semantics from the three planes."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.geometry import GridSpec
from lifting.lifters.tpvformer.tpv_planes import TPVPlanes


class TPVAggregator(nn.Module):
    """Sum plane features at a 3D location and classify it with a small MLP.

    Mirrors ``tpvformer10/tpv_aggregator.py`` without mmcv / mmseg.

    Args:
        grid: region the planes cover (used to normalise query points).
        in_channels: plane channels.
        num_classes: semantic classes (including "empty").
        hidden_channels: MLP width.
        scale: bilinear upsampling of the planes before decoding voxels.
    """

    def __init__(
        self,
        grid: GridSpec,
        in_channels: int,
        num_classes: int,
        hidden_channels: int | None = None,
        scale: int = 1,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.scale = scale
        hidden = hidden_channels or 2 * in_channels
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.Softplus(), nn.Linear(hidden, in_channels)
        )
        self.classifier = nn.Linear(in_channels, num_classes)

    def classify(self, features: Tensor) -> Tensor:
        """``(..., C)`` -> ``(..., K)``."""
        return self.classifier(self.decoder(features))

    def forward(self, planes: TPVPlanes, points: Tensor | None = None) -> dict[str, Tensor]:
        """
        Args:
            planes: TPV features.
            points: optional ``(B, P, 3)`` metric ego-frame query points (e.g. a LiDAR sweep).

        Returns:
            ``voxel_logits`` ``(B, K, X*s, Y*s, Z*s)`` and, with ``points``, ``point_logits`` ``(B, K, P)``.
        """
        up = planes.upsample(self.scale)
        voxels = up.to_voxels().permute(0, 2, 3, 4, 1)
        out = {"voxel_logits": self.classify(voxels).permute(0, 4, 1, 2, 3)}
        if points is not None:
            feats = up.sample_points(self.grid.to_normalized(points)).transpose(1, 2)
            out["point_logits"] = self.classify(feats).transpose(1, 2)
        return out
