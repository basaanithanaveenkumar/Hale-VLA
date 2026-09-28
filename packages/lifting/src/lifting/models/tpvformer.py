"""Full TPVFormer for 3D semantic occupancy / LiDAR segmentation."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.encoders import ImageEncoder
from lifting.geometry import Cameras
from lifting.lifters.tpvformer import TPVAggregator, TPVFormerLifter


class TPVFormer(nn.Module):
    """Image encoder -> TPV encoder -> TPV aggregator."""

    def __init__(
        self, encoder: ImageEncoder, lifter: TPVFormerLifter, aggregator: TPVAggregator
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lifter = lifter
        self.aggregator = aggregator

    def forward(self, images: Tensor, cameras: Cameras, points: Tensor | None = None) -> dict:
        """
        Args:
            images: ``(B, N, 3, H, W)`` in ``[0, 1]``.
            cameras: calibration.
            points: optional ``(B, P, 3)`` metric points to classify.

        Returns:
            ``planes`` (:class:`TPVPlanes`), ``voxel_logits`` ``(B, K, X, Y, Z)`` and
            optionally ``point_logits`` ``(B, K, P)``.
        """
        b, n = images.shape[:2]
        feats = [f.view(b, n, *f.shape[1:]) for f in self.encoder(images.flatten(0, 1))]
        planes = self.lifter.lift_planes(feats, cameras)
        return {"planes": planes, **self.aggregator(planes, points)}
