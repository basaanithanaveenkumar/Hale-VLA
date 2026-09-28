"""Encoder -> lifter -> BEV decoder -> head."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.decoders import BEVDecoder, SegmentationHead
from lifting.encoders import ImageEncoder
from lifting.geometry import Cameras
from lifting.lifters import BaseLifter


class BEVSegmentationModel(nn.Module):
    """Multi-camera BEV semantic segmentation network.

    All four parts are injected, so any encoder can be paired with any lifter.
    """

    def __init__(
        self,
        encoder: ImageEncoder,
        lifter: BaseLifter,
        decoder: BEVDecoder,
        head: SegmentationHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lifter = lifter
        self.decoder = decoder
        self.head = head

    def encode_images(self, images: Tensor) -> list[Tensor]:
        """``(B, N, 3, H, W)`` -> pyramid of ``(B, N, C, H_l, W_l)``."""
        b, n = images.shape[:2]
        return [f.view(b, n, *f.shape[1:]) for f in self.encoder(images.flatten(0, 1))]

    def forward(self, images: Tensor, cameras: Cameras) -> dict[str, Tensor]:
        """
        Args:
            images: ``(B, N, 3, H, W)`` in ``[0, 1]``.
            cameras: calibration of the ``N`` cameras.

        Returns:
            ``bev_features`` ``(B, C, X, Y)`` and ``logits`` ``(B, K, X, Y)``.
        """
        bev = self.lifter(self.encode_images(images), cameras)
        return {"bev_features": bev, "logits": self.head(self.decoder(bev))}
