"""
Multi-camera bird's-eye-view (BEV) scene encoder for Halo-VLA.

Lifts surround-view camera images into a metric BEV grid with any lifter from the
``lifting`` package (Simple-BEV, Lift-Splat, BEVFormer, TIIM, TPVFormer, ...) and turns
the grid into a short sequence of tokens that the decoder transformer can attend to.

``lifting`` (Halos-lift) is an optional dependency: it is imported when an encoder is built,
so this module imports without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from hale_core.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from lifting import Cameras, GridSpec

    from config.model_config import HaloVLMConfig


def _lifting() -> Any:
    """The ``lifting`` package, or a :class:`MissingDependencyError` with the install hint."""
    try:
        import lifting
        import lifting.decoders
        import lifting.encoders
        import lifting.models
    except ImportError as exc:
        raise MissingDependencyError(
            "lifting", "multi-camera BEV tokens need Halos-lift: pip install 'lifting @ git+https://github.com/"
            "basaanithanaveenkumar/Halos-lift.git'"
        ) from exc
    return lifting


class BEVSceneEncoder(nn.Module):
    """[B, N_cam, 3, H, W] images + calibration -> [B, num_tokens, emb_dim] BEV tokens.

    Args:
        emb_dim:      output token width (the VLA embedding size).
        lifter:       name of a registered ``lifting`` lifter.
        grid:         metric BEV region and resolution (default ``lifting.GridSpec()``).
        image_size:   (H, W) of the camera images.
        num_cameras:  cameras in the rig.
        channels:     width of the image encoder / BEV features.
        token_pool:   BEV cells averaged into one token (per side).

    Raises:
        MissingDependencyError: if the ``lifting`` package is not installed.
    """

    def __init__(
        self,
        emb_dim: int,
        lifter: str = "bevformer",
        grid: GridSpec | None = None,
        image_size: tuple[int, int] = (224, 224),
        num_cameras: int = 6,
        channels: int = 64,
        token_pool: int = 4,
    ) -> None:
        super().__init__()
        lifting = _lifting()
        self.grid = grid or lifting.GridSpec()
        self.image_encoder = lifting.encoders.SimpleConvEncoder(out_channels=channels, num_levels=2)
        self.lifter = lifting.models.build_lifter(
            lifter, self.grid, channels, image_size, num_cameras, self.image_encoder
        )
        self.bev_decoder = lifting.decoders.BEVDecoder(channels, channels)
        self.pool = nn.AvgPool2d(token_pool, ceil_mode=True)
        nx, ny = self.grid.bev_shape
        self.num_tokens = -(-nx // token_pool) * -(-ny // token_pool)
        self.proj = nn.Linear(channels, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @classmethod
    def from_config(cls, config: HaloVLMConfig) -> BEVSceneEncoder:
        """Build the encoder from the ``bev_*`` fields of a :class:`HaloVLMConfig`."""
        grid = _lifting().GridSpec(*config.bev_bounds, resolution=tuple(config.bev_resolution))
        return cls(
            emb_dim=config.emb_dim,
            lifter=config.bev_lifter,
            grid=grid,
            image_size=tuple(config.bev_image_size),
            num_cameras=config.bev_num_cameras,
            channels=config.bev_channels,
            token_pool=config.bev_token_pool,
        )

    def bev_features(self, images: torch.Tensor, cameras: Cameras) -> torch.Tensor:
        """[B, N, 3, H, W] -> [B, C, X, Y] BEV feature map."""
        B, N = images.shape[:2]
        feats = [f.view(B, N, *f.shape[1:]) for f in self.image_encoder(images.flatten(0, 1))]
        return self.bev_decoder(self.lifter(feats, cameras))

    def forward(self, images: torch.Tensor, cameras: Cameras) -> torch.Tensor:
        """Returns BEV tokens [B, num_tokens, emb_dim]."""
        bev = self.pool(self.bev_features(images, cameras))  # [B, C, X', Y']
        tokens = bev.flatten(2).transpose(1, 2)               # [B, X'*Y', C]
        return self.proj(tokens) + self.pos_embed
