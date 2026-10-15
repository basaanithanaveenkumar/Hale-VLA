"""``model.bev``: multi-camera bird's-eye-view scene tokens (Halos-lift ``lifting`` package)."""

from __future__ import annotations

from pydantic import Field

from config.sections.base import VLASection


class BEVConfig(VLASection):
    """Surround-view cameras lifted to a metric BEV grid, pooled into tokens before the image patches.

    Batches must then carry ``bev_images`` ``[B, N_cam, 3, H, W]`` and ``bev_cameras``
    (``lifting.Cameras``); the shipped EO / DROID / MoMa loaders do not provide them.
    """

    OVERRIDES = {"enabled": "use_bev", "lifter": "bev_lifter", "num_cameras": "bev_num_cameras",
                 "image_size": "bev_image_size", "bounds": "bev_bounds", "resolution": "bev_resolution",
                 "channels": "bev_channels", "token_pool": "bev_token_pool"}

    enabled: bool = False
    lifter: str = "bevformer"              # simple_bev | lift_splat | bevformer | tpvformer | tiim ...
    num_cameras: int = 6
    image_size: list[int] = Field(default_factory=lambda: [224, 224])
    bounds: list[list[float]] = Field(default_factory=lambda: [[-25.6, 25.6], [-25.6, 25.6], [-2.0, 4.0]])
    resolution: list[int] = Field(default_factory=lambda: [64, 64, 6])
    channels: int = 64
    token_pool: int = 4                    # BEV cells per token side: 64x64 -> 16x16 = 256 tokens
