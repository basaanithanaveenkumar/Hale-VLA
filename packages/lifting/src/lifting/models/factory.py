"""One-line constructors for complete models."""

from __future__ import annotations

import inspect
import math
from typing import Any

from lifting.decoders import BEVDecoder, SegmentationHead
from lifting.encoders import ImageEncoder, SimpleConvEncoder
from lifting.geometry import DepthBins, GridSpec
from lifting.lifters import LIFTERS, TPVAggregator, TPVFormerLifter
from lifting.models.bev_segmentation_model import BEVSegmentationModel
from lifting.models.tpvformer import TPVFormer


def _feature_size(image_size: tuple[int, int], stride: int) -> tuple[int, int]:
    return math.ceil(image_size[0] / stride), math.ceil(image_size[1] / stride)


def _depth_bins_for(grid: GridSpec) -> DepthBins:
    """Default depth bins, widened when the grid reaches beyond their range."""
    default = DepthBins()
    reach = max(
        math.hypot(x, y) for x in grid.x_bounds for y in grid.y_bounds
    )  # farthest BEV corner from the ego origin
    if reach <= default.max_depth:
        return default
    return DepthBins(default.min_depth, math.ceil(reach), default.num_bins)


def build_lifter(
    name: str,
    grid: GridSpec,
    channels: int,
    image_size: tuple[int, int],
    num_cameras: int,
    encoder: ImageEncoder,
    **kwargs: Any,
):
    """Build a registered lifter, injecting the context it asks for.

    ``feature_size`` and ``num_cameras`` are derived from the encoder / rig, and ``depth_bins``
    are sized to cover the grid, unless given explicitly in ``kwargs``.
    """
    cls = LIFTERS.get(name)
    params = inspect.signature(cls).parameters
    context = {
        "feature_size": _feature_size(image_size, encoder.strides[kwargs.get("level", -1)]),
        "num_cameras": num_cameras,
        "depth_bins": _depth_bins_for(grid),
    }
    for key, value in context.items():
        if key in params and key not in kwargs:
            kwargs[key] = value
    return cls(grid=grid, in_channels=channels, out_channels=channels, **kwargs)


def build_bev_model(
    lifter: str = "simple_bev",
    grid: GridSpec | None = None,
    num_classes: int = 3,
    image_size: tuple[int, int] = (48, 96),
    num_cameras: int = 4,
    channels: int = 64,
    encoder: ImageEncoder | None = None,
    **lifter_kwargs: Any,
) -> BEVSegmentationModel:
    """Build a BEV segmentation model around any registered lifter.

    Example::

        model = build_bev_model("bevformer", GridSpec(), num_classes=3)
        out = model(images, cameras)["logits"]  # (B, 3, X, Y)
    """
    grid = grid or GridSpec()
    encoder = encoder or SimpleConvEncoder(out_channels=channels, num_levels=2)
    lift = build_lifter(
        lifter, grid, encoder.out_channels, image_size, num_cameras, encoder, **lifter_kwargs
    )
    decoder = BEVDecoder(encoder.out_channels, channels)
    return BEVSegmentationModel(encoder, lift, decoder, SegmentationHead(channels, num_classes))


def build_tpvformer(
    grid: GridSpec | None = None,
    num_classes: int = 3,
    channels: int = 64,
    encoder: ImageEncoder | None = None,
    scale: int = 1,
    **lifter_kwargs: Any,
) -> TPVFormer:
    """Build a TPVFormer for 3D semantic occupancy."""
    grid = grid or GridSpec()
    encoder = encoder or SimpleConvEncoder(out_channels=channels, num_levels=2)
    lifter = TPVFormerLifter(grid, encoder.out_channels, encoder.out_channels, **lifter_kwargs)
    aggregator = TPVAggregator(grid, encoder.out_channels, num_classes, scale=scale)
    return TPVFormer(encoder, lifter, aggregator)
