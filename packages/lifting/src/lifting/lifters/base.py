"""The single interface every 2D -> BEV lifter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor, nn

from lifting.geometry import Cameras, GridSpec
from lifting.registry import Registry


class BaseLifter(nn.Module, ABC):
    """Turns multi-camera image features into a BEV feature map.

    Args:
        grid: metric region and resolution of the output.
        in_channels: channels of every image feature level.
        out_channels: channels of the returned BEV map.

    Subclasses implement :meth:`forward` with the signature::

        forward(features: Sequence[Tensor], cameras: Cameras) -> Tensor

    where ``features[l]`` is ``(B, N, C, H_l, W_l)`` (fine -> coarse) and the result is
    ``(B, out_channels, X, Y)``. ``cameras`` refer to the *input image* resolution; lifters
    rescale intrinsics to each feature level themselves.
    """

    #: whether the lifter uses camera geometry (False only for geometry-free baselines).
    uses_geometry: bool = True

    def __init__(self, grid: GridSpec, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.grid = grid
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, features: Sequence[Tensor], cameras: Cameras) -> Tensor:
        """Lift ``features`` to a ``(B, out_channels, X, Y)`` BEV map."""


#: Global registry of lifters: ``LIFTERS.build(name, grid=..., in_channels=..., out_channels=...)``.
LIFTERS: Registry[BaseLifter] = Registry("lifter")
