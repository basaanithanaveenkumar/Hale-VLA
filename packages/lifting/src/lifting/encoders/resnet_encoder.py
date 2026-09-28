"""torchvision ResNet backbone + FPN (optional ``torchvision`` dependency)."""

from __future__ import annotations

from torch import Tensor, nn

from lifting.encoders.base import ImageEncoder
from lifting.encoders.fpn import FPN

_STAGE_CHANNELS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
}


class ResNetEncoder(ImageEncoder):
    """ResNet stages C3..C5 (strides 8, 16, 32) fused by an FPN.

    Args:
        arch: one of ``resnet18/34/50/101``.
        out_channels: FPN width.
        num_levels: how many levels (starting at stride 8, as in Simple-BEV) to return.
        pretrained: load ImageNet weights (downloads on first use).
    """

    def __init__(
        self,
        arch: str = "resnet50",
        out_channels: int = 128,
        num_levels: int = 3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        try:
            import torchvision
        except ImportError as err:  # pragma: no cover - exercised only without torchvision
            raise ImportError("ResNetEncoder needs `pip install lifting[vision]`") from err
        if arch not in _STAGE_CHANNELS:
            raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_STAGE_CHANNELS)}")
        if not 1 <= num_levels <= 3:
            raise ValueError("num_levels must be in [1, 3]")
        net = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.stages = nn.ModuleList([net.layer2, net.layer3, net.layer4][:num_levels])
        self.out_channels = out_channels
        self.strides = tuple(8 * 2**i for i in range(num_levels))
        self.neck = FPN(_STAGE_CHANNELS[arch][1 : 1 + num_levels], out_channels)

    def extract(self, images: Tensor) -> list[Tensor]:
        x = self.stem(images)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return self.neck(feats)
