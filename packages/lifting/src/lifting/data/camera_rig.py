"""A ring of outward-looking pinhole cameras mounted on the ego vehicle."""

from __future__ import annotations

import math

import torch
from torch import Tensor


class CameraRig:
    """``N`` cameras evenly spaced in yaw (like nuScenes' surround rig).

    Args:
        num_cameras: number of cameras.
        image_size: ``(H, W)``.
        horizontal_fov_deg: horizontal field of view of every camera.
        height: mounting height above the ground (metres).
        pitch_deg: downward tilt.
        yaw_jitter_deg: the whole rig is rotated by ``U(-j, j)`` per sample. Large values
            force a network to *use* the extrinsics instead of memorising a fixed layout.
        position_jitter: ``U(-p, p)`` metres of per-camera translation noise.
    """

    def __init__(
        self,
        num_cameras: int = 4,
        image_size: tuple[int, int] = (48, 96),
        horizontal_fov_deg: float = 100.0,
        height: float = 1.6,
        pitch_deg: float = 6.0,
        yaw_jitter_deg: float = 180.0,
        position_jitter: float = 0.0,
    ) -> None:
        self.num_cameras = num_cameras
        self.image_size = image_size
        self.horizontal_fov_deg = horizontal_fov_deg
        self.height = height
        self.pitch_deg = pitch_deg
        self.yaw_jitter_deg = yaw_jitter_deg
        self.position_jitter = position_jitter

    def intrinsics(self) -> Tensor:
        """``(N, 3, 3)`` shared intrinsics (square pixels, centred principal point)."""
        h, w = self.image_size
        f = 0.5 * w / math.tan(math.radians(self.horizontal_fov_deg) / 2)
        k = torch.tensor([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])
        return k.expand(self.num_cameras, 3, 3).clone()

    @staticmethod
    def look_rotation(yaw: float, pitch: float) -> Tensor:
        """Camera-to-ego rotation with columns ``[right, down, forward]``."""
        forward = torch.tensor(
            [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)]
        )
        right = torch.tensor([math.sin(yaw), -math.cos(yaw), 0.0])
        down = torch.linalg.cross(forward, right)
        return torch.stack([right, down, forward], dim=1)

    def extrinsics(self, generator: torch.Generator | None = None) -> Tensor:
        """``(N, 4, 4)`` camera-to-ego transforms (randomised by the jitter settings)."""
        jitter = 0.0
        if self.yaw_jitter_deg > 0:
            jitter = (torch.rand((), generator=generator).item() * 2 - 1) * math.radians(
                self.yaw_jitter_deg
            )
        pitch = math.radians(self.pitch_deg)
        out = torch.zeros(self.num_cameras, 4, 4)
        for i in range(self.num_cameras):
            yaw = jitter + 2 * math.pi * i / self.num_cameras
            out[i, :3, :3] = self.look_rotation(yaw, pitch)
            pos = torch.tensor([0.0, 0.0, self.height])
            if self.position_jitter > 0:
                pos = pos + (torch.rand(3, generator=generator) * 2 - 1) * self.position_jitter
            out[i, :3, 3] = pos
            out[i, 3, 3] = 1.0
        return out
