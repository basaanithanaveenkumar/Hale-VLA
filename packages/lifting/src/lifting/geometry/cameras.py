"""Batched multi-camera pinhole model.

Conventions
-----------
* Ego / world frame: x forward, y left, z up (ROS / robotics convention).
* Camera frame: OpenCV, x right, y down, z forward (optical axis).
* Pixels: continuous coordinates, pixel ``i`` covers ``[i, i + 1)`` so its centre
  is ``i + 0.5``. This matches ``F.grid_sample(..., align_corners=False)`` and makes
  rescaling intrinsics to a feature-map resolution exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Cameras:
    """A rig of ``N`` pinhole cameras for a batch of ``B`` samples.

    Attributes:
        intrinsics: ``(B, N, 3, 3)`` pixel-from-camera matrices.
        cam_to_ego: ``(B, N, 4, 4)`` rigid transforms mapping camera points to ego.
        image_size: ``(H, W)`` of the image the intrinsics refer to.
    """

    intrinsics: Tensor
    cam_to_ego: Tensor
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"intrinsics must be (B, N, 3, 3), got {tuple(self.intrinsics.shape)}")
        if self.cam_to_ego.shape[-2:] != (4, 4):
            raise ValueError(f"cam_to_ego must be (B, N, 4, 4), got {tuple(self.cam_to_ego.shape)}")
        if self.intrinsics.shape[:2] != self.cam_to_ego.shape[:2]:
            raise ValueError("intrinsics and cam_to_ego disagree on (B, N)")

    # ------------------------------------------------------------------ properties
    @property
    def batch_size(self) -> int:
        return self.intrinsics.shape[0]

    @property
    def num_cameras(self) -> int:
        return self.intrinsics.shape[1]

    @property
    def device(self) -> torch.device:
        return self.intrinsics.device

    @property
    def ego_to_cam(self) -> Tensor:
        """``(B, N, 4, 4)`` inverse extrinsics (closed form for rigid transforms)."""
        rot = self.cam_to_ego[..., :3, :3]
        trans = self.cam_to_ego[..., :3, 3:]
        rot_t = rot.transpose(-1, -2)
        out = torch.zeros_like(self.cam_to_ego)
        out[..., :3, :3] = rot_t
        out[..., :3, 3:] = -rot_t @ trans
        out[..., 3, 3] = 1.0
        return out

    # ------------------------------------------------------------------ transforms
    def to(self, device: torch.device | str) -> Cameras:
        return Cameras(self.intrinsics.to(device), self.cam_to_ego.to(device), self.image_size)

    def scaled_to(self, height: int, width: int) -> Cameras:
        """Return cameras whose intrinsics address a ``(height, width)`` feature map."""
        sy = height / self.image_size[0]
        sx = width / self.image_size[1]
        scale = torch.tensor([sx, sy, 1.0], device=self.device, dtype=self.intrinsics.dtype)
        return Cameras(self.intrinsics * scale[:, None], self.cam_to_ego, (height, width))

    def transformed(self, ego_transform: Tensor) -> Cameras:
        """Left-multiply the extrinsics by a ``(4, 4)`` or ``(B, 4, 4)`` ego transform."""
        if ego_transform.dim() == 2:
            ego_transform = ego_transform.expand(self.batch_size, 4, 4)
        cam_to_ego = ego_transform[:, None].to(self.cam_to_ego) @ self.cam_to_ego
        return Cameras(self.intrinsics, cam_to_ego, self.image_size)

    # ------------------------------------------------------------------ projection
    def project(self, points: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor, Tensor]:
        """Project ego-frame points into every camera.

        Args:
            points: ``(B, P, 3)`` ego-frame points.

        Returns:
            uv: ``(B, N, P, 2)`` continuous pixel coordinates.
            depth: ``(B, N, P)`` camera-frame z (distance along the optical axis).
            valid: ``(B, N, P)`` point is in front of the camera and inside the image.
        """
        rot = self.ego_to_cam[..., :3, :3]  # B N 3 3
        trans = self.ego_to_cam[..., :3, 3]  # B N 3
        cam_pts = torch.einsum("bnij,bpj->bnpi", rot, points) + trans[:, :, None]
        depth = cam_pts[..., 2]
        pix = torch.einsum("bnij,bnpj->bnpi", self.intrinsics, cam_pts)
        uv = pix[..., :2] / depth.clamp(min=eps)[..., None]
        height, width = self.image_size
        valid = (
            (depth > eps)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] < width)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < height)
        )
        return uv, depth, valid

    def normalize_uv(self, uv: Tensor) -> Tensor:
        """Map pixel coordinates to ``[0, 1]`` (deformable-attention convention)."""
        height, width = self.image_size
        return uv / uv.new_tensor([width, height])

    def unproject(self, uv: Tensor, depth: Tensor) -> Tensor:
        """Lift pixels with known depth to ego-frame points.

        Args:
            uv: ``(B, N, P, 2)`` pixel coordinates.
            depth: ``(B, N, P)`` camera-frame z.

        Returns:
            ``(B, N, P, 3)`` ego-frame points.
        """
        ones = torch.ones_like(uv[..., :1])
        rays = torch.einsum(
            "bnij,bnpj->bnpi", torch.linalg.inv(self.intrinsics), torch.cat([uv, ones], -1)
        )
        cam_pts = rays * depth[..., None]
        rot = self.cam_to_ego[..., :3, :3]
        trans = self.cam_to_ego[..., :3, 3]
        return torch.einsum("bnij,bnpj->bnpi", rot, cam_pts) + trans[:, :, None]
