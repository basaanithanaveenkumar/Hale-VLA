"""Vectorised ray casting of box scenes - produces geometrically exact synthetic camera views."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lifting.data.scene_objects import SceneObjects


@dataclass
class RenderResult:
    """Rendered views.

    Attributes:
        images: ``(N, 3, H, W)`` RGB in ``[0, 1]``.
        depth: ``(N, H, W)`` camera-frame z of the first hit (``inf`` for sky).
        labels: ``(N, H, W)`` semantic id per pixel (0 = ground / sky).
    """

    images: Tensor
    depth: Tensor
    labels: Tensor


class RayCastRenderer:
    """Renders a checkerboard ground plane, a sky gradient and Lambertian-shaded boxes.

    Args:
        checker_size: ground tile size in metres (gives the network texture/parallax cues).
        light_dir: direction *towards* the light in the ego frame.
    """

    def __init__(
        self, checker_size: float = 2.0, light_dir: tuple[float, float, float] = (0.4, 0.3, 0.85)
    ) -> None:
        self.checker_size = checker_size
        light = torch.tensor(light_dir)
        self.light = light / light.norm()

    @staticmethod
    def camera_rays(
        intrinsics: Tensor, cam_to_ego: Tensor, image_size: tuple[int, int]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Ray origins ``(N, 3)``, unit directions ``(N, H*W, 3)`` and per-ray z-scale ``(N, H*W)``."""
        h, w = image_size
        v, u = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
        pix = torch.stack([u, v, torch.ones_like(u)], -1).view(-1, 3)
        dirs_cam = pix @ torch.linalg.inv(intrinsics).transpose(-1, -2)  # (N, HW, 3), z = 1
        norm = dirs_cam.norm(dim=-1)
        dirs = dirs_cam @ cam_to_ego[:, :3, :3].transpose(-1, -2)
        return cam_to_ego[:, :3, 3], dirs / norm[..., None], 1.0 / norm

    def _intersect_boxes(
        self, origins: Tensor, dirs: Tensor, objects: SceneObjects
    ) -> tuple[Tensor, Tensor]:
        """Slab test against every box: hit distance ``(R, K)`` (``inf`` = miss) and local normal ``(R, K, 3)``."""
        k = len(objects)
        cos, sin = objects.yaws.cos(), objects.yaws.sin()
        centers = torch.cat([objects.centers, objects.sizes[:, 2:] / 2], -1)  # (K, 3)
        half = objects.sizes / 2
        rel = origins[:, None] - centers  # (R, K, 3)

        def rotate(v: Tensor) -> Tensor:
            return torch.stack(
                [cos * v[..., 0] + sin * v[..., 1], -sin * v[..., 0] + cos * v[..., 1], v[..., 2]],
                -1,
            )

        o = rotate(rel)
        d = rotate(dirs[:, None].expand(-1, k, -1))
        d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t1 = (-half - o) / d
        t2 = (half - o) / d
        t_near, t_far = torch.minimum(t1, t2), torch.maximum(t1, t2)
        t_enter, axis = t_near.max(-1)
        t_exit = t_far.min(-1).values
        hit = (t_exit >= t_enter) & (t_enter > 1e-4)
        dist = torch.where(hit, t_enter, torch.full_like(t_enter, float("inf")))
        normal_local = torch.nn.functional.one_hot(axis, 3).float() * -torch.sign(
            torch.gather(d, -1, axis[..., None])
        )
        # back to ego frame (rotate by +yaw)
        nx = cos * normal_local[..., 0] - sin * normal_local[..., 1]
        ny = sin * normal_local[..., 0] + cos * normal_local[..., 1]
        return dist, torch.stack([nx, ny, normal_local[..., 2]], -1)

    def render(
        self,
        objects: SceneObjects,
        intrinsics: Tensor,
        cam_to_ego: Tensor,
        image_size: tuple[int, int],
    ) -> RenderResult:
        """Render ``N`` views of ``objects`` for the given calibration."""
        h, w = image_size
        n = intrinsics.shape[0]
        origins, dirs, z_scale = self.camera_rays(intrinsics, cam_to_ego, image_size)
        o = origins[:, None].expand(-1, h * w, -1).reshape(-1, 3)
        d = dirs.reshape(-1, 3)

        # sky
        elevation = d[:, 2].clamp(0, 1)[:, None]
        color = (1 - elevation) * torch.tensor([0.75, 0.85, 0.95]) + elevation * torch.tensor(
            [0.35, 0.55, 0.9]
        )
        dist = torch.full((d.shape[0],), float("inf"))
        labels = torch.zeros(d.shape[0], dtype=torch.long)

        # ground plane z = 0
        t_ground = torch.where(
            d[:, 2] < -1e-6, -o[:, 2] / d[:, 2], torch.full_like(dist, float("inf"))
        )
        ground = torch.isfinite(t_ground)
        p = o + t_ground.nan_to_num(posinf=0)[:, None] * d
        checker = (
            (p[:, 0] / self.checker_size).floor() + (p[:, 1] / self.checker_size).floor()
        ) % 2
        fog = (t_ground.nan_to_num(posinf=1e3) / 60.0).clamp(0, 1)[:, None]
        ground_color = (0.42 + 0.16 * checker)[:, None].expand(-1, 3) * torch.tensor(
            [1.0, 1.0, 0.95]
        )
        ground_color = (1 - fog) * ground_color + fog * torch.tensor([0.75, 0.85, 0.95])
        color = torch.where(ground[:, None], ground_color, color)
        dist = torch.where(ground, t_ground, dist)

        # boxes
        if len(objects) > 0:
            box_dist, normals = self._intersect_boxes(o, d, objects)
            nearest, idx = box_dist.min(-1)
            hit = nearest < dist
            normal = normals[torch.arange(len(idx)), idx]
            shade = 0.45 + 0.55 * (normal @ self.light).clamp(min=0)
            box_color = objects.colors[idx] * shade[:, None]
            color = torch.where(hit[:, None], box_color, color)
            dist = torch.where(hit, nearest, dist)
            labels = torch.where(hit, objects.labels[idx], labels)

        depth = dist.view(n, h * w) * z_scale
        return RenderResult(
            images=color.view(n, h, w, 3).permute(0, 3, 1, 2).clamp(0, 1).contiguous(),
            depth=depth.view(n, h, w),
            labels=labels.view(n, h, w),
        )
