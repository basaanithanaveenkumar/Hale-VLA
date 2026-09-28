import torch

from lifting.geometry import Cameras
from tests.helpers import rotation_z


def test_ego_to_cam_is_inverse(cameras: Cameras) -> None:
    eye = cameras.ego_to_cam @ cameras.cam_to_ego
    assert torch.allclose(eye, torch.eye(4).expand_as(eye), atol=1e-5)


def test_project_unproject_roundtrip(cameras: Cameras) -> None:
    b, n = cameras.batch_size, cameras.num_cameras
    uv = torch.rand(b, n, 50, 2) * torch.tensor([64.0, 32.0])
    depth = 1 + 20 * torch.rand(b, n, 50)
    points = cameras.unproject(uv, depth)
    for cam in range(n):
        uv2, d2, valid = cameras.project(points[:, cam])
        assert torch.allclose(uv2[:, cam], uv[:, cam], atol=1e-3)
        assert torch.allclose(d2[:, cam], depth[:, cam], atol=1e-3)
        assert valid[:, cam].all()


def test_point_on_optical_axis_hits_principal_point(cameras: Cameras) -> None:
    origin = cameras.cam_to_ego[0, 0, :3, 3]
    forward = cameras.cam_to_ego[0, 0, :3, 2]
    point = (origin + 7.0 * forward).view(1, 1, 3).expand(cameras.batch_size, 1, 3)
    uv, depth, valid = cameras.project(point)
    assert torch.allclose(uv[0, 0, 0], torch.tensor([32.0, 16.0]), atol=1e-4)
    assert torch.isclose(depth[0, 0, 0], torch.tensor(7.0), atol=1e-4)
    assert valid[0, 0, 0]


def test_points_behind_camera_are_invalid(cameras: Cameras) -> None:
    origin = cameras.cam_to_ego[0, 0, :3, 3]
    forward = cameras.cam_to_ego[0, 0, :3, 2]
    point = (origin - 5.0 * forward).view(1, 1, 3).expand(cameras.batch_size, 1, 3)
    _, _, valid = cameras.project(point)
    assert not valid[0, 0, 0]


def test_scaled_to_rescales_pixels(cameras: Cameras) -> None:
    pts = torch.randn(cameras.batch_size, 20, 3) * 8
    uv, _, _ = cameras.project(pts)
    uv_half, _, _ = cameras.scaled_to(16, 32).project(pts)
    assert torch.allclose(uv_half, uv / 2, atol=1e-4)
    assert torch.allclose(
        cameras.normalize_uv(uv), cameras.scaled_to(16, 32).normalize_uv(uv_half), atol=1e-5
    )


def test_transformed_cameras_see_transformed_points(cameras: Cameras) -> None:
    t = rotation_z(37.0)
    pts = torch.randn(cameras.batch_size, 30, 3) * 8
    pts_rot = pts @ t[:3, :3].T
    uv, d, _ = cameras.project(pts)
    uv2, d2, _ = cameras.transformed(t).project(pts_rot)
    assert torch.allclose(uv, uv2, atol=1e-3)
    assert torch.allclose(d, d2, atol=1e-4)
