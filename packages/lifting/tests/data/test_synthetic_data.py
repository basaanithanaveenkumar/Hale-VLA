"""The synthetic world must be geometrically exact, otherwise the verifier proves nothing."""

import torch

from lifting.data import (
    CameraRig,
    LabelRasterizer,
    ObjectSampler,
    RayCastRenderer,
    SceneBatch,
    SceneObjects,
    SyntheticSceneDataset,
)
from lifting.geometry import Cameras, GridSpec


def test_dataset_is_deterministic(grid: GridSpec, rig: CameraRig) -> None:
    a = SyntheticSceneDataset(3, grid, rig, seed=5)[1]
    b = SyntheticSceneDataset(3, grid, rig, seed=5)[1]
    c = SyntheticSceneDataset(3, grid, rig, seed=6)[1]
    assert torch.equal(a["images"], b["images"]) and torch.equal(a["cam_to_ego"], b["cam_to_ego"])
    assert not torch.equal(a["images"], c["images"])


def test_sample_shapes(batch: SceneBatch, grid: GridSpec) -> None:
    assert batch.images.shape == (2, 4, 3, 32, 64)
    assert batch.images.min() >= 0 and batch.images.max() <= 1
    assert batch.bev_labels.shape == (2, *grid.bev_shape)
    assert batch.occupancy.shape == (2, *grid.resolution)
    assert batch.depth.shape == (2, 4, 32, 64)
    assert (batch.bev_labels > 0).any()


def test_look_rotation_is_a_rotation() -> None:
    r = CameraRig.look_rotation(0.7, 0.1)
    assert torch.allclose(r.T @ r, torch.eye(3), atol=1e-6)
    assert torch.isclose(torch.linalg.det(r), torch.tensor(1.0))


def test_rig_yaw_jitter_rotates_all_cameras_together() -> None:
    rig = CameraRig(num_cameras=4, yaw_jitter_deg=180)
    t = rig.extrinsics(torch.Generator().manual_seed(1))
    fwd = t[:, :2, 2]
    angles = torch.atan2(fwd[:, 1], fwd[:, 0])
    diffs = torch.remainder(angles.roll(-1) - angles, 2 * torch.pi)
    assert torch.allclose(diffs, torch.full((4,), torch.pi / 2), atol=1e-5)
    assert not torch.allclose(fwd[0], torch.tensor([1.0, 0.0]), atol=1e-3)


def one_box_scene() -> SceneObjects:
    return SceneObjects(
        centers=torch.tensor([[8.0, 0.5]]),
        sizes=torch.tensor([[4.0, 2.0, 1.5]]),
        yaws=torch.tensor([0.3]),
        labels=torch.tensor([1]),
        colors=torch.tensor([[1.0, 0.0, 0.0]]),
    )


def render(objects: SceneObjects, rig: CameraRig):
    k = rig.intrinsics()
    t = rig.extrinsics(torch.Generator().manual_seed(0))
    view = RayCastRenderer().render(objects, k, t, rig.image_size)
    cams = Cameras(k[None], t[None], rig.image_size)
    return view, cams


def test_rendered_depth_is_consistent_with_the_scene() -> None:
    rig = CameraRig(image_size=(32, 64), yaw_jitter_deg=0)
    objects = one_box_scene()
    view, cams = render(objects, rig)
    h, w = rig.image_size
    v, u = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    uv = torch.stack([u, v], -1).view(1, 1, -1, 2).expand(1, 4, -1, -1)
    depth = view.depth.view(1, 4, -1)
    finite = torch.isfinite(depth)
    pts = cams.unproject(uv, depth.nan_to_num(posinf=0.0))[0]  # (4, HW, 3)
    labels = view.labels.view(4, -1)

    ground = finite[0] & (labels == 0)
    assert ground.sum() > 100
    assert pts[ground][:, 2].abs().max() < 1e-3  # ground pixels lie on z = 0

    box = labels == 1
    assert box.sum() > 10
    local = objects.to_local(pts[box][:, :2])[:, 0]
    half = objects.sizes[0] / 2
    assert (local.abs() <= half[:2] + 1e-3).all()  # box pixels lie on the box
    z = pts[box][:, 2]
    assert (z >= -1e-3).all() and (z <= objects.sizes[0, 2] + 1e-3).all()

    sky = ~finite[0]
    assert (view.images.permute(0, 2, 3, 1).reshape(4, -1, 3)[sky][:, 2] > 0.8).all()


def test_box_is_visible_where_it_projects() -> None:
    rig = CameraRig(image_size=(32, 64), yaw_jitter_deg=0)
    objects = one_box_scene()
    view, cams = render(objects, rig)
    centre = torch.tensor([[[8.0, 0.5, 0.75]]])
    uv, _, valid = cams.project(centre)
    cam = int(valid[0].float().argmax())
    u, v = uv[0, cam, 0].long().tolist()
    assert view.labels[cam, v, u] == 1
    assert view.images[cam, 0, v, u] > view.images[cam, 2, v, u]  # red box


def test_labels_match_objects() -> None:
    grid = GridSpec(resolution=(32, 32, 4))
    objects = one_box_scene()
    r = LabelRasterizer(grid)
    bev = r.bev(objects)
    occ = r.occupancy(objects)
    ix, iy, _ = grid.to_index(torch.tensor([8.0, 0.5, 0.0])).floor().long().tolist()
    assert bev[ix, iy] == 1
    assert bev.sum() > 4  # a 4 x 2 m car covers several 1 m cells
    assert torch.equal(occ.max(-1).values, bev)
    zs = grid.axis_centers(2)
    assert (occ[ix, iy][zs < 0] == 0).all() and (occ[ix, iy][(zs > 0) & (zs < 1.5)] == 1).all()
    assert (occ[ix, iy][zs > 1.5] == 0).all()


def test_sampler_respects_ego_keep_out() -> None:
    sampler = ObjectSampler(extent=14, num_objects=(8, 8), min_ego_distance=3.0)
    objects = sampler.sample(torch.Generator().manual_seed(0))
    assert len(objects) > 0
    assert (objects.centers.norm(dim=-1) > 3.0).all()
    assert set(objects.labels.tolist()) <= {1, 2}
