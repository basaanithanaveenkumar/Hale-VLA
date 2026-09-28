"""Geometric correctness of every lifting technique, verified with analytic probes.

The probes use *linear ramps* as image features (feature = pixel coordinate or depth).
Bilinear / trilinear interpolation reproduces a linear ramp exactly, so a correct
lifter must return, for every voxel, the pixel coordinate / depth at which that voxel
is observed - no learning involved.
"""

from __future__ import annotations

import pytest
import torch

from lifting.encoders import SimpleConvEncoder
from lifting.geometry import Cameras, DepthBins, GridSpec
from lifting.lifters import (
    LIFTERS,
    BEVFormerLifter,
    BilinearSamplingLifter,
    DepthWarpLifter,
    LiftSplatLifter,
    MLPViewLifter,
    PolarRayLifter,
    TPVAggregator,
    TPVFormerLifter,
    TPVPlanes,
    TPVReferencePoints,
)
from lifting.models import build_lifter
from tests.helpers import rotation_z

H, W = 32, 64


def pixel_ramp(cameras: Cameras, h: int = H, w: int = W) -> torch.Tensor:
    """``(B, N, 2, h, w)`` features whose value is the normalised pixel centre ``(u/W, v/H)``."""
    v, u = torch.meshgrid((torch.arange(h) + 0.5) / h, (torch.arange(w) + 0.5) / w, indexing="ij")
    ramp = torch.stack([u, v])
    return ramp.expand(cameras.batch_size, cameras.num_cameras, 2, h, w).clone()


def interior(uv_norm: torch.Tensor, h: int = H, w: int = W) -> torch.Tensor:
    """Points at least half a pixel from the border (where the ramp is exactly linear)."""
    lo = torch.tensor([0.5 / w, 0.5 / h])
    return ((uv_norm >= lo) & (uv_norm <= 1 - lo)).all(-1)


def single_view_voxels(grid: GridSpec, cameras: Cameras):
    pts = grid.voxel_centers().view(1, -1, 3).expand(cameras.batch_size, -1, -1)
    uv, depth, valid = cameras.project(pts)
    uv_n = cameras.normalize_uv(uv)
    # observed by exactly one camera, away from that image's border
    single = (valid.sum(1) == 1) & (valid & interior(uv_n)).any(1)
    cam = valid.float().argmax(1)
    return uv_n, depth, single, cam


# --------------------------------------------------------------------------- Simple-BEV
def test_bilinear_unprojection_reads_the_projected_pixel(grid: GridSpec, cameras: Cameras) -> None:
    lifter = BilinearSamplingLifter(grid, 2, 8)
    volume = lifter.unproject(pixel_ramp(cameras), cameras)  # (B, 2, X, Y, Z)
    uv_n, _, single, cam = single_view_voxels(grid, cameras)
    got = volume.flatten(2).transpose(1, 2)  # (B, P, 2)
    expected = torch.gather(uv_n, 1, cam[:, None, :, None].expand(-1, 1, -1, 2))[:, 0]
    assert single.sum() > 50
    assert torch.allclose(got[single], expected[single], atol=1e-4)


def test_bilinear_unprojection_zero_where_unseen(grid: GridSpec, cameras: Cameras) -> None:
    lifter = BilinearSamplingLifter(grid, 2, 8)
    volume = lifter.unproject(torch.ones(cameras.batch_size, cameras.num_cameras, 2, H, W), cameras)
    pts = grid.voxel_centers().view(1, -1, 3).expand(cameras.batch_size, -1, -1)
    _, _, valid = cameras.project(pts)
    seen = valid.any(1)
    flat = volume[:, 0].flatten(1)
    assert torch.all(flat[~seen] == 0)
    assert (flat[seen] > 0).all() and (flat[seen] <= 1 + 1e-5).all()


@pytest.mark.parametrize("lifter_cls", [BilinearSamplingLifter, DepthWarpLifter])
def test_rotation_equivariance(lifter_cls, grid: GridSpec, cameras: Cameras) -> None:
    """Rotating the rig by 90 deg must rotate the lifted volume by 90 deg (same images)."""
    torch.manual_seed(0)
    lifter = lifter_cls(grid, 4, 4).eval()
    feats = torch.rand(cameras.batch_size, cameras.num_cameras, 4, H, W)
    if lifter_cls is BilinearSamplingLifter:
        vol = lifter.unproject(feats, cameras)
        vol_rot = lifter.unproject(feats, cameras.transformed(rotation_z(90)))
    else:
        frustum = lifter.frustum_volume(feats)
        vol = lifter.warp(frustum, cameras)
        vol_rot = lifter.warp(frustum, cameras.transformed(rotation_z(90)))
    nx, ny, _ = grid.resolution
    i, j = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing="ij")
    # (x, y) -> (-y, x): cell (i, j) moves to (ny - 1 - j, i)
    expected = torch.empty_like(vol)
    expected[:, :, ny - 1 - j, i] = vol[:, :, i, j]
    assert torch.allclose(vol_rot, expected, atol=1e-4)


# --------------------------------------------------------------------------- depth warp (liftnet)
def test_depth_warp_reads_pixel_and_depth(grid: GridSpec, cameras: Cameras) -> None:
    bins = DepthBins(1.0, 25.0, 24)
    lifter = DepthWarpLifter(grid, 2, 8, depth_bins=bins)
    b, n = cameras.batch_size, cameras.num_cameras
    ramp = pixel_ramp(cameras)[:, :, :, None].expand(-1, -1, -1, bins.num_bins, -1, -1)
    depth_ramp = bins.centers().view(1, 1, 1, -1, 1, 1).expand(b, n, 1, -1, H, W)
    volume = torch.cat([ramp, depth_ramp], 2)  # (B, N, 3, D, H, W)
    vox = lifter.warp(volume, cameras).flatten(2).transpose(1, 2)  # (B, P, 3)
    uv_n, depth, single, cam = single_view_voxels(grid, cameras)
    d = torch.gather(depth, 1, cam[:, None]).squeeze(1)
    inside = single & (d > bins.min_depth + bins.step / 2) & (d < bins.max_depth - bins.step / 2)
    expected_uv = torch.gather(uv_n, 1, cam[:, None, :, None].expand(-1, 1, -1, 2))[:, 0]
    assert inside.sum() > 50
    assert torch.allclose(vox[..., :2][inside], expected_uv[inside], atol=1e-4)
    assert torch.allclose(vox[..., 2][inside], d[inside], atol=1e-3)


# --------------------------------------------------------------------------- Lift-Splat
def test_lift_splat_puts_mass_in_the_right_voxel(grid: GridSpec, cameras: Cameras) -> None:
    bins = DepthBins(1.0, 25.0, 24)
    lifter = LiftSplatLifter(grid, 4, 4, depth_bins=bins, context_channels=1)
    b, n = cameras.batch_size, cameras.num_cameras
    depth = torch.zeros(b, n, bins.num_bins, H, W)
    context = torch.zeros(b, n, 1, H, W)
    cam, u, v, d = 2, 40, 17, 5
    depth[:, cam, d, v, u] = 1.0
    context[:, cam, 0, v, u] = 3.0
    vox = lifter.splat(depth, context, cameras)  # (B, 1, X, Y, Z)

    point = cameras.unproject(
        torch.tensor([u + 0.5, v + 0.5]).expand(b, n, 1, 2), bins.centers()[d].expand(b, n, 1)
    )[:, cam, 0]
    idx = grid.to_index(point).floor().long()
    for bi in range(b):
        assert grid.contains(point[bi])
        assert vox[bi, 0].sum() == pytest.approx(3.0)
        assert vox[bi, 0, idx[bi, 0], idx[bi, 1], idx[bi, 2]] == pytest.approx(3.0)


def test_lift_splat_frustum_points_project_back(grid: GridSpec, cameras: Cameras) -> None:
    lifter = LiftSplatLifter(grid, 4, 4)
    scaled = cameras.scaled_to(4, 8)
    pts = lifter.frustum_points(scaled, 4, 8)
    uv, depth, _ = scaled.project(pts[:, 1])
    grid_uv = torch.stack(
        torch.meshgrid(torch.arange(4) + 0.5, torch.arange(8) + 0.5, indexing="ij")[::-1], -1
    ).reshape(-1, 2)
    assert torch.allclose(uv[:, 1], grid_uv.repeat(lifter.depth_bins.num_bins, 1), atol=1e-3)
    assert torch.allclose(depth[:, 1], lifter.depth_bins.centers().repeat_interleave(32), atol=1e-3)


# --------------------------------------------------------------------------- TIIM polar rays
def test_polar_to_bev_reads_column_and_depth(grid: GridSpec, cameras: Cameras) -> None:
    bins = DepthBins(1.0, 25.0, 24)
    lifter = PolarRayLifter(grid, 8, 8, depth_bins=bins)
    b, n = cameras.batch_size, cameras.num_cameras
    cols = ((torch.arange(W) + 0.5) / W).view(1, 1, 1, 1, W).expand(b, n, 1, bins.num_bins, W)
    deps = bins.centers().view(1, 1, 1, -1, 1).expand(b, n, 1, -1, W)
    bev = lifter.polar_to_bev(torch.cat([cols, deps], 2), cameras)  # (B, 2, X, Y)

    centers = grid.voxel_centers()[:, :, 0].clone()
    centers[..., 2] = sum(grid.z_bounds) / 2
    uv, depth, _ = cameras.project(centers.view(1, -1, 3).expand(b, -1, -1))
    u_n = uv[..., 0] / W
    ok = (depth > bins.min_depth + bins.step / 2) & (depth < bins.max_depth - bins.step / 2)
    ok &= (u_n > 0.5 / W) & (u_n < 1 - 0.5 / W)
    single = ok.sum(1) == 1
    cam = ok.float().argmax(1)
    got = bev.flatten(2).transpose(1, 2)
    exp_u = torch.gather(u_n, 1, cam[:, None]).squeeze(1)
    exp_d = torch.gather(depth, 1, cam[:, None]).squeeze(1)
    assert single.sum() > 50
    assert torch.allclose(got[..., 0][single], exp_u[single], atol=1e-4)
    assert torch.allclose(got[..., 1][single], exp_d[single], atol=1e-3)


# --------------------------------------------------------------------------- BEVFormer
def test_bevformer_reference_points_are_projected_pillars(grid: GridSpec, cameras: Cameras) -> None:
    lifter = BEVFormerLifter(grid, 16, 16, num_anchors=4)
    ref, visible = lifter.camera_reference_points(cameras)
    b, n = cameras.batch_size, cameras.num_cameras
    assert ref.shape == (b, n, grid.num_cells_bev, 4, 2)
    pillars = grid.pillar_points(4).view(1, -1, 3).expand(b, -1, -1)
    uv, _, valid = cameras.project(pillars)
    assert torch.allclose(ref, cameras.normalize_uv(uv).view_as(ref))
    assert torch.equal(visible, valid.view_as(visible))
    # a 360 deg rig sees every pillar that is not right next to the ego vehicle
    far = grid.voxel_centers()[:, :, 0, :2].reshape(-1, 2).norm(dim=-1) > 4
    assert visible.any(-1).any(1)[:, far].all()


def test_bev_reference_points_are_cell_centres(grid: GridSpec) -> None:
    lifter = BEVFormerLifter(grid, 16, 16)
    ref = lifter.bev_reference_points(torch.device("cpu")).view(16, 16, 2)
    assert torch.allclose(ref[3, 5], torch.tensor([(5 + 0.5) / 16, (3 + 0.5) / 16]))


# --------------------------------------------------------------------------- TPVFormer
def test_tpv_anchor_points_lie_on_plane_normals(grid: GridSpec) -> None:
    refs = TPVReferencePoints(grid, (4, 6, 6))
    xs, ys, zs = (grid.axis_centers(a) for a in range(3))
    xy = refs.anchor_points(0).view(16, 16, 4, 3)
    xz = refs.anchor_points(1).view(16, 4, 6, 3)
    yz = refs.anchor_points(2).view(16, 4, 6, 3)
    assert torch.allclose(xy[2, 9, :, :2], torch.stack([xs[2], ys[9]]).expand(4, 2))
    assert torch.allclose(xz[2, 3, :, 0::2], torch.stack([xs[2], zs[3]]).expand(6, 2))
    assert torch.allclose(yz[7, 1, :, 1:], torch.stack([ys[7], zs[1]]).expand(6, 2))
    assert xz[..., 1].std() > 5 and yz[..., 0].std() > 5  # spread along the normal


def test_tpv_hybrid_references_point_at_the_same_3d_location(grid: GridSpec) -> None:
    refs = TPVReferencePoints(grid, (4, 6, 6)).hybrid_reference_points()
    nx, ny, nz = grid.resolution
    assert refs.shape == (nx * ny + nx * nz + ny * nz, 3, 2)
    # xy query (i, j): its own plane reference is its cell centre (col=y, row=x)
    i, j = 5, 11
    r = refs[i * ny + j]
    assert torch.allclose(r[0], torch.tensor([(j + 0.5) / ny, (i + 0.5) / nx]))
    assert torch.allclose(r[1, 1], torch.tensor((i + 0.5) / nx))  # same x on the xz plane
    assert torch.allclose(r[2, 1], torch.tensor((j + 0.5) / ny))  # same y on the yz plane
    # xz query (i, k): same x and z on xz plane and z on yz plane
    k = 2
    r = refs[nx * ny + i * nz + k]
    assert torch.allclose(r[1], torch.tensor([(k + 0.5) / nz, (i + 0.5) / nx]))
    assert torch.allclose(r[2, 0], torch.tensor((k + 0.5) / nz))


def test_tpv_planes_voxels_and_points(grid: GridSpec) -> None:
    nx, ny, nz = grid.resolution
    planes = TPVPlanes(
        torch.randn(2, 3, nx, ny), torch.randn(2, 3, nx, nz), torch.randn(2, 3, ny, nz)
    )
    vox = planes.to_voxels()
    assert vox.shape == (2, 3, nx, ny, nz)
    assert torch.allclose(
        vox[1, :, 4, 7, 2], planes.xy[1, :, 4, 7] + planes.xz[1, :, 4, 2] + planes.yz[1, :, 7, 2]
    )
    # sampling at voxel centres reproduces the voxel features
    centers = grid.voxel_centers().view(1, -1, 3).expand(2, -1, -1)
    sampled = planes.sample_points(grid.to_normalized(centers))
    assert torch.allclose(sampled, vox.flatten(2), atol=1e-5)
    # token round trip
    again = TPVPlanes.from_tokens(planes.to_tokens(), grid.resolution)
    assert torch.equal(again.xz, planes.xz)


def test_tpv_aggregator_points_match_voxels(grid: GridSpec) -> None:
    nx, ny, nz = grid.resolution
    planes = TPVPlanes(
        torch.randn(1, 8, nx, ny), torch.randn(1, 8, nx, nz), torch.randn(1, 8, ny, nz)
    )
    agg = TPVAggregator(grid, 8, num_classes=3)
    pts = grid.voxel_centers()[3:5, 6:8, 1:3].reshape(1, -1, 3)
    out = agg(planes, pts)
    assert out["voxel_logits"].shape == (1, 3, nx, ny, nz)
    assert torch.allclose(
        out["point_logits"], out["voxel_logits"][:, :, 3:5, 6:8, 1:3].reshape(1, 3, -1), atol=1e-5
    )
    assert TPVAggregator(grid, 8, 3, scale=2)(planes)["voxel_logits"].shape == (
        1,
        3,
        2 * nx,
        2 * ny,
        2 * nz,
    )


def test_tpvformer_lift_planes_shapes(grid: GridSpec, cameras: Cameras) -> None:
    lifter = TPVFormerLifter(grid, 16, 16, num_layers=1)
    feats = [torch.randn(cameras.batch_size, cameras.num_cameras, 16, 8, 16)]
    planes = lifter.lift_planes(feats, cameras)
    nx, ny, nz = grid.resolution
    assert planes.xy.shape == (2, 16, nx, ny)
    assert planes.xz.shape == (2, 16, nx, nz)
    assert planes.yz.shape == (2, 16, ny, nz)


# --------------------------------------------------------------------------- all lifters
def make_lifter(name: str, grid: GridSpec, cameras: Cameras, channels: int = 16):
    enc = SimpleConvEncoder(channels, num_levels=2)
    return build_lifter(name, grid, channels, (H, W), cameras.num_cameras, enc).eval(), enc


@pytest.mark.parametrize("name", LIFTERS.names())
def test_lifter_output_shape_and_gradients(name: str, grid: GridSpec, cameras: Cameras) -> None:
    lifter, enc = make_lifter(name, grid, cameras)
    lifter.train()
    b, n = cameras.batch_size, cameras.num_cameras
    feats = [torch.randn(b, n, 16, H // s, W // s, requires_grad=True) for s in enc.strides]
    out = lifter(feats, cameras)
    assert out.shape == (b, 16, *grid.bev_shape)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert sum(f.grad.abs().sum() for f in feats if f.grad is not None) > 0


@pytest.mark.parametrize("name", [n for n in LIFTERS.names() if n != "mlp_view"])
def test_geometric_lifters_depend_on_calibration(
    name: str, grid: GridSpec, cameras: Cameras
) -> None:
    lifter, enc = make_lifter(name, grid, cameras)
    assert lifter.uses_geometry
    b, n = cameras.batch_size, cameras.num_cameras
    feats = [torch.randn(b, n, 16, H // s, W // s) for s in enc.strides]
    with torch.no_grad():
        a = lifter(feats, cameras)
        r = lifter(feats, cameras.transformed(rotation_z(90)))
    assert not torch.allclose(a, r, atol=1e-3)


def test_mlp_view_ignores_calibration(grid: GridSpec, cameras: Cameras) -> None:
    lifter, enc = make_lifter("mlp_view", grid, cameras)
    assert isinstance(lifter, MLPViewLifter) and not lifter.uses_geometry
    feats = [torch.randn(2, 4, 16, H // s, W // s) for s in enc.strides]
    with torch.no_grad():
        assert torch.equal(
            lifter(feats, cameras), lifter(feats, cameras.transformed(rotation_z(90)))
        )
