"""End-to-end model assembly: encoder -> lifter -> decoder -> head, on rendered scenes."""

import io

import pytest
import torch

from lifting import LIFTERS, GridSpec, build_bev_model, build_tpvformer
from lifting.data import CameraRig, SceneBatch, SyntheticSceneDataset
from lifting.encoders import ResNetEncoder

IMAGE = (32, 64)


@pytest.mark.parametrize("name", LIFTERS.names())
def test_bev_model_forward_backward(name: str, batch: SceneBatch, grid: GridSpec) -> None:
    model = build_bev_model(name, grid, num_classes=3, image_size=IMAGE, channels=16)
    out = model(batch.images, batch.cameras)
    assert out["logits"].shape == (2, 3, *grid.bev_shape)
    assert out["bev_features"].shape == (2, 16, *grid.bev_shape)
    torch.nn.functional.cross_entropy(out["logits"], batch.bev_labels).backward()
    enc_grad = sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    assert enc_grad > 0, "gradient must flow through the lifter into the image encoder"


@pytest.mark.parametrize("name", [n for n in LIFTERS.names() if n != "mlp_view"])
def test_geometric_models_accept_any_number_of_cameras(name: str, grid: GridSpec) -> None:
    rig = CameraRig(num_cameras=6, image_size=IMAGE, horizontal_fov_deg=70)
    ds = SyntheticSceneDataset(1, grid, rig)
    b = SceneBatch.collate([ds[0]])
    model = build_bev_model(
        name, grid, num_classes=3, image_size=IMAGE, num_cameras=6, channels=16
    ).eval()
    with torch.no_grad():
        assert model(b.images, b.cameras)["logits"].shape == (1, 3, *grid.bev_shape)


def test_multi_scale_bevformer_with_resnet(batch: SceneBatch, grid: GridSpec) -> None:
    pytest.importorskip("torchvision")
    encoder = ResNetEncoder("resnet18", out_channels=16, num_levels=2)
    model = build_bev_model("bevformer", grid, 3, IMAGE, encoder=encoder, channels=16, num_levels=2)
    assert model(batch.images, batch.cameras)["logits"].shape == (2, 3, *grid.bev_shape)


def test_tpvformer_occupancy_and_points(batch: SceneBatch, grid: GridSpec) -> None:
    model = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1)
    points = torch.randn(2, 50, 3) * 6
    out = model(batch.images, batch.cameras, points)
    assert out["voxel_logits"].shape == (2, 3, *grid.resolution)
    assert out["point_logits"].shape == (2, 3, 50)
    torch.nn.functional.cross_entropy(out["voxel_logits"], batch.occupancy).backward()
    assert model.lifter.queries.grad.abs().sum() > 0
    up = build_tpvformer(grid, num_classes=3, channels=16, num_layers=1, scale=2)
    nx, ny, nz = grid.resolution
    assert up(batch.images, batch.cameras)["voxel_logits"].shape == (2, 3, 2 * nx, 2 * ny, 2 * nz)


@pytest.mark.parametrize("name", ["simple_bev", "bevformer", "tpvformer"])
def test_state_dict_round_trip(name: str, batch: SceneBatch, grid: GridSpec) -> None:
    model = build_bev_model(name, grid, 3, IMAGE, channels=16).eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = build_bev_model(name, grid, 3, IMAGE, channels=16).eval()
    clone.load_state_dict(torch.load(buffer))
    with torch.no_grad():
        assert torch.allclose(
            model(batch.images, batch.cameras)["logits"],
            clone(batch.images, batch.cameras)["logits"],
        )


def test_non_default_grid(batch: SceneBatch) -> None:
    grid = GridSpec((-10, 20), (-12, 12), (-1, 3), (15, 12, 2))
    for name in ("simple_bev", "lift_splat", "tpvformer"):
        model = build_bev_model(name, grid, 3, IMAGE, channels=16)
        assert model(batch.images, batch.cameras)["logits"].shape == (2, 3, 15, 12)


def test_depth_bins_cover_large_grids() -> None:
    from lifting.encoders import SimpleConvEncoder
    from lifting.models import build_lifter

    enc = SimpleConvEncoder(16)
    small = build_lifter("lift_splat", GridSpec(), 16, IMAGE, 4, enc)
    assert small.depth_bins.max_depth == 25.0  # default range already covers +-16 m
    large = GridSpec((-40, 40), (-40, 40), (-2, 4), (20, 20, 2))
    for name in ("lift_splat", "depth_warp", "tiim"):
        lifter = build_lifter(name, large, 16, IMAGE, 4, enc)
        assert lifter.depth_bins.max_depth >= (2 * 40**2) ** 0.5
