"""Integration of the `lifting` package (multi-camera BEV lifting) into Halo-VLA."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

lifting = pytest.importorskip("lifting")

from lifting.data import CameraRig, SceneBatch, SyntheticSceneDataset  # noqa: E402

from config import HaloVLMConfig  # noqa: E402
from models.bev_encoder import BEVSceneEncoder  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402

IMAGE = (32, 64)


def small_config(**overrides) -> HaloVLMConfig:
    base = dict(
        emb_dim=64, img_size=32, patch_size=8, vit_num_layers=1, vit_num_heads=4,
        vit_mlp_dim=64, dec_num_layers=1, dec_num_heads=4, dec_mlp_dim=64, use_moe=False,
        action_hidden_dims=(64,), state_hidden_dims=(64,),
        bev_num_cameras=4, bev_image_size=IMAGE, bev_resolution=(16, 16, 4),
        bev_bounds=((-16.0, 16.0), (-16.0, 16.0), (-1.0, 3.0)), bev_channels=16,
    )
    base.update(overrides)
    return HaloVLMConfig(**base)


@pytest.fixture(scope="module")
def scene() -> SceneBatch:
    rig = CameraRig(num_cameras=4, image_size=IMAGE)
    grid = lifting.GridSpec(resolution=(16, 16, 4))
    ds = SyntheticSceneDataset(2, grid, rig)
    return SceneBatch.collate([ds[0], ds[1]])


@pytest.mark.parametrize("lifter", ["simple_bev", "lift_splat", "bevformer", "tpvformer"])
def test_bev_scene_encoder_tokens(lifter: str, scene: SceneBatch) -> None:
    enc = BEVSceneEncoder.from_config(small_config(bev_lifter=lifter))
    tokens = enc(scene.images, scene.cameras)
    assert tokens.shape == (2, enc.num_tokens, 64)
    assert enc.num_tokens == 16  # 16x16 BEV pooled 4x4
    tokens.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.image_encoder.parameters())


def test_halo_vlm_with_bev_tokens(scene: SceneBatch) -> None:
    cfg = small_config(use_bev=True, bev_lifter="simple_bev")
    model = HaloVLM(cfg)
    ids = torch.tensor([[cfg.image_token_id, 5, 6, cfg.state_token_id, cfg.action_token_id]] * 2)
    args = (torch.rand(2, 1, 3, 32, 32), ids, torch.ones_like(ids), torch.rand(2, 1, cfg.state_dim))
    logits_plain, actions_plain = model(*args)
    logits, actions = model(*args, bev_images=scene.images, bev_cameras=scene.cameras)
    bev_tokens = model.bev_encoder.num_tokens
    assert logits.shape[1] == logits_plain.shape[1] + bev_tokens
    assert actions.shape == actions_plain.shape == (2, 1, cfg.action_chunk_size, cfg.action_dim)


def test_bev_inputs_require_use_bev(scene: SceneBatch) -> None:
    cfg = small_config()
    model = HaloVLM(cfg)
    ids = torch.tensor([[cfg.image_token_id, 5]])
    with pytest.raises(ValueError):
        model(
            torch.rand(1, 1, 3, 32, 32), ids, torch.ones_like(ids), torch.rand(1, 0, cfg.state_dim),
            bev_images=scene.images[:1], bev_cameras=scene.cameras,
        )
