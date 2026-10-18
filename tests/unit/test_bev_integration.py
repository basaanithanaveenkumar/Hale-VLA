"""Integration of the `lifting` package (multi-camera BEV lifting) into Halo-VLA.

BEV tokens are prepended before the image patches; every consumer of the token prefix (language
loss offsets, action / world-video query positions, the DiT world model's image slots, the
dual-system System 2, a pretrained LLM decoder, the YAML run config) must account for them.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

lifting = pytest.importorskip("lifting")

from lifting.data import CameraRig, SceneBatch, SyntheticSceneDataset  # noqa: E402

from config import HaloVLMConfig  # noqa: E402
from models.bev_encoder import BEVSceneEncoder  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402
from training import HaloVLATrainingStep  # noqa: E402

IMAGE = (32, 64)


def small_config(**overrides) -> HaloVLMConfig:
    base = dict(
        emb_dim=64, img_size=32, patch_size=8, vit_num_layers=1, vit_num_heads=4,
        vit_mlp_dim=64, dec_num_layers=1, dec_num_heads=4, dec_mlp_dim=64, use_moe=False,
        action_hidden_dims=(64,), state_hidden_dims=(64,), action_dim=4, state_dim=3, action_chunk_size=2,
        flow_hidden_dim=32, enable_visual_dit=False,
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


def vla_batch(cfg: HaloVLMConfig, scene: SceneBatch | None = None) -> dict:
    ids = torch.tensor([[cfg.image_token_id, 5, 6, cfg.state_token_id, cfg.action_token_id]] * 2)
    batch = {"images": torch.rand(2, 1, 3, 32, 32), "input_ids": ids, "attention_mask": torch.ones_like(ids),
             "labels": ids.clone(), "states": torch.rand(2, 1, cfg.state_dim),
             "actions": torch.randn(2, cfg.action_chunk_size, cfg.action_dim),
             "action_mask": torch.ones(2, cfg.action_chunk_size)}
    if scene is not None:
        batch["bev_images"], batch["bev_cameras"] = scene.images, scene.cameras
    return batch


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
    b = vla_batch(cfg)
    args = (b["images"], b["input_ids"], b["attention_mask"], b["states"])
    logits_plain, hiddens_plain, _, _ = model(*args)
    logits, hiddens, _, _ = model(*args, bev_images=scene.images, bev_cameras=scene.cameras)
    assert logits.shape[1] == logits_plain.shape[1] + model.bev_encoder.num_tokens
    # forward returns the decoder states at <halo_action>; the flow decoder turns them into chunks
    assert hiddens.shape == hiddens_plain.shape == (2, 1, cfg.emb_dim)
    assert not torch.allclose(hiddens, hiddens_plain)  # the action query attends to the BEV map
    actions = model.sample_actions(hiddens, num_steps=2)
    assert actions.shape == (2, 1, cfg.action_chunk_size, cfg.action_dim)


def test_bev_inputs_require_use_bev(scene: SceneBatch) -> None:
    cfg = small_config()
    model = HaloVLM(cfg)
    ids = torch.tensor([[cfg.image_token_id, 5]])
    with pytest.raises(ValueError, match="use_bev"):
        model(torch.rand(1, 1, 3, 32, 32), ids, torch.ones_like(ids), torch.rand(1, 0, cfg.state_dim),
              bev_images=scene.images[:1], bev_cameras=scene.cameras)
    with pytest.raises(ValueError, match="bev_cameras"):
        HaloVLM(small_config(use_bev=True, bev_lifter="simple_bev"))(
            torch.rand(1, 1, 3, 32, 32), ids, torch.ones_like(ids), torch.rand(1, 0, cfg.state_dim),
            bev_images=scene.images[:1])


def test_world_model_reads_image_slots_after_the_bev_tokens(scene: SceneBatch) -> None:
    """The DiT conditions on decoder states at the image positions, which BEV tokens shift."""
    cfg = small_config(use_bev=True, bev_lifter="simple_bev", enable_visual_dit=True, dit_patch_size=8,
                       dit_hidden_size=32, dit_depth=1, dit_num_heads=4, dit_max_resolution=32,
                       dit_time_freq_dim=32, num_visual_predict_frames=1)
    model = HaloVLM(cfg).eval()
    b = vla_batch(cfg)
    b["images"] = torch.rand(2, 2, 3, 32, 32)
    b["input_ids"] = torch.tensor([[cfg.image_token_id, cfg.image_token_id, 5, cfg.action_token_id]] * 2)
    b["attention_mask"] = torch.ones_like(b["input_ids"])
    with torch.no_grad():
        logits, _, visual_ctx, _ = model(b["images"], b["input_ids"], b["attention_mask"], b["states"],
                                         bev_images=scene.images, bev_cameras=scene.cameras)
        hidden = model.decoder_transformer(  # recompute the decoder states to read the image slots
            model._embed_inputs(b["images"], b["input_ids"], b["states"], scene.images, scene.cameras)[0]
            + model.pos_embed(torch.arange(logits.size(1))).unsqueeze(0),
            pad_mask=torch.ones(2, logits.size(1)))
    num_bev, per_image = model.bev_encoder.num_tokens, cfg.num_image_tokens_per_image
    first_frame = hidden[:, num_bev:num_bev + per_image].mean(1)  # past frame = frame 0
    torch.testing.assert_close(visual_ctx, first_frame)


def test_training_step_offsets_the_language_loss_by_the_bev_tokens(scene: SceneBatch) -> None:
    cfg = small_config(use_bev=True, bev_lifter="simple_bev")
    torch.manual_seed(0)
    model = HaloVLM(cfg)
    step = HaloVLATrainingStep(model, visual_loss_weight=0.0)
    out = step(vla_batch(cfg, scene))
    assert torch.isfinite(out.total) and out.lang > 0 and out.act > 0
    out.total.backward()
    assert any(p.grad is not None for p in model.bev_encoder.parameters())
    # Same loss as slicing the logits by hand after the BEV + image prefix.
    from training.training_step import compute_language_loss

    b = vla_batch(cfg, scene)
    model.eval()  # no dropout: both passes see the same network
    with torch.no_grad():
        logits = model(b["images"], b["input_ids"], b["attention_mask"], b["states"],
                       bev_images=scene.images, bev_cameras=scene.cameras)[0]
        prefix = model.bev_encoder.num_tokens + cfg.num_image_tokens_per_image
        torch.testing.assert_close(compute_language_loss(logits, b["labels"], prefix), step(b).lang.detach())


def test_dual_system_system2_sees_the_bev_map(scene: SceneBatch) -> None:
    from dual_system import DualSystemVLA, resolve_dual_system_config

    cfg = small_config(use_bev=True, bev_lifter="simple_bev")
    model = DualSystemVLA(HaloVLM(cfg), resolve_dual_system_config(
        "basic", {"system1_dim": 32, "system1_layers": 1, "system1_heads": 4, "sample_steps": 2})).eval()
    with torch.no_grad():
        with_bev = model.encode_latent(vla_batch(cfg, scene))[0]
        without = model.encode_latent(vla_batch(cfg))[0]
    assert not torch.allclose(with_bev, without)
    model.train()
    model(vla_batch(cfg, scene)).total.backward()
    assert any(p.grad is not None for p in model.system2.model.bev_encoder.parameters())


def test_bev_yaml_section_builds_the_encoder() -> None:
    pytest.importorskip("hale_core")
    from config.overrides import apply_model_overrides
    from config.run_config import load_vla_config

    run = load_vla_config(ROOT / "configs" / "examples" / "bev_tokens.yaml")
    config = apply_model_overrides(HaloVLMConfig(), run.model_overrides())
    assert config.use_bev and config.bev_lifter == "bevformer"
    assert config.bev_bounds == ((-25.6, 25.6), (-25.6, 25.6), (-2.0, 4.0))
    enc = BEVSceneEncoder.from_config(config)
    assert enc.num_tokens == 256
