"""Flow-matching action decoder: training loss, gradients and ODE sampling."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import HaloVLMConfig  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402


def tiny_model() -> tuple[HaloVLM, HaloVLMConfig]:
    cfg = HaloVLMConfig(emb_dim=64, img_size=32, patch_size=8, vit_num_layers=1, vit_num_heads=4, vit_mlp_dim=64,
                        dec_num_layers=1, dec_num_heads=4, dec_mlp_dim=64, use_moe=False, enable_visual_dit=False,
                        state_hidden_dims=(64,), flow_hidden_dim=64, flow_time_embed_dim=16)
    torch.manual_seed(0)
    return HaloVLM(cfg), cfg


def test_flow_loss_trains_and_sampling_returns_chunks() -> None:
    model, cfg = tiny_model()
    ids = torch.tensor([[cfg.image_token_id, 5, 6, cfg.state_token_id, cfg.action_token_id]] * 2)
    logits, hiddens, _, _ = model(torch.rand(2, 1, 3, 32, 32), ids, torch.ones_like(ids), torch.rand(2, 1, cfg.state_dim))
    assert hiddens.shape == (2, 1, cfg.emb_dim)
    targets = torch.randn(2, cfg.action_chunk_size, cfg.action_dim)
    loss = model.compute_flow_loss(hiddens, targets, torch.ones(2, cfg.action_chunk_size))
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(p.grad is not None for p in model.flow_decoder.parameters())
    actions = model.sample_actions(hiddens.detach(), num_steps=3)
    assert actions.shape == (2, 1, cfg.action_chunk_size, cfg.action_dim) and torch.isfinite(actions).all()


def test_no_action_tokens_gives_zero_flow_loss() -> None:
    model, cfg = tiny_model()
    ids = torch.tensor([[cfg.image_token_id, 5, 6]])
    _, hiddens, _, _ = model(torch.rand(1, 1, 3, 32, 32), ids, torch.ones_like(ids), torch.rand(1, 0, cfg.state_dim))
    assert hiddens is None
    assert model.compute_flow_loss(None, torch.zeros(1, 2, cfg.action_dim), torch.ones(1, 2)).item() == 0.0
