---
name: hale-vla-dev
description: Set up, navigate and smoke-test the Hale-VLA (HaloVLM) codebase. Use when starting work in this repo, when imports fail, when you need to know where a component lives, or before changing model code in src/Halo_VLA/models.
---

# Hale-VLA development

Hale-VLA is a from-scratch PyTorch vision-language-action model. The model class is
`HaloVLM` (`src/Halo_VLA/models/halo_vla.py`); everything is configured by the
`HaloVLMConfig` dataclass in `config/model_config.py`.

## Environment

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Imports are **not** package-relative. Model files do `from models.x import ...` and
`from config import ...`, so both the repo root and `src/Halo_VLA` must be on the path:

```bash
export PYTHONPATH=.:src/Halo_VLA
```

The scripts in `scripts/` insert these paths themselves; ad-hoc snippets and tests need
the export above.

## Code map

| Path | What it is |
|---|---|
| `config/model_config.py` | `HaloVLMConfig`: every hyper-parameter (vision, decoder / LLM, BEV, action, world model, dual system) |
| `config/run_config.py`, `config/sections/` | YAML run-config schema (`scripts/train.py --config`), Hale-VLM layout |
| `config/special_tokens.json` | `<image>`, `<halo_action>`, `<state>`, `<halo_world_video>` on top of the cosmo2 tokenizer (ids rewritten from the LLM's tokenizer when `llm_backbone` is set) |
| `src/Halo_VLA/models/halo_vla.py` | `HaloVLM.forward(...)` → `(logits, action_hiddens, visual_context_emb, world_video_query_hiddens)`; `sample_actions`, `compute_flow_loss`, `encode_sequence` |
| `src/Halo_VLA/models/vision_encoders.py`, `vit.py`, `image_proj.py` | ViT / SigLIP / CLIP / HaleBlocks encoders; MLP / Q-Former / gated cross-attention projectors |
| `src/Halo_VLA/models/language_decoder.py` | pretrained HaleBlocks LLM as the decoder (LoRA / full / frozen) |
| `src/Halo_VLA/models/transformer.py`, `moe.py` | built-in decoder: pre-norm blocks with DeepSeekMoE FFNs |
| `src/Halo_VLA/models/bev_encoder.py` | multi-camera BEV scene tokens (Halos-lift `lifting`) |
| `src/Halo_VLA/models/flow_action_decoder.py`, `state_encoder.py` | flow-matching action head, proprioception encoder |
| `src/Halo_VLA/models/dit_frame_prediction.py` | optional DiT world model (future frames) |
| `src/Halo_VLA/dual_system/` | dual-system VLA (System 2 / latent / System 1 / controller) |
| `src/Halo_VLA/training/` | `HaloVLATrainingStep`, evaluator, DDP / Ray / WebDataset helpers |
| `dataloader/` | EO-Data1.5M, DROID, AIRoA MoMa loaders; `stages/` dataset registries |
| `scripts/train.py` / `inference.py` / `visualize.py` / `train_{reinforce,grpo,dpo,dreamer}.py` | training, generation, visualisation, RL |

`vlm.py`, `ha_vlm.py` and `action_decoder.py` are older prototypes, not used by `HaloVLM`.

## Smoke test (CPU, ~20 s)

```bash
PYTHONPATH=.:src/Halo_VLA python - <<'PY'
import torch
from config import HaloVLMConfig
from models.halo_vla import HaloVLM
c = HaloVLMConfig(enable_visual_dit=False)
m = HaloVLM(c).eval()
ids = torch.tensor([[c.image_token_id, 1, 2, c.state_token_id, 3, c.action_token_id]])
logits, act_h, _, _ = m(torch.randn(1, 1, 3, 224, 224), ids, torch.ones_like(ids), torch.randn(1, 1, c.state_dim))
print(logits.shape, m.sample_actions(act_h).shape)   # [1, 202, 49156]  [1, 1, 16, 7]
PY
```

Or a whole training run: `python scripts/train.py --config configs/overfit.yaml`.

## Things that surprise people

- BEV tokens (optional) and image patches are **prepended** to the sequence (196 per 224px
  image with the MLP projector), so logits are `[B, T_bev + N_img*P + seq_len, V]`. Anything
  that indexes text positions must offset by `num_prepended = logits.size(1) - input_ids.size(1)`
  (see `HaloVLATrainingStep`).
- `<state>` tokens are replaced **in place**; `<halo_action>` / `<halo_world_video>` tokens stay
  as text tokens and their hidden states are read out after the decoder.
- `VisTransformer` builds `TransformerBlock` with its defaults, so the ViT also uses a
  `DeepseekMoE` FFN (16 routed experts, top-4) — not the MLP `vit_mlp_dim` suggests.
- `HeadAttn` scales scores by `1/sqrt(emb_dim)`, not `1/sqrt(head_size)`.
- `moe.BestKRouter` and `transformer.SelfAttn` contain bugs but are unused; don't wire
  them in without fixing them.

## Checks before committing

```bash
black --check . && ruff check . && pytest
```

Run `pytest -o addopts=""` (offline, CPU). Add tests beside any model change, using a tiny
config (`emb_dim=64, dec_num_layers=1, vit_num_layers=1, use_moe=False, enable_visual_dit=False`).
