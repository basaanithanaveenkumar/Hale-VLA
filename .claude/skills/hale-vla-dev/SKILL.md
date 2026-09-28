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
| `config/model_config.py` | `HaloVLMConfig` — every hyper-parameter (dims, MoE, action/state heads) |
| `config/special_tokens.json` | `<image>`=151665, `<action>`=151666, `<state>`=151667 on top of the Qwen2.5 tokenizer |
| `src/Halo_VLA/models/halo_vla.py` | `HaloVLM.forward(images, input_ids, attention_mask, states)` → `(logits, action_preds)` |
| `src/Halo_VLA/models/vit.py` | `PatchEmb` + `VisTransformer` (16x16 patches, learned pos-emb) |
| `src/Halo_VLA/models/transformer.py` | `HeadAttn`, `MultiHeadAttn`, pre-norm `TransformerBlock`, `DecoderTransformer` |
| `src/Halo_VLA/models/moe.py` | `NoiseBestKRouter`, SwiGLU `Expert`, `DeepseekMoE` (shared + routed experts) |
| `src/Halo_VLA/models/state_encoder.py` / `action_decoder.py` | MLP heads for proprioception in and continuous actions out |
| `src/Halo_VLA/models/image_proj.py` | 3-layer MLP vision→LLM projector |
| `dataloader/eo_dataset.py` | `EODataset`, `eo_collate_fn`, `build_eo_dataloader` for IPEC-COMMUNITY/EO-Data1.5M |
| `scripts/train.py` / `inference.py` / `visualize.py` | training loop, generation, video visualisation |
| `utils/param_count.py` | `log_module_parameters` |

`vlm.py` and `ha_vlm.py` are older prototypes and are not used by the training scripts.

## Smoke test (CPU, ~20 s)

```bash
PYTHONPATH=.:src/Halo_VLA python - <<'PY'
import torch
from config import HaloVLMConfig
from models.halo_vla import HaloVLM
c = HaloVLMConfig()
m = HaloVLM(c).eval()
ids = torch.tensor([[c.image_token_id, 1, 2, c.state_token_id, 3, c.action_token_id]])
logits, acts = m(torch.randn(1, 1, 3, 224, 224), ids, torch.ones_like(ids), torch.randn(1, 1, c.state_dim))
print(logits.shape, acts.shape)   # [1, 202, 151668]  [1, 1, 1, 7]
PY
```

Default config is ~356M parameters (≈155M of them are the tied-size token embedding
and LM head over the 151,668-token vocabulary).

## Things that surprise people

- Image patches are **prepended** to the sequence (196 per 224px image), so logits are
  `[B, N_img*196 + seq_len, V]`. Anything that indexes text positions must offset by
  `num_prepended = N_img * num_patches` (see `compute_language_loss` in `scripts/train.py`).
- `<state>` tokens are replaced **in place**; `<action>` tokens stay as text tokens and
  their hidden states are read out after the decoder.
- `VisTransformer` builds `TransformerBlock` with its defaults, so the ViT also uses a
  `DeepseekMoE` FFN (16 routed experts, top-4) — not the MLP `vit_mlp_dim` suggests.
- `attention_mask` is accepted but not applied inside attention; the decoder uses a
  causal mask only, so right-padding is safe but left-padding is not.
- `HeadAttn` scales scores by `1/sqrt(emb_dim)`, not `1/sqrt(head_size)`.
- `HaloVLM.forward` prints the decoder sequence length every call; silence it before
  benchmarking.
- `moe.BestKRouter` and `transformer.SelfAttn` contain bugs but are unused; don't wire
  them in without fixing them.

## Checks before committing

```bash
black --check . && ruff check . && pytest
```

`tests/test_models.py` is currently empty — add tests beside any model change
(shape tests with a tiny config: `emb_dim=64, dec_num_layers=2, vit_num_layers=1`).
