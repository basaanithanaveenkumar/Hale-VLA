# Halo-VLA: Vision-Language Assistant

A PyTorch-based Vision-Language Model (VLA) implementation combining visual and linguistic understanding.

## Resources

| | |
|---|---|
| Paper (arXiv source) | [`paper/main.tex`](paper/main.tex) — build with `make -C paper` |
| Project page | [basaanithanaveenkumar.github.io/Hale-VLA](https://basaanithanaveenkumar.github.io/Hale-VLA/) ([source](project-page/index.html)) |
| Documentation | [`docs/`](docs/README.md) — getting started, configuration, data, API |
| Architecture diagrams | [`docs/architecture.md`](docs/architecture.md) (Mermaid) |
| Blog | [Three tokens and a router: building a VLA from scratch](docs/blog/2026-09-28-building-a-vla-from-scratch.md) |
| Claude Code skills | [`.claude/skills/`](.claude/skills) — `hale-vla-dev`, `hale-vla-train`, `hale-vla-extend`, `hale-publish` |

## Features

| | |
|---|---|
| **Vision** | built-in ViT, pretrained SigLIP / CLIP, or any HaleBlocks vision backbone (`model.vision.encoder`); MLP, BLIP-2 Q-Former or Flamingo gated cross-attention projector |
| **Language decoder** | built-in MoE decoder, or any of HaleBlocks' 49 fine-tunable LLMs (`model.llm.backbone`; LoRA / full / frozen). These include the 2026 small backbones, many of them text towers of natively multimodal models: Qwen3.5, Gemma 4, Ministral 3, Cosmos-Reason2, LFM2.5-VL, SmolVLM2, SmolLM3, Granite 4.2 ([docs/llm_backbone.md](docs/llm_backbone.md)) |
| **Actions** | flow-matching action chunks at `<halo_action>` tokens; proprioception tokens at `<state>` |
| **Multi-camera BEV** | optional bird's-eye-view scene tokens from Halos-lift (TPVFormer, BEVFormer, Lift-Splat, Simple-BEV, TIIM), placed before the image patches (below) |
| **Dual system** | slow System 2 (this VLM) + fast System 1 policy; presets `basic`, `lcb`, `dp_vla`, `hirt`, `robodual`, `dexvla`, `openhelix`, `custom` ([docs/dual_system.md](docs/dual_system.md)) |
| **World model (optional)** | DiT future-frame prediction at `<halo_world_video>` tokens (`model.world_model`, [docs/world_model.md](docs/world_model.md)) |
| **RL fine-tuning** | REINFORCE, GRPO, DPO and Dreamer-style hooks (`src/Halo_VLA/rl`, `scripts/train_*.py`) |
| **Config** | Hale-VLM-style YAML run configs with `inherits:`, validated against a schema; CLI flags override ([docs/configuration.md](docs/configuration.md)) |
| **Data** | EO-Data1.5M, DROID, AIRoA MoMa loaders; pretrain / mid-train / post-train dataset registries (`dataloader/stages`) |
| **Scale-out** | torchrun DDP, Ray Train, WebDataset shards ([docs/distributed.md](docs/distributed.md)) |
| **Evaluation** | held-out validation with action metrics (L1, RMSE, chunk accuracy, smoothness), language / generation metrics (`src/Halo_VLA/metrics`) |
| **Notebooks** | backbone guides and the dual-system tutorial (`notebooks/`) |

## Multi-camera BEV scene tokens (Halos-lift)

[Halos-lift](https://github.com/basaanithanaveenkumar/Halos-lift) is a standalone library that is
declared as a git dependency, so `uv sync` installs it. To use it in another project:

```bash
uv add "halos-lift @ git+https://github.com/basaanithanaveenkumar/Halos-lift"
```

Enable BEV tokens in the VLA:

```python
from config import HaloVLMConfig
from models.halo_vla import HaloVLM
from lifting import Cameras

cfg = HaloVLMConfig(use_bev=True, bev_lifter="bevformer", bev_num_cameras=6, bev_image_size=(224, 224))
model = HaloVLM(cfg)
cameras = Cameras(intrinsics=K, cam_to_ego=T, image_size=(224, 224))  # K (B,6,3,3), T (B,6,4,4)
logits, action_hiddens, _, _ = model(images, input_ids, attention_mask, states,
                                     bev_images=surround_images, bev_cameras=cameras)  # (B,6,3,H,W)
actions = model.sample_actions(action_hiddens)
```

`models/bev_encoder.py` (`BEVSceneEncoder`) turns the BEV map into `(X/4)·(Y/4)` tokens, which are
prepended to the image patches. The language-loss offsets, the action / world-video query positions,
the DiT world model's image slots and the dual-system System 2 all account for them. In a YAML
config, set `model.bev` (`configs/examples/bev_tokens.yaml`). Batches must then carry `bev_images`
and `bev_cameras`; the shipped EO / DROID / MoMa loaders don't provide them. Any registered lifter name works for `bev_lifter`. See
the [Halos-lift README](https://github.com/basaanithanaveenkumar/Halos-lift) for the library, its tests, and the
synthetic-scene verifier that checks each lifting technique.

## Project Structure

```
Hale-VLA/
├── config/                 # HaloVLMConfig, special tokens, YAML run-config schema (run_config.py, sections/)
├── configs/                # run configs: base, overfit, projectors, dual_system/, examples/, roadmap/
├── dataloader/             # EO-Data, DROID, AIRoA MoMa loaders; stages/ dataset registries
├── src/Halo_VLA/
│   ├── models/             # HaloVLM, vision encoders, LLM bridge, BEV encoder, flow head, DiT world model
│   ├── dual_system/        # System 2 / latent bridge / System 1 / controller
│   ├── training/           # training step, evaluator, DDP / Ray / WebDataset helpers
│   ├── metrics/            # action, generation, policy metrics
│   └── rl/                 # REINFORCE / GRPO / DPO / Dreamer
├── scripts/                # train.py (--config), inference, visualisation, RL trainers, profiling
├── notebooks/              # backbone guides, dual-system tutorial
├── tests/                  # pytest suite (offline, CPU)
└── docs/                   # architecture, configuration, data, guides
```

## Installation

### Using UV (Recommended)

```bash
# Install the package in development mode
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
pip install -e ".[dev]"
```

## Quick Start

```bash
uv sync --extra config --extra finetune
python scripts/train.py --config configs/overfit.yaml                       # CPU smoke test
python scripts/train.py --config configs/examples/llm_qwen3_5_4b_base.yaml  # Qwen3.5 decoder, LoRA
python scripts/train.py --config configs/dual_system/openhelix.yaml         # dual-system VLA
python scripts/train.py --config configs/examples/bev_tokens.yaml           # + multi-camera BEV tokens
```

```python
from config import HaloVLMConfig
from models.halo_vla import HaloVLM

model = HaloVLM(HaloVLMConfig(llm_backbone="smollm2-360m", action_dim=7, state_dim=8))
logits, action_hiddens, visual_ctx, world_queries = model(images, input_ids, attention_mask, states)
actions = model.sample_actions(action_hiddens)          # [B, n_action_tokens, chunk, action_dim]
```

## Development

### Setup Development Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Run Tests

```bash
pytest -o addopts=""          # whole suite, CPU, offline
pytest                        # with coverage
```

### Code Formatting and Linting

```bash
black .           # Format code
ruff check .      # Lint
isort .           # Sort imports
mypy models       # Type checking
```

## Requirements

- Python ≥ 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- numpy >= 1.24.0

See [pyproject.toml](pyproject.toml) for complete dependency list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

```bibtex
@misc{naveenkumar2026halevla,
  title  = {Hale-VLA: An Interleaved Vision--Language--Action Transformer
            with DeepSeek-Style Mixture-of-Experts, Built From Scratch},
  author = {NaveenKumar, B A},
  year   = {2026},
  url    = {https://github.com/basaanithanaveenkumar/Hale-VLA}
}
```
