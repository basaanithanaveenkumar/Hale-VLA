# Getting started

## Install

```bash
git clone https://github.com/basaanithanaveenkumar/Hale-VLA
cd Hale-VLA
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Python ≥ 3.10 and PyTorch ≥ 2.0 are required. A GPU is recommended for training; the
smoke test below runs on CPU.

## Import paths

Model modules import each other as `models.*` and read `config`, so put both the repo
root and `src/Halo_VLA` on `PYTHONPATH` for your own scripts (the bundled scripts do this
automatically):

```bash
export PYTHONPATH=.:src/Halo_VLA
```

## Smoke test

```python
import torch
from config import HaloVLMConfig
from models.halo_vla import HaloVLM

cfg = HaloVLMConfig()
model = HaloVLM(cfg).eval()

# one image, some text, one state, one action query
ids = torch.tensor([[cfg.image_token_id, 1, 2, cfg.state_token_id, 3, cfg.action_token_id]])
images = torch.randn(1, 1, 3, 224, 224)          # [B, N_img, 3, H, W]
states = torch.randn(1, 1, cfg.state_dim)         # [B, N_state, state_dim]

logits, actions = model(images, ids, torch.ones_like(ids), states)
print(logits.shape)   # torch.Size([1, 202, 151668])  -> 196 patches + 6 tokens
print(actions.shape)  # torch.Size([1, 1, 1, 7])      -> [B, n_action_tokens, chunk, action_dim]
```

The default configuration has 355.9M parameters.

## First training run

```bash
python scripts/train.py --subset interleave-temporal --max_samples 100 \
  --batch_size 2 --epochs 1 --action_dim 32 --state_dim 32
```

This downloads a slice of `IPEC-COMMUNITY/EO-Data1.5M` from the Hugging Face Hub,
trains for one epoch and writes checkpoints to `checkpoints/`. See
[configuration](configuration.md) for all flags.

## Inference

```bash
python scripts/inference.py --help
python scripts/visualize.py --help
```

## Development

```bash
black . && ruff check . && isort . && pytest
```
