# Halo-VLA: Vision-Language Assistant

A PyTorch-based Vision-Language Model (VLA) implementation combining visual and linguistic understanding.

## Features

- **Vision Transformer (ViT)**: State-of-the-art image encoding
- **Transformer Backbone**: Multi-head attention mechanism for sequence modeling
- **Language Model Head**: Causal language modeling for text generation
- **Mixture of Experts**: Efficient multi-expert model architecture
- **Positional Embeddings**: Learnable positional encoding for sequences
- **Image Projection**: Efficient image-to-embedding projection layer
- **Multi-camera BEV tokens** (optional): surround-view images are lifted to a metric bird's-eye view
  by the [Halos-lift](https://github.com/basaanithanaveenkumar/Halos-lift) package (`import lifting`) (TPVFormer, BEVFormer, Lift-Splat, Simple-BEV, TIIM,
  all in pure PyTorch) and fed to the decoder as extra tokens

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
logits, actions = model(images, input_ids, attention_mask, states,
                        bev_images=surround_images, bev_cameras=cameras)  # surround_images (B,6,3,H,W)
```

`models/bev_encoder.py` (`BEVSceneEncoder`) turns the BEV map into `(X/4)·(Y/4)` tokens, which are
prepended to the image patches. Any registered lifter name works for `bev_lifter`. See
the [Halos-lift README](https://github.com/basaanithanaveenkumar/Halos-lift) for the library, its tests, and the
synthetic-scene verifier that checks each lifting technique.

## Project Structure

```
Halo-VLA/
├── models/
│   ├── __init__.py
│   ├── ha_vlm.py              # Main VLA model
│   ├── vlm.py                 # Base VLM class
│   ├── vit.py                 # Vision Transformer
│   ├── transformer.py         # Transformer encoder/decoder
│   ├── lm_head.py            # Language model head
│   ├── image_proj.py         # Image projection layer
│   ├── moe.py                # Mixture of Experts
│   └── positional_embeddings.py  # Positional encoding
├── tests/
│   ├── __init__.py
│   └── test_models.py        # Model tests
├── pyproject.toml            # Project configuration
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
└── README.md               # This file
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

```python
from models import VLM, ViT, Transformer

# Initialize components
vit = ViT(...)
transformer = Transformer(...)
vla_model = VLM(vision_encoder=vit, language_model=transformer)

# Forward pass
output = vla_model(images, text_tokens)
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
pytest
pytest --cov=models  # With coverage
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
@software{halo_vla_2026,
  title = {Halo-VLA: Vision-Language Assistant},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/halo-vla}
}
```