# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Production-spec pass 2 (mirrors HALE-WAM): registries replace the if/elif chains on dual-system presets (`DUAL_SYSTEM_PRESETS`), latent extractors (`LATENT_EXTRACTORS`), System 1 policies (`SYSTEM1_POLICIES`, a strategy per policy kind shared by training, sampling and the deploy graphs), bridge projectors, image projectors (`IMAGE_PROJECTORS`), vision encoders (`VISION_ENCODERS`) and training datasets (`TRAIN_DATASETS`); `scripts/train.py` delegates to `CheckpointManager`, `build_model_config` and `build_train_loader`; `dual_system.config` keeps `PRESETS` / `resolve_dual_system_config` / `parse_overrides` importable (they live in `dual_system.presets` / `dual_system.resolution`)
- The first production-spec pass (shared `hale_core` exceptions, loguru, one class per file, thin scripts, profiling package) is ported from HALE-WAM; BEV tokens (`model.bev`) are unchanged, and `lifting` is imported only when a BEV encoder is built
- Tests split into `tests/unit`, `tests/integration` and `tests/support`; the run-config test no longer globs `configs/deploy`, and `configs/deploy/_shared/parts` is tracked (the build-directory `.gitignore` rule had hidden it)
- Google-style docstrings and type hints on the public API, `docs/registries.md` and mermaid diagrams, ruff configuration and formatting

### Added
- Initial project setup with uv package manager
- Vision Transformer (ViT) implementation
- Transformer encoder/decoder backbone
- Language model head for token generation
- Image projection layer
- Mixture of Experts module
- Positional embedding implementations
- Comprehensive test suite
- Development tooling (pytest, black, ruff, mypy)
- Documentation and contribution guidelines

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2026-02-14

### Added
- Initial release
- Basic VLA architecture components
- Project scaffolding

[Unreleased]: https://github.com/yourusername/halo-vla/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/halo-vla/releases/tag/v0.1.0
