# Hale-VLA documentation

Hale-VLA (the Halo vision-language-action model, with an optional DiT world model) is one transformer that **perceives** camera frames and
robot state, **reasons** in language, **acts** with a flow-matching action head and
**imagines** future frames with a DiT world model.

| Page | Contents |
|---|---|
| [Getting started](getting-started.md) | install, smoke test, first runs |
| [Architecture](architecture.md) | Mermaid diagrams: system, token routing, flow head, DiT conditioning, training step |
| [Configuration](configuration.md) | YAML run configs (`--config`), `HaloVLMConfig` fields, training flags |
| [LLM backbones](llm_backbone.md) | pretrained, LoRA-finetunable language decoders (incl. multimodal text towers) |
| [Vision encoders & validation](vision_encoders_and_validation.md) | SigLIP / CLIP / HaleBlocks encoders, held-out metrics |
| [Dual-system VLA](dual_system.md) | System 2 / System 1 from first principles, every published variant with a diagram |
| [Distributed training](distributed.md) | torchrun, Ray Train, WebDataset |
| [Data](data.md) | datasets and batch format |
| [API](api.md) | module reference |
| [Driving to humanoids](driving-to-humanoids/README.md) | guide: from driving stacks to humanoid control |
| [World model reference](world_model.md) | detailed write-up of the DiT head, losses and bug fixes |
| [Deployment](deploy.md) | export to TensorRT / Jetson, MLX, AMD MIGraphX / Ryzen AI, Qualcomm AI Hub / ExecuTorch |
| [Blog](blog/README.md) | long-form posts |

Also: the [paper](../paper/main.tex) and the [project page](../project-page/index.html).
