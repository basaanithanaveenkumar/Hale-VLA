# Hale-VLA documentation

Hale-VLA is a from-scratch PyTorch vision-language-action model: a ViT, a causal
decoder with DeepSeek-style mixture-of-experts, a proprioceptive state encoder and a
continuous action head, all driven by three special tokens (`<image>`, `<state>`,
`<action>`) in a Qwen2.5 chat transcript.

| Page | Contents |
|---|---|
| [Getting started](getting-started.md) | install, smoke test, first training run |
| [Architecture](architecture.md) | Mermaid diagrams of the model, token flow, MoE layer and training loop |
| [Configuration](configuration.md) | every field of `HaloVLMConfig` and the training CLI flags |
| [Data](data.md) | EO-Data1.5M subsets, sample format, collate output |
| [API reference](api.md) | public classes and their tensor shapes |
| [Blog](blog/README.md) | long-form write-ups |

Also: the [paper](../paper/main.tex) and the [project page](../project-page/index.html).
