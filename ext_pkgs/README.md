# ext_pkgs — vendor SDK sources (git submodules)

Shallow (`--depth 1`, `shallow = true`) submodules of the toolchains `hale-vla-deploy` targets.
They are **reference sources** (samples, op-support tables, converters, docs) pinned next to the
code; the pipeline itself uses the installed packages (`pip install -e ".[deploy-<vendor>]"`).

```bash
git submodule update --init --depth 1 ext_pkgs/<name>   # fetch one (LFS: GIT_LFS_SKIP_SMUDGE=1)
git submodule update --remote --depth 1 ext_pkgs/<name> # move to the latest upstream commit
```
Nested submodules (e.g. ExecuTorch's third-party deps) are intentionally not initialised.

| submodule | upstream | used for |
|-----------|----------|----------|
| `TensorRT` | NVIDIA/TensorRT | ONNX parser, plugins, samples (TensorRT 10/11 strong typing) |
| `TensorRT-LLM` | NVIDIA/TensorRT-LLM | LLM serving (PyTorch backend, `trtllm-serve`) |
| `TensorRT-Edge-LLM` | NVIDIA/TensorRT-Edge-LLM | on-device LLM/VLM runtime for Jetson Thor / Orin |
| `Model-Optimizer` | NVIDIA/Model-Optimizer | AutoCast (FP16/BF16) and PTQ (INT8/FP8/INT4) |
| `Torch-TensorRT` | pytorch/TensorRT | partial TensorRT lowering of `torch.export` graphs |
| `mlx` | ml-explore/mlx | MLX arrays, `mx.export_function`, fused kernels |
| `mlx-lm` | ml-explore/mlx-lm | HF → MLX LLM conversion / serving |
| `mlx-vlm` | Blaizzy/mlx-vlm | HF → MLX VLM conversion |
| `AMDMIGraphX` | ROCm/AMDMIGraphX | AMD GPU graph compiler (`.mxr`), ORT MIGraphX EP |
| `Quark` | amd/Quark | quantization for the Ryzen AI NPU (A16W8, A8W8, BF16) |
| `RyzenAI-SW` | amd/RyzenAI-SW | Ryzen AI NPU examples, Vitis AI EP setup |
| `ai-hub-models` | qualcomm/ai-hub-models | AI Hub export recipes for Snapdragon NPUs |
| `executorch` | pytorch/executorch | ExecuTorch + QNN (HTP) backend for Android |
