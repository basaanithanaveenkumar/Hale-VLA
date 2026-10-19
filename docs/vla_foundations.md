# VLA foundations: nine concept clusters, built and measured

`src/Halo_VLA/vla_foundations/` implements, at toy scale, the design choices that sit under every VLA, and
`notebooks/vla_foundations/` explains each cluster step by step in the style of the BEVFormer notebook (numbered steps, a diagram, a verification
against a slow or by-hand computation, an end-to-end experiment, controls, recap + exercises). Every notebook states where it **corrects**
common summaries (several claims in popular study outlines do not match the papers) and reports what it actually measured.

| notebook | covers |
|---|---|
| `action_representations_step_by_step.ipynb` | RT-2 / OpenVLA 256-bin tokens (quantisation error, quantile ranges, KV-cached sequential decoding), conditional flow matching vs DDPM / DDIM (2-D toy, analytic optimal field, NFE curves, path straightness), Diffusion Policy, chunking (chunk length vs noise, receding horizon, absolute vs relative chunks, ACT temporal ensembling, control-rate arithmetic) |
| `vlm_backbone_step_by_step.ipynb` | SigLIP sigmoid loss, DINOv2-style self-distillation, patch-PCA and probes under an appearance shift, frozen / fused encoders in a policy, RT-2 from-scratch / fine-tune / co-fine-tune at three sizes, early vs late vs pooled fusion with word→image attention |
| `fast_tokenization_step_by_step.ipynb` | FAST = DCT + quantisation + BPE built from scratch: energy compaction, frequency-major order, compression/error frontier, bins vs FAST policies at a coarse and a 4× finer control rate, malformed sequences |
| `finetuning_methods_step_by_step.ipynb` | full fine-tuning, LoRA, OFT (orthogonal, Cayley parametrisation; **not** OpenVLA-OFT), frozen backbone + head: mechanics verified by hand, learning-rate sweep, learning vs forgetting, replay |
| `training_pipeline_step_by_step.ipynb` | behaviour cloning as maximum likelihood, covariate shift made visible, compounding error, DAgger, noise-injected demonstrations, chunking, pre-training vs data scale, data quality, the data reality of OXE-scale training |
| `generalization_step_by_step.ipynb` | do VLAs use language (shortcut and counterfactual tests), six distribution shifts, embodiment as an ID vs a kinematic descriptor (zero- and few-shot), long-horizon skill chaining |
| `inference_control_step_by_step.ipynb` | where a VLA call's time goes, query vs action rate, GR00T System 2 / System 1 at different refresh rates, a lightly damped servo with and without a classical tracker, embodied thinking |
| `perception_interface_step_by_step.ipynb` | pinhole cameras, depth and pointmaps (camera vs robot frame), RGB ambiguity, five input representations, occlusion and multi-view fusion, classical vs learned precision, design options |
| `evaluation_step_by_step.ipynb` | what the benchmarks test, a toy four-suite benchmark, memoriser vs generalist, confidence intervals / seed variance / paired tests, a SIMPLER-style sim-vs-real study (Pearson, MMRV), success vs progress and failure taxonomy |

## Modules

| module | public classes / functions |
|---|---|
| `discretization.py` | `ActionBinner` (uniform or quantile bins) |
| `autoregressive.py` | `ARConfig`, `AutoregressiveActionPolicy` (naive and KV-cached decoding, variable-length hooks), `make_cofine_tune_data` |
| `fast_policy.py` | `FastActionPolicy` (π0-FAST-style policy over `pi_family.FastActionTokenizer` tokens) |
| `diffusion.py` | `DDPMSchedule`, `DiffusionPolicy` (ε-prediction, DDPM and DDIM) |
| `toy2d.py` | 2-D flow / DDPM / DDIM reference implementations, `path_straightness` |
| `chunking.py` | `ChunkedRunner` (receding horizon, ACT ensembling, noise, pushes, relative or absolute chunks), `effective_control_hz` |
| `multimodality.py` | `mode_coverage`, `separated_any_scenarios` |
| `encoders.py`, `probes.py`, `analysis.py` | `ToyViT`, `SigLIPLite`, `DinoLite`; linear probes; word→image attention breakdown |
| `fusion.py`, `tasks.py` | `FusionPolicy` (early / late / pooled), `FusedVision`; colour, side, synonym task families |
| `finetuning.py` | `LoRALinear`, `OFTLinear`, `inject_adapters`, `freeze_all_but`, `hyperspherical_energy` |
| `training.py` | `narrow_start_demos`, `dagger_episodes`, `perturbed_expert_episodes`, `state_action_error`, `ReplayMixture` |
| `control.py` | `PlantRunner` (servo + tracker), `DualRateRunner` (System 2 / System 1) |
| `thinking.py` | `ThinkingPolicy` (reasoning trace before the actions) |
| `generalization.py` | `shortcut_demos`, `language_sensitivity`, `MorphPolicy`, `adapt_new_id`, `run_chain` |
| `world3d.py`, `policy3d.py` | pinhole cameras, ray-cast renderer, depth / pointmaps; `Policy3D`, input representations, `pointmaps_to_bev` |
| `evaluation_stats.py` | `wilson_interval`, `mcnemar_exact`, `required_episodes`, `bootstrap_difference`, `pearson`, `mmrv`, `classify_failures` |

## Corrections to common summaries (the notebooks follow the papers)

* RT-2 actions are end-effector deltas (8 integers per step); the ~5 Hz belongs to the 5B model (55B: 1-3 Hz). π0 uses PaliGemma (SigLIP + Gemma 2B), not PaLI-X.
* SigLIP's contribution is the sigmoid loss; dense spatial strength comes from DINOv2, hence OpenVLA fuses both. RT-2's size gain is substantial (Table 6: 44 → 63 co-fine-tuned), not marginal.
* Flow matching's *conditional* paths are straight, the learned marginal field is not; a one-step generative head collapses to the mean.
* **OpenVLA-OFT** = an optimised fine-tuning *recipe* (parallel decoding, action chunking, continuous L1 head; 26× throughput, 76.5 → 97.1 % LIBERO), not orthogonal fine-tuning.
* FAST = DCT + quantisation + BPE; the FAST+ universal tokenizer is trained on ~1M real trajectories. Open X-Embodiment = 60 datasets, 22 embodiments, > 1M trajectories.
* GR00T N1's System 1 is a flow-matching DiT (not an MLP head); Hz figures are not comparable across systems (query rate vs action rate).
* SIMPLER is a set of environments *validated by sim-vs-real rank agreement* (Pearson, MMRV); LIBERO-PRO / LIBERO-Plus show standard LIBERO scores partly reflect memorisation.
* PointVLA, SpatialVLA and SeeR-VLA (robot-frame pointmaps) are described as their papers state; unverified numbers from third-party outlines are not used.

## Running

```bash
cd Hale-VLA
PYTHONPATH=.:src/Halo_VLA python -m pytest tests/test_vla_foundations.py tests/test_vla_foundations_more.py -o addopts=""
# notebooks: run from notebooks/vla_foundations with a kernel that has torch + matplotlib (20-60 minutes each on a laptop CPU)
```

Notebook numbers vary from run to run (training is stochastic and multi-threaded): single-seed differences of ~10 points are noise, and the
notebooks say so wherever it matters.
