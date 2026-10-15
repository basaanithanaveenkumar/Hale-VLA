---
name: hale-vla-train
description: Train, evaluate, run inference and render videos for Hale-VLA on EO-Data1.5M. Use when asked to train the model, debug losses, resume from a checkpoint, run generation, or visualise predicted vs ground-truth actions.
---

# Training and running Hale-VLA

## Data

`dataloader/eo_dataset.py` streams `IPEC-COMMUNITY/EO-Data1.5M` from the Hugging Face Hub.
17 subsets: 5 interleaved (`interleave-free_chat`, `-random_qa`, `-temporal`,
`-trajectory`, `-video_caption`) and 12 QA (`qa-affordance_qa`, `qa-task_planning`, ...).
Each sample becomes a ChatML transcript (`<|im_start|>system/user/assistant`) with
`<image>`, `<state>`, `<halo_action>` and `<halo_world_video>` placeholders. Only assistant tokens carry labels;
everything else is `-100`.

`eo_collate_fn` pads per batch and returns `images [B,N,3,H,W]`, `input_ids`,
`attention_mask`, `labels`, `actions [B,T,action_dim]`, `action_mask`, `states`,
`state_mask`.

Hub access is required on first use (`huggingface-cli login` if the dataset is gated).
Use `--max_samples` to keep experiments small.

## Train

Prefer a YAML run config (Hale-VLM layout, validated; flags override it):

```bash
python scripts/train.py --config configs/overfit.yaml                        # smoke test
python scripts/train.py --config configs/examples/llm_qwen3_5_4b_base.yaml   # pretrained LLM decoder
python scripts/train.py --config configs/dual_system/openhelix.yaml          # dual system
python scripts/train.py --config configs/base.yaml --dataset droid --droid_data_root /data/droid_100
```

Objective: `L = CE(text) + action_loss_weight * flow_matching(actions) + visual_loss_weight * DiT(frames)`.
In dual-system mode the System 1 policy loss replaces the action and video terms. Validation
uses `--val_fraction` (action L1 / RMSE / chunk accuracy), and Ray / WebDataset come from
`train.ray` / `data.webdataset`. The resolved YAML is saved to `<ckpt_dir>/run_config.yaml`.

EO-Data actions are 32-dimensional; the training CLI and `configs/base.yaml` default to
`action_dim: 32` (the dataclass default of 7 is for 6-DoF + gripper robots).

## Inference and visualisation

```bash
python scripts/inference.py --help     # interactive generation and dataset evaluation
python scripts/visualize.py --help     # MP4s with frames, Q/A text and action plots
```

## Debugging checklist

1. Loss is NaN → check that `labels` isn't all `-100` for the batch and that
   `action_mask.sum() > 0`; lower `--lr`.
2. Shape mismatch in the action loss → `--action_dim` differs from the dataset.
3. `IndexError` in image encoding → a sample has more `<image>` tokens than images in
   `images`; check `max_images` in `EODatasetConfig`.
4. OOM → reduce `--max_seq_len` first: every image adds 196 tokens before the text.
