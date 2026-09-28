---
name: hale-vla-train
description: Train, evaluate, run inference and render videos for Hale-VLA on EO-Data1.5M. Use when asked to train the model, debug losses, resume from a checkpoint, run generation, or visualise predicted vs ground-truth actions.
---

# Training and running Hale-VLA

## Data

`dataloader/eo_dataset.py` streams `IPEC-COMMUNITY/EO-Data1.5M` from the Hugging Face Hub.
17 subsets: 5 interleaved (`interleave-free_chat`, `-random_qa`, `-temporal`,
`-trajectory`, `-video_caption`) and 12 QA (`qa-affordance_qa`, `qa-task_planning`, ...).
Each sample becomes a Qwen chat transcript (`<|im_start|>system/user/assistant`) with
`<image>`, `<state>` and `<action>` placeholders. Only assistant tokens carry labels;
everything else is `-100`.

`eo_collate_fn` pads per batch and returns `images [B,N,3,H,W]`, `input_ids`,
`attention_mask`, `labels`, `actions [B,T,action_dim]`, `action_mask`, `states`,
`state_mask`.

Hub access is required on first use (`huggingface-cli login` if the dataset is gated).
Use `--max_samples` to keep experiments small.

## Train

```bash
python scripts/train.py \
  --subset interleave-temporal --batch_size 2 --max_seq_len 512 \
  --action_dim 32 --state_dim 32 --action_chunk_size 1 \
  --epochs 5 --lr 1e-4 --weight_decay 0.01 --grad_clip 1.0 \
  --action_loss_weight 1.0 --max_samples 100 --ckpt_dir checkpoints
```

Objective: `L = CE(text) + action_loss_weight * masked_MSE(actions)`.
- `compute_language_loss` skips the prepended image patches and shifts by one.
- `compute_action_loss` flattens `[B, n_act, chunk, dim]` to `[B, n_act*chunk, dim]`
  and masks padded steps.

Optimiser is AdamW with a per-step cosine schedule. Checkpoints go to `--ckpt_dir`
every `--save_every` epochs; `--vis_every` renders videos of predictions during training.

EO-Data actions are 32-dimensional, so pass `--action_dim 32` (the config default of 7
is for 6-DoF + gripper robots).

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
