# Configuration

All model hyper-parameters live in `HaloVLMConfig` (`config/model_config.py`). Pass a
config object or keyword overrides:

```python
HaloVLM(HaloVLMConfig(emb_dim=256, dec_num_layers=6))
HaloVLM(emb_dim=256, dec_num_layers=6)          # same thing
```

## Model (`HaloVLMConfig`)

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 151668 | Qwen2.5 vocabulary + 3 custom tokens |
| `image_token_id` / `action_token_id` / `state_token_id` | 151665 / 151666 / 151667 | must match `config/special_tokens.json` |
| `emb_dim` | 512 | shared width of ViT, decoder and heads |
| `img_size`, `patch_size`, `in_chans` | 224, 16, 3 | → 196 patches per image |
| `vit_num_layers`, `vit_num_heads` | 4, 16 | vision tower depth / heads |
| `vit_mlp_dim`, `vit_drop` | 512, 0.0 | only used if the ViT is built dense |
| `dec_num_layers`, `dec_num_heads` | 12, 16 | decoder depth / heads |
| `dec_mlp_dim`, `dec_drop` | 512, 0.0 | dense FFN width (when `use_moe=False`) / dropout |
| `use_moe` | True | DeepSeekMoE FFN in decoder blocks |
| `moe_hid_scale` | 1.2 | expert hidden width = `round(emb_dim * scale)` |
| `moe_num_routed_experts` | 8 | routed experts per layer |
| `moe_top_k` | 2 | routed experts used per token |
| `moe_num_shared_experts` | 2 | always-on experts per layer |
| `max_position_embeddings` | 2000 | learned absolute positions over patches + text |
| `proj_vision_dim`, `proj_llm_dim` | None → `emb_dim` | image projector in/out widths |
| `action_dim` | 7 | action dimensions (use 32 for EO-Data) |
| `action_hidden_dims` | (512, 256) | action MLP hidden widths |
| `action_chunk_size` | 1 | future steps predicted per `<action>` token |
| `action_dropout`, `action_use_layernorm` | 0.1, True | action MLP regularisation |
| `state_dim` | 32 | proprioceptive state size |
| `state_hidden_dims` | (256, 512) | state MLP hidden widths |
| `state_dropout`, `state_use_layernorm` | 0.1, True | state MLP regularisation |
| `system_prompt` | robotic assistant prompt | inserted at the start of each transcript |

## Training script (`scripts/train.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--subset` | `interleave-temporal` | EO-Data1.5M subset |
| `--batch_size` | 2 | |
| `--num_workers` | 2 | dataloader workers |
| `--max_seq_len` | 512 | text tokens (images add 196 each on top) |
| `--action_dim`, `--state_dim` | 32, 32 | override the config |
| `--action_chunk_size` | 1 | |
| `--max_samples` | 100 | cap dataset size for quick runs |
| `--epochs` | 5 | |
| `--lr`, `--weight_decay` | 1e-4, 0.01 | AdamW |
| `--grad_clip` | 1.0 | max grad norm |
| `--action_loss_weight` | 1.0 | λ in `L = CE + λ·MSE` |
| `--device` | cuda if available | |
| `--log_every`, `--save_every` | 10 steps, 1 epoch | |
| `--ckpt_dir` | `checkpoints` | |
| `--vis_every`, `--vis_max_videos`, `--vis_min_frames` | 100, 4, 3 | training-time video rendering |

## Special tokens (`config/special_tokens.json`)

```json
{
  "image_token": "<image>", "action_token": "<action>", "state_token": "<state>",
  "token_ids": {"image_token_id": 151665, "action_token_id": 151666, "state_token_id": 151667},
  "base_vocab_size": 151643,
  "vocab_size_with_custom_tokens": 151668,
  "tokenizer_name": "Qwen/Qwen2.5-VL-3B-Instruct"
}
```
