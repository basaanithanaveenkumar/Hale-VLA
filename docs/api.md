# API reference

Shapes use `B` batch, `L` text length, `N` images per sample, `P = (img_size/patch_size)^2`
patches per image, `d = emb_dim`, `V = vocab_size`.

## `HaloVLM(config=None, **kwargs)` — `models/halo_vla.py`

`forward(images, input_ids, attention_mask, states) -> (logits, action_preds)`

| Argument | Shape |
|---|---|
| `images` | `[B, N, 3, H, W]`, N ≥ max number of `<image>` tokens in the batch |
| `input_ids` | `[B, L]` with `<image>`, `<state>`, `<action>` placeholders |
| `attention_mask` | `[B, L]` (currently not applied inside attention; right-pad inputs) |
| `states` | `[B, N_state, state_dim]`, one per `<state>` token |

| Output | Shape |
|---|---|
| `logits` | `[B, N·P + L, V]` — image positions first |
| `action_preds` | `[B, n_action_tokens, chunk_size, action_dim]` or `None` |

Submodules: `vis_enc`, `image_projector`, `token_emb`, `pos_embed`,
`decoder_transformer`, `layer_norm`, `lm_head`, `state_encoder`, `action_decoder`.

## `VisTransformer(img_size, p_size, in_chans, emb_dim, num_layers, num_heads, mlp_dim, drop_fact)` — `models/vit.py`

`[B, 3, H, W] -> [B, P, d]`. Conv patch embedding + learned positions + transformer blocks
(MoE FFN by default) + LayerNorm.

## `DecoderTransformer(num_layers, emb_dim, num_heads, mlp_dim, drop_fact, use_moe, ...)` — `models/transformer.py`

`[B, T, d] -> [B, T, d]`. Stack of causal pre-norm `TransformerBlock`s and a final LayerNorm.

- `TransformerBlock(emb_dim, num_heads, mlp_dim, drop_fact, causal_mask, use_moe, moe_hid_scale, moe_num_routed_experts, moe_top_k, moe_num_shared_experts)`
- `MultiHeadAttn(emb_dim, num_heads, drop_fact, causal_mask)` — concatenation of `HeadAttn`s plus an output projection.

## `DeepseekMoE(emb_dim, hid_dim, num_router_exprts, best_k, num_shared_exprts)` — `models/moe.py`

`[B, T, d] -> [B, T, d]`. Sum of shared experts plus gate-weighted top-k routed experts.

- `NoiseBestKRouter(emb_dim, num_exprts, best_k)` → `(probs [B,T,E], idxs [B,T,k])`; adds
  softplus-scaled Gaussian noise only in training mode.
- `Expert(emb_dim, hid_dim, dropout)` — SwiGLU MLP.

## `ImageProjector(vision_dim, llm_dim)` — `models/image_proj.py`

`[..., vision_dim] -> [..., llm_dim]` via `llm_dim/4 → llm_dim/2 → llm_dim` with LayerNorm + GELU.

## `StateEncoder(config)` — `models/state_encoder.py`

`[B, state_dim] -> [B, d]`.

## `ActionDecoder(config)` — `models/action_decoder.py`

`[B, d] -> [B, action_chunk_size, action_dim]`. Exposes `.chunk_size` and `.action_dim`.

## `LMHead(hidden_size, vocab_size)` — `models/lm_head.py`

`[B, T, d] -> [B, T, V]`.

## Config helpers — `config/tokens.py`

`get_token(name)`, `get_token_id(name)`, `get_all_custom_tokens()`, `get_vocab_size()`,
`SPECIAL_TOKENS`.
