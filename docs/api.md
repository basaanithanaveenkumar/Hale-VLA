# API reference

Shapes use `B` batch, `L` text length, `N` images per sample, `P = (img_size/patch_size)^2`
patches per image, `d = emb_dim`, `V = vocab_size`.

## `HaloVLM(config=None, **kwargs)` — `models/halo_vla.py`

`forward(images, input_ids, attention_mask, states, image_mask=None, bev_images=None, bev_cameras=None)
-> (logits, action_hiddens, visual_context_emb, world_video_query_hiddens)`

| Argument | Shape |
|---|---|
| `images` | `[B, N, 3, H, W]`, N ≥ max number of `<image>` tokens in the batch |
| `input_ids` | `[B, L]` with `<image>`, `<state>`, `<halo_action>`, `<halo_world_video>` placeholders |
| `attention_mask` | `[B, L]`, 1 = real token |
| `states` | `[B, N_state, state_dim]`, one per `<state>` token |
| `image_mask` | `[B, N]` optional, 1 = real image slot |
| `bev_images`, `bev_cameras` | optional `[B, N_cam, 3, H, W]` + `lifting.Cameras` (needs `use_bev`) |

| Output | Shape |
|---|---|
| `logits` | `[B, T_bev + N·P + L, V]`: BEV tokens, then image positions, then text |
| `action_hiddens` | `[B, n_action_tokens, d]` decoder states at `<halo_action>`, or `None` |
| `visual_context_emb` | `[B, d]` past-frame context for the DiT world model, or `None` |
| `world_video_query_hiddens` | `[B, n_wv, d]` states at `<halo_world_video>`, or `None` |

* `sample_actions(action_hiddens, num_steps=None) -> [B, n_action_tokens, chunk_size, action_dim]`
  integrates the flow-matching ODE.
* `compute_flow_loss(action_hiddens, actions, action_mask)` is the training loss for that head.
* `compute_visual_prediction_loss(...) -> (loss, parts)` is the DiT world-model loss.
* `encode_sequence(images, input_ids, attention_mask, states, *, extra_embeds=None,
  capture_layer=None, bev_images=None, bev_cameras=None) -> dict` returns the hidden states that
  the dual-system System 2 reads.

Submodules: `vis_enc`, `image_projector`, `state_encoder`, `bev_encoder` (optional),
`decoder_transformer` + `token_emb` + `pos_embed` + `lm_head` (built-in decoder) or `language`
(a pretrained LLM, `config.llm_backbone`), `flow_decoder`, `visual_predictor` (optional).

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

## `FlowActionDecoder(action_dim_flat, obs_dim, hidden_dim, time_embed_dim)` — `models/flow_action_decoder.py`

Velocity field `v(x_t, t, cond)` over a flattened action chunk; used by `HaloVLM.compute_flow_loss`
and `HaloVLM.sample_actions`. (`models/action_decoder.py`, the earlier MLP head, is unused by `HaloVLM`.)

## `BEVSceneEncoder(emb_dim, lifter, grid, image_size, num_cameras, channels, token_pool)` — `models/bev_encoder.py`

`[B, N_cam, 3, H, W]` + `lifting.Cameras` -> `[B, num_tokens, d]` BEV scene tokens.

## `LMHead(hidden_size, vocab_size)` — `models/lm_head.py`

`[B, T, d] -> [B, T, V]`.

## Config helpers — `config/tokens.py`

`get_token(name)`, `get_token_id(name)`, `get_all_custom_tokens()`, `get_vocab_size()`,
`SPECIAL_TOKENS`.
