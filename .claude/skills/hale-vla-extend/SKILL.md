---
name: hale-vla-extend
description: Add a new modality token, encoder, action head or config option to Hale-VLA end to end (config, special tokens, model forward, dataloader, tests). Use when extending the architecture, e.g. adding depth images, language-conditioned chunking, or a diffusion/flow action head.
---

# Extending Hale-VLA

Every new input or output travels the same path. Touch all of these, in order:

1. **Special token** — add the string and id to `config/special_tokens.json`
   (next free id is 151668) and bump `vocab_size_with_custom_tokens`.
   Add a getter in `config/tokens.py` (`get_all_custom_tokens` must list it so the
   tokenizer learns it).
2. **Config** — add fields to `HaloVLMConfig` (`config/model_config.py`), including the new
   `*_token_id` and `vocab_size`. Keep defaults backwards compatible.
3. **Module** — new file in `src/Halo_VLA/models/`, taking `config: HaloVLMConfig | None`
   and `**kwargs` like `StateEncoder`/`ActionDecoder`. Export it in `models/__init__.py`.
4. **Forward** — in `HaloVLM.forward` choose one of the existing patterns:
   - *prepend* (like images): encode, concatenate before `text_embeds`, and add to
     `num_prepended` so action readout and the LM loss stay aligned;
   - *in-place* (like `<state>`): zero the placeholder embedding and write the encoding
     at the token position;
   - *read-out* (like `<action>`): gather `transformer_out` at token positions, offset by
     `num_prepended`, and decode.
5. **Data** — emit the placeholder in `EODataset` text and return the tensor from
   `__getitem__`; pad it in `eo_collate_fn`.
6. **Loss** — add a term in `scripts/train.py` with its own `--*_loss_weight` flag.
7. **Tests** — shape test in `tests/test_models.py` with a tiny config.
8. **Docs** — update `docs/architecture.md` (Mermaid) and the config table in
   `docs/configuration.md`.

## Swapping the action head

`ActionDecoder` maps one hidden state to `[chunk_size, action_dim]`. A replacement must
keep the call signature `decoder(hidden[B*n_act, emb_dim]) -> [B*n_act, chunk, dim]`
and expose `.chunk_size` and `.action_dim`, because `HaloVLM.forward` reshapes with them.
A flow-matching head needs the ground-truth actions at train time; pass them through
`forward` as an optional argument rather than changing the return type.

## Turning MoE off

`HaloVLMConfig(use_moe=False)` switches the decoder FFN to a GELU MLP of width
`dec_mlp_dim`. The ViT ignores this flag (see `hale-vla-dev`); pass `use_moe` into
`VisTransformer` if you need a dense vision tower.
