# Architecture

Diagrams are Mermaid and render on GitHub. The same diagrams appear on the
[project page](../project-page/index.html) and are described in Section 3 of the
[paper](../paper/main.tex).

## 1. System overview

```mermaid
flowchart TB
  subgraph Inputs
    IMG["Images<br/>[B, N, 3, 224, 224]"]
    TXT["Chat tokens<br/>with &lt;image&gt; &lt;state&gt; &lt;action&gt;"]
    ST["Robot state<br/>[B, N_state, 32]"]
  end

  IMG --> VIT["VisTransformer<br/>16×16 patches · 4 blocks · MoE FFN"]
  VIT --> PROJ["ImageProjector<br/>512→128→256→512"]
  TXT --> EMB["Token embedding<br/>151,668 × 512"]
  ST --> SENC["StateEncoder<br/>32→256→512→512"]

  EMB --> ZERO["zero &lt;image&gt; / &lt;state&gt; placeholders"]
  SENC --> INJ["write state embeddings<br/>at &lt;state&gt; positions"]
  ZERO --> INJ
  PROJ --> CAT["concat: [patches ; text]"]
  INJ --> CAT
  CAT --> POS["+ learned positions (≤ 2000)"]
  POS --> DEC["DecoderTransformer<br/>12 causal blocks · MHA + DeepSeekMoE"]
  DEC --> LN["LayerNorm"]
  LN --> LM["LMHead → logits<br/>[B, N·196 + L, V]"]
  LN --> GATHER["gather hidden states at<br/>&lt;action&gt; positions (+ N·196 offset)"]
  GATHER --> ACT["ActionDecoder<br/>512→512→256→chunk·action_dim"]
  ACT --> OUT["actions<br/>[B, n_act, chunk, action_dim]"]
```

## 2. How the three special tokens are handled

```mermaid
flowchart TB
  IDS["input_ids:  &lt;image&gt; Pick up &lt;state&gt; the cup &lt;action&gt;"]
  IDS -->|"&lt;image&gt; · prepend"| P["replaced by 196 ViT patch embeddings<br/>placed before all text"]
  IDS -->|"&lt;state&gt; · in place"| S["embedding overwritten by<br/>StateEncoder(state) at the same position"]
  IDS -->|"&lt;action&gt; · read out"| A["hidden state after the decoder<br/>→ ActionDecoder → action chunk"]
```

Resulting decoder sequence for one image:

```mermaid
flowchart LR
  p1["patch 1"] --> p2["…"] --> p196["patch 196"] --> w1["Pick"] --> w2["up"] --> s["state emb"] --> w3["the"] --> w4["cup"] --> a["&lt;action&gt; → ActionDecoder"]
```

## 3. Decoder block with DeepSeek-style MoE

```mermaid
flowchart TB
  X["x  [B, T, 512]"] --> N1["LayerNorm"]
  N1 --> MHA["Causal multi-head attention<br/>16 heads × 32"]
  MHA --> R1["+ residual"]
  X --> R1
  R1 --> N2["LayerNorm"]
  N2 --> ROUTER["NoiseBestKRouter<br/>logits + softplus-scaled noise (train)<br/>top-2 of 8 → softmax"]
  N2 --> SH1["Shared expert 1<br/>SwiGLU 512→614→512"]
  N2 --> SH2["Shared expert 2"]
  ROUTER -- "p₁" --> E1["Routed expert i"]
  ROUTER -- "p₂" --> E2["Routed expert j"]
  N2 --> E1
  N2 --> E2
  SH1 --> SUM["Σ shared + p-weighted routed"]
  SH2 --> SUM
  E1 --> SUM
  E2 --> SUM
  SUM --> R2["+ residual"]
  R1 --> R2
  R2 --> Y["x'  [B, T, 512]"]
```

The vision encoder uses the same block without the causal mask and with 16 routed
experts (top-4).

## 4. Training step

```mermaid
sequenceDiagram
  participant DL as EO dataloader
  participant M as HaloVLM
  participant L as Losses
  participant O as AdamW + cosine LR
  DL->>M: images, input_ids, attention_mask, states
  M-->>L: logits [B, N·196+L, V], action_preds
  DL->>L: labels (-100 outside assistant), actions, action_mask
  L->>L: CE on logits[:, N·196 : -1] vs labels[:, 1:]
  L->>L: masked MSE on flattened action chunks
  L->>O: total = CE + λ·MSE
  O->>M: clip grad (1.0), step, scheduler.step()
```

## 5. Module map

```mermaid
classDiagram
  class HaloVLM {
    +HaloVLMConfig config
    +forward(images, input_ids, attention_mask, states)
  }
  class VisTransformer
  class PatchEmb
  class ImageProjector
  class DecoderTransformer
  class TransformerBlock
  class MultiHeadAttn
  class HeadAttn
  class DeepseekMoE
  class NoiseBestKRouter
  class Expert
  class StateEncoder
  class ActionDecoder
  class LMHead
  HaloVLM *-- VisTransformer
  HaloVLM *-- ImageProjector
  HaloVLM *-- DecoderTransformer
  HaloVLM *-- StateEncoder
  HaloVLM *-- ActionDecoder
  HaloVLM *-- LMHead
  VisTransformer *-- PatchEmb
  VisTransformer *-- TransformerBlock
  DecoderTransformer *-- TransformerBlock
  TransformerBlock *-- MultiHeadAttn
  TransformerBlock *-- DeepseekMoE
  MultiHeadAttn *-- HeadAttn
  DeepseekMoE *-- NoiseBestKRouter
  DeepseekMoE *-- Expert
```

## Parameter budget (default config, measured)

| Module | Params |
|---|---|
| Vision encoder | 72.7M |
| Image projector | 0.23M |
| Token embedding | 77.7M |
| Position embedding | 1.0M |
| Decoder | 125.9M (≈58M active per token) |
| LM head | 77.7M |
| State encoder | 0.40M |
| Action decoder | 0.40M |
| **Total** | **355.9M** |
