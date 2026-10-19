"""A tiny vision-language model that stands in for SmolVLM-2 (SmolVLA) and Eagle-2 (GR00T N1 / N1.5).

Both published models read the scene with a pretrained VLM and condition an action network on its *intermediate* features:

* SmolVLA keeps only the first ``N = L/2`` layers of the VLM's language decoder;
* GR00T N1 takes the 12th-layer features of Eagle-2's 24-layer LLM, and throws the rest away.

:class:`ToyVLM` has the same anatomy: a vision encoder that turns an image into a few tokens (patch embedding followed by
a **pixel shuffle** that merges 2x2 neighbouring patches, the trick both models use to get 64 tokens per frame), word
embeddings, and a stack of transformer layers over ``[image tokens | prompt tokens | extra tokens]`` with a bidirectional
prefix. ``forward_layers`` returns the hidden state after *every* layer so a policy can pick any of them.

It is pre-trained once on visual question answering (:mod:`open_vlas.vlm_pretraining`) before any robot data, the toy
equivalent of web-scale VLM pre-training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from pi_family.layers import ExpertLayer
from pi_family.masks import make_attention_mask
from pi_family.tokenizers import PiVocabulary

from open_vlas.attention import RMSNorm, init_linear


@dataclass
class VLMConfig:
    """Size of the toy VLM. Published values in the comments."""

    image_size: int = 32
    patch_size: int = 4  # 32 / 4 = 8x8 = 64 patches
    shuffle: int = 2  # 2x2 pixel shuffle -> 16 tokens per frame (SmolVLM-2 / Eagle-2: 64 tokens from a 256-token grid)
    width: int = 64  # SmolVLM-2 500M: 960; Eagle-2 (SmolLM2 1.7B): 2048
    depth: int = 6  # SmolLM2-360M: 32 layers; Eagle-2: 24 layers
    num_heads: int = 4
    head_dim: int = 16
    mlp_ratio: float = 2.0
    max_len: int = 96

    @property
    def num_image_tokens(self) -> int:
        grid = self.image_size // self.patch_size
        return (grid // self.shuffle) ** 2 if self.shuffle > 1 else grid * grid


class PatchShuffleVisionEncoder(nn.Module):
    """Image → tokens: patch embedding (+ 2 coordinate channels) → learned positions → pixel-shuffle merge.

    The pixel shuffle (space-to-depth) concatenates each 2x2 block of patch tokens into one token of 4x the width and projects
    it back, cutting the sequence length 4x with no information thrown away. Tokens are what the language model pays
    quadratic attention for, so this is the cheapest efficiency lever in a VLA.
    """

    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg
        grid = cfg.image_size // cfg.patch_size
        self.grid = grid
        axis = torch.linspace(-1.0, 1.0, cfg.image_size)
        self.register_buffer(
            "coords", torch.stack(torch.meshgrid(axis, axis, indexing="xy"))[None], persistent=False
        )
        self.patchify = nn.Conv2d(5, cfg.width, cfg.patch_size, stride=cfg.patch_size)
        self.position = nn.Parameter(torch.zeros(1, grid * grid, cfg.width))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.kaiming_normal_(self.patchify.weight, nonlinearity="relu")
        nn.init.zeros_(self.patchify.bias)
        self.merge = (
            nn.Linear(cfg.width * cfg.shuffle**2, cfg.width) if cfg.shuffle > 1 else nn.Identity()
        )
        if cfg.shuffle > 1:
            init_linear(self.merge)

    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) → (B, grid², width), before the pixel shuffle."""
        coords = self.coords.expand(images.shape[0], -1, -1, -1)
        x = self.patchify(torch.cat([images, coords], 1))
        return x.flatten(2).transpose(1, 2) + self.position

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_tokens(images)
        s = self.cfg.shuffle
        if s > 1:
            b, _, w = x.shape
            g = self.grid
            x = x.view(b, g // s, s, g // s, s, w).permute(0, 1, 3, 2, 4, 5)
            x = self.merge(x.reshape(b, (g // s) ** 2, s * s * w))
        return x


@dataclass
class VLMInputs:
    """A padded token sequence for the VLM: embeddings, attention bits and where the pieces are."""

    embeds: torch.Tensor  # (B, L, W)
    ar: torch.Tensor  # (B, L) bool, True = starts a new (causal) block
    valid: torch.Tensor  # (B, L) bool, False = padding
    num_image_tokens: int
    prompt_slices: list = field(
        default_factory=list
    )  # per example (start, end) of the prompt tokens
    segments: list = field(
        default_factory=list
    )  # (name, start, end) of the image / text / extra parts

    @property
    def mask(self) -> torch.Tensor:
        return make_attention_mask(self.ar, self.valid)


class ToyVLM(nn.Module):
    """Vision encoder + word embeddings + prefix-LM transformer, with access to every layer's hidden state.

    Args:
        cfg: sizes.
        vocab: shared word vocabulary (:class:`pi_family.PiVocabulary`).
    """

    def __init__(self, cfg: VLMConfig, vocab: PiVocabulary | None = None):
        super().__init__()
        self.cfg, self.vocab = cfg, vocab or PiVocabulary()
        self.vision = PatchShuffleVisionEncoder(cfg)
        self.tok_emb = nn.Embedding(self.vocab.size, cfg.width)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.width)
        self.layers = nn.ModuleList(
            ExpertLayer(cfg.width, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        )
        self.final_norm = RMSNorm(cfg.width)
        self.lm_head = nn.Linear(cfg.width, self.vocab.size, bias=False)
        for p in (self.tok_emb.weight, self.pos_emb.weight):
            nn.init.trunc_normal_(p, std=0.02)
        init_linear(self.lm_head)
        self.num_layers = cfg.depth

    # ---------------------------------------------------------------------------------------- inputs
    def prompt_ids(self, text: str) -> list:
        return [self.vocab["<bos>"]] + self.vocab.encode(text) + [self.vocab["<sep>"]]

    def build_inputs(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        extra: torch.Tensor | None = None,
        answers: Sequence[Sequence[int]] | None = None,
        image_tokens: torch.Tensor | None = None,
    ) -> VLMInputs:
        """``[image tokens | prompt tokens | extra tokens (e.g. state) | answer tokens]``.

        The image, prompt and extra tokens form one **bidirectional** block (a prefix LM); answer tokens (VQA pre-training) are
        causal. ``image_tokens`` lets a caller reuse already-computed visual tokens.
        """
        device = self.tok_emb.weight.device
        b = images.shape[0]
        img = image_tokens if image_tokens is not None else self.vision(images.to(device))
        n_img = img.shape[1]
        n_extra = 0 if extra is None else extra.shape[1]
        prompts_ids = [self.prompt_ids(t) for t in prompts]
        answer_ids = [list(a) for a in answers] if answers is not None else [[] for _ in range(b)]
        length = n_img + max(len(p) + n_extra + len(a) for p, a in zip(prompts_ids, answer_ids))
        embeds = img.new_zeros(b, length, self.cfg.width)
        ar = torch.zeros(b, length, dtype=torch.bool, device=device)
        valid = torch.zeros(b, length, dtype=torch.bool, device=device)
        embeds[:, :n_img] = img
        valid[:, :n_img] = True
        ar[:, 0] = True  # the image tokens start the bidirectional prefix block
        prompt_slices = []
        for i, (ids, ans) in enumerate(zip(prompts_ids, answer_ids)):
            pos = n_img
            embeds[i, pos : pos + len(ids)] = self.tok_emb(torch.tensor(ids, device=device))
            valid[i, pos : pos + len(ids)] = True
            prompt_slices.append((pos, pos + len(ids)))
            pos += len(ids)
            if extra is not None:
                embeds[i, pos : pos + n_extra] = extra[i]
                valid[i, pos : pos + n_extra] = True
                pos += n_extra
            if ans:
                embeds[i, pos : pos + len(ans)] = self.tok_emb(torch.tensor(ans, device=device))
                valid[i, pos : pos + len(ans)] = True
                ar[i, pos : pos + len(ans)] = True  # answer tokens are causal
        embeds = embeds + self.pos_emb(torch.arange(length, device=device))[None]
        segments = [("image", 0, n_img), ("text", n_img, length)]
        return VLMInputs(embeds, ar, valid, n_img, prompt_slices, segments)

    # ---------------------------------------------------------------------------------------- forward
    def forward_layers(self, inputs: VLMInputs, upto: int | None = None) -> list:
        """Hidden state after each of the first ``upto`` layers (all layers by default)."""
        mask = inputs.mask
        x = inputs.embeds
        out = []
        for layer in self.layers[: upto or len(self.layers)]:
            q, k, v, gate = layer.project(x)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
            x = layer.finish(x, attended, gate)
            out.append(x)
        return out

    def logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        """Language-model logits from the *last* layer's hidden state (only valid on an untruncated VLM)."""
        if len(self.layers) != self.num_layers:
            raise RuntimeError("this VLM was truncated: it has no language-model head any more")
        return self.lm_head(self.final_norm(hidden_last))

    def truncate(self, n_layers: int) -> ToyVLM:
        """Drop every layer after the first ``n_layers`` (SmolVLA's layer skipping, GR00T N1's "use layer 12 only")."""
        if not 0 < n_layers <= len(self.layers):
            raise ValueError(f"n_layers must be in [1, {len(self.layers)}]")
        self.layers = nn.ModuleList(list(self.layers)[:n_layers])
        return self

    def freeze(self, *, vision: bool = True, language: bool = True) -> ToyVLM:
        """Freeze the vision encoder and / or the language part (embeddings + layers + head)."""
        for p in self.vision.parameters():
            p.requires_grad_(not vision)
        for module in (self.tok_emb, self.pos_emb, self.layers, self.final_norm, self.lm_head):
            for p in module.parameters():
                p.requires_grad_(not language)
        return self
