"""A tiny ViT stem standing in for SigLIP: patchify, project, add a learned 2-D position."""

from __future__ import annotations

import torch
from torch import nn


class PatchEncoder(nn.Module):
    """``(B, 3, S, S)`` image → ``(B, (S/p)², width)`` patch tokens.

    π models use SigLIP (400M) as the image encoder; its outputs are projected into the LLM
    width and placed before the text, like PaliGemma. A Conv2d with kernel = stride = patch size
    plus learned positions is the same computation at toy scale, followed by one small MLP.

    Args:
        image_size: input resolution.
        patch: patch size.
        width: output token width (the VLM width).
    """

    def __init__(self, image_size: int = 32, patch: int = 8, width: int = 64) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, (image_size // patch) ** 2, width) * 0.02)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(width, width))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images).flatten(2).transpose(1, 2) + self.pos
        return x + self.mlp(x)
