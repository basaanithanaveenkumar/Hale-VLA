"""Image encoders: a patch encoder (stand-in for SigLIP / Gemma3's vision tower) and π0.7's MEM-style history encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from pi_family.layers import _init_linear


class PatchImageEncoder(nn.Module):
    """(B, 3, S, S) image → (B, P, width) patch tokens.

    A small conv stem sees colours and edges, a stride-``patch`` conv turns each patch into one token, and a learned
    position embedding tells the backbone *where* the patch is (the toy tasks are all about locations). Two
    coordinate channels (CoordConv) are appended to the image so a token also knows *where inside its patch* an
    object sits - with only a few patches per image this sub-patch precision is what makes reaching accurate.
    """

    def __init__(self, image_size: int, patch_size: int, width: int, stem_channels: int = 16):
        super().__init__()
        # stem_channels = 0: plain ViT patch embedding (one linear map per patch), ~10x cheaper on CPU
        self.stem = (
            nn.Sequential(nn.Conv2d(5, stem_channels, 3, padding=1), nn.GELU())
            if stem_channels
            else nn.Identity()
        )
        self.patchify = nn.Conv2d(stem_channels or 5, width, patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        axis = torch.linspace(-1.0, 1.0, image_size)
        self.register_buffer(
            "coords", torch.stack(torch.meshgrid(axis, axis, indexing="xy"))[None], persistent=False
        )
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        coords = self.coords.expand(images.shape[0], -1, -1, -1)
        x = self.patchify(self.stem(torch.cat([images, coords], 1)))  # (B, width, S/p, S/p)
        return x.flatten(2).transpose(1, 2) + self.position


class HistoryVideoEncoder(nn.Module):
    """MEM-style video history encoder (Torne et al., 2026; used by π0.7).

    Every frame goes through the same :class:`PatchImageEncoder`; then, *for each patch location*, a small
    transformer mixes that location across time and the output at the current frame is kept. So any number of
    history frames is compressed into exactly the token count of a single frame (π0.7: "compressed to the same
    number of tokens as a single frame"). Missing frames (episode start, history dropout) are masked out.

    Frame index 0 is the *current* frame, index k is k·stride steps ago.
    """

    def __init__(
        self, patch_encoder: PatchImageEncoder, width: int, max_frames: int, num_heads: int = 4
    ):
        super().__init__()
        self.patch_encoder = patch_encoder
        self.frame_age = nn.Parameter(
            torch.zeros(1, max_frames, 1, width)
        )  # "how long ago" embedding
        nn.init.trunc_normal_(self.frame_age, std=0.02)
        self.temporal = nn.TransformerEncoderLayer(
            width,
            num_heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.temporal.apply(_init_linear)

    def forward(
        self, frames: torch.Tensor, frame_valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``frames`` (B, T, 3, S, S), ``frame_valid`` (B, T) bool → (B, P, width)."""
        b, t = frames.shape[:2]
        tokens = self.patch_encoder(frames.flatten(0, 1)).view(b, t, -1, self.frame_age.shape[-1])
        tokens = tokens + self.frame_age[:, :t]
        p = tokens.shape[2]
        per_location = tokens.permute(0, 2, 1, 3).reshape(b * p, t, -1)  # (B·P, T, width)
        pad = None
        if frame_valid is not None:
            valid = frame_valid.clone()
            valid[:, 0] = True  # the current frame always exists
            pad = (~valid)[:, None].expand(b, p, t).reshape(b * p, t)
        mixed = self.temporal(per_location, src_key_padding_mask=pad)
        return mixed[:, 0].view(b, p, -1)
