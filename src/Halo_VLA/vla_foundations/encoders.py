"""Vision encoders that differ in *how they were pre-trained*: SigLIP-style language supervision vs DINO-style self-supervision.

All of them share one small ViT (:class:`ToyViT`); only the pre-training objective differs, which is exactly the axis the OpenVLA paper
(and its Prismatic ablations) compares:

* **SigLIP** (Zhai et al., 2023) trains an image tower and a text tower with a **pairwise sigmoid loss** on (image, caption) pairs:
  every pair in the batch is an independent binary "do these match?" problem, with no softmax normalisation over the batch. Captions say
  *what* is in the picture (and coarsely how things relate), so the features are semantic and language-aligned, but nothing in the
  objective asks for precise locations.
* **DINOv2** (Oquab et al., 2023) uses no text: a *student* is trained to match an EMA *teacher*'s sharpened, centred outputs on
  augmented views, at the image level (DINO) **and at masked patches** (iBOT). Patch-level targets force every patch token to carry its own
  local content, which is why the features are strong on dense / spatial tasks.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from open_vlas.attention import RMSNorm, init_linear
from pi_family.layers import ExpertLayer
from pi_family.tokenizers import PiVocabulary
from pi_family.toy.world import PointWorld

CAPTION_WORDS = ["of", "and", "a", "square", "scene", "above", "below", "next"]


def caption_vocab() -> PiVocabulary:
    """The shared vocabulary of the encoder experiments (policy words + a few caption words)."""
    from pi_family.tokenizers import DEFAULT_WORDS

    return PiVocabulary(words=list(DEFAULT_WORDS) + CAPTION_WORDS)


class ToyViT(nn.Module):
    """Patch embedding + learned positions + transformer layers: ``images (B, 3, S, S) → tokens (B, N, width)``.

    Args:
        image_size / patch_size: 32 and 4 give 8×8 = 64 tokens.
        width / depth / heads: transformer size.
        mask_token: adds a learnable token that replaces masked patches (needed by iBOT-style training).
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        width: int = 64,
        depth: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        self.grid = image_size // patch_size
        self.width = width
        self.patchify = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, self.grid**2, width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.layers = nn.ModuleList(
            ExpertLayer(width, heads, width // heads, 2.0) for _ in range(depth)
        )
        self.norm = RMSNorm(width)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``mask`` (B, N) bool replaces those patches by the mask token before the transformer."""
        x = self.patchify(images).flatten(2).transpose(1, 2)
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.expand_as(x), x)
        x = x + self.position
        for layer in self.layers:
            q, k, v, gate = layer.project(x)
            x = layer.finish(x, F.scaled_dot_product_attention(q, k, v), gate)
        return self.norm(x)

    def pooled(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images).mean(1)


class TextTower(nn.Module):
    """Word embedding + a small transformer + mean pooling: ``captions → (B, width)``."""

    def __init__(
        self,
        vocab: PiVocabulary,
        width: int = 64,
        depth: int = 2,
        heads: int = 4,
        max_len: int = 16,
    ):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab.size, width)
        self.pos = nn.Embedding(max_len, width)
        self.layers = nn.ModuleList(
            ExpertLayer(width, heads, width // heads, 2.0) for _ in range(depth)
        )
        self.norm = RMSNorm(width)

    def tokens(self, texts: list) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocab.encode(t) for t in texts]
        n = max(len(i) for i in ids)
        dev = self.emb.weight.device
        tok = torch.zeros(len(ids), n, dtype=torch.long, device=dev)
        valid = torch.zeros(len(ids), n, dtype=torch.bool, device=dev)
        for k, i in enumerate(ids):
            tok[k, : len(i)] = torch.tensor(i, device=dev)
            valid[k, : len(i)] = True
        x = self.emb(tok) + self.pos(torch.arange(n, device=dev))[None]
        mask = valid[:, None, None, :]
        for layer in self.layers:
            q, k, v, gate = layer.project(x)
            x = layer.finish(x, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), gate)
        return self.norm(x), valid

    def forward(self, texts: list) -> torch.Tensor:
        x, valid = self.tokens(texts)
        return (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)


# ================================================================================================ SigLIP
def scene_caption(scene, rng: np.random.Generator) -> str:
    """A web-style caption: which colours are present, and sometimes a *coarse* left/right/above/below relation.

    Like real alt-text it never gives coordinates or distances: that is the point of the experiment.
    """
    items = list(scene.objects.items())
    if len(items) >= 2 and rng.random() < 0.5:
        (a, pa), (b, pb) = (items[i] for i in rng.permutation(len(items))[:2])
        dx, dy = pa[0] - pb[0], pa[1] - pb[1]
        rel = (
            ("left" if dx < 0 else "right")
            if abs(dx) > abs(dy)
            else ("top" if dy < 0 else "bottom")
        )
        return f"{a} {rel} of {b}"
    names = [c for c, _ in items]
    rng.shuffle(names)
    return " and ".join(names) if len(names) > 1 else f"a {names[0]} square"


class SigLIPLite(nn.Module):
    """Two towers + the pairwise **sigmoid** loss: ``L = −(1/B) Σ_i Σ_j log σ( z_ij · (t·⟨x_i, y_j⟩ + b) )``, ``z_ij = +1`` iff ``i = j``."""

    def __init__(self, vocab: PiVocabulary, width: int = 64, depth: int = 3):
        super().__init__()
        self.vision = ToyViT(width=width, depth=depth)
        self.text = TextTower(vocab, width)
        self.log_t = nn.Parameter(
            torch.tensor(float(np.log(10.0)))
        )  # learnable temperature, init 10 (paper)
        self.bias = nn.Parameter(
            torch.tensor(-10.0)
        )  # learnable bias, init -10 (paper): most pairs are negatives

    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.vision.pooled(images), dim=-1)

    def embed_text(self, texts: list) -> torch.Tensor:
        return F.normalize(self.text(texts), dim=-1)

    def loss(self, images: torch.Tensor, texts: list) -> torch.Tensor:
        logits = self.log_t.exp() * self.embed_images(images) @ self.embed_text(texts).T + self.bias
        labels = 2 * torch.eye(len(texts)) - 1  # +1 on the diagonal, -1 elsewhere
        return -F.logsigmoid(labels * logits).sum() / len(texts)


# ================================================================================================ DINO / iBOT
class DinoLite(nn.Module):
    """Self-distillation with an EMA teacher, an image-level (DINO) and a masked-patch (iBOT) loss, with centring and sharpening.

    The projection head maps tokens to ``prototypes`` logits; the teacher's softmax (low temperature = *sharpening*, minus a running
    *centre*) is the target for the student's softmax on a differently augmented, partially masked view.
    """

    def __init__(
        self,
        width: int = 64,
        depth: int = 3,
        prototypes: int = 64,
        momentum: float = 0.99,
        t_student: float = 0.1,
        t_teacher: float = 0.04,
    ):
        super().__init__()
        self.student = ToyViT(width=width, depth=depth)
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, prototypes))
        self.teacher_head = copy.deepcopy(self.head).requires_grad_(False)
        self.register_buffer("center_global", torch.zeros(1, prototypes))
        self.register_buffer("center_patch", torch.zeros(1, prototypes))
        self.momentum, self.t_s, self.t_t = momentum, t_student, t_teacher
        self.head.apply(init_linear)

    @staticmethod
    def augment(images: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        """Photometric augmentation only (brightness / contrast / noise): geometric crops would destroy the position signal we study."""
        b = images.shape[0]
        gain = 0.7 + 0.6 * torch.rand(b, 1, 1, 1, generator=gen)
        bias = 0.1 * (torch.rand(b, 1, 1, 1, generator=gen) - 0.5)
        return (images * gain + bias + 0.03 * torch.randn(images.shape, generator=gen)).clamp(0, 1)

    @torch.no_grad()
    def ema_update(self) -> None:
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.mul_(self.momentum).add_(s.detach(), alpha=1 - self.momentum)
        for s, t in zip(self.head.parameters(), self.teacher_head.parameters()):
            t.mul_(self.momentum).add_(s.detach(), alpha=1 - self.momentum)

    def loss(self, images: torch.Tensor, mask_ratio: float = 0.4) -> tuple[torch.Tensor, dict]:
        v1, v2 = self.augment(images), self.augment(images)
        b, n = v1.shape[0], self.student.grid**2
        mask = torch.rand(b, n) < mask_ratio
        with torch.no_grad():
            tt = self.teacher(v2)
            tp = self.teacher_head(tt)  # (B, N, K)
            tg = self.teacher_head(tt.mean(1))  # (B, K)
            t_patch = F.softmax((tp - self.center_patch) / self.t_t, -1)
            t_glob = F.softmax((tg - self.center_global) / self.t_t, -1)
        st = self.student(v1, mask=mask)
        sp = self.head(st)
        sg = self.head(st.mean(1))
        l_glob = -(t_glob * F.log_softmax(sg / self.t_s, -1)).sum(-1).mean()
        l_patch = -(t_patch * F.log_softmax(sp / self.t_s, -1)).sum(-1)[mask].mean()
        with torch.no_grad():  # centre update (prevents one prototype from dominating)
            self.center_global.mul_(0.9).add_(tg.mean(0, keepdim=True), alpha=0.1)
            self.center_patch.mul_(0.9).add_(tp.mean((0, 1))[None], alpha=0.1)
        return l_glob + l_patch, dict(global_loss=l_glob.item(), patch_loss=l_patch.item())


# ================================================================================================ trainers
@dataclass
class PretrainLog:
    losses: list = field(default_factory=list)
    seconds: float = 0.0


def _scenes(world: PointWorld, rng: np.random.Generator, n: int):
    scenes = [world.sample_scene(rng, int(rng.integers(1, 4))) for _ in range(n)]
    images = torch.from_numpy(np.stack([world.render(s.robot, s.objects) for s in scenes]))
    return scenes, images


def train_siglip(
    model: SigLIPLite,
    world: PointWorld,
    steps: int = 2500,
    batch: int = 128,
    lr: float = 2e-3,
    seed: int = 0,
) -> PretrainLog:
    """Image-caption pre-training on rendered scenes."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    log, t0 = PretrainLog(), time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 50)
        scenes, images = _scenes(world, rng, batch)
        loss = model.loss(images, [scene_caption(s, rng) for s in scenes])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log.losses.append(loss.item())
    log.seconds = time.time() - t0
    return log


def train_dino(
    model: DinoLite,
    world: PointWorld,
    steps: int = 2500,
    batch: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> PretrainLog:
    """Self-supervised pre-training on rendered scenes (no captions, no labels)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    params = list(model.student.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    log, t0 = PretrainLog(), time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 50)
        _, images = _scenes(world, rng, batch)
        loss, _ = model.loss(images)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.ema_update()
        log.losses.append(loss.item())
    log.seconds = time.time() - t0
    return log
