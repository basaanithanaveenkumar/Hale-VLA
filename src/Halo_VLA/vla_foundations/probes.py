"""Linear probes that measure what a frozen vision encoder knows.

* :func:`semantic_probe`: from the **pooled** feature, can a linear classifier tell whether a colour is present? (language-like content)
* :func:`localization_probe`: from the **patch tokens**, can a *per-patch* linear scorer find the patch containing the object's centre?
  (dense, spatial content)
* :func:`zero_shot_colors`: for language-aligned encoders only: classify colour presence by comparing the image embedding with the
  embedding of the word.

A probe is deliberately weak (one linear layer) so that the score reflects the *features*, not the probe.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from pi_family.toy.world import COLORS, PointWorld


def _data(world: PointWorld, n: int, seed: int, color: str = "red"):
    rng = np.random.default_rng(seed)
    scenes = [world.sample_scene(rng, int(rng.integers(1, 4))) for _ in range(n)]
    images = torch.from_numpy(np.stack([world.render(s.robot, s.objects) for s in scenes]))
    return scenes, images


@torch.no_grad()
def _tokens(encode, images: torch.Tensor, chunk: int = 500) -> torch.Tensor:
    return torch.cat([encode(images[i : i + chunk]) for i in range(0, len(images), chunk)])


@torch.enable_grad()
def semantic_probe(
    encode,
    world: PointWorld,
    color: str = "blue",
    n_train: int = 3000,
    n_test: int = 1000,
    seed: int = 0,
    test_transform=None,
) -> float:
    """Accuracy of a linear classifier on the pooled feature for "is ``color`` present?". ``encode(images) → (B, N, W)`` tokens.

    The probe is trained on clean images; ``test_transform(images) → images`` (e.g. an appearance shift) is applied to the *test* images only,
    so the score measures how well the features survive the shift."""
    scenes, images = _data(world, n_train + n_test, seed)
    y = torch.tensor([float(color in s.objects) for s in scenes])
    x = _tokens(encode, images).mean(1)
    mu, sd = x[:n_train].mean(0), x[:n_train].std(0) + 1e-6
    if test_transform is not None:
        x_test = _tokens(encode, test_transform(images[n_train:])).mean(1)
        x = torch.cat([x[:n_train], x_test])
    x = (x - mu) / sd
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(x[:n_train] @ w + b, y[:n_train])
        opt.zero_grad()
        loss.backward()
        opt.step()
    pred = (x[n_train:] @ w + b > 0).float()
    return float((pred == y[n_train:]).float().mean())


@torch.enable_grad()
def localization_probe(
    encode,
    world: PointWorld,
    color: str = "red",
    n_train: int = 3000,
    n_test: int = 1000,
    seed: int = 1,
) -> dict:
    """A per-patch linear scorer trained to put its highest score on the patch containing the ``color`` object's centre.

    Returns ``{"patch_accuracy", "error_px"}`` on held-out scenes (error: distance in pixels between the predicted patch centre and the
    object's true centre; chance for 64 patches is ≈ 1.5 % accuracy).
    """
    scenes, images = _data(world, n_train + n_test, seed)
    keep = [i for i, s in enumerate(scenes) if color in s.objects]
    scenes, images = [scenes[i] for i in keep], images[keep]
    tok = _tokens(encode, images)  # (M, N, W)
    n = tok.shape[1]
    g = int(np.sqrt(n))
    size = images.shape[-1]
    cell = size // g
    pix = np.array([np.asarray(s.objects[color]) * (size - 1) for s in scenes])  # (x, y) in pixels
    label = torch.tensor(
        [int(min(p[1] // cell, g - 1)) * g + int(min(p[0] // cell, g - 1)) for p in pix]
    )
    split = int(len(scenes) * n_train / (n_train + n_test))
    mu, sd = tok[:split].mean((0, 1)), tok[:split].std((0, 1)) + 1e-6
    tok = (tok - mu) / sd
    w = torch.zeros(tok.shape[2], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(400):
        loss = F.cross_entropy(tok[:split] @ w, label[:split])
        opt.zero_grad()
        loss.backward()
        opt.step()
    pred = (tok[split:] @ w).argmax(1)
    centers = np.stack([(pred.numpy() % g + 0.5) * cell, (pred.numpy() // g + 0.5) * cell], 1)
    err = np.linalg.norm(centers - pix[split:], axis=1).mean()
    return {"patch_accuracy": float((pred == label[split:]).float().mean()), "error_px": float(err)}


@torch.no_grad()
def zero_shot_colors(siglip, world: PointWorld, n: int = 1000, seed: int = 2) -> float:
    """Zero-shot colour-presence accuracy of a language-aligned encoder: the image embedding is compared with the text "a <colour> square"."""
    scenes, images = _data(world, n, seed)
    img = siglip.embed_images(images)
    names = list(COLORS)
    txt = siglip.embed_text([f"a {c} square" for c in names])
    scores = img @ txt.T
    correct = 0
    for i, s in enumerate(scenes):
        present = {names[j] for j in range(len(names)) if names[j] in s.objects}
        best = names[int(scores[i].argmax())]
        correct += best in present
    return correct / n
