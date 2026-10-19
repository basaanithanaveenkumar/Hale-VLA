"""Looking inside a trained :class:`FusionPolicy`: where does the instruction's colour word look?"""

from __future__ import annotations

import numpy as np
import torch
from open_vlas.evaluation import EmbodiedEvaluator
from pi_family.toy import PointWorld

from vla_foundations.fusion import FusionPolicy


@torch.no_grad()
def word_to_image_attention(model: FusionPolicy, batch, word_index: int = -1):
    """Attention from one instruction token to the image tokens, for every layer of an **early-fusion** joint transformer.

    Returns ``(maps, n_img)`` where ``maps[l]`` is ``(B, n_img)`` (averaged over heads). ``word_index = -1`` is the last instruction
    token, i.e. the colour word of ``go to <colour>``.
    """
    device = model.action_position.device
    img = model._visual_tokens(batch.images.to(device))
    txt, tvalid = model._text_tokens(batch.task, device)
    x = (
        torch.cat([img, txt], 1)
        + model.pos(torch.arange(img.shape[1] + txt.shape[1], device=device))[None]
    )
    valid = torch.cat([torch.ones(img.shape[:2], dtype=torch.bool, device=device), tvalid], 1)
    n_img, maps = img.shape[1], []
    mask = valid[:, None, None, :]
    for layer in model.llm:
        q, k, v, gate = layer.project(x)
        scores = (q @ k.transpose(-1, -2) / layer.head_dim**0.5).masked_fill(~mask, float("-inf"))
        w = scores.softmax(-1)  # (B, heads, L, L)
        maps.append(w.mean(1)[:, n_img + word_index, :n_img])
        x = layer.finish(x, w @ v, gate)
    return maps, n_img


def attention_breakdown(model: FusionPolicy, world: PointWorld, scenarios: list) -> dict:
    """Where the colour word's attention goes, per layer: the *target* object's patch, the *other objects'* patches (mean per patch),
    the *robot's* patch and the *background* patches (mean per patch). Chance level is ``1 / n_img`` for every category.

    Returns ``{"target": (L,), "other": (L,), "robot": (L,), "background": (L,), "chance": float}``.
    """
    ev = EmbodiedEvaluator(world, 16, 8, 1)
    robots = np.stack([s.scene.robot for s in scenarios])[:, None, :].astype(np.float32)
    batch = ev.observe(scenarios, robots, np.full(len(scenarios), -1.0), 0)
    maps, n_img = word_to_image_attention(model, batch)
    g = int(np.sqrt(n_img))
    cell = world.image_size / g
    out = {k: [] for k in ("target", "other", "robot", "background")}
    for layer_map in maps:
        acc = {k: [] for k in out}
        for i, s in enumerate(scenarios):

            def patch(xy):
                r, c = world._pixel(xy)
                return int(r // cell) * g + int(c // cell)

            t, rb = patch(s.scene.objects[s.target]), patch(s.scene.robot)
            others = {patch(xy) for name, xy in s.scene.objects.items() if name != s.target} - {
                t,
                rb,
            }
            used = {t, rb} | others
            w = layer_map[i].numpy()
            acc["target"].append(w[t])
            acc["robot"].append(w[rb])
            if others:
                acc["other"].append(np.mean([w[j] for j in others]))
            rest = [j for j in range(n_img) if j not in used]
            acc["background"].append(np.mean(w[rest]))
        for k in out:
            out[k].append(float(np.mean(acc[k])))
    return {**{k: np.array(v) for k, v in out.items()}, "chance": 1.0 / n_img}
