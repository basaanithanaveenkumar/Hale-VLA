"""Schematic diagrams for the action-representation and VLM-backbone notebooks (matplotlib only).

Visual grammar (shared with the other notebooks): **blue** = vision / VLM weights, **red** = action network, **grey** = data or fixed
steps, **orange** = text, **green** = state / proprioception, **purple** = learned auxiliary parts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

C = dict(
    vis="#2b6cb0",
    vis_fc="#e6f0fa",
    act="#c53030",
    act_fc="#fdeaea",
    data="#555555",
    data_fc="#f2f2f2",
    text="#f58518",
    text_fc="#fff1e0",
    state="#54a24b",
    aux="#8e44ad",
    aux_fc="#f1e6f8",
    ok="#2f855a",
    note="#555555",
)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False, lw=1.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        zorder=4,
        fontweight="bold" if bold else "normal",
        linespacing=1.3,
    )


def _arrow(ax, p0, p1, color="#333333", lw=1.5, ls="-", rad=0.0, ms=12):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            mutation_scale=ms,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
        )
    )


def _tokens(ax, x, y, n, color, w=0.26, h=0.3, label=None, fs=8, texts=None):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * w * 1.12, y), w, h, fc=color, ec="white", lw=0.6, zorder=4))
        if texts:
            ax.text(
                x + i * w * 1.12 + w / 2,
                y + h / 2,
                texts[i],
                ha="center",
                va="center",
                fontsize=5.5,
                color="white",
                zorder=5,
            )
    if label:
        ax.text(x + n * w * 1.12 / 2, y + h + 0.1, label, ha="center", va="bottom", fontsize=fs)
    return x + n * w * 1.12


# ================================================================================================ action representations
def fig_action_heads_overview():
    """Four ways to turn the VLM's features into actions."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), facecolor="white")
    titles = [
        "discrete tokens\n(RT-2, OpenVLA)",
        "flow matching\n(π0, SmolVLA, GR00T)",
        "diffusion\n(Diffusion Policy)",
        "chunked prediction\n(ACT, π0.5, SmolVLA)",
    ]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    # 1: bins
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    xs = np.linspace(0.5, 9.5, 400)
    ax.plot(xs, 3 + 1.4 * np.sin(xs * 0.9), color=C["act"], lw=2)
    for k in np.linspace(0.5, 9.5, 12):
        ax.axvline(k, color="#cccccc", lw=0.8)
    ax.step(
        np.linspace(0.5, 9.5, 12),
        3 + 1.4 * np.sin(np.linspace(0.5, 9.5, 12) * 0.9),
        where="mid",
        color=C["vis"],
        lw=2,
    )
    ax.text(
        5,
        0.5,
        "continuous → bin id → text token\n(256 bins per dimension)",
        ha="center",
        fontsize=8.5,
    )
    # 2: flow
    ax = axes[1]
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(40, 2)) * 0.7 + np.array([0, 0])
    data = np.where(
        rng.random(40)[:, None] < 0.5, np.array([-1.6, 0]), np.array([1.6, 0])
    ) + 0.15 * rng.normal(size=(40, 2))
    for a, b in zip(noise, data):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbbbbb", lw=0.7)
    ax.scatter(*noise.T, s=8, color="#555555")
    ax.scatter(*data.T, s=10, color=C["act"])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.text(
        0, -1.9, "straight-line paths noise → action\nlearn the velocity", ha="center", fontsize=8.5
    )
    # 3: diffusion
    ax = axes[2]
    for _ in range(14):
        t = np.linspace(0, 1, 40)
        end = np.array([-1.6, 0]) if rng.random() < 0.5 else np.array([1.6, 0])
        start = rng.normal(size=2) * 0.7
        path = (
            start[None] * (1 - t[:, None])
            + end[None] * t[:, None]
            + 0.18 * np.cumsum(rng.normal(size=(40, 2)), 0) * (1 - t[:, None]) / 3
        )
        ax.plot(path[:, 0], path[:, 1], color="#999999", lw=0.8)
        ax.scatter(*path[-1], s=10, color=C["act"])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.text(
        0, -1.9, "many noisy denoising steps\n(stochastic reverse chain)", ha="center", fontsize=8.5
    )
    # 4: chunk
    ax = axes[3]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    for i in range(10):
        ax.add_patch(Rectangle((i * 0.95 + 0.2, 3.3), 0.85, 0.9, fc=C["act"], ec="white"))
    ax.annotate(
        "", xy=(9.6, 2.9), xytext=(0.2, 2.9), arrowprops=dict(arrowstyle="<->", color=C["note"])
    )
    ax.text(5, 2.4, "one model call → H actions", ha="center", fontsize=8.5)
    ax.text(
        5,
        0.7,
        "execute k ≤ H, then re-plan\n(or ensemble overlapping chunks)",
        ha="center",
        fontsize=8.5,
    )
    plt.tight_layout()
    plt.show()


def fig_rt2_sequence(n_bins_show=(311, 423, 170, 55, 244, 128)):
    """RT-2's sequence: image tokens, instruction, then the action as ordinary tokens (one per dimension)."""
    fig, ax = _fig(12.5, 2.8)
    x = _tokens(ax, 0.2, 1.0, 8, C["vis"], label="image tokens")
    x = _tokens(ax, x + 0.2, 1.0, 4, C["text"], label='"pick the red cube"')
    x = _tokens(
        ax,
        x + 0.3,
        1.0,
        len(n_bins_show),
        C["act"],
        label="action tokens (one per dimension): the bin ids",
        texts=[str(b) for b in n_bins_show],
    )
    ax.text(
        0.2,
        0.35,
        "the VLM predicts them left to right, each conditioned on all the tokens before it, with its ordinary next-token cross-entropy",
        fontsize=9,
        color=C["note"],
    )
    ax.annotate(
        "",
        xy=(x - 0.1, 1.7),
        xytext=(x - 1.9, 1.7),
        arrowprops=dict(arrowstyle="-|>", color=C["act"], lw=1.4),
    )
    ax.text(x - 1.0, 1.95, "sequential decoding", ha="center", fontsize=8.5, color=C["act"])
    ax.set_xlim(0, x + 0.4)
    ax.set_ylim(0, 2.4)
    ax.set_title("RT-2: actions are just more tokens", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()


def fig_decode_timelines(tokens=16, nfe=4):
    """Sequential token decoding vs a few parallel denoising steps vs one-shot regression, as timelines of model calls."""
    fig, ax = _fig(12.5, 3.2)
    rows = [
        ("autoregressive tokens", tokens, C["act"], f"{tokens} sequential calls (H·D tokens)"),
        ("flow / diffusion", nfe, C["aux"], f"{nfe} calls (NFE), each producing the whole chunk"),
        ("regression", 1, C["ok"], "1 call"),
    ]
    for r, (name, n, col, label) in enumerate(rows):
        y = 2.2 - r * 0.9
        ax.text(-0.2, y + 0.25, name, ha="right", va="center", fontsize=9)
        for i in range(n):
            ax.add_patch(Rectangle((i * 0.62, y), 0.55, 0.5, fc=col, ec="white"))
        ax.text(n * 0.62 + 0.15, y + 0.25, label, va="center", fontsize=8.5, color=C["note"])
    ax.set_xlim(-3.2, 14)
    ax.set_ylim(0, 3.1)
    ax.set_title("how many sequential model evaluations per chunk?", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()


def fig_chunk_execution(H=8, k=4):  # noqa: N803
    """Receding horizon vs temporal ensembling timelines."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.8), facecolor="white", sharex=True)
    total = 3 * k + H
    for ax, mode in zip(
        axes,
        (
            "receding horizon: query every k steps, execute k of H",
            "temporal ensembling: query every step, average the overlapping chunks",
        ),
    ):
        ax.axis("off")
        ax.set_title(mode, fontsize=9.5, loc="left")
        starts = range(0, total - H + 1, k if mode.startswith("receding") else 1)
        cols = plt.cm.Blues(np.linspace(0.35, 0.9, max(len(list(starts)), 1)))
        for j, s in enumerate(starts):
            y = 2.2 - (j % 4) * 0.5
            if not mode.startswith("receding") and j >= 4:
                break
            for t in range(H):
                used = t < k if mode.startswith("receding") else True
                ax.add_patch(
                    Rectangle((s + t, y), 0.9, 0.38, fc=cols[j] if used else "#e4e4e4", ec="white")
                )
            ax.text(s - 0.2, y + 0.19, f"t={s}", ha="right", va="center", fontsize=7)
        ax.set_xlim(-2, total + 1)
        ax.set_ylim(0, 2.9)
    plt.tight_layout()
    plt.show()


# ================================================================================================ backbone
def fig_vla_backbone():
    """The VLA stack: vision encoder(s) → projector → LLM (with text) → action head."""
    fig, ax = _fig(13, 3.6)
    _box(
        ax,
        0.1,
        1.9,
        2.4,
        0.9,
        "vision encoder 1\n(SigLIP: language-aligned)",
        C["vis_fc"],
        C["vis"],
        fs=8.5,
    )
    _box(
        ax,
        0.1,
        0.5,
        2.4,
        0.9,
        "vision encoder 2\n(DINOv2: dense / spatial)",
        C["vis_fc"],
        C["vis"],
        fs=8.5,
    )
    _box(
        ax, 3.2, 1.2, 1.8, 0.9, "fuse + projector\n(MLP → LLM width)", C["aux_fc"], C["aux"], fs=8.5
    )
    _box(
        ax,
        5.7,
        0.6,
        3.1,
        2.1,
        "language backbone (LLM)\nimage tokens + words\nin one sequence\nworld knowledge · grounding\nzero-shot instructions",
        C["text_fc"],
        C["text"],
        fs=8.5,
    )
    _box(
        ax,
        9.5,
        1.2,
        2.0,
        0.9,
        "action head\ntokens · flow · diffusion",
        C["act_fc"],
        C["act"],
        fs=8.5,
    )
    _tokens(ax, 5.8, -0.35, 4, C["text"], label=None)
    ax.text(7.3, -0.6, '"put the strawberry in the bowl"', ha="center", fontsize=8.5)
    _arrow(ax, (7.3, -0.02), (7.3, 0.55))
    _arrow(ax, (2.55, 2.3), (3.15, 1.8))
    _arrow(ax, (2.55, 0.95), (3.15, 1.4))
    _arrow(ax, (5.05, 1.65), (5.65, 1.65))
    _arrow(ax, (8.85, 1.65), (9.45, 1.65))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.9, 3.2)
    ax.set_title("the stack every VLA builds on", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()


def fig_fusion_types():
    """Early, late and pooled fusion: where the two modalities meet."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor="white")
    specs = [
        (
            "early fusion\n(OpenVLA, RT-2, π0)",
            "one transformer sees image + words together;\nevery layer lets them interact",
        ),
        (
            "late fusion\n(towers meet at the action head)",
            "separate image and text towers;\nonly the expert's cross-attention joins them",
        ),
        (
            "pooled / global fusion\n(CLIP-style)",
            "each modality pooled to ONE vector;\nan MLP mixes the two vectors",
        ),
    ]
    for ax, (title, sub) in zip(axes, specs):
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(title, fontsize=10)
        ax.text(5, 0.2, sub, ha="center", fontsize=8.3, color=C["note"])
    ax = axes[0]
    _box(
        ax,
        2.2,
        1.4,
        5.6,
        3.4,
        "shared transformer\n(image + text tokens,\nbidirectional)",
        C["text_fc"],
        C["text"],
        fs=9,
    )
    _tokens(ax, 2.4, 0.9, 5, C["vis"], w=0.3)
    _tokens(ax, 5.3, 0.9, 3, C["text"], w=0.3)
    _box(ax, 3.0, 5.4, 4.0, 1.0, "action head", C["act_fc"], C["act"], fs=9)
    _arrow(ax, (5, 4.85), (5, 5.35))
    ax = axes[1]
    _box(ax, 0.4, 1.4, 3.8, 2.4, "vision tower", C["vis_fc"], C["vis"], fs=9)
    _box(ax, 5.8, 1.4, 3.8, 2.4, "text tower", C["text_fc"], C["text"], fs=9)
    _box(ax, 2.2, 5.0, 5.6, 1.4, "action head\n(cross-attends both)", C["act_fc"], C["act"], fs=9)
    _arrow(ax, (2.3, 3.85), (4.0, 4.95))
    _arrow(ax, (7.7, 3.85), (6.0, 4.95))
    ax = axes[2]
    _box(ax, 0.4, 1.4, 3.8, 1.6, "vision tower\n→ mean pool", C["vis_fc"], C["vis"], fs=9)
    _box(ax, 5.8, 1.4, 3.8, 1.6, "text tower\n→ mean pool", C["text_fc"], C["text"], fs=9)
    _box(ax, 3.2, 3.8, 3.6, 0.9, "MLP([v ; t]) = 1 token", C["aux_fc"], C["aux"], fs=9)
    _box(ax, 2.2, 5.5, 5.6, 1.1, "action head", C["act_fc"], C["act"], fs=9)
    _arrow(ax, (2.3, 3.05), (4.0, 3.75))
    _arrow(ax, (7.7, 3.05), (6.0, 3.75))
    _arrow(ax, (5, 4.75), (5, 5.45))
    plt.tight_layout()
    plt.show()


def fig_siglip_loss(batch=6):
    """The pairwise sigmoid loss: every (image, text) pair is an independent binary problem."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), facecolor="white")
    lab = 2 * np.eye(batch) - 1
    axes[0].imshow(lab, cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_title(
        "labels z_ij: +1 for the matching pair,\n−1 for every other pair", fontsize=9.5
    )
    axes[0].set_xlabel("text j")
    axes[0].set_ylabel("image i")
    rng = np.random.default_rng(0)
    logit = 4 * lab + rng.normal(size=(batch, batch)) * 1.5
    axes[1].imshow(1 / (1 + np.exp(-logit * lab)), cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title(
        "σ(z_ij · (t·⟨x_i, y_j⟩ + b)): each cell is its own\nbinary classification (no softmax over the batch)",
        fontsize=9.5,
    )
    axes[1].set_xlabel("text j")
    for a in axes:
        a.set_xticks(range(batch))
        a.set_yticks(range(batch))
    plt.tight_layout()
    plt.show()


def fig_dino_scheme():
    """DINO / iBOT self-distillation with an EMA teacher."""
    fig, ax = _fig(12.5, 3.6)
    _box(ax, 0.1, 1.6, 1.9, 0.9, "image", C["data_fc"], C["data"])
    _box(
        ax,
        2.7,
        2.6,
        2.4,
        0.9,
        "view 1 (photometric aug.)\n+ random patch MASK",
        C["data_fc"],
        C["data"],
        fs=8.5,
    )
    _box(ax, 2.7, 0.5, 2.4, 0.9, "view 2 (photometric aug.)", C["data_fc"], C["data"], fs=8.5)
    _box(ax, 5.8, 2.6, 2.0, 0.9, "STUDENT ViT\n+ head", C["vis_fc"], C["vis"], fs=9)
    _box(ax, 5.8, 0.5, 2.0, 0.9, "TEACHER ViT\n+ head (EMA)", C["aux_fc"], C["aux"], fs=9)
    _box(
        ax,
        8.5,
        2.6,
        3.8,
        0.9,
        "student softmax (T=0.1)\nat image level AND at masked patches",
        C["vis_fc"],
        C["vis"],
        fs=8.5,
    )
    _box(
        ax,
        8.5,
        0.5,
        3.8,
        0.9,
        "teacher softmax: centred, sharpened (T=0.04)\nimage level AND every patch",
        C["aux_fc"],
        C["aux"],
        fs=8.5,
    )
    _arrow(ax, (2.05, 2.2), (2.65, 2.9))
    _arrow(ax, (2.05, 1.9), (2.65, 1.0))
    _arrow(ax, (5.15, 3.05), (5.75, 3.05))
    _arrow(ax, (5.15, 0.95), (5.75, 0.95))
    _arrow(ax, (7.85, 3.05), (8.45, 3.05))
    _arrow(ax, (7.85, 0.95), (8.45, 0.95))
    _arrow(ax, (10.4, 1.45), (10.4, 2.55), color=C["act"], lw=2)
    ax.text(10.55, 2.0, "cross-entropy: match the teacher", fontsize=8, color=C["act"])
    _arrow(ax, (6.8, 2.55), (6.8, 1.45), color=C["aux"], ls="--")
    ax.text(6.9, 2.0, "EMA", fontsize=8.5, color=C["aux"])
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.0)
    ax.set_title("DINOv2-style self-distillation (no text, no labels)", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()


def fig_fast_pipeline(chunk, tok):
    """The FAST pipeline on one real chunk ``(H, A)``: actions → DCT → quantised integers → flattened symbols → BPE tokens."""
    import torch

    chunk = torch.as_tensor(chunk, dtype=torch.float32)
    coeff = tok.dct @ chunk
    q = tok.quantize(chunk)
    symbols = tok._symbols(q)
    tokens = tok.encode(chunk)
    h, a = chunk.shape
    fig = plt.figure(figsize=(14, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for d in range(a):
        ax.plot(range(h), chunk[:, d], "o-", ms=3, label=f"dim {d}")
    ax.set_title("① action chunk (H steps × D dims)", fontsize=10, loc="left")
    ax.set_xlabel("time step")
    ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[0, 1])
    width = 0.8 / a
    for d in range(a):
        ax.bar(np.arange(h) + d * width, coeff[:, d], width, label=f"dim {d}")
    ax.set_title("② DCT along time", fontsize=10, loc="left")
    ax.set_xlabel("frequency index")
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(
        q.T.numpy(),
        cmap="RdBu_r",
        vmin=-tok.max_coefficient,
        vmax=tok.max_coefficient,
        aspect="auto",
    )
    for i in range(h):
        for d in range(a):
            ax.text(i, d, int(q[i, d]), ha="center", va="center", fontsize=7)
    ax.set_title(f"③ round(scale·coefficient), scale = {tok.scale:g}", fontsize=10, loc="left")
    ax.set_xlabel("frequency index")
    ax.set_yticks(range(a))
    ax.set_yticklabels([f"dim {d}" for d in range(a)])
    del im
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0,
        0.93,
        f"④ flatten ({tok.order}-major, {len(symbols)} integers)  →  ⑤ byte-pair encoding ({len(tokens)} tokens)",
        fontsize=10,
        va="top",
    )
    cell = min(0.026, 0.96 / max(len(symbols), 1))
    for i, sym in enumerate(symbols):
        v = sym - tok.max_coefficient
        ax.add_patch(
            Rectangle(
                (i * cell, 0.52),
                cell * 0.92,
                0.2,
                fc=C["data_fc"] if v == 0 else C["text_fc"],
                ec=C["data"],
                lw=0.6,
            )
        )
        ax.text(i * cell + cell * 0.46, 0.62, str(v), ha="center", va="center", fontsize=6.5)
    ax.text(
        0,
        0.47,
        "integers (0 = grey: the long runs of zeros are what BPE compresses)",
        fontsize=8,
        color=C["note"],
        va="top",
    )
    cell2 = min(0.05, 0.96 / max(len(tokens), 1))
    for i, t in enumerate(tokens):
        merged = t >= tok.base_size
        ax.add_patch(
            Rectangle(
                (i * cell2, 0.08),
                cell2 * 0.92,
                0.22,
                fc=C["aux_fc"] if merged else C["vis_fc"],
                ec=C["aux"] if merged else C["vis"],
                lw=1.2,
            )
        )
        ax.text(i * cell2 + cell2 * 0.46, 0.19, str(t), ha="center", va="center", fontsize=7.5)
    ax.text(
        0,
        0.03,
        "tokens (purple = a merged pair/run, blue = a single integer)",
        fontsize=8,
        color=C["note"],
        va="top",
    )
    return fig
