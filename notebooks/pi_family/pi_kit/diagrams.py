"""Schematic diagrams for the π-family notebooks (matplotlib only, no trained model needed).

Visual grammar used in every figure:

* **blue** boxes = backbone / VLM weights, **red** boxes = action-expert weights, **grey** = data or fixed steps;
* solid arrows = data flow, dashed arrows = attention reads, a red ✕ on an arrow = stop-gradient;
* the same token colours as :mod:`pi_kit.plots` (images blue, text orange, state green, actions red, FAST purple).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

C = dict(
    vlm="#2b6cb0",
    vlm_fc="#e6f0fa",
    exp="#c53030",
    exp_fc="#fdeaea",
    data="#555555",
    data_fc="#f2f2f2",
    img="#4c78a8",
    text="#f58518",
    state="#54a24b",
    act="#e45756",
    fast="#b279a2",
    goal="#72b7b2",
    meta="#9d755d",
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


def _arrow(ax, p0, p1, color="#333333", lw=1.5, ls="-", rad=0.0, style="-|>", ms=12):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            mutation_scale=ms,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
        )
    )


def _tokens(ax, x, y, n, color, w=0.28, h=0.32, label=None, fs=8):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * w * 1.1, y), w, h, fc=color, ec="white", lw=0.6, zorder=4))
    if label:
        ax.text(x + n * w * 1.1 / 2, y + h + 0.12, label, ha="center", va="bottom", fontsize=fs)
    return x + n * w * 1.1


def _cross(ax, x, y, color="#c53030", s=0.12):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=2.2, zorder=6)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=2.2, zorder=6)


def pipeline(steps, title="", learned=()):
    """Horizontal flow chart; ``learned`` indices are drawn red (trained), others grey."""
    n = len(steps)
    bw, gap = 2.3, 0.65
    total = n * bw + (n - 1) * gap
    fig, ax = _fig(max(8, total * 0.8), 2.3)
    for i, (name, detail) in enumerate(steps):
        x = i * (bw + gap)
        fc, ec = (C["exp_fc"], C["exp"]) if i in learned else (C["data_fc"], C["data"])
        _box(ax, x, 0.3, bw, 1.25, f"{name}\n{detail}", fc, ec, fs=8.6)
        if i < n - 1:
            _arrow(ax, (x + bw + 0.04, 0.92), (x + bw + gap - 0.04, 0.92), lw=1.7)
    ax.set_xlim(-0.2, total + 0.2)
    ax.set_ylim(0, 2.1)
    ax.text(0, 1.85, title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------- family
def fig_family_overview():
    """What each generation adds, as three stacked rows sharing one skeleton."""
    rows = [
        (
            "π0  (Oct 2024)",
            [
                "PaliGemma 3B VLM",
                "300M action expert",
                "flow matching, H = 50",
                "block-causal mask",
            ],
        ),
        (
            "π0.5  (Apr 2025)",
            [
                "+ discrete state tokens",
                "+ subtask text (hierarchical)",
                "+ FAST tokens & KI",
                "+ adaRMS time, co-training",
            ],
        ),
        (
            "π0.7  (Apr 2026)",
            [
                "+ MEM video history",
                "+ subgoal images (world model)",
                "+ metadata & control mode",
                "+ prompt dropout, CFG, RTC",
            ],
        ),
    ]
    fig, ax = _fig(13, 3.6)
    for r, (name, items) in enumerate(rows):
        y = 2.4 - r * 1.05
        ax.text(0, y + 0.33, name, fontsize=11, fontweight="bold", va="center")
        for k, item in enumerate(items):
            fc, ec = (
                (C["vlm_fc"], C["vlm"])
                if r == 0
                else (C["exp_fc"], C["exp"])
                if r == 1
                else ("#eaf6f4", C["goal"])
            )
            _box(ax, 2.3 + k * 2.75, y, 2.55, 0.66, item, fc, ec, fs=8.8)
        if r:
            _arrow(ax, (1.0, y + 1.0), (1.0, y + 0.72), lw=1.4)
    ax.set_xlim(-0.1, 13.4)
    ax.set_ylim(0.1, 3.3)
    ax.text(
        0,
        3.15,
        "Each generation keeps the previous one and changes what the model is told and how it is trained",
        fontsize=9.5,
        color=C["note"],
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------- π0
def fig_pi0_architecture():
    """π0: prefix through the VLM weights, state + noisy actions through the action expert, joint attention."""
    fig, ax = _fig(12.5, 5.0)
    # token row
    x = _tokens(ax, 0.2, 0.3, 6, C["img"], label="image patches")
    x = _tokens(ax, x + 0.15, 0.3, 4, C["text"], label='"go to red"')
    x2 = _tokens(ax, x + 0.6, 0.3, 1, C["state"], label=r"$q_t$")
    x3 = _tokens(ax, x2 + 0.25, 0.3, 8, C["act"], label=r"noisy actions $a^{\tau}$ (+ τ)")
    # experts
    _box(
        ax,
        0.1,
        1.35,
        x - 0.05,
        2.0,
        "VLM backbone weights\n(PaliGemma, 3B)\n\nrun ONCE per observation\n→ keys / values cached",
        C["vlm_fc"],
        C["vlm"],
        fs=9.5,
        bold=False,
    )
    _box(
        ax,
        x + 0.5,
        1.35,
        x3 - x - 0.4,
        2.0,
        "action expert weights\n(300M, narrower)\n\nrun every denoising step",
        C["exp_fc"],
        C["exp"],
        fs=9.5,
    )
    for xx in (1.5, 3.5, x2 - 0.15, (x2 + x3) / 2):
        _arrow(ax, (xx, 0.75), (xx, 1.3), lw=1.2)
    # shared attention
    _arrow(ax, (x + 0.48, 2.65), (x - 0.02, 2.65), color=C["exp"], ls="--", lw=1.8)
    ax.text((2 * x + 0.46) / 2, 2.8, "reads", ha="center", fontsize=8.5, color=C["exp"])
    ax.text(
        x / 2,
        1.12,
        "the backbone never reads the expert's tokens",
        ha="center",
        fontsize=8,
        color=C["note"],
    )
    # output + loop
    _box(
        ax,
        x3 + 0.5,
        1.9,
        2.4,
        0.9,
        "velocity v_θ\n(B, H, action_dim)",
        C["data_fc"],
        C["data"],
        fs=9,
    )
    _arrow(ax, (x3 - 0.05, 2.35), (x3 + 0.48, 2.35))
    _box(
        ax,
        x3 + 0.5,
        0.3,
        2.4,
        1.05,
        "Euler step\n$a^{\\tau-\\delta} = a^{\\tau} - \\delta\\, v$\n10 steps, τ: 1 → 0",
        C["data_fc"],
        C["data"],
        fs=8.8,
    )
    _arrow(ax, (x3 + 1.7, 1.88), (x3 + 1.7, 1.38))
    _arrow(ax, (x3 + 0.48, 0.55), ((x2 + x3) / 2 + 0.6, 0.55), rad=0.0, ls="--", color=C["act"])
    ax.text(
        x3 + 0.5,
        3.45,
        "same number of heads & head size,\nso both experts share one attention",
        fontsize=8.5,
        color=C["note"],
    )
    ax.set_xlim(0, x3 + 3.1)
    ax.set_ylim(0, 4.0)
    ax.set_title(
        "π0 = VLM + flow-matching action expert (mixture of transformers)", fontsize=11, loc="left"
    )
    plt.tight_layout()
    plt.show()


def fig_flow_matching_paths(seed: int = 0):
    """Straight-line probability paths from Gaussian noise to a bimodal action distribution."""
    rng = np.random.default_rng(seed)
    n = 60
    modes = np.where(rng.random(n) < 0.5, -1.6, 1.6)
    data = np.stack([modes + 0.15 * rng.normal(size=n), 0.15 * rng.normal(size=n)], 1)
    noise = rng.normal(size=(n, 2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor="white")
    ax = axes[0]
    for d, e in zip(data, noise):
        ax.plot([e[0], d[0]], [e[1], d[1]], color="#bbbbbb", lw=0.7)
    ax.scatter(*noise.T, s=10, color="#555555", label="ε ~ N(0, I)   (τ = 1)")
    ax.scatter(*data.T, s=12, color=C["act"], label="actions a   (τ = 0)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("training pairs: x_τ = τ ε + (1-τ) a", fontsize=9.5)
    ax = axes[1]
    for tau, alpha in ((1.0, 0.25), (0.6, 0.45), (0.3, 0.7), (0.0, 1.0)):
        x = tau * noise + (1 - tau) * data
        ax.scatter(*x.T, s=9, alpha=alpha, label=f"τ = {tau}")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("the noisy inputs the network sees", fontsize=9.5)
    ax = axes[2]
    inside = (np.abs(noise) < 1.8).all(1) & (noise[:, 1] > 0.6)
    i = int(
        np.argmax(np.linalg.norm(noise - data, axis=1) * inside)
    )  # a long pair that fits the axes
    e, d = noise[i], data[i]
    xt = 0.6 * e + 0.4 * d
    ax.plot([e[0], d[0]], [e[1], d[1]], ":", color="#999999")
    ax.scatter(*e, s=45, color="#555555", zorder=4)
    ax.scatter(*d, s=45, color=C["act"], zorder=4)
    ax.scatter(*xt, s=45, color=C["vlm"], zorder=4)
    tip = xt + 0.3 * (e - d)
    ax.annotate("", xytext=xt, xy=tip, arrowprops=dict(arrowstyle="-|>", color=C["vlm"], lw=2))
    ax.annotate(
        "ε (noise)", e, xytext=(6, 6), textcoords="offset points", fontsize=8.5, color="#555555"
    )
    ax.annotate(
        "a (action)", d, xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=C["act"]
    )
    ax.annotate(
        "x_τ", xt, xytext=(-22, -14), textcoords="offset points", fontsize=9, color=C["vlm"]
    )
    ax.annotate(
        "target u = ε − a",
        tip,
        xytext=(10, -6),
        textcoords="offset points",
        fontsize=8.5,
        color=C["vlm"],
    )
    ax.set_title("loss: ‖v_θ(x_τ, τ | o) - (ε - a)‖²", fontsize=9.5)
    for a in axes:
        a.set_xlim(-3.2, 3.2)
        a.set_ylim(-2.6, 2.6)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------- π0.5
def fig_pi05_hierarchy():
    """π0.5's two-step inference with one model: text subtask first, then continuous actions."""
    fig, ax = _fig(12.5, 3.4)
    _box(
        ax,
        0.1,
        1.1,
        2.2,
        1.2,
        'image + "collect red\nthen blue" + state bins',
        C["data_fc"],
        C["data"],
        fs=8.8,
    )
    _box(
        ax,
        3.0,
        1.1,
        2.4,
        1.2,
        "VLM\n(high-level step)\ngreedy text decoding",
        C["vlm_fc"],
        C["vlm"],
        fs=8.8,
    )
    _box(ax, 6.1, 1.1, 1.9, 1.2, 'subtask\n"go to red"', "#fff6dc", "#c9a227", fs=9)
    _box(
        ax,
        8.7,
        1.1,
        2.0,
        1.2,
        "action expert\n(low-level step)\nflow matching",
        C["exp_fc"],
        C["exp"],
        fs=8.8,
    )
    _box(ax, 11.3, 1.1, 1.3, 1.2, "chunk\na_1..a_H", C["data_fc"], C["data"], fs=8.8)
    for a, b in ((2.32, 2.98), (5.42, 6.08), (8.02, 8.68), (10.72, 11.28)):
        _arrow(ax, (a, 1.7), (b, 1.7))
    _arrow(ax, (7.0, 2.35), (4.2, 2.35), rad=0.35, ls="--", color=C["note"])
    ax.text(
        5.6,
        3.05,
        "written into the prompt, then the same backbone is run again",
        fontsize=8.5,
        ha="center",
        color=C["note"],
    )
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0.6, 3.3)
    ax.set_title("π0.5 hierarchical inference: think in words, then act", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()


def fig_knowledge_insulation():
    """Where the gradients of the two losses go, with and without knowledge insulation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), facecolor="white")
    for ax, insulated in zip(axes, (False, True)):
        ax.axis("off")
        _box(
            ax,
            0.2,
            1.2,
            3.0,
            1.4,
            "VLM backbone\n(+ vision encoder)",
            C["vlm_fc"],
            C["vlm"],
            fs=9.5,
        )
        _box(ax, 4.6, 1.2, 2.4, 1.4, "action expert", C["exp_fc"], C["exp"], fs=9.5)
        _box(ax, 0.2, 3.1, 3.0, 0.6, "CE on subtask + FAST tokens", "#f6eefa", C["fast"], fs=8.8)
        _box(ax, 4.6, 3.1, 2.4, 0.6, "α · flow-matching MSE", "#fdeaea", C["exp"], fs=8.8)
        _arrow(ax, (1.7, 3.08), (1.7, 2.65), color=C["fast"], lw=2)
        _arrow(ax, (5.8, 3.08), (5.8, 2.65), color=C["exp"], lw=2)
        _arrow(ax, (4.55, 1.9), (3.25, 1.9), color=C["exp"], lw=2, ls="--")
        ax.text(3.9, 2.05, "gradient", fontsize=8, ha="center", color=C["exp"])
        if insulated:
            _cross(ax, 3.9, 1.9)
            ax.text(3.9, 1.35, "detach()", ha="center", fontsize=8.5, color=C["exp"])
        ax.set_xlim(0, 7.3)
        ax.set_ylim(0.9, 4.1)
        ax.set_title(
            "with knowledge insulation (π0.5-KI, π0.7)" if insulated else "without (π0)",
            fontsize=10,
            loc="left",
        )
    plt.tight_layout()
    plt.show()


def fig_fast_tokenizer(chunk, dct_coeff, quantized, tokens):
    """Real numbers from one chunk: actions → DCT → integers → BPE tokens."""
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13, 2.8),
        facecolor="white",
        gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.2, 1.6]),
    )
    axes[0].plot(chunk[:, 0], "-o", ms=3, label="dx")
    axes[0].plot(chunk[:, 1], "-o", ms=3, label="dy")
    axes[0].set_title("1. action chunk (H × 2)", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].bar(np.arange(len(dct_coeff)) - 0.2, dct_coeff[:, 0], 0.4)
    axes[1].bar(np.arange(len(dct_coeff)) + 0.2, dct_coeff[:, 1], 0.4)
    axes[1].set_title("2. DCT per dimension\n(energy in low frequencies)", fontsize=9)
    axes[2].imshow(quantized.T, cmap="coolwarm", vmin=-6, vmax=6, aspect="auto")
    axes[2].set_title("3. round(γ · DCT): mostly 0", fontsize=9)
    axes[2].set_xlabel("frequency", fontsize=8)
    axes[3].axis("off")
    axes[3].text(
        0,
        0.85,
        f"4. frequency-major + BPE\n{quantized.size} integers → {len(tokens)} tokens",
        fontsize=9.5,
        va="top",
    )
    axes[3].text(
        0,
        0.3,
        " ".join(str(t) for t in tokens[:24]) + (" …" if len(tokens) > 24 else ""),
        fontsize=8.5,
        family="monospace",
        wrap=True,
    )
    for a in axes[:3]:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------- π0.7
def fig_pi07_prompt():
    """The four kinds of context in π0.7's prompt and how they are dropped out during training."""
    fig, ax = _fig(13, 3.2)
    items = [
        ("observation\n(MEM history)", C["img"], "always"),
        ("subgoal image", C["goal"], "25 % of examples"),
        ('task + subtask\n"go to red"', C["text"], "subtask dropped 30 %\nwhen a subgoal is given"),
        ("metadata\nspeed / quality / mistake", C["meta"], "all dropped 15 %,\neach field 5 %"),
        ("control mode\nee | joint", "#666666", "never dropped"),
        ("state tokens\nq_t, q_t-2, …", C["state"], "with the history"),
    ]
    for k, (name, color, rule) in enumerate(items):
        x = 0.1 + k * 2.15
        _box(ax, x, 1.4, 1.95, 1.0, name, "white", color, fs=8.6, lw=2.2)
        ax.text(x + 0.97, 1.1, rule, ha="center", va="top", fontsize=8, color=C["note"])
    ax.set_xlim(0, 13.1)
    ax.set_ylim(0.6, 2.75)
    ax.set_title(
        "π0.7 prompt = everything you know about the episode, each part optional",
        fontsize=11,
        loc="left",
    )
    plt.tight_layout()
    plt.show()


def fig_pi07_runtime():
    """Algorithm 1: high-level policy, world model and action expert running at different rates."""
    fig, ax = _fig(12.5, 4.2)
    _box(
        ax,
        0.1,
        2.6,
        2.6,
        1.0,
        "high-level policy\n(same arch., text out)",
        C["vlm_fc"],
        C["vlm"],
        fs=8.8,
    )
    _box(ax, 3.4, 2.6, 2.6, 1.0, "world model g_ψ\nsubgoal image", "#eaf6f4", C["goal"], fs=8.8)
    _box(
        ax,
        6.7,
        2.6,
        2.8,
        1.0,
        "π0.7 VLA\nmetadata: quality 5,\nfast, no mistakes",
        C["exp_fc"],
        C["exp"],
        fs=8.4,
    )
    _box(ax, 10.2, 2.6, 2.1, 1.0, "robot executes\nĤ of H actions", C["data_fc"], C["data"], fs=8.8)
    _arrow(ax, (2.72, 3.1), (3.38, 3.1))
    _arrow(ax, (6.02, 3.1), (6.68, 3.1))
    _arrow(ax, (9.52, 3.1), (10.18, 3.1))
    _arrow(ax, (1.4, 2.58), (7.2, 2.58), rad=0.25, color=C["text"])
    ax.text(4.3, 1.9, "subtask text", fontsize=8.5, color=C["text"])
    _arrow(ax, (11.25, 2.58), (1.4, 2.58), rad=-0.35, ls="--", color=C["note"])
    ax.text(6.3, 0.75, "new observation", fontsize=8.5, color=C["note"], ha="center")
    ax.text(
        3.4,
        3.85,
        "refresh when the subtask changes or every Δ seconds",
        fontsize=8.3,
        color=C["goal"],
    )
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0.4, 4.2)
    ax.set_title("π0.7 at run time (Algorithm 1)", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()


def fig_rtc(chunk: int = 16, exec_horizon: int = 8, delay: int = 3):
    """Real-time chunking timeline: while the next chunk is computed the old one keeps running."""
    fig, ax = _fig(12, 2.9)
    for row, (start, label) in enumerate(((0, "chunk k"), (exec_horizon, "chunk k+1"))):
        y = 1.6 - row * 0.9
        for j in range(chunk):
            t = start + j
            color = "#cfd8e3"
            if row == 0 and j < exec_horizon + delay:
                color = C["vlm"]
            if row == 1:
                color = (
                    C["goal"]
                    if j < delay
                    else (C["exp"] if j < delay + exec_horizon else "#f3c7c7")
                )
            ax.add_patch(Rectangle((t, y), 0.92, 0.6, fc=color, ec="white"))
        ax.text(start - 0.3, y + 0.3, label, ha="right", va="center", fontsize=9)
    ax.text(
        (exec_horizon + delay) / 2,
        2.3,
        "executed from chunk k",
        ha="center",
        fontsize=8,
        color=C["vlm"],
    )
    ax.add_patch(Rectangle((exec_horizon, 2.35), delay, 0.25, fc="#ffd166", ec="none"))
    ax.text(
        exec_horizon + delay / 2, 2.7, f"inference latency d = {delay}", ha="center", fontsize=8.5
    )
    ax.text(
        exec_horizon + delay / 2,
        0.45,
        "committed\n(τ = 0, given)",
        ha="center",
        va="top",
        fontsize=8,
        color=C["goal"],
    )
    ax.text(
        exec_horizon + delay + exec_horizon / 2,
        0.45,
        "generated to continue them",
        ha="center",
        va="top",
        fontsize=8,
        color=C["exp"],
    )
    ax.set_xlim(-2.5, exec_horizon + chunk + 0.5)
    ax.set_ylim(-0.3, 3.0)
    ax.set_title("training-time real-time chunking (RTC)", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()
