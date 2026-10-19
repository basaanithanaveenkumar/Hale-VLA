"""Schematic diagrams for the SmolVLA and GR00T notebooks (matplotlib only, no trained model needed).

Visual grammar (same as the π-family notebooks): **blue** = VLM / System 2 weights, **red** = action network / System 1 weights,
**grey** = data or fixed steps, **green** = state, **orange** = text, **purple** = learned / auxiliary heads, dashed arrows =
attention reads, a red ✕ = a frozen / blocked path.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

C = dict(
    vlm="#2b6cb0",
    vlm_fc="#e6f0fa",
    act="#c53030",
    act_fc="#fdeaea",
    data="#555555",
    data_fc="#f2f2f2",
    img="#4c78a8",
    text="#f58518",
    state="#54a24b",
    fut="#b279a2",
    fut_fc="#f6eefa",
    note="#555555",
    ok="#2f855a",
    warn="#c53030",
)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False, lw=1.6, tc="k"):
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
        color=tc,
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


def _tokens(ax, x, y, n, color, w=0.26, h=0.3, label=None, fs=8):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * w * 1.12, y), w, h, fc=color, ec="white", lw=0.6, zorder=4))
    if label:
        ax.text(x + n * w * 1.12 / 2, y + h + 0.1, label, ha="center", va="bottom", fontsize=fs)
    return x + n * w * 1.12


def _cross(ax, x, y, color="#c53030", s=0.12):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=2.2, zorder=6)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=2.2, zorder=6)


# ================================================================================================ SmolVLA
def fig_smolvla_architecture(layers: int = 6, skip_to: int = 3):
    """SmolVLA: truncated frozen VLM (left), interleaved cross/self-attention action expert (right), layer-i features feeding CA blocks."""
    fig, ax = _fig(13, 6.2)
    # token row
    x = _tokens(ax, 0.2, 0.2, 4, C["img"], label="image\n(pixel-shuffled)")
    x = _tokens(ax, x + 0.15, 0.2, 3, C["text"], label="prompt")
    x = _tokens(ax, x + 0.15, 0.2, 1, C["state"], label="state q_t")
    # VLM stack
    for i in range(layers):
        y = 1.05 + i * 0.62
        kept = i < skip_to
        _box(
            ax,
            0.1,
            y,
            3.4,
            0.52,
            f"VLM layer {i + 1}" + ("" if kept else "   (skipped)"),
            C["vlm_fc"] if kept else "#f4f4f4",
            C["vlm"] if kept else "#bbbbbb",
            fs=8.5,
            tc="k" if kept else "#999999",
        )
    _arrow(ax, (1.8, 0.62), (1.8, 1.0))
    ax.text(
        1.8,
        5.35,
        "pre-trained, frozen VLM\n(top layers deleted)",
        ha="center",
        fontsize=9,
        color=C["vlm"],
        fontweight="bold",
    )
    # expert stack
    kinds = ["self" if i % 2 == 0 else "cross" for i in range(skip_to)]
    for i, kind in enumerate(kinds):
        y = 1.05 + i * 0.62
        cross = kind == "cross"
        _box(
            ax,
            6.2,
            y,
            3.2,
            0.52,
            "cross-attention → VLM features" if cross else "causal self-attention (actions)",
            C["act_fc"] if cross else "#fff3e0",
            C["act"] if cross else "#c77c00",
            fs=8.2,
        )
        if cross:
            _arrow(ax, (3.55, y + 0.26), (6.15, y + 0.26), color=C["act"], ls="--", lw=1.6)
            ax.text(
                4.85, y + 0.33, f"layer {i + 1} features", ha="center", fontsize=7.5, color=C["act"]
            )
    ax.text(
        7.8,
        1.05 + skip_to * 0.62 + 0.2,
        "action expert  (0.75 × VLM width)\ninterleaved SA / CA, one block per kept VLM layer",
        ha="center",
        fontsize=9,
        color=C["act"],
        fontweight="bold",
    )
    # expert inputs / outputs
    _box(
        ax,
        6.2,
        0.1,
        3.2,
        0.62,
        "action-time MLP\nW3·swish(W2·[W1·a^τ , φ(τ)])",
        C["data_fc"],
        C["data"],
        fs=8.3,
    )
    _tokens(ax, 6.55, -0.35, 8, C["act"], label=None)
    ax.text(
        7.7, -0.55, "noisy action chunk a^τ  (n actions)", ha="center", fontsize=8, color=C["note"]
    )
    _arrow(ax, (7.8, 0.12), (7.8, 0.0), lw=1.2)
    _arrow(ax, (7.8, 0.75), (7.8, 1.0))
    # output
    _box(ax, 10.1, 1.0, 2.5, 0.9, "velocity v_θ\n(n, action_dim)", C["data_fc"], C["data"], fs=9)
    _arrow(ax, (9.45, 1.4), (10.05, 1.4))
    _box(
        ax,
        10.1,
        2.4,
        2.5,
        1.1,
        "Euler step  (10 steps)\na^{τ−δ} = a^τ − δ·v\nVLM runs once",
        C["data_fc"],
        C["data"],
        fs=8.5,
    )
    _arrow(ax, (11.35, 1.92), (11.35, 2.38))
    _arrow(ax, (10.05, 3.0), (8.9, 0.75), color=C["act"], ls="--", rad=-0.2)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(-0.9, 6.2)
    ax.set_title(
        "SmolVLA = frozen truncated VLM + thin interleaved flow-matching action expert",
        fontsize=11,
        loc="left",
    )
    plt.tight_layout()
    plt.show()


def fig_pixel_shuffle(grid: int = 8, shuffle: int = 2):
    """Pixel shuffle: each 2x2 block of patch tokens becomes one token (4x fewer tokens, nothing discarded)."""
    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 3.6), facecolor="white", gridspec_kw=dict(width_ratios=[1, 0.35, 1])
    )
    rng = np.random.default_rng(1)
    colors = rng.uniform(0.35, 0.95, size=(grid, grid, 3))
    ax = axes[0]
    ax.imshow(colors, interpolation="nearest")
    for k in range(0, grid + 1, shuffle):
        ax.axhline(k - 0.5, color="k", lw=2)
        ax.axvline(k - 0.5, color="k", lw=2)
    ax.set_title(f"{grid}×{grid} = {grid * grid} patch tokens\n(each 4×4 pixels)", fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])
    axes[1].axis("off")
    axes[1].annotate(
        "",
        xy=(1, 0.5),
        xytext=(0, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=2),
        xycoords="axes fraction",
    )
    axes[1].text(
        0.5,
        0.62,
        "concatenate\neach 2×2 block\n+ Linear",
        ha="center",
        fontsize=8.5,
        transform=axes[1].transAxes,
    )
    ax = axes[2]
    g2 = grid // shuffle
    big = np.zeros((g2, g2, 3))
    for i in range(g2):
        for j in range(g2):
            big[i, j] = colors[
                i * shuffle : (i + 1) * shuffle, j * shuffle : (j + 1) * shuffle
            ].mean((0, 1))
    ax.imshow(big, interpolation="nearest")
    for k in range(g2 + 1):
        ax.axhline(k - 0.5, color="k", lw=1)
        ax.axvline(k - 0.5, color="k", lw=1)
    ax.set_title(f"{g2}×{g2} = {g2 * g2} tokens\n(each carries 4× the channels)", fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def fig_async_timeline(chunk: int = 10, latency: int = 3, g: float = 0.6):
    """Sync vs async inference on one timeline: when queries are sent, when chunks arrive, where the robot idles."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.4), facecolor="white", sharex=True)
    total = 3 * chunk
    for ax, mode in zip(axes, ("synchronous (g = 0)", f"asynchronous (g = {g})")):
        ax.axis("off")
        sync = mode.startswith("sync")
        cells, sends, arrivals = [], [], []
        queue = 0
        # simulate
        pending = None
        for step in range(total):
            if pending is not None and step >= pending:
                queue = chunk - (0 if sync else latency)
                arrivals.append(step)
                pending = None
            need = queue == 0 if sync else queue / chunk < g
            if pending is None and need:
                sends.append(step)
                pending = step + latency
            if queue > 0:
                cells.append("act")
                queue -= 1
            else:
                cells.append("idle")
        for i, kind in enumerate(cells):
            ax.add_patch(
                Rectangle(
                    (i, 0), 0.92, 0.5, fc=C["vlm"] if kind == "act" else "#f2c94c", ec="white"
                )
            )
        for s in sends:
            ax.annotate(
                "",
                xy=(s + 0.45, 0.55),
                xytext=(s + 0.45, 1.1),
                arrowprops=dict(arrowstyle="-|>", color=C["act"], lw=1.6),
            )
            ax.text(s + 0.45, 1.15, "send obs", ha="center", fontsize=7.5, color=C["act"])
        for a in arrivals:
            ax.annotate(
                "",
                xy=(a + 0.45, 0.55),
                xytext=(a + 0.45, 1.1),
                arrowprops=dict(arrowstyle="-|>", color=C["ok"], lw=1.6, ls="--"),
            )
            ax.text(a + 0.45, 1.15, "chunk", ha="center", fontsize=7.5, color=C["ok"])
        ax.set_xlim(-0.5, total + 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(
            mode + f":  {sum(k == 'idle' for k in cells)} idle steps in {total}",
            fontsize=10,
            loc="left",
        )
    axes[1].text(
        0,
        -0.45,
        f"blue = executing an action · yellow = idle, waiting for the model · latency ℓ = {latency} steps, chunk n = {chunk}",
        fontsize=8,
        color=C["note"],
    )
    plt.tight_layout()
    plt.show()


# ================================================================================================ GR00T
def fig_groot_architecture(layers: int = 3, flare: bool = False):
    """GR00T N1: VLM (System 2) → mid-layer features → DiT (System 1) with embodiment-specific encoders / decoder."""
    fig, ax = _fig(13.5, 5.6)
    # System 2
    _box(
        ax,
        0.1,
        2.5,
        3.2,
        2.2,
        "System 2\npre-trained VLM\n(vision encoder + LLM)\nmiddle-layer features\n(N1: layer 12 of 24)",
        C["vlm_fc"],
        C["vlm"],
        fs=9,
    )
    x = _tokens(ax, 0.3, 1.5, 4, C["img"], label="image")
    _tokens(ax, x + 0.1, 1.5, 3, C["text"], label="instruction")
    _arrow(ax, (1.7, 1.9), (1.7, 2.45))
    # DiT
    for i in range(4):
        y = 1.0 + i * 0.72
        cross = i % 2 == 0
        _box(
            ax,
            5.9,
            y,
            3.4,
            0.6,
            "cross-attention → VL features"
            if cross
            else "self-attention  [state | actions" + (" | future]" if flare else "]"),
            C["act_fc"] if cross else "#fff3e0",
            C["act"] if cross else "#c77c00",
            fs=8.2,
        )
        if cross:
            _arrow(ax, (3.35, 3.6), (5.85, y + 0.3), color=C["act"], ls="--", lw=1.3, rad=0.1)
    ax.text(
        7.6,
        4.05,
        "System 1: diffusion transformer (DiT)\nadaptive LayerNorm ← denoising step τ",
        ha="center",
        fontsize=9,
        color=C["act"],
        fontweight="bold",
    )
    # embodiment-specific
    _box(ax, 4.0, 0.0, 1.7, 0.75, "state encoder\n(per embodiment)", "#eaf6ea", C["state"], fs=8)
    _box(
        ax,
        6.1,
        -0.35,
        1.9,
        0.75,
        "action encoder\n(per embodiment)\n[a^τ, φ(τ)]",
        "#fff0f0",
        C["act"],
        fs=7.8,
    )
    _box(ax, 9.9, 1.8, 1.9, 1.0, "action decoder\n(per embodiment)", "#fff0f0", C["act"], fs=8.5)
    _arrow(ax, (4.85, 0.78), (6.4, 1.0))
    _arrow(ax, (7.0, 0.42), (7.0, 0.98))
    _arrow(ax, (9.35, 2.4), (9.85, 2.3))
    _tokens(ax, 10.0, 0.7, 4, C["act"], label="action chunk")
    _arrow(ax, (10.85, 1.78), (10.85, 1.05))
    if flare:
        _tokens(ax, 6.2, 4.45, 3, C["fut"], label="M future tokens")
        _box(
            ax,
            10.3,
            3.6,
            2.6,
            0.9,
            "FLARE: cos-sim to\nfuture-frame embedding",
            C["fut_fc"],
            C["fut"],
            fs=8.3,
        )
        _arrow(ax, (9.35, 3.5), (10.25, 3.9), color=C["fut"], ls="--")
    ax.text(0.2, 0.9, "frozen in N1.5 (N1: only the LLM)", fontsize=8, color=C["warn"])
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.7, 5.2)
    ax.set_title(
        "GR00T " + ("N1.5" if flare else "N1") + ": dual system, one network for many robot bodies",
        fontsize=11,
        loc="left",
    )
    plt.tight_layout()
    plt.show()


def fig_embodiment_routing():
    """The same transformer, different MLPs per embodiment (CategorySpecificLinear)."""
    fig, ax = _fig(12.5, 3.8)
    names = [
        ("point2d", "state 2 → action 2", "#4c78a8"),
        ("joint2d", "state 2 → action 2\n(joint-space)", "#f58518"),
        ("gripper3d", "state 3 → action 3\n(+ gripper)", "#54a24b"),
        ("lapa", "no state → latent 4\n(from video)", "#b279a2"),
    ]
    for i, (n, d, col) in enumerate(names):
        y = 3.0 - i * 0.85
        _box(ax, 0.1, y, 2.0, 0.65, f"{n}\n{d}", "white", col, fs=8, lw=2)
        _box(ax, 2.6, y, 1.7, 0.65, "state / action\nencoder W[e]", "white", col, fs=8, lw=2)
        _arrow(ax, (2.12, y + 0.32), (2.58, y + 0.32), color=col)
        _arrow(ax, (4.32, y + 0.32), (5.4, 1.9), color=col, lw=1.2, rad=0.0)
        _box(ax, 9.3, y, 1.7, 0.65, "action\ndecoder W[e]", "white", col, fs=8, lw=2)
        _arrow(ax, (8.1, 1.9), (9.28, y + 0.32), color=col, lw=1.2)
        _arrow(ax, (11.02, y + 0.32), (11.6, y + 0.32), color=col)
        ax.text(11.7, y + 0.25, "a_e", fontsize=9, color=col)
    _box(
        ax,
        5.4,
        1.0,
        2.7,
        1.8,
        "shared DiT\n(one set of weights)\n+ shared VLM",
        C["act_fc"],
        C["act"],
        fs=10,
        bold=True,
    )
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.0)
    ax.set_title(
        "Embodiment-specific MLPs in front of and behind one shared transformer",
        fontsize=11,
        loc="left",
    )
    plt.tight_layout()
    plt.show()


def fig_data_pyramid():
    """GR00T N1's data pyramid: web + human video (base), synthetic (middle), real robot teleoperation (peak)."""
    fig, ax = _fig(10.5, 4.8)
    levels = [
        ("real robot teleoperation (88 h)", "#c53030", 1.0),
        (
            "synthetic: simulation (DexMimicGen, 780K traj.) + neural trajectories (video generation, 827 h)",
            "#f58518",
            2.2,
        ),
        ("human egocentric video  +  web data (latent actions, VQ-VAE)", "#4c78a8", 3.6),
    ]
    for i, (name, col, width) in enumerate(levels):
        y = 2.6 - i * 1.0
        x0 = 5.2 - width * 1.3
        ax.add_patch(
            plt.Polygon(
                [[x0, y], [10.4 - x0, y], [10.4 - x0 - 0.5, y + 0.9], [x0 + 0.5, y + 0.9]],
                fc=col,
                ec="white",
                alpha=0.9,
            )
        )
        ax.text(
            5.2,
            y + 0.45,
            name,
            ha="center",
            va="center",
            fontsize=8.6,
            color="white",
            fontweight="bold",
            wrap=True,
        )
    ax.text(
        5.2,
        3.75,
        "amount of data ↓            cost per trajectory ↑",
        ha="center",
        fontsize=9,
        color=C["note"],
    )
    ax.set_xlim(0, 10.4)
    ax.set_ylim(-0.2, 4.1)
    ax.set_title("GR00T N1's data pyramid", fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()


def fig_latent_action_model():
    """VQ-VAE inverse dynamics: (x_t, x_t+H) → latent → decode x_t+H."""
    fig, ax = _fig(12.5, 3.2)
    _box(ax, 0.1, 1.0, 1.6, 0.9, "frame x_t", C["data_fc"], C["data"], fs=9)
    _box(ax, 0.1, 0.0, 1.6, 0.9, "future x_t+H", C["data_fc"], C["data"], fs=9)
    _box(ax, 2.4, 0.5, 2.0, 1.0, "encoder\n(sees both frames)", C["vlm_fc"], C["vlm"], fs=9)
    _box(
        ax, 5.1, 0.5, 2.2, 1.0, "z_e  →  nearest code\n(codebook of 8)", C["fut_fc"], C["fut"], fs=9
    )
    _box(ax, 8.0, 0.5, 2.0, 1.0, "decoder\n(x_t, code) → x_t+H", C["act_fc"], C["act"], fs=9)
    _box(ax, 10.7, 0.5, 1.7, 1.0, "reconstruct\nx_t+H", C["data_fc"], C["data"], fs=9)
    for a, b in (
        ((1.72, 1.45), (2.38, 1.1)),
        ((1.72, 0.45), (2.38, 0.9)),
        ((4.42, 1.0), (5.08, 1.0)),
        ((7.32, 1.0), (7.98, 1.0)),
        ((10.02, 1.0), (10.68, 1.0)),
    ):
        _arrow(ax, a, b)
    _arrow(ax, (1.0, 1.95), (8.9, 1.55), ls="--", color=C["act"], rad=-0.25)
    ax.text(5.0, 2.55, "x_t is also given to the decoder", fontsize=8, color=C["act"], ha="center")
    ax.text(
        6.2,
        -0.25,
        "after training: the pre-quantisation z_e is the 'action' label for action-less video (the LAPA pseudo-embodiment)",
        fontsize=8.5,
        color=C["note"],
        ha="center",
    )
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-0.6, 3.0)
    ax.set_title(
        "Latent-action model: inverse dynamics through a discrete bottleneck",
        fontsize=11,
        loc="left",
    )
    plt.tight_layout()
    plt.show()


def fig_flare():
    """FLARE: future tokens in the DiT are aligned (cosine) with the future frame's embedding at a middle block."""
    fig, ax = _fig(12.5, 4.2)
    _tokens(ax, 0.3, 2.6, 1, C["state"], label="state")
    _tokens(ax, 1.0, 2.6, 6, C["act"], label="noised actions")
    _tokens(ax, 3.4, 2.6, 3, C["fut"], label="M future tokens", w=0.3)
    for i in range(4):
        y = 0.4 + i * 0.5
        mid = i == 1
        _box(
            ax,
            0.3,
            y,
            4.3,
            0.4,
            f"DiT block {i + 1}" + ("   ← future tokens read here" if mid else ""),
            C["fut_fc"] if mid else C["act_fc"],
            C["fut"] if mid else C["act"],
            fs=8.4,
        )
    _arrow(ax, (2.4, 2.55), (2.4, 2.35))
    _box(ax, 6.0, 0.55, 2.3, 0.7, "MLP head", C["fut_fc"], C["fut"], fs=9)
    _arrow(ax, (4.65, 0.9), (5.95, 0.9), color=C["fut"])
    _box(ax, 6.0, 1.7, 2.3, 0.7, "cosine\nsimilarity", C["fut_fc"], C["fut"], fs=9)
    _arrow(ax, (7.15, 1.27), (7.15, 1.67), color=C["fut"])
    _box(
        ax,
        9.3,
        1.7,
        2.8,
        0.7,
        "frozen encoder of the\nfuture frame x_t+H",
        C["vlm_fc"],
        C["vlm"],
        fs=8.5,
    )
    _arrow(ax, (9.25, 2.05), (8.35, 2.05), color=C["fut"])
    ax.text(
        10.7,
        1.45,
        "(no action label needed → works on human video)",
        ha="center",
        fontsize=8,
        color=C["note"],
    )
    ax.text(
        3.0,
        3.3,
        "L = L_flow-matching + 0.2 · (1 − cos)",
        fontsize=10,
        color=C["fut"],
        fontweight="bold",
    )
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0.0, 3.8)
    ax.set_title(
        "FLARE (GR00T N1.5): align the policy's hidden state with the future",
        fontsize=11,
        loc="left",
    )
    plt.tight_layout()
    plt.show()
