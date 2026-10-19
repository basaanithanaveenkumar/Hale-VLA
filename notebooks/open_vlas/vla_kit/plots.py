"""Data plots for the SmolVLA and GR00T notebooks: scenes, token layouts, masks, attention overlays, rollouts, queues, latent actions."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

COLORS = {
    "image": "#4c78a8",
    "text": "#f58518",
    "state": "#54a24b",
    "answer": "#b279a2",
    "actions": "#e45756",
    "pad": "#dddddd",
}
INK, MUTED = "#222222", "#8a8a8a"


def show_image(ax, image, title: str = ""):
    """Draw one (3, S, S) image without axes."""
    img = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def _px(xy, size: int = 32):
    return np.asarray(xy) * (size - 1)


def show_batch(batch, n: int = 6, title: str = "", max_step: float = 0.05):
    """Observations with the (end-effector) path implied by each action chunk, for embodiments whose first two action dims are motion."""
    n = min(n, batch.size)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.7), facecolor="white")
    for i, ax in enumerate(np.atleast_1d(axes)):
        show_image(ax, batch.images[i], batch.task[i])
        if batch.actions is not None and int(batch.embodiment[i]) != 1:
            start = batch.state[i, :2].numpy()
            path = np.vstack(
                [start, start + max_step * np.cumsum(batch.actions[i, :, :2].numpy(), 0)]
            )
            px = _px(path)
            ax.plot(px[:, 0], px[:, 1], "-", color="white", lw=1.2)
            ax.plot(px[-1, 0], px[-1, 1], "o", color="white", ms=3)
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plot_token_layout(
    inputs,
    vocab,
    example: int = 0,
    title: str = "",
    extra_label: str = "state",
    suffix: Sequence | None = None,
):
    """One row of coloured cells: image tokens, prompt words, extra (state) token(s), answer tokens, and an optional action suffix."""
    labels, colors = [], []
    n_img = inputs.num_image_tokens
    s, e = inputs.prompt_slices[example]
    for j in range(inputs.embeds.shape[1]):
        if not inputs.valid[example, j]:
            continue
        if j < n_img:
            labels.append("img")
            colors.append(COLORS["image"])
        elif j < e:
            labels.append("•")
            colors.append(COLORS["text"])
        else:
            labels.append(extra_label if not inputs.ar[example, j] else "ans")
            colors.append(COLORS["state"] if not inputs.ar[example, j] else COLORS["answer"])
    for name, n in suffix or []:
        labels += [name] * n
        colors += [COLORS["actions"]] * n
    n = len(labels)
    fig, ax = plt.subplots(figsize=(min(15, 0.32 * n + 1), 1.2), facecolor="white")
    for j, (lab, col) in enumerate(zip(labels, colors)):
        ax.add_patch(plt.Rectangle((j, 0), 0.94, 1, color=col))
        if lab not in ("img", "•"):
            ax.text(j + 0.47, 0.5, lab[0], ha="center", va="center", color="white", fontsize=7)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.3, 1.1)
    ax.axis("off")
    ax.set_title(
        title or f"example {example}: image tokens · prompt words · {extra_label}",
        fontsize=9.5,
        loc="left",
    )
    plt.tight_layout()
    plt.show()


def plot_mask(
    mask: torch.Tensor,
    groups: Sequence,
    title: str = "attention mask (row = query, column = key)",
    size: float = 5.2,
):
    """A boolean (Lq, Lk) mask with named groups drawn as bands. ``groups``: list of (name, start, end) along the key axis."""
    m = mask.detach().cpu().numpy()
    fig, ax = plt.subplots(
        figsize=(size, size * m.shape[0] / max(m.shape[1], 1) + 0.3), facecolor="white"
    )
    ax.imshow(
        m.astype(float),
        cmap=ListedColormap(["#f3f3f3", "#2b6cb0"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    for _, a, _ in groups:
        ax.axvline(a - 0.5, color="#e45756", lw=0.8)
    centers = [(a + b) / 2 for _, a, b in groups]
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_cross_attention(
    images,
    weights,
    num_image_tokens: int,
    rows: Sequence[int] = (0,),
    titles: Sequence[str] | None = None,
    targets: Sequence | None = None,
    ncols: int = 4,
):
    """Overlay the cross-attention of chosen action tokens onto the image.

    ``weights`` is (B, heads, H, L) from a cross-attention block; the first ``num_image_tokens`` keys are the image tokens (a g×g grid).
    Averaged over heads. ``targets`` optionally gives the target object's pixel position per example (drawn as a white ring).
    """
    g = int(np.sqrt(num_image_tokens))
    b = min(weights.shape[0], ncols)
    fig, axes = plt.subplots(
        len(rows), b, figsize=(2.6 * b, 2.7 * len(rows)), facecolor="white", squeeze=False
    )
    for r, row in enumerate(rows):
        for i in range(b):
            w = weights[i].mean(0)[row, :num_image_tokens].reshape(g, g).cpu().numpy()
            ax = axes[r, i]
            show_image(ax, images[i], "")
            ax.imshow(
                w,
                cmap="magma",
                alpha=0.6,
                extent=(-0.5, images.shape[-1] - 0.5, images.shape[-2] - 0.5, -0.5),
                interpolation="nearest",
            )
            if targets is not None:
                ax.scatter(
                    [targets[i][0]],
                    [targets[i][1]],
                    s=140,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=1.5,
                )
            ax.set_title(
                (titles[i] if titles else "") + (f"\naction token {row}" if i == 0 else ""),
                fontsize=8.5,
            )
    plt.tight_layout()
    plt.show()


def bar_compare(
    values: dict,
    ylabel: str,
    title: str = "",
    ylim=None,
    fmt: str = "{:.2f}",
    colors=None,
    rotate: int = 0,
):
    """Labelled bar chart for ablations."""
    names = list(values)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.15 * len(names)), 3.1), facecolor="white")
    bars = ax.bar(
        names, [values[k] for k in names], color=colors or ["#4c78a8"] * len(names), width=0.62
    )
    for b, k in zip(bars, names):
        v = values[k]
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    plt.xticks(fontsize=8, rotation=rotate, ha="right" if rotate else "center")
    plt.tight_layout()
    plt.show()


def grouped_bars(table: dict, ylabel: str, title: str = "", ylim=(0, 1.05), colors=None):
    """``table[group][series] = value`` as grouped bars."""
    groups = list(table)
    series = list(next(iter(table.values())))
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(groups) + 1.5), 3.2), facecolor="white")
    w = 0.8 / len(series)
    for k, s in enumerate(series):
        xs = np.arange(len(groups)) + k * w - 0.4 + w / 2
        vals = [table[g][s] for g in groups]
        ax.bar(xs, vals, w * 0.92, label=s, color=(colors or [None] * len(series))[k])
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    plt.tight_layout()
    plt.show()


def plot_loss(history, keys=("flow",), title: str = "training"):
    fig, axes = plt.subplots(
        1, len(keys) + 1, figsize=(3.6 * (len(keys) + 1), 2.6), facecolor="white"
    )
    axes = np.atleast_1d(axes)
    axes[0].plot(history.losses, alpha=0.25, color=MUTED)
    axes[0].plot(history.smoothed(), color=INK)
    axes[0].set_title("total loss", fontsize=9)
    axes[0].set_yscale("log")
    for ax, k in zip(axes[1:], keys):
        s = history.series(k)
        ax.plot(s, alpha=0.25, color=MUTED)
        if len(s) > 25:
            ax.plot(np.convolve(s, np.ones(25) / 25, mode="valid"), color=INK)
        ax.set_title(k, fontsize=9)
        ax.set_yscale("log")
    for ax in axes:
        ax.set_xlabel("step", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plot_rollouts(result, scenarios, world, n: int = 6, title: str = "closed-loop rollouts"):
    """First frame of each scenario with the executed path; green title = success."""
    n = min(n, len(scenarios))
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), facecolor="white")
    for i, ax in enumerate(np.atleast_1d(axes)):
        s = scenarios[i]
        show_image(ax, world.render(s.scene.robot, s.scene.objects), "")
        steps = int(result.steps[i]) if result.success[i] else result.trajectories.shape[1] - 1
        px = _px(result.trajectories[i][: steps + 1])
        ax.plot(px[:, 0], px[:, 1], "-", color="white", lw=1.3)
        ax.plot(px[-1, 0], px[-1, 1], "x", color="white", ms=6)
        ok = bool(result.success[i])
        ax.set_title(
            f"{s.task}\n{'✓' if ok else '✗'} {steps} steps",
            fontsize=8.5,
            color="#2f855a" if ok else "#c53030",
        )
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plot_denoising(
    batch,
    trajectory: list,
    example: int = 0,
    steps_to_show: Sequence[int] | None = None,
    max_step: float = 0.05,
    tau_zero_is_noise: bool = False,
):
    """The action chunk at several Euler steps, drawn as the path it would make the robot follow."""
    n = len(trajectory) - 1
    steps_to_show = (
        list(steps_to_show) if steps_to_show is not None else sorted({0, n // 3, 2 * n // 3, n})
    )
    fig, axes = plt.subplots(
        1, len(steps_to_show), figsize=(2.6 * len(steps_to_show), 2.9), facecolor="white"
    )
    state = batch.state[example, :2].numpy()
    for ax, k in zip(np.atleast_1d(axes), steps_to_show):
        tau = k / n if tau_zero_is_noise else 1 - k / n
        show_image(ax, batch.images[example], f"step {k}/{n}  (τ = {tau:.2f})")
        a = trajectory[k][example, :, :2].cpu().numpy()
        px = _px(np.vstack([state, state + max_step * np.cumsum(a, 0)]))
        ax.plot(px[:, 0], px[:, 1], "-", color="white", lw=1.4)
        ax.scatter(
            px[1:, 0], px[1:, 1], c=np.linspace(0, 1, len(px) - 1), cmap="autumn", s=8, zorder=3
        )
        ax.set_xlim(-0.5, 31.5)
        ax.set_ylim(31.5, -0.5)
    fig.suptitle(
        f'denoising an action chunk: noise → plan - "{batch.task[example]}"',
        fontsize=10,
        x=0.01,
        ha="left",
    )
    plt.tight_layout()
    plt.show()


def plot_queue_traces(
    traces: dict, chunk_size: int, title: str = "action-queue size over time (paper Fig. 3)"
):
    """Queue length per step for several runs (dict name → AsyncTrace); idle steps are shaded and sends marked."""
    fig, axes = plt.subplots(
        len(traces),
        1,
        figsize=(10.5, 1.9 * len(traces)),
        facecolor="white",
        sharex=True,
        squeeze=False,
    )
    for ax, (name, tr) in zip(axes[:, 0], traces.items()):
        t = np.arange(len(tr.queue_sizes))
        ax.fill_between(
            t,
            0,
            chunk_size,
            where=np.array(tr.idle_flags),
            color="#f2c94c",
            alpha=0.6,
            step="mid",
            label="idle",
        )
        ax.step(t, tr.queue_sizes, where="mid", color="#2b6cb0")
        for s in tr.send_times:
            ax.axvline(s, color="#c53030", lw=0.8, ls=":")
        ax.set_ylim(0, chunk_size + 1)
        ax.set_ylabel("|queue|", fontsize=8)
        ax.set_title(
            f"{name}  ·  {tr.idle_steps} idle steps · {tr.inferences} model calls · {'success at step ' + str(tr.steps) if tr.success else 'failed'}",
            fontsize=8.8,
            loc="left",
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("control step (red dotted = observation sent)", fontsize=8)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plot_latent_counterfactuals(
    lam, x_t, x_future, codes: Sequence[int] | None = None, n_examples: int = 3
):
    """For each example: the current frame, the true future, and the model's *imagined* future for every code in the codebook."""
    k = lam.codebook.num_embeddings
    codes = list(codes) if codes is not None else list(range(k))
    fig, axes = plt.subplots(
        n_examples,
        2 + len(codes),
        figsize=(1.35 * (2 + len(codes)), 1.5 * n_examples),
        facecolor="white",
        squeeze=False,
    )
    with torch.no_grad():
        for i in range(n_examples):
            show_image(axes[i, 0], x_t[i], "now" if i == 0 else "")
            show_image(axes[i, 1], x_future[i], "true future" if i == 0 else "")
            for j, c in enumerate(codes):
                z = lam.codebook.weight[c][None]
                img = lam.decode(x_t[i : i + 1], z)[0].clamp(0, 1)
                show_image(axes[i, 2 + j], img, f"code {c}" if i == 0 else "")
    plt.tight_layout()
    plt.show()
