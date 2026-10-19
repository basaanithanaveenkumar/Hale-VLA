"""Data plots for the action-representation and VLM-backbone notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

INK, MUTED = "#222222", "#8a8a8a"


def show_image(ax, image, title: str = ""):
    img = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def curves(
    x,
    ys: dict,
    xlabel: str,
    ylabel: str,
    title: str = "",
    logx: bool = False,
    ylim=None,
    markers: bool = True,
    logy: bool = False,
):
    """Several named curves over a shared x axis."""
    fig, ax = plt.subplots(figsize=(6.2, 3.3), facecolor="white")
    for name, y in ys.items():
        ax.plot(x, y, "o-" if markers else "-", ms=4, label=name)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    plt.tight_layout()
    plt.show()


def plot_2d_samples(panels: dict, ncols: int = 4, lim: float = 3.4, title: str = ""):
    """Scatter plots of 2-D samples, one panel per method (dict name → (N, 2) tensor)."""
    n = len(panels)
    rows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        rows, ncols, figsize=(3.2 * ncols, 3.0 * rows), facecolor="white", squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, pts) in zip(axes.ravel(), panels.items()):
        ax.axis("on")
        p = pts.numpy() if torch.is_tensor(pts) else pts
        ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.5, color="#c53030")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plot_paths(trajectories: dict, n_paths: int = 25, lim: float = 3.4):
    """Sampling paths of several samplers (dict name → list of (N, 2) tensors, noise → sample)."""
    fig, axes = plt.subplots(
        1,
        len(trajectories),
        figsize=(3.6 * len(trajectories), 3.4),
        facecolor="white",
        squeeze=False,
    )
    for ax, (name, traj) in zip(axes[0], trajectories.items()):
        pts = torch.stack(traj, 0).numpy()  # (K+1, N, 2)
        for i in range(min(n_paths, pts.shape[1])):
            ax.plot(
                pts[:, i, 0], pts[:, i, 1], lw=0.8, alpha=0.7, color=plt.cm.viridis(i / n_paths)
            )
            ax.scatter(*pts[-1, i], s=10, color="#c53030", zorder=3)
            ax.scatter(*pts[0, i], s=6, color="#555555", zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_noise_schedules(schedules: dict, steps: int):
    """ᾱ_t (fraction of the signal left) and the signal-to-noise ratio of several DDPM schedules."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), facecolor="white")
    for name, s in schedules.items():
        axes[0].plot(np.arange(steps), s.alpha_bar.numpy(), label=name)
        axes[1].plot(np.arange(steps), (s.alpha_bar / (1 - s.alpha_bar)).numpy(), label=name)
    axes[0].set_title("ᾱ_t: how much of x_0 survives at step t", fontsize=9.5)
    axes[1].set_title("signal-to-noise ratio ᾱ/(1 − ᾱ)", fontsize=9.5)
    axes[1].set_yscale("log")
    for a in axes:
        a.set_xlabel("diffusion step t")
        a.legend(fontsize=8, frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_chunk_noising(chunk, schedule, steps=(0, 15, 40, 70, 99), max_step: float = 0.05):
    """A real action chunk (as the path it would draw) progressively noised by the forward process."""
    fig, axes = plt.subplots(1, len(steps), figsize=(2.7 * len(steps), 2.7), facecolor="white")
    x0 = torch.as_tensor(chunk, dtype=torch.float32) / 1.5
    g = torch.Generator().manual_seed(0)
    eps = torch.randn(x0.shape, generator=g)
    for ax, t in zip(axes, steps):
        xt = schedule.q_sample(x0[None], torch.tensor([t]), eps[None])[0] * 1.5
        path = np.vstack([[0, 0], np.cumsum(xt.numpy(), 0) * max_step])
        ax.plot(path[:, 0], path[:, 1], "o-", ms=2.5, lw=1.2, color="#c53030")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {t}  (ᾱ = {float(schedule.alpha_bar[t]):.2f})", fontsize=9)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.show()


def plot_token_distribution(
    probs, true_bin: int | None = None, title: str = "", xlabel: str = "bin id"
):
    """The categorical distribution an autoregressive policy puts over the bins of one action dimension."""
    fig, ax = plt.subplots(figsize=(6.2, 2.4), facecolor="white")
    ax.bar(np.arange(len(probs)), probs, color="#4c78a8", width=1.0)
    if true_bin is not None:
        ax.axvline(true_bin, color="#c53030", ls="--", lw=1.2, label="demonstrated bin")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability")
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_patch_pca(encoders: dict, images, size: int = 32):
    """Project every patch token onto the 3 leading principal components (computed per encoder over the shown images) and show it as RGB.

    Structure that the encoder separates (objects vs background, one object vs another) appears as distinct colours.
    """
    n = images.shape[0]
    fig, axes = plt.subplots(
        len(encoders) + 1,
        n,
        figsize=(1.9 * n, 1.9 * (len(encoders) + 1)),
        facecolor="white",
        squeeze=False,
    )
    for j in range(n):
        show_image(axes[0, j], images[j], "image" if j == 0 else "")
    for r, (name, enc) in enumerate(encoders.items(), start=1):
        with torch.no_grad():
            tok = enc(images)  # (n, N, W)
        g = int(np.sqrt(tok.shape[1]))
        flat = tok.reshape(-1, tok.shape[-1]).numpy()
        flat = flat - flat.mean(0)
        u, s, vt = np.linalg.svd(flat, full_matrices=False)
        proj = flat @ vt[:3].T
        proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-9)
        proj = proj.reshape(n, g, g, 3)
        for j in range(n):
            axes[r, j].imshow(proj[j], interpolation="nearest")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_similarity(sim, row_labels, col_labels, title: str = ""):
    """An annotated image-text similarity matrix."""
    fig, ax = plt.subplots(
        figsize=(1.1 * len(col_labels) + 2, 0.7 * len(row_labels) + 1.6), facecolor="white"
    )
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title(title, fontsize=9.5, loc="left")
    plt.colorbar(im, fraction=0.04)
    plt.tight_layout()
    plt.show()


def plot_text_to_image_attention(images, attn, words, n_img_tokens: int, titles=None):
    """Overlay, for each example, the attention of the word tokens onto the image-token grid.

    ``attn``: (B, W, N_img) attention from W word tokens to the image tokens (already averaged over heads / layers).
    """
    b = images.shape[0]
    g = int(np.sqrt(n_img_tokens))
    fig, axes = plt.subplots(
        b, len(words), figsize=(2.3 * len(words), 2.4 * b), facecolor="white", squeeze=False
    )
    for i in range(b):
        for j, w in enumerate(words):
            ax = axes[i, j]
            show_image(ax, images[i], "")
            heat = attn[i, j].reshape(g, g).cpu().numpy()
            ax.imshow(
                heat,
                cmap="magma",
                alpha=0.65,
                extent=(-0.5, images.shape[-1] - 0.5, images.shape[-2] - 0.5, -0.5),
                interpolation="nearest",
            )
            if i == 0:
                ax.set_title(f'"{w}"', fontsize=9)
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
    names = list(values)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.15 * len(names)), 3.1), facecolor="white")
    bars = ax.bar(
        names, [values[k] for k in names], color=colors or ["#4c78a8"] * len(names), width=0.62
    )
    for b, k in zip(bars, names):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            fmt.format(values[k]),
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
    fig, ax = plt.subplots(figsize=(max(5, 1.7 * len(groups) + 1.5), 3.2), facecolor="white")
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
    for ax in axes:
        ax.set_xlabel("step", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plot_rollouts(result, scenarios, world, n: int = 6, title: str = "closed-loop rollouts"):
    n = min(n, len(scenarios))
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), facecolor="white")
    for i, ax in enumerate(np.atleast_1d(axes)):
        s = scenarios[i]
        show_image(ax, world.render(s.scene.robot, s.scene.objects), "")
        steps = int(result.steps[i]) if result.success[i] else result.trajectories.shape[1] - 1
        px = result.trajectories[i][: steps + 1] * 31
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


def plot_first_chunks(world, scenario, chunks: dict, ncols: int = 4):
    """For one scene, draw the path of the first action chunk of every sample, one panel per policy.

    ``chunks[name]`` is ``(N, H, ≥2)`` actions (normalised units); paths start at the robot and integrate ``max_step · action``.
    """
    from pi_family.toy.world import COLORS

    names = list(chunks)
    rows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols, figsize=(3.2 * ncols, 3.3 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        for colour, xy in scenario.scene.objects.items():
            ax.add_patch(
                plt.Rectangle(
                    (xy[0] - 0.04, xy[1] - 0.04), 0.08, 0.08, color=COLORS[colour], zorder=2
                )
            )
        start = scenario.scene.robot
        for a in chunks[name]:
            path = start + np.cumsum(np.asarray(a)[:, :2], axis=0) * world.max_step
            ax.plot(*np.vstack([start, path]).T, color="#222222", alpha=0.45, lw=1.2, zorder=3)
            ax.plot(*path[-1], "o", color="#222222", ms=3, alpha=0.6, zorder=3)
        ax.plot(*start, "s", color="#f5e9a8", mec="k", ms=7, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
