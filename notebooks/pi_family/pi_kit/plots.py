"""Data plots for the π-family notebooks: scenes, demonstrations, attention masks, denoising, rollouts.

Every function draws into matplotlib and calls ``plt.show()``. Images use ``origin="upper"`` and world coordinates
are mapped with ``pixel = xy · (S - 1)`` so trajectories line up with the rendered robot.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

INK = "#222222"
MUTED = "#8a8a8a"
SEGMENT_COLORS = {
    "image": "#4c78a8",
    "observation": "#4c78a8",
    "subgoal": "#72b7b2",
    "text": "#f58518",
    "state": "#54a24b",
    "actions": "#e45756",
    "fast": "#b279a2",
    "subtask": "#eeca3b",
}


def _to_px(xy, size: int = 32):
    return np.asarray(xy) * (size - 1)


def _img(ax, image, title: str = ""):
    img = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def show_image(ax, image, title: str = ""):
    """Draw one (3, S, S) image without axes."""
    _img(ax, image, title)


def show_batch(batch, n: int = 6, title: str = "", show_chunk: bool = True):
    """Observation images with the robot's future path implied by the action chunk (ee actions only)."""
    n = min(n, batch.size)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.6), facecolor="white")
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        label = (
            batch.task[i]
            if not batch.subtask or batch.subtask[i] in (None, batch.task[i])
            else f"{batch.task[i]}\n→ {batch.subtask[i]}"
        )
        _img(ax, batch.images[i], label)
        if (
            show_chunk
            and batch.actions is not None
            and (not batch.control_mode or batch.control_mode[i] == "ee")
        ):
            path = batch.state[i].numpy() + 0.05 * np.cumsum(batch.actions[i].numpy(), 0)
            path = np.vstack([batch.state[i].numpy(), path])
            px = _to_px(path)
            ax.plot(px[:, 0], px[:, 1], "-", color="white", lw=1.2, alpha=0.9)
            ax.plot(px[-1, 0], px[-1, 1], "o", color="white", ms=3)
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def show_episode(dataset, episode, steps: Sequence[int] = None, title: str = ""):
    """A filmstrip of one demonstration with its subtask label per frame."""
    steps = (
        list(steps)
        if steps is not None
        else list(np.linspace(0, episode.active_steps, 5).astype(int))
    )
    fig, axes = plt.subplots(1, len(steps), figsize=(2.2 * len(steps), 2.6), facecolor="white")
    for ax, t in zip(np.atleast_1d(axes), steps):
        _img(
            ax,
            dataset.frame(episode, t),
            f"t = {t}\n{episode.subtask[min(t, len(episode.subtask) - 1)]}",
        )
    fig.suptitle(
        title or f'"{episode.task}" ({episode.style.name})', fontsize=11, x=0.01, ha="left"
    )
    plt.tight_layout()
    plt.show()


def plot_token_layout(
    prefix, suffix_len: int, example: int = 0, vocab=None, title: str = "", max_label: int = 60
):
    """One row of coloured cells: what each token of one example's sequence is."""
    labels, colors = [], []
    ids = prefix.ids[example].tolist()
    valid = prefix.valid[example].tolist()
    for seg_name, start, end in prefix.segments:
        for j in range(start, end):
            if not valid[j]:
                continue
            if ids[j] == -2:
                labels.append("goal" if seg_name == "subgoal" else "img")
                colors.append(SEGMENT_COLORS.get(seg_name, "#4c78a8"))
            elif ids[j] == -1:
                labels.append("q")
                colors.append(SEGMENT_COLORS["state"])
            else:
                word = vocab.decode([ids[j]]) if vocab is not None else str(ids[j])
                is_fast = vocab is not None and ids[j] >= vocab.fast_offset
                is_state = vocab is not None and vocab.state_offset <= ids[j] < vocab.fast_offset
                labels.append(word)
                colors.append(
                    SEGMENT_COLORS["fast"]
                    if is_fast or word in ("<boa>", "<eoa>")
                    else SEGMENT_COLORS["state"]
                    if is_state
                    else SEGMENT_COLORS["text"]
                )
    labels += ["a"] * suffix_len
    colors += [SEGMENT_COLORS["actions"]] * suffix_len
    n = len(labels)
    fig, ax = plt.subplots(figsize=(min(16, 0.24 * n + 1), 1.5), facecolor="white")
    for j, (lab, col) in enumerate(zip(labels, colors)):
        ax.add_patch(plt.Rectangle((j, 0), 0.95, 1, color=col))
        if n <= max_label or lab not in ("img", "goal", "a"):
            ax.text(j + 0.47, -0.15, lab, rotation=90, ha="center", va="top", fontsize=7)
    ax.set_xlim(0, n)
    ax.set_ylim(-2.2, 1.1)
    ax.axis("off")
    ax.set_title(
        title or f"token sequence of example {example} ({n} tokens)", fontsize=10, loc="left"
    )
    plt.tight_layout()
    plt.show()


def plot_attention_mask(
    prefix_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    prefix,
    example: int = 0,
    suffix_labels: list | None = None,
    title: str = "attention mask (row = query, column = key)",
):
    """The full joint mask of one example; shaded bands mark the token groups."""
    lp = prefix_mask.shape[-1]
    ls = suffix_mask.shape[1]
    full = np.zeros((lp + ls, lp + ls), bool)
    full[:lp, :lp] = prefix_mask[example].cpu().numpy()
    full[lp:] = suffix_mask[example].cpu().numpy()
    valid = np.concatenate([prefix.valid[example].cpu().numpy(), np.ones(ls, bool)])
    keep = np.nonzero(valid)[0]
    full = full[np.ix_(keep, keep)]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="white")
    ax.imshow(full, cmap=ListedColormap(["#f3f3f3", "#2b6cb0"]), interpolation="nearest")
    # group boundaries
    bounds, names = [], []
    pos = 0
    for name, start, end in prefix.segments:
        n_valid = int(prefix.valid[example, start:end].sum())
        if n_valid:
            bounds.append(pos)
            names.append(name)
            pos += n_valid
    for name, n in suffix_labels or [("actions", ls)]:
        bounds.append(pos)
        names.append(name)
        pos += n
    for b0, name in zip(bounds, names):
        ax.axhline(b0 - 0.5, color="#e45756", lw=0.8)
        ax.axvline(b0 - 0.5, color="#e45756", lw=0.8)
    centers = [(b0 + b1) / 2 for b0, b1 in zip(bounds, bounds[1:] + [pos])]
    ax.set_xticks(centers)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(centers)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def chunk_to_path(state, chunk, max_step: float = 0.05, joint_map=None):
    a = np.asarray(chunk)
    if joint_map is not None:
        a = a @ np.asarray(joint_map).T
    return np.vstack([np.asarray(state), np.asarray(state) + max_step * np.cumsum(a, 0)])


def plot_denoising(
    batch,
    trajectory: list,
    example: int = 0,
    steps_to_show: Sequence[int] = None,
    title: str = "denoising an action chunk: noise → plan",
):
    """The chunk at several Euler steps, drawn as the path it would make the robot follow."""
    n = len(trajectory) - 1
    steps_to_show = (
        list(steps_to_show)
        if steps_to_show is not None
        else sorted({0, n // 4, n // 2, 3 * n // 4, n})
    )
    fig, axes = plt.subplots(
        1, len(steps_to_show), figsize=(2.5 * len(steps_to_show), 2.8), facecolor="white"
    )
    state = batch.state[example].numpy()
    for ax, k in zip(np.atleast_1d(axes), steps_to_show):
        _img(ax, batch.images[example], f"step {k}/{n}  (τ = {1 - k / n:.2f})")
        px = _to_px(chunk_to_path(state, trajectory[k][example].cpu().numpy()))
        ax.plot(px[:, 0], px[:, 1], "-", color="white", lw=1.4)
        ax.scatter(
            px[1:, 0], px[1:, 1], c=np.linspace(0, 1, len(px) - 1), cmap="autumn", s=8, zorder=3
        )
        ax.set_xlim(-0.5, 31.5)
        ax.set_ylim(31.5, -0.5)
    fig.suptitle(f'{title} - "{batch.task[example]}"', fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plot_samples(batch, samples: torch.Tensor, example: int = 0, title: str = "", ax=None):
    """Many sampled chunks for the same observation (shows multimodality)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="white")
    _img(ax, batch.images[example], title or batch.task[example])
    state = batch.state[example].numpy()
    for s in samples:
        px = _to_px(chunk_to_path(state, s.cpu().numpy()))
        ax.plot(px[:, 0], px[:, 1], "-", color="white", lw=0.8, alpha=0.5)
        ax.plot(px[-1, 0], px[-1, 1], "o", color="#ffd166", ms=2.5)
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(31.5, -0.5)
    if own:
        plt.tight_layout()
        plt.show()


def plot_rollouts(
    result, scenarios, world, n: int = 6, title: str = "closed-loop rollouts", labels=None
):
    """First frame of each scenario with the executed path; green title = success."""
    n = min(n, len(scenarios))
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), facecolor="white")
    for i, ax in enumerate(np.atleast_1d(axes)):
        s = scenarios[i]
        cue = s.scene.marked
        _img(ax, world.render(s.scene.robot, s.scene.objects, set(), cue), "")
        steps = int(result.steps[i]) if result.success[i] else len(result.trajectories[i]) - 1
        px = _to_px(result.trajectories[i][: steps + 1])
        ax.plot(px[:, 0], px[:, 1], "-", color="white", lw=1.3)
        ax.plot(px[-1, 0], px[-1, 1], "x", color="white", ms=6)
        ok = bool(result.success[i])
        text = labels[i] if labels else s.task
        ax.set_title(
            f"{text}\n{'✓' if ok else '✗'} {steps} steps",
            fontsize=8.5,
            color="#2f855a" if ok else "#c53030",
        )
        ax.set_xlim(-0.5, 31.5)
        ax.set_ylim(31.5, -0.5)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def bar_compare(
    values: dict, ylabel: str, title: str = "", ylim=None, fmt: str = "{:.2f}", colors=None
):
    """Simple labelled bar chart for ablations."""
    names = list(values)
    fig, ax = plt.subplots(figsize=(max(4, 1.25 * len(names)), 3), facecolor="white")
    bars = ax.bar(
        names, [values[k] for k in names], color=colors or ["#4c78a8"] * len(names), width=0.6
    )
    for b, k in zip(bars, names):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            fmt.format(values[k]),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    plt.xticks(fontsize=8.5)
    plt.tight_layout()
    plt.show()


def plot_loss(history, keys=("flow",), title: str = "training"):
    fig, axes = plt.subplots(
        1, len(keys) + 1, figsize=(3.6 * (len(keys) + 1), 2.7), facecolor="white"
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
