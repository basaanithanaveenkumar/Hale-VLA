"""Statistics for evaluating policies: confidence intervals, paired comparisons, sim-vs-real agreement, and a failure taxonomy."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well behaved near 0 and 1, unlike the normal approximation)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for two policies evaluated on the **same** episodes (boolean success arrays).

    Only the discordant episodes matter: ``n10`` (A succeeds, B fails) and ``n01``; under the null each discordant pair is a fair coin.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def required_episodes(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Episodes *per policy* needed to detect ``p1`` vs ``p2`` with an unpaired two-proportion z-test (normal approximation)."""
    z_a, z_b = 1.959964, 0.841621  # two-sided 5 %, 80 % power
    if (alpha, power) != (0.05, 0.8):
        from statistics import NormalDist

        z_a, z_b = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (
        z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(math.ceil(num / (p1 - p2) ** 2))


def bootstrap_difference(
    a: np.ndarray, b: np.ndarray, reps: int = 4000, seed: int = 0
) -> tuple[float, float]:
    """95 % bootstrap interval of ``mean(a) − mean(b)`` for paired episode outcomes (resampling episodes)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (reps, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mmrv(sim, real) -> float:
    """Mean Maximum Rank Violation (SIMPLER, Li et al. 2024): for every policy ``i`` take the largest real-performance gap
    ``|r_i − r_j|`` among the policies ``j`` that the simulator **ranks the other way round**; average over policies. 0 = the simulator never
    mis-ranks policies that differ in reality; small values mean mis-rankings only occur between policies that are close in reality."""
    sim, real = np.asarray(sim, float), np.asarray(real, float)
    n = len(sim)
    total = 0.0
    for i in range(n):
        worst = 0.0
        for j in range(n):
            if (sim[i] - sim[j]) * (real[i] - real[j]) < 0:
                worst = max(worst, abs(real[i] - real[j]))
        total += worst
    return total / n


def classify_failures(
    trajectories, steps, success, goal_xy, other_xy, radius: float = 0.05, near: float = 0.15
) -> dict:
    """Split failed episodes into **wrong object** (ended within ``radius`` of a different object), **near miss** (ended within ``near`` of the
    target), **lost** (ended farther): ``trajectories`` (N, T+1, 2); ``goal_xy`` (N, 2); ``other_xy`` list of (k, 2) arrays."""
    counts = {"wrong object": 0, "near miss": 0, "lost": 0}
    fails = 0
    for traj, ok, g, others in zip(trajectories, success, goal_xy, other_xy):
        if ok:
            continue
        fails += 1
        end = traj[-1]
        if any(np.linalg.norm(end - o) <= radius for o in others):
            counts["wrong object"] += 1
        elif np.linalg.norm(end - g) <= near:
            counts["near miss"] += 1
        else:
            counts["lost"] += 1
    counts["failures"] = fails
    return counts
