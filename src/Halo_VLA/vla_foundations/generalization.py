"""Generalisation experiments: do VLAs use language?, embodiment descriptors, and chained (long-horizon) skills.

* **Shortcuts** - :func:`shortcut_demos` builds ``go to <colour>`` demonstrations in which each colour always sits in the same band of the
  table (red left, green middle, blue right) except with probability ``counterfactual``. A policy may then solve the task by going to a
  *place* without reading the instruction. :func:`language_sensitivity`
  measures whether swapping the instruction changes what the policy does.
* **Morphology** - :class:`MorphPolicy` maps (displacement to the goal, embodiment) to an action for robots whose action→motion map is a
  random 2x2 matrix ``J``. The embodiment is given either as an **ID** (a learned vector per training embodiment) or as a **descriptor**
  (the entries of ``J``: kinematic structure as an input).
* **Chains** - :func:`run_chain` executes ``go to A, then B, then C`` with a single-skill policy and a scripted planner.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from open_vlas.embodiments import EMBODIMENTS
from open_vlas.evaluation import EmbodiedEvaluator, EmbodiedScenario
from pi_family.toy import DemoGenerator, PointWorld
from pi_family.toy.episodes import Episode
from pi_family.toy.world import COLORS, Scene

# ------------------------------------------------------------------------------------------------ shortcuts

SLOTS = {
    "red": 0.2,
    "green": 0.5,
    "blue": 0.8,
}  # the usual x position of each colour in the "shortcut" data


def slot_scene(world: PointWorld, rng: np.random.Generator, permuted: bool):
    """Three objects in three vertical bands (x ≈ 0.2 / 0.5 / 0.8, random y). In the *usual* arrangement red is left, green middle,
    blue right; ``permuted=True`` assigns the colours to the bands at random (a counterfactual layout)."""
    colours = list(rng.permutation(list(COLORS))) if permuted else list(COLORS)
    for _ in range(1000):
        objects = {
            c: np.array(
                [SLOTS["red"] + 0.3 * k + rng.uniform(-0.07, 0.07), rng.uniform(0.12, 0.88)],
                dtype=np.float32,
            )
            for k, c in enumerate(colours)
        }
        robot = rng.uniform(0.1, 0.9, 2).astype(np.float32)
        if all(np.linalg.norm(robot - xy) > 0.2 for xy in objects.values()):
            return Scene(objects=objects, robot=robot)
    raise RuntimeError("could not sample a scene")


def shortcut_demos(
    gen: DemoGenerator, n: int, rng: np.random.Generator, counterfactual: float
) -> list[Episode]:
    """``n`` demonstrations of ``go to <colour>``; with probability ``counterfactual`` the colours are placed in the bands at random,
    otherwise in the usual arrangement (red left, green middle, blue right), where "go to red" can be solved by *going left*."""
    out = []
    for _ in range(n):
        scene = slot_scene(gen.world, rng, permuted=bool(rng.random() < counterfactual))
        target = str(rng.choice(list(COLORS)))
        out.append(
            gen.demo("reach", rng, "expert", task=(scene, f"go to {target}", [target], [target]))
        )
    return out


def shortcut_scenarios(
    gen: DemoGenerator, n: int, seed: int, permuted: bool
) -> list[EmbodiedScenario]:
    """Evaluation scenes in the usual arrangement (``permuted=False``: the shortcut works) or with permuted colours (only language works)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        scene = slot_scene(gen.world, rng, permuted)
        target = str(rng.choice(list(COLORS)))
        out.append(EmbodiedScenario(scene, f"go to {target}", target, "point2d"))
    return out


@torch.no_grad()
def language_sensitivity(policy, world: PointWorld, gen: DemoGenerator, n: int, seed: int) -> float:
    """Fraction of scenes in which naming a *different* object changes the first action by more than 60°.

    Only scenes whose two named objects lie in clearly different directions from the robot (≥ 70° apart) are used, so a policy that
    follows the instruction scores 1.0 and one that ignores it scores 0.0. ``policy(batch)`` must be deterministic.
    """
    rng = np.random.default_rng(seed)
    ev = EmbodiedEvaluator(world, 1, 1, 1)
    first, second = [], []
    while len(first) < n:
        scene = world.sample_scene(rng, colors=list(rng.permutation(list(COLORS))))
        a, b = (str(c) for c in rng.permutation(list(COLORS))[:2])
        da, db = scene.objects[a] - scene.robot, scene.objects[b] - scene.robot
        if float(da @ db) / (np.linalg.norm(da) * np.linalg.norm(db)) > 0.34:  # cos 70°
            continue
        first.append(EmbodiedScenario(scene, f"go to {a}", a, "point2d"))
        second.append(EmbodiedScenario(scene, f"go to {b}", b, "point2d"))

    def action(scenarios):
        robots = np.stack([s.scene.robot for s in scenarios])[:, None, :].astype(np.float32)
        return (
            policy(ev.observe(scenarios, robots, np.full(len(scenarios), -1.0), 0))[:, 0, :2]
            .cpu()
            .numpy()
        )

    u, v = action(first), action(second)
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-8)
    return float((cos < 0.5).mean())  # cos 60° = 0.5


# ------------------------------------------------------------------------------------------------ morphology


def random_jacobian(rng: np.random.Generator) -> np.ndarray:
    """A random well-conditioned 2x2 action→motion map (a rotation, a reflection or not, and two axis gains in [0.6, 1.6])."""
    theta = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    gains = np.diag(rng.uniform(0.6, 1.6, 2))
    flip = np.diag([1.0, rng.choice([-1.0, 1.0])])
    return (rot @ gains @ flip).astype(np.float32)


class MorphPolicy(nn.Module):
    """MLP policy ``(d, embodiment) → a`` with ``d`` the displacement to the goal (in units of one step).

    ``conditioning='id'``: a learned embedding per training embodiment (an unseen ID has no meaning).
    ``conditioning='descriptor'``: the four entries of the embodiment's ``J`` are an input (kinematic structure as an inductive bias).
    """

    def __init__(self, conditioning: str, num_ids: int = 1, width: int = 128, id_dim: int = 8):
        super().__init__()
        if conditioning not in ("id", "descriptor"):
            raise ValueError("conditioning must be 'id' or 'descriptor'")
        self.conditioning = conditioning
        self.embedding = nn.Embedding(num_ids, id_dim) if conditioning == "id" else None
        cond_dim = id_dim if conditioning == "id" else 4
        self.net = nn.Sequential(
            nn.Linear(2 + cond_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

    def condition(self, ids: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids) if self.conditioning == "id" else jac.flatten(1)

    def forward(self, d: torch.Tensor, ids: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([d, self.condition(ids, jac)], 1))


def train_morph(
    model: MorphPolicy, jacobians: np.ndarray, steps: int = 3000, batch: int = 256, seed: int = 0
) -> float:
    """Behaviour cloning on random displacements: the expert action solves ``J a = d`` (clipped to one step)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    jt = torch.from_numpy(jacobians)
    for _ in range(steps):
        ids = torch.from_numpy(rng.integers(0, len(jacobians), batch))
        d = torch.from_numpy(rng.uniform(-1, 1, (batch, 2)).astype(np.float32))
        j = jt[ids]
        target = torch.linalg.solve(j, d[:, :, None])[:, :, 0].clamp(-1.5, 1.5)
        loss = (model(d, ids, j) - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


@torch.no_grad()
def morph_success(
    model: MorphPolicy,
    jac: np.ndarray,
    world: PointWorld,
    n: int = 200,
    seed: int = 0,
    ident: int = 0,
) -> float:
    """Closed loop: the robot with motion map ``jac`` reaches random goals within 40 steps (``ident`` is the ID given to an ID-conditioned model)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, (n, 2)).astype(np.float32)
    goal = rng.uniform(0.1, 0.9, (n, 2)).astype(np.float32)
    j = torch.from_numpy(np.repeat(jac[None], n, 0))
    ids = torch.full((n,), ident, dtype=torch.long)
    done = np.zeros(n, bool)
    for _ in range(40):
        d = torch.from_numpy((goal - p) / world.max_step).clamp(-1.0, 1.0)
        a = model(d, ids, j).clamp(-1.5, 1.5).numpy()
        move = (a @ jac.T) * world.max_step
        p = np.where(done[:, None], p, np.clip(p + move, 0, 1)).astype(np.float32)
        done |= np.linalg.norm(p - goal, axis=1) <= world.success_radius
    return float(done.mean())


# ------------------------------------------------------------------------------------------------ chains


@torch.no_grad()
def run_chain(
    world: PointWorld,
    policy,
    scenarios: list,
    sequences: list,
    per_skill_steps: int = 30,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Execute ``go to A`` then ``go to B`` … for every scenario with a one-action-per-call ``policy``.

    A scripted planner issues the next instruction as soon as the current object is reached (within the success radius); a skill that
    does not finish within ``per_skill_steps`` ends the episode. Returns the number of *completed* skills per scenario.
    """
    rng = np.random.default_rng(seed)
    n = len(scenarios)
    ev = EmbodiedEvaluator(world, 1, 1, 1)
    robots = np.stack([s.scene.robot for s in scenarios]).astype(np.float32)
    stage = np.zeros(n, int)
    spent = np.zeros(n, int)
    alive = np.ones(n, bool)
    emb = EMBODIMENTS["point2d"]
    total = max(len(s) for s in sequences) * per_skill_steps
    for _ in range(total):
        if not alive.any():
            break
        cur = [
            EmbodiedScenario(
                s.scene,
                f"go to {sequences[i][min(stage[i], len(sequences[i]) - 1)]}",
                sequences[i][min(stage[i], len(sequences[i]) - 1)],
                "point2d",
            )
            for i, s in enumerate(scenarios)
        ]
        a = policy(ev.observe(cur, robots[:, None, :], np.full(n, -1.0), 0))[:, 0, :2].cpu().numpy()
        if noise:
            a = a + rng.normal(0, noise, a.shape)
        for i in range(n):
            if not alive[i] or stage[i] >= len(sequences[i]):
                continue
            robots[i] = world.step(robots[i], emb.ee_from_action(a[i]), "ee")
            spent[i] += 1
            goal = scenarios[i].scene.objects[sequences[i][stage[i]]]
            if np.linalg.norm(robots[i] - goal) <= world.success_radius:
                stage[i] += 1
                spent[i] = 0
            elif spent[i] >= per_skill_steps:
                alive[i] = False
    return stage


def adapt_new_id(
    model: MorphPolicy, jac: np.ndarray, n_demos: int, steps: int = 400, seed: int = 0
) -> int:
    """Few-shot adaptation of an ID-conditioned model to a **new** embodiment: add one embedding row and train only that row on ``n_demos``
    demonstrations of the new robot (``d → a`` pairs). Returns the new ID."""
    if model.embedding is None:
        raise ValueError("only ID-conditioned models have an embedding to extend")
    old = model.embedding
    new = nn.Embedding(old.num_embeddings + 1, old.embedding_dim)
    with torch.no_grad():
        new.weight[:-1] = old.weight
        new.weight[-1] = old.weight.mean(0)
    model.embedding = new
    for p in model.parameters():
        p.requires_grad_(False)
    new.weight.requires_grad_(True)
    rng = np.random.default_rng(seed)
    d = torch.from_numpy(rng.uniform(-1, 1, (n_demos, 2)).astype(np.float32))
    j = torch.from_numpy(np.repeat(jac[None], n_demos, 0))
    target = torch.linalg.solve(j, d[:, :, None])[:, :, 0].clamp(-1.5, 1.5)
    ids = torch.full((n_demos,), old.num_embeddings, dtype=torch.long)
    opt = torch.optim.Adam([new.weight], lr=3e-2)
    for _ in range(steps):
        loss = (model(d, ids, j) - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return old.num_embeddings
