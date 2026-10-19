"""Extra task families for the backbone experiments.

* ``go to <colour>``: bind a colour *word* to an object (language grounding).
* ``go to <left|right|top|bottom>``: pick the object at an extreme of the scene (a *spatial relation* between objects: nothing in the
  word names a colour, so the model must compare object positions).

Both use three objects (one of each colour) so that the relational task has something to compare.
"""

from __future__ import annotations

import numpy as np
from open_vlas.evaluation import EmbodiedScenario
from pi_family.toy import DemoGenerator, PointWorld
from pi_family.toy.world import COLORS

SIDES = ("left", "right", "top", "bottom")


def side_task(
    world: PointWorld, rng: np.random.Generator, side: str | None = None, margin: float = 0.12
):
    """A scene with three objects and the instruction ``go to <side>``; returns ``(scene, text, [target colour], [target colour])``."""
    side = side or str(rng.choice(SIDES))
    axis = 0 if side in ("left", "right") else 1
    pick = np.argmin if side in ("left", "top") else np.argmax
    for _ in range(200):
        scene = world.sample_scene(rng, colors=list(rng.permutation(list(COLORS))))
        coords = np.array([scene.objects[c][axis] for c in scene.objects])
        order = np.sort(coords)
        gap = (order[1] - order[0]) if side in ("left", "top") else (order[-1] - order[-2])
        if gap > margin:
            break
    target = list(scene.objects)[int(pick(coords))]
    return scene, f"go to {side}", [target], [target]


def make_side_demos(generator: DemoGenerator, n: int, rng: np.random.Generator) -> list:
    """``n`` expert demonstrations of ``go to <side>``."""
    return [
        generator.demo("reach", rng, "expert", task=side_task(generator.world, rng))
        for _ in range(n)
    ]


def make_side_scenarios(
    generator: DemoGenerator, n: int, seed: int, embodiment: str = "point2d"
) -> list:
    """``n`` reproducible evaluation scenes of ``go to <side>``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        scene, text, targets, _ = side_task(generator.world, rng)
        out.append(EmbodiedScenario(scene, text, targets[0], embodiment))
    return out


def colour_scenarios_three(
    generator: DemoGenerator,
    n: int,
    seed: int,
    embodiment: str = "point2d",
    word: str | None = None,
) -> list:
    """Three-object ``go to <colour>`` scenes; ``word`` overrides the instruction's colour word (e.g. a synonym) while the target stays the true colour."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        target = str(rng.choice(list(COLORS)))
        world_scene = generator.world.sample_scene(rng, colors=list(rng.permutation(list(COLORS))))
        text = f"go to {word if word else target}"
        out.append(EmbodiedScenario(world_scene, text, target, embodiment))
    return out


def make_colour_demos_three(generator: DemoGenerator, n: int, rng: np.random.Generator) -> list:
    """Three-object ``go to <colour>`` demonstrations."""
    demos = []
    for _ in range(n):
        target = str(rng.choice(list(COLORS)))
        scene = generator.world.sample_scene(rng, colors=list(rng.permutation(list(COLORS))))
        demos.append(
            generator.demo(
                "reach", rng, "expert", task=(scene, f"go to {target}", [target], [target])
            )
        )
    return demos


def synonym_scenarios(
    generator: DemoGenerator, n: int, seed: int, embodiment: str = "point2d"
) -> list:
    """Three-object scenes whose instruction names the target colour with a **synonym** ("go to crimson").

    The robot demonstrations only ever use the plain colour words, so following these instructions requires language knowledge that
    came from somewhere else (the VQA pre-training data), the toy version of RT-2's "web knowledge transfer".
    """
    from pi_family.toy.world import SYNONYMS

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        target = str(rng.choice(list(COLORS)))
        word = str(rng.choice([w for w, c in SYNONYMS.items() if c == target]))
        scene = generator.world.sample_scene(rng, colors=list(rng.permutation(list(COLORS))))
        out.append(EmbodiedScenario(scene, f"go to {word}", target, embodiment))
    return out


def colour_demos(
    generator: DemoGenerator, n: int, rng: np.random.Generator, colours=tuple(COLORS)
) -> list:
    """Three-object ``go to <colour>`` demonstrations whose *target* is drawn from ``colours`` (the other colours are distractors)."""
    demos = []
    for _ in range(n):
        target = str(rng.choice(list(colours)))
        scene = generator.world.sample_scene(rng, colors=list(rng.permutation(list(COLORS))))
        demos.append(
            generator.demo(
                "reach", rng, "expert", task=(scene, f"go to {target}", [target], [target])
            )
        )
    return demos


def colour_scenarios(
    generator: DemoGenerator,
    n: int,
    seed: int,
    colours=tuple(COLORS),
    synonyms: bool = False,
    embodiment: str = "point2d",
) -> list:
    """Three-object evaluation scenes whose target is drawn from ``colours``; with ``synonyms`` the instruction uses a synonym word."""
    from pi_family.toy.world import SYNONYMS

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        target = str(rng.choice(list(colours)))
        word = (
            str(rng.choice([w for w, c in SYNONYMS.items() if c == target])) if synonyms else target
        )
        scene = generator.world.sample_scene(rng, colors=list(rng.permutation(list(COLORS))))
        out.append(EmbodiedScenario(scene, f"go to {word}", target, embodiment))
    return out
