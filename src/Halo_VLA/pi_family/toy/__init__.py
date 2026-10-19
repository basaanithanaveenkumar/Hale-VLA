"""The point world: a toy robot environment small enough to train every π generation on a CPU in about a minute.

* :class:`PointWorld` - rendering (32×32 RGB) and dynamics (``ee`` / ``joint`` control)
* :class:`DemoGenerator` - task sampling and scripted demonstrations in several styles (quality / speed / mistakes)
* :class:`ChunkDataset` / :class:`PiBatch` - (observation, prompt, action chunk) training batches
* :class:`ClosedLoopEvaluator` - chunked closed-loop rollouts with optional inference delay
"""

from pi_family.toy.dataset import ChunkDataset, PiBatch, collate
from pi_family.toy.episodes import STYLES, DemoGenerator, Episode, EpisodeStyle
from pi_family.toy.rollout import ClosedLoopEvaluator, RolloutResult, Scenario, make_scenarios
from pi_family.toy.world import COLORS, JOINT_MAP, SYNONYMS, PointWorld, Scene

__all__ = [
    "COLORS",
    "JOINT_MAP",
    "STYLES",
    "SYNONYMS",
    "ChunkDataset",
    "ClosedLoopEvaluator",
    "DemoGenerator",
    "Episode",
    "EpisodeStyle",
    "PiBatch",
    "PointWorld",
    "RolloutResult",
    "Scenario",
    "Scene",
    "collate",
    "make_scenarios",
]
