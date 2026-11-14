"""One episode's world state in the 2-D toy workspace."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ToyScene:
    """Coloured targets, an optional obstacle and the robot, in the unit square ``[0, 1]²``.

    Attributes:
        robot: robot (end-effector) position, shape ``(2,)``.
        targets: colour name → position ``(2,)``.
        obstacle: ``(centre (2,), radius)`` or ``None``.
        visited: colours already reached (their targets are no longer drawn).
    """

    robot: np.ndarray
    targets: dict[str, np.ndarray]
    obstacle: tuple[np.ndarray, float] | None = None
    visited: set[str] = field(default_factory=set)

    def copy(self) -> ToyScene:
        """A deep copy (episodes mutate ``robot`` and ``visited``)."""
        obstacle = None if self.obstacle is None else (self.obstacle[0].copy(), self.obstacle[1])
        return ToyScene(self.robot.copy(), {k: v.copy() for k, v in self.targets.items()},
                        obstacle, set(self.visited))
