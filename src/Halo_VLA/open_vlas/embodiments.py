"""Robot embodiments for the cross-embodiment experiments (GR00T N1).

All of them move the same point in the same toy world, but they differ in what the policy *sees and says*:

=========  ===========  ============  =============================================================
name       state dim    action dim    what is different
=========  ===========  ============  =============================================================
point2d    2 (x, y)     2             action = end-effector displacement
joint2d    2 (x, y)     2             action = joint-space command (a fixed linear map of the above)
gripper3d  3 (x, y, g)  3             extra gripper command that must close near the target
lapa       0            4             not a robot: the "action" is a *latent action* from video
=========  ===========  ============  =============================================================

GR00T N1 handles such differences with one MLP per embodiment in front of and behind a shared transformer; this module only
describes the embodiments and converts demonstrations to/from each action space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pi_family.toy.world import JOINT_MAP, JOINT_MAP_INV

MAX_STATE_DIM = 3  # state vectors are zero-padded to this size inside a batch
MAX_ACTION_DIM = 4  # action vectors are zero-padded to this size (the latent action is 4-dim)
LATENT_DIM = 4
GRIP_RADIUS = 0.15  # the gripper command flips to "close" within this distance of the target


class Embodiment(ABC):
    """One robot body: dimensions, and conversions between the world's end-effector motion and its own action space."""

    name: str = ""
    id: int = -1
    state_dim: int = 0
    action_dim: int = 0
    is_robot: bool = True

    @abstractmethod
    def state(self, xy: np.ndarray, grip: float) -> np.ndarray:
        """Proprioception (state_dim,) from the point's position and the last gripper command."""

    @abstractmethod
    def action_from_ee(
        self, ee_action: np.ndarray, xy: np.ndarray, target_xy: np.ndarray
    ) -> np.ndarray:
        """Express a normalised end-effector displacement in this embodiment's action space."""

    @abstractmethod
    def ee_from_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`action_from_ee`: the 2-D displacement the world should apply."""

    def grip_of(self, action: np.ndarray) -> float:
        """Gripper state after executing ``action`` (embodiments without a gripper keep -1)."""
        return -1.0


class Point2D(Embodiment):
    """The default point robot: state (x, y), action = normalised end-effector displacement."""

    name, id, state_dim, action_dim = "point2d", 0, 2, 2

    def state(self, xy, grip):
        return np.asarray(xy, dtype=np.float32)

    def action_from_ee(self, ee_action, xy, target_xy):
        return np.asarray(ee_action, dtype=np.float32)

    def ee_from_action(self, action):
        return np.asarray(action[:2], dtype=np.float32)


class Joint2D(Embodiment):
    """Same body, joint-space commands: ``ee = J @ a`` for a fixed, non-trivial 2x2 ``J``."""

    name, id, state_dim, action_dim = "joint2d", 1, 2, 2

    def state(self, xy, grip):
        return np.asarray(xy, dtype=np.float32)

    def action_from_ee(self, ee_action, xy, target_xy):
        return (JOINT_MAP_INV @ np.asarray(ee_action, dtype=np.float32)).astype(np.float32)

    def ee_from_action(self, action):
        return (JOINT_MAP @ np.asarray(action[:2], dtype=np.float32)).astype(np.float32)


class Gripper3D(Embodiment):
    """Point robot with a gripper: state (x, y, g_prev), action (dx, dy, g_cmd) with g_cmd = +1 near the target, else -1."""

    name, id, state_dim, action_dim = "gripper3d", 2, 3, 3

    def state(self, xy, grip):
        return np.array([xy[0], xy[1], grip], dtype=np.float32)

    def action_from_ee(self, ee_action, xy, target_xy):
        near = np.linalg.norm(np.asarray(target_xy) - np.asarray(xy)) < GRIP_RADIUS
        return np.array([ee_action[0], ee_action[1], 1.0 if near else -1.0], dtype=np.float32)

    def ee_from_action(self, action):
        return np.asarray(action[:2], dtype=np.float32)

    def grip_of(self, action):
        return 1.0 if action[2] > 0 else -1.0


class LatentAction(Embodiment):
    """GR00T N1's "LAPA" pseudo-embodiment: video without actions, labelled with latent actions (see :mod:`latent_actions`)."""

    name, id, state_dim, action_dim, is_robot = "lapa", 3, 0, LATENT_DIM, False

    def state(self, xy, grip):
        return np.zeros(0, dtype=np.float32)

    def action_from_ee(self, ee_action, xy, target_xy):
        raise NotImplementedError("a latent-action 'embodiment' has no action space of its own")

    def ee_from_action(self, action):
        raise NotImplementedError("a latent-action 'embodiment' cannot act in the world")


EMBODIMENTS: dict[str, Embodiment] = {
    e.name: e for e in (Point2D(), Joint2D(), Gripper3D(), LatentAction())
}
NUM_EMBODIMENTS = len(EMBODIMENTS)


def embodied_arrays(ep, emb: Embodiment) -> tuple[np.ndarray, np.ndarray]:
    """Convert one demonstration to ``(states (T + 1, state_dim), actions (T, action_dim))`` of ``emb``.

    The gripper state at step t is the command given at t - 1 (initially "open", -1), exactly as a real gripper reports it.
    """
    target = ep.scene.objects[ep.targets[0]]
    states, actions, grip = [], [], -1.0
    for t in range(len(ep.actions)):
        states.append(emb.state(ep.robot[t], grip))
        a = emb.action_from_ee(ep.actions[t], ep.robot[t], target)
        actions.append(a)
        grip = emb.grip_of(a)
    states.append(emb.state(ep.robot[len(ep.actions)], grip))
    return np.stack(states), np.stack(actions)
