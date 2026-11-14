"""Demonstrations for "go to <colour>": observations sampled along expert paths."""

from __future__ import annotations

import numpy as np
import torch

from pi_lab.toy_reach_env import ToyReachEnv
from pi_lab.toy_vocabulary import ToyVocabulary


class ReachDemoDataset:
    """Pre-generates ``(image, instruction, state, action chunk)`` training examples.

    Scenes hold **two red targets** and one blue: "go to blue" has one answer, "go to red" two.
    The expert picks a red target at random, so for the same first observation the
    demonstrations split into two modes. Regression averages them; flow matching must not.

    Each scene yields several examples: the robot is placed at random points along the
    expert's path, the camera image is re-rendered there, and the next ``H`` expert actions
    form the target chunk.

    Args:
        env: the toy environment.
        vocab: vocabulary for instructions.
        scenes: number of scenes to generate.
        per_scene: examples drawn from each scene's path.
    """

    def __init__(self, env: ToyReachEnv, vocab: ToyVocabulary, scenes: int = 1500,
                 per_scene: int = 4) -> None:
        images, texts, states, chunks = [], [], [], []
        for _ in range(scenes):
            scene = env.sample_scene(colours=("red#1", "red#2", "blue"))
            colour = "red" if env.rng.random() < 0.6 else "blue"
            path = env.expert_path(scene, colour)
            starts = np.concatenate([scene.robot[None], path[:-1]], axis=0)
            for i in env.rng.choice(len(starts), size=min(per_scene, len(starts)), replace=False):
                s = scene.copy()
                s.robot = starts[i]
                images.append(env.render(s))
                texts.append(vocab.encode(f"go to {colour}"))
                states.append(starts[i])
                chunks.append(env.chunk_from_path(starts[i], path[i:]))
        self.images = torch.tensor(np.array(images))
        self.text = torch.tensor(texts)
        self.state = torch.tensor(np.array(states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(chunks))

    def __len__(self) -> int:
        return len(self.actions)

    def batch(self, size: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """A random minibatch as a dict of tensors."""
        idx = torch.randint(len(self), (size,), generator=generator).to(self.actions.device)
        return {"images": self.images[idx], "text": self.text[idx], "state": self.state[idx],
                "actions": self.actions[idx]}
