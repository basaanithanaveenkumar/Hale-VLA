"""Stage 0 for both models: pre-train the toy VLM on visual question answering before any robot data.

SmolVLA and GR00T N1 both start from a VLM trained on web-scale image-text data (SmolVLM-2, Eagle-2) and keep it (largely)
**frozen**; that only makes sense if the features already encode "where is the red thing". Our toy VLM gets the same head
start from a minute of VQA on point-world images::

    <image> <bos> where is red <sep>   →   "70 30 <eos>"        (x and y in tens of percent)

trained with next-token cross-entropy on the answer tokens only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from pi_family.toy.world import SYNONYMS, PointWorld

from open_vlas.vlm import ToyVLM


@dataclass
class VLMPretrainHistory:
    losses: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    seconds: float = 0.0


class VLMPretrainer:
    """Trains a :class:`ToyVLM` (all of it) on point-world VQA.

    Args:
        vlm: the model to pre-train (must still have all its layers: the language-model head sits on the last one).
        world: renders the images.
        lr / batch_size / seed: optimiser settings.
    """

    def __init__(
        self,
        vlm: ToyVLM,
        world: PointWorld,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
        synonym_prob: float = 0.0,
    ):
        self.vlm, self.world, self.batch_size = vlm, world, batch_size
        # share of questions that name a colour with a synonym ("crimson" = red): the toy's "web world knowledge"
        self.synonym_prob = synonym_prob
        self.rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.AdamW(vlm.parameters(), lr=lr, weight_decay=1e-4)
        self.base_lr = lr
        self.history = VLMPretrainHistory()

    @staticmethod
    def answer(xy) -> str:
        """Coordinates as two number words, e.g. (0.68, 0.31) → "70 30"."""
        x, y = (int(10 * round(10 * float(v))) for v in xy)
        return f"{x} {y}"

    def sample(self, n: int):
        """``n`` (image, question, answer) triples about an object's or the robot's position."""
        images, questions, answers = [], [], []
        for _ in range(n):
            scene = self.world.sample_scene(self.rng, int(self.rng.integers(1, 4)))
            images.append(self.world.render(scene.robot, scene.objects))
            if self.rng.random() < 0.2:
                questions.append(
                    "where is the object"
                )  # "the object" = the robot dot (the vocabulary is tiny)
                answers.append(self.answer(scene.robot))
            else:
                color = str(self.rng.choice(list(scene.objects)))
                word = color
                if self.rng.random() < self.synonym_prob:
                    word = str(self.rng.choice([w for w, c in SYNONYMS.items() if c == color]))
                questions.append(f"where is {word}")
                answers.append(self.answer(scene.objects[color]))
        return torch.from_numpy(np.stack(images)), questions, answers

    def loss(self, images, questions, answers):
        v = self.vlm.vocab
        ans_ids = [v.encode(a) + [v["<eos>"]] for a in answers]
        inputs = self.vlm.build_inputs(images, questions, answers=ans_ids)
        hidden = self.vlm.forward_layers(inputs)[-1]
        logits = self.vlm.logits(hidden)
        # predict answer token j+1 from position j: targets are the answer tokens, positions are shifted by one
        targets = torch.full(inputs.valid.shape, -100, dtype=torch.long, device=hidden.device)
        for i, ids in enumerate(ans_ids):
            start = int(inputs.valid[i].sum()) - len(ids)
            targets[i, start - 1 : start - 1 + len(ids)] = torch.tensor(ids, device=hidden.device)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        hit = (logits.argmax(-1) == targets)[targets != -100].float().mean()
        return loss, hit.item()

    def fit(self, steps: int, log_every: int = 0) -> VLMPretrainHistory:
        self.vlm.train()
        start = time.time()
        for step in range(steps):
            for g in self.optimizer.param_groups:
                g["lr"] = self.base_lr * min(1.0, (step + 1) / 50)
            loss, acc = self.loss(*self.sample(self.batch_size))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), 1.0)
            self.optimizer.step()
            self.history.losses.append(loss.item())
            self.history.accuracy.append(acc)
            if log_every and (step + 1) % log_every == 0:
                print(
                    f"VLM pre-training step {step + 1:5d}  loss {loss.item():.3f}  token accuracy {acc:.2f}"
                )
        self.history.seconds += time.time() - start
        self.vlm.eval()
        return self.history

    @torch.no_grad()
    def ask(self, images: torch.Tensor, question: str) -> list:
        """Greedy answer ("70 30") to the same question for each image."""
        v = self.vlm.vocab
        out = [[] for _ in range(images.shape[0])]
        for _ in range(3):
            inputs = self.vlm.build_inputs(images, [question] * len(out), answers=out)
            hidden = self.vlm.forward_layers(inputs)[-1]
            last = inputs.valid.long().cumsum(1).argmax(1)
            nxt = self.vlm.logits(hidden[torch.arange(len(out)), last]).argmax(-1).tolist()
            for o, t in zip(out, nxt):
                o.append(t)
        return [v.decode([t for t in o if t != v["<eos>"]][:2]) for o in out]
