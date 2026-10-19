"""Stage 0: pre-train the toy backbone as a tiny *vision-language model* before any robot data.

Every π model starts from a pre-trained VLM (PaliGemma for π0 / π0.5, Gemma3 for π0.7): the backbone already knows
how to find "the red thing" in an image and say where it is. Our toy backbone starts from random weights, and
learning perception *only* through the flow-matching gradient of a fresh action expert is slow and fragile. This
module gives it the equivalent of web pre-training: visual question answering on point-world images,

    <image> <bos> where is red <sep>  →  "70 30 <eos>"     (x and y of the object, rounded to tens of %)

trained with next-token cross-entropy through the policy's own image encoder, embeddings, backbone and LM head.
The answer words are ordinary vocabulary words, so nothing robot-specific is added to the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from pi_family.policy import ImageBlock, PiPolicy, TokenStream
from pi_family.toy.world import COLORS, PointWorld


@dataclass
class PretrainHistory:
    losses: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    seconds: float = 0.0


class VQAPretrainer:
    """Trains a :class:`PiPolicy`'s VLM parts (vision, embeddings, backbone, LM head) on point-world VQA."""

    def __init__(
        self,
        policy: PiPolicy,
        world: PointWorld,
        lr: float = 2e-3,
        batch_size: int = 64,
        seed: int = 0,
    ):
        self.policy, self.world, self.batch_size = policy, world, batch_size
        self.rng = np.random.default_rng(seed)
        params = [
            p
            for n, p in policy.named_parameters()
            if not n.startswith(("transformer.expert", "action"))
        ]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        self.base_lr = lr
        self.history = PretrainHistory()

    @staticmethod
    def answer(xy) -> str:
        """Coordinates as two number words, e.g. (0.68, 0.31) → "70 30"."""
        x, y = (int(10 * round(10 * float(v))) for v in xy)
        return f"{x} {y}"

    def sample(self, n: int):
        """``n`` (image, question, answer) triples; questions ask about an object or the robot itself."""
        images, questions, answers = [], [], []
        for _ in range(n):
            scene = self.world.sample_scene(self.rng, int(self.rng.integers(1, 4)))
            images.append(self.world.render(scene.robot, scene.objects))
            if self.rng.random() < 0.2:
                questions.append("where is the object")  # "object" = the robot dot here ...
                answers.append(
                    self.answer(scene.robot)
                )  # ... the vocabulary is tiny, so we reuse it
            else:
                color = str(self.rng.choice(list(scene.objects)))
                questions.append(f"where is {color}")
                answers.append(self.answer(scene.objects[color]))
        return torch.from_numpy(np.stack(images)), questions, answers

    def loss(self, images, questions, answers):
        p, v = self.policy, self.policy.vocab
        device = p.prefix_position.weight.device
        streams = []
        for q, a in zip(questions, answers):
            s = TokenStream().add([v["<bos>"]] + v.encode(q) + [v["<sep>"]], causal=False)
            s.add(v.encode(a) + [v["<eos>"]], causal=True, target=True)
            streams.append(s)
        prefix = p.assemble_prefix([ImageBlock(p.encode_observation(images.to(device)))], streams)
        hidden, _ = p.encode_prefix(prefix)
        logits = p.lm_head(hidden)
        loss = F.cross_entropy(logits.flatten(0, 1), prefix.targets.flatten(), ignore_index=-100)
        hit = (logits.argmax(-1) == prefix.targets)[prefix.targets != -100].float().mean()
        return loss, hit.item()

    def fit(self, steps: int, log_every: int = 0) -> PretrainHistory:
        self.policy.train()
        start = time.time()
        for step in range(steps):
            for group in self.optimizer.param_groups:
                group["lr"] = self.base_lr * min(1.0, (step + 1) / 50)
            loss, acc = self.loss(*self.sample(self.batch_size))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            self.history.losses.append(loss.item())
            self.history.accuracy.append(acc)
            if log_every and (step + 1) % log_every == 0:
                print(
                    f"VQA pre-training step {step + 1:5d}  loss {loss.item():.3f}  token accuracy {acc:.2f}"
                )
        self.history.seconds += time.time() - start
        self.policy.eval()
        return self.history

    @torch.no_grad()
    def ask(self, images: torch.Tensor, question: str) -> list:
        """Greedy answer to the same question for each image (for the notebooks)."""
        p, v = self.policy, self.policy.vocab
        out = [[] for _ in range(images.shape[0])]
        for _ in range(3):
            streams = [
                TokenStream()
                .add([v["<bos>"]] + v.encode(question) + [v["<sep>"]], causal=False)
                .add(o, causal=True)
                for o in out
            ]
            prefix = p.assemble_prefix([ImageBlock(p.encode_observation(images))], streams)
            hidden, _ = p.encode_prefix(prefix)
            nxt = (
                p.lm_head(hidden[torch.arange(len(out)), prefix.last_valid_index()])
                .argmax(-1)
                .tolist()
            )
            for o, t in zip(out, nxt):
                o.append(t)
        return [v.decode([t for t in o if t != v["<eos>"]][:2]) for o in out]


__all__ = ["COLORS", "PretrainHistory", "VQAPretrainer"]
