"""π0.5: open-world generalisation through co-training and hierarchical inference.

References: π0.5 (Physical Intelligence, 2025, arXiv:2504.16054); knowledge insulation "π0.5 + KI"
(Driess et al., 2025, arXiv:2505.23705); FAST tokens (Pertsch et al., 2025, arXiv:2501.09747).

What changes relative to :mod:`pi_family.pi0`::

    [ image  <bos> task <sep> s_1 s_2 <sep> ] [ subtask words <eos> ] [ <boa> FAST ... <eoa> ] [ a_1 ... a_H ]
      bidirectional prefix (state = text!)     causal, CE-trained       causal, CE-trained       action expert
                                                 action expert sees it   action expert must NOT

1. **Discrete state**: q_t is binned and written as tokens in the prefix (no separate state token).
2. **High-level subtask**: the backbone predicts the next subtask as text ("go to red"); at run time the robot
   first decodes the subtask, then generates actions conditioned on it (hierarchical inference, one model).
3. **FAST tokens + knowledge insulation**: the backbone also learns to emit the action chunk as discrete FAST
   tokens (a stable CE loss). The action expert reads the backbone through ``detach()`` - its flow-matching
   gradient never reaches the VLM - and is not allowed to attend the FAST tokens (they would give the answer away).
4. **adaRMSNorm time conditioning** in the action expert instead of π0's action-time MLP.
5. Loss ``CE(subtask + FAST) + α · flow`` with α = 10.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pi_family.config import Pi05Config
from pi_family.flow_matching import sinusoidal_time_embedding
from pi_family.layers import _init_linear
from pi_family.policy import ImageBlock, PiPolicy, PrefixSequence, SuffixSequence, TokenStream
from pi_family.tokenizers import FastActionTokenizer, PiVocabulary, StateDiscretizer


class TimeConditioner(nn.Module):
    """Flow time τ → conditioning vector for adaRMSNorm, per action token: ``W2 · swish(W1 · φ(τ))``."""

    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.net.apply(_init_linear)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_time_embedding(time, self.embed_dim))


def build_pi05_vocabulary(config: Pi05Config, fast: FastActionTokenizer) -> PiVocabulary:
    return PiVocabulary(state_bins=config.state_bins, fast_size=fast.vocab_size)


class Pi05Policy(PiPolicy):
    """π0.5 at toy scale: one model that predicts subtasks (text), FAST tokens (text) and continuous actions."""

    def __init__(
        self,
        config: Pi05Config,
        fast_tokenizer: FastActionTokenizer,
        vocab: PiVocabulary | None = None,
    ):
        vocab = vocab or build_pi05_vocabulary(config, fast_tokenizer)
        super().__init__(config, vocab, expert_cond_dim=config.expert_width)
        self.fast = fast_tokenizer
        self.state_discretizer = StateDiscretizer(config.state_bins)
        self.action_in = nn.Linear(config.action_dim, config.expert_width)
        self.time_cond = TimeConditioner(config.time_embed_dim, config.expert_width)
        _init_linear(self.action_in)
        if config.ordinal_state_init:
            self._init_state_embeddings()

    def _init_state_embeddings(self) -> None:
        """Give the state-bin tokens an *ordered* starting point (bin k is close to bin k + 1).

        A pre-trained VLM already knows that "117" lies between "116" and "118"; our from-scratch toy backbone does
        not, and learning 32 unrelated bin embeddings from robot data alone is slow (``ordinal_state_init=False``
        to see it). Sinusoidal features of the bin centre are a cheap stand-in for that pre-trained knowledge.
        """
        bins = torch.arange(self.config.state_bins, dtype=torch.float32)
        centre = (bins + 0.5) / self.config.state_bins
        features = sinusoidal_time_embedding(
            centre, self.config.width, min_period=0.05, max_period=4.0
        )
        with torch.no_grad():
            start = self.vocab.state_offset
            self.token_embedding.weight[start : start + self.config.state_bins] = 0.5 * features

    @property
    def insulate(self) -> bool:
        return self.config.knowledge_insulation

    # ------------------------------------------------------------------------------------------ prompt
    def prompt_head(self, task: str, state: torch.Tensor) -> list:
        """``<bos> task <sep> state-bin tokens <sep>`` - the bidirectional part of the text."""
        bins = self.vocab.state_token(self.state_discretizer.encode(state)).tolist()
        return (
            [self.vocab["<bos>"]]
            + self.vocab.encode(task)
            + [self.vocab["<sep>"]]
            + bins
            + [self.vocab["<sep>"]]
        )

    def fast_ids(self, chunk: torch.Tensor) -> list:
        return self.vocab.fast_token(self.fast.encode(chunk.cpu()))

    def build_prefix(
        self, batch, *, train: bool, mode: str = "act", generated=None
    ) -> PrefixSequence:
        c, v = self.config, self.vocab
        device = self.prefix_position.weight.device
        image = ImageBlock(self.image_encoder(batch.images.to(device)), name="image")
        streams = []
        for i in range(batch.size):
            s = TokenStream().add(self.prompt_head(batch.task[i], batch.state[i]), causal=False)
            if mode == "subtask":  # high-level inference: decode the subtask
                s.add(generated[i] if generated else [], causal=True)
                streams.append(s)
                continue
            s.add(
                v.encode(batch.subtask[i]) + [v["<eos>"]],
                causal=True,
                target=train and c.predict_subtask,
            )
            if mode == "fast":  # decode the chunk as FAST tokens
                s.add(
                    [v["<boa>"]] + list(generated[i] if generated else []),
                    causal=True,
                    visible=False,
                )
            elif train and c.train_fast_tokens:
                s.add([v["<boa>"]], causal=True, visible=False)
                s.add(
                    self.fast_ids(batch.actions[i]) + [v["<eoa>"]],
                    causal=True,
                    target=True,
                    visible=False,
                )
            streams.append(s)
        return self.assemble_prefix([image], streams)

    def build_suffix(
        self, batch, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> SuffixSequence:
        tokens = self.action_in(noisy_actions) + self.action_position
        ar = torch.zeros(self.config.chunk_size, dtype=torch.bool, device=noisy_actions.device)
        ar[0] = True
        return SuffixSequence(tokens, ar, self.time_cond(time), action_start=0)

    # ------------------------------------------------------------------------------------------ inference
    def subtask_vocabulary(self) -> list:
        words = "go to red green blue <eos>".split()
        return [self.vocab[w] for w in words]

    def predict_subtask(self, batch) -> list:
        """High-level step: decode "what to do next" as text from the image, task and state."""
        out = self.generate(
            batch, "subtask", max_new_tokens=5, allowed_ids=self.subtask_vocabulary()
        )
        return [self.vocab.decode(ids) for ids in out]

    @torch.no_grad()
    def act(self, batch, committed=None, num_steps=None, return_subtask: bool = False):
        """Hierarchical inference: predict the subtask, put it in the prompt, then sample the action chunk."""
        subtasks = self.predict_subtask(batch)
        actions = self.sample_actions(
            batch.replace(subtask=subtasks), num_steps=num_steps, committed=committed
        )
        return (actions, subtasks) if return_subtask else actions

    @torch.no_grad()
    def decode_fast_actions(self, batch, max_new_tokens: int = 40) -> torch.Tensor:
        """The *discrete* way to act (π0-FAST / π0.5 pre-training): decode FAST tokens autoregressively."""
        fast_ids = list(range(self.vocab.fast_offset, self.vocab.size)) + [self.vocab["<eoa>"]]
        out = self.generate(
            batch, "fast", max_new_tokens=max_new_tokens, stop_token="<eoa>", allowed_ids=fast_ids
        )
        return torch.stack(
            [self.fast.decode([t - self.vocab.fast_offset for t in ids]) for ids in out]
        )
