"""π0: a vision-language-action flow model (Black et al., 2024, arXiv:2410.24164).

Sequence layout and attention (three blocks, each sees itself and every block to its left)::

    [ image patches  <bos> task words ]  [ state ]  [ a_1 ... a_H ]
      backbone weights, bidirectional    expert     expert, bidirectional among actions

* the **prefix** (images + language) is processed by the VLM weights and never sees the robot-specific tokens,
  so it keeps behaving like the pre-trained VLM, and it can be KV-cached across denoising steps;
* the **state** token q_t is a linear projection, processed by the action expert; it is its own block so it does
  not change with the noise (π0 caches it too);
* each **action token** embeds one noisy action together with the flow time τ through a small MLP:
  ``W3 · swish(W2 · [W1 · a^τ_k , φ(τ)])``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from pi_family.config import Pi0Config
from pi_family.flow_matching import sinusoidal_time_embedding
from pi_family.layers import _init_linear
from pi_family.policy import ImageBlock, PiPolicy, PrefixSequence, SuffixSequence, TokenStream
from pi_family.tokenizers import PiVocabulary


class ActionTimeMLP(nn.Module):
    """π0's action-token embedding: the noisy action and the flow time are fused by an MLP (no adaptive norm)."""

    def __init__(self, action_dim: int, width: int):
        super().__init__()
        self.width = width
        self.action_in = nn.Linear(action_dim, width)
        self.fuse = nn.Linear(2 * width, width)
        self.out = nn.Linear(width, width)
        self.apply(_init_linear)

    def forward(self, noisy_actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """(B, H, A), (B, H) → (B, H, width)."""
        tau = sinusoidal_time_embedding(time, self.width)
        return self.out(F.silu(self.fuse(torch.cat([self.action_in(noisy_actions), tau], dim=-1))))


class Pi0Policy(PiPolicy):
    """π0 at toy scale. See the module docstring for the token layout."""

    def __init__(self, config: Pi0Config | None = None, vocab: PiVocabulary | None = None):
        config = config or Pi0Config()
        super().__init__(config, vocab or PiVocabulary(), expert_cond_dim=None)
        self.state_proj = nn.Linear(config.state_dim, config.expert_width)
        self.action_time = ActionTimeMLP(config.action_dim, config.expert_width)
        _init_linear(self.state_proj)

    def build_prefix(
        self, batch, *, train: bool, mode: str = "act", generated=None
    ) -> PrefixSequence:
        device = self.prefix_position.weight.device
        image = ImageBlock(self.image_encoder(batch.images.to(device)), name="image")
        streams = []
        for task in batch.task:
            # ar = False: the words continue the image block, so the whole prefix is one bidirectional block
            streams.append(
                TokenStream().add([self.vocab["<bos>"]] + self.vocab.encode(task), causal=False)
            )
        return self.assemble_prefix([image], streams)

    def build_suffix(
        self, batch, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> SuffixSequence:
        device = noisy_actions.device
        state = self.state_proj(batch.state.to(device))[:, None]  # (B, 1, De)
        actions = self.action_time(noisy_actions, time) + self.action_position
        ar = torch.zeros(1 + self.config.chunk_size, dtype=torch.bool, device=device)
        ar[0] = True  # the state token starts its own block ...
        ar[1] = True  # ... and the action tokens start the next one
        return SuffixSequence(torch.cat([state, actions], 1), ar, None, action_start=1)
