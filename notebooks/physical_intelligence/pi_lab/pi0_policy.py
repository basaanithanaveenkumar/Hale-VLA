"""π0: a PaliGemma-style VLM plus a flow-matching action expert (Black et al., 2024)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pi_lab.block_attention_mask import BlockAttentionMask
from pi_lab.flow_matching import FlowMatching
from pi_lab.mask_block import MaskBlock
from pi_lab.mixture_of_transformers import KVCache, MixtureOfTransformers
from pi_lab.patch_encoder import PatchEncoder


class Pi0Policy(nn.Module):
    """Toy-scale π0 with the real structure.

    * **prefix** (VLM expert, width ``vlm_width``): image patches + instruction tokens,
      bidirectional, as in PaliGemma;
    * **suffix** (action expert, width ``expert_width``): one continuous **state** token plus
      ``H`` noisy **action** tokens. Each action token is ``MLP([W·a_τ ; φ(τ)])``, π0's
      "action-time mixer" that injects the flow time by concatenation;
    * **mask**: three blocks ``[prefix] → [state] → [actions]``, bidirectional inside a block,
      causal between blocks (see :class:`BlockAttentionMask`);
    * **output**: the action tokens' final states are projected to a velocity ``v(a_τ, τ | o, ℓ)``.

    ``objective="regression"`` trains the same network to output the action chunk directly
    (MSE). It is a baseline showing why π0 uses flow matching: regression averages multimodal
    demonstrations.

    Args:
        vocab_size: text vocabulary size.
        text_len: number of instruction tokens.
        horizon: actions per chunk ``H``.
        action_dim: action size.
        state_dim: proprioceptive state size.
        vlm_width: VLM expert width.
        expert_width: action expert width (smaller, as in π0: 1024 vs 2048).
        depth: layers.
        image_size: camera resolution.
        patch: patch size.
        objective: ``"flow"`` (π0) or ``"regression"`` (baseline).
        time_beta: Beta parameters of the training-time distribution (π0: ``(1.5, 1)``).
    """

    def __init__(self, vocab_size: int, text_len: int = 3, horizon: int = 10, action_dim: int = 2,
                 state_dim: int = 2, vlm_width: int = 64, expert_width: int = 32, depth: int = 3,
                 image_size: int = 32, patch: int = 8, objective: str = "flow",
                 time_beta: tuple[float, float] = (1.5, 1.0)) -> None:
        super().__init__()
        if objective not in ("flow", "regression"):
            raise ValueError(f"objective must be 'flow' or 'regression', got {objective!r}")
        self.objective, self.horizon, self.action_dim = objective, horizon, action_dim
        self.image = PatchEncoder(image_size, patch, vlm_width)
        self.text = nn.Embedding(vocab_size, vlm_width)
        self.text_pos = nn.Parameter(torch.randn(1, text_len, vlm_width) * 0.02)
        self.state_in = nn.Linear(state_dim, expert_width)
        self.action_in = nn.Linear(action_dim, expert_width)
        self.action_time_mlp = nn.Sequential(nn.Linear(2 * expert_width, expert_width), nn.SiLU(),
                                             nn.Linear(expert_width, expert_width))
        self.action_pos = nn.Parameter(torch.randn(1, horizon, expert_width) * 0.02)
        self.action_out = nn.Linear(expert_width, action_dim)
        self.backbone = MixtureOfTransformers([vlm_width, expert_width], depth)
        self.n_prefix = (image_size // patch) ** 2 + text_len
        self.layout = BlockAttentionMask([MaskBlock("image+text", self.n_prefix),
                                          MaskBlock("state", 1), MaskBlock("actions", horizon)])
        self.register_buffer("mask", self.layout.build(), persistent=False)
        self.flow = FlowMatching(*time_beta)
        self.expert_width = expert_width

    # ------------------------------------------------------------------ token builders
    def prefix_tokens(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Image patches then instruction tokens, in the VLM width."""
        return torch.cat([self.image(images), self.text(text) + self.text_pos], dim=1)

    def suffix_tokens(self, state: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """State token + action tokens (action ⊕ time embedding → MLP), in the expert width."""
        time = FlowMatching.time_embedding(t, self.expert_width)[:, None].expand(-1, self.horizon, -1)
        actions = self.action_time_mlp(torch.cat([self.action_in(noisy), time], dim=-1))
        return torch.cat([self.state_in(state)[:, None], actions + self.action_pos], dim=1)

    # ------------------------------------------------------------------ forward passes
    def encode_prefix(self, images: torch.Tensor, text: torch.Tensor) -> KVCache:
        """Run the VLM once per observation and keep its keys/values (they never change)."""
        n = self.n_prefix
        _, kv = self.backbone([self.prefix_tokens(images, text)], [0], self.mask[:n, :n],
                              return_kv=True)
        return kv

    def velocity(self, kv: KVCache, state: torch.Tensor, noisy: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """Action-expert pass over the suffix only, attending to the cached prefix."""
        n = self.n_prefix
        (out,), _ = self.backbone([self.suffix_tokens(state, noisy, t)], [1], self.mask[n:, :],
                                  past_kv=kv)
        return self.action_out(out[:, 1:])

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flow-matching (or regression) loss on a batch, as one full forward pass."""
        actions = batch["actions"]
        if self.objective == "regression":
            noisy = torch.zeros_like(actions)
            t = torch.ones(actions.shape[0], device=actions.device)
        else:
            noise = torch.randn_like(actions)
            t = self.flow.sample_time(actions.shape[0], actions.device)
            noisy = self.flow.interpolate(actions, noise, t)
        streams = [self.prefix_tokens(batch["images"], batch["text"]),
                   self.suffix_tokens(batch["state"], noisy, t)]
        (_, suffix), _ = self.backbone(streams, [0, 1], self.mask)
        pred = self.action_out(suffix[:, 1:])
        target = actions if self.objective == "regression" else self.flow.target(actions, noise)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, images: torch.Tensor, text: torch.Tensor, state: torch.Tensor,
               steps: int = 10, noise: torch.Tensor | None = None, return_path: bool = False):
        """Action chunks ``(B, H, D)``: cache the prefix once, then ``steps`` Euler steps."""
        kv = self.encode_prefix(images, text)
        B = images.shape[0]
        if self.objective == "regression":
            dev = images.device
            out = self.velocity(kv, state, torch.zeros(B, self.horizon, self.action_dim, device=dev),
                                torch.ones(B, device=dev))
            return (out, [out]) if return_path else out
        if noise is None:
            noise = torch.randn(B, self.horizon, self.action_dim, device=images.device)
        return self.flow.sample(lambda x, t: self.velocity(kv, state, x, t), noise, steps, return_path)
