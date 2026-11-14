"""A mixture of transformers: one joint attention over all tokens, separate weights per group."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from pi_lab.expert_block import ExpertBlock

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class MixtureOfTransformers(nn.Module):
    """The π backbone: a VLM expert and an action expert that attend to each other.

    Every layer holds one :class:`ExpertBlock` per expert. Each token stream is processed by its
    expert's weights, then queries, keys and values of **all** streams are concatenated and go
    through **one** masked attention. So the action expert reads the VLM's features at every
    layer (not just the last one), while the VLM keeps its own pretrained weights.

    Supports a key/value cache: run the prefix once (``return_kv=True``), then run only the
    suffix streams with ``past_kv=`` at each denoising step.

    Args:
        widths: residual width of each expert (e.g. ``[64, 32]``).
        depth: number of layers.
        heads: attention heads.
        head_dim: per-head size.
        cond_widths: per expert, size of the AdaRMSNorm condition (``None`` = plain RMSNorm).
    """

    def __init__(self, widths: Sequence[int], depth: int, heads: int = 4, head_dim: int = 16,
                 cond_widths: Sequence[int | None] | None = None) -> None:
        super().__init__()
        cond_widths = list(cond_widths or [None] * len(widths))
        self.layers = nn.ModuleList(
            nn.ModuleList(ExpertBlock(w, heads, head_dim, cond_width=c)
                          for w, c in zip(widths, cond_widths, strict=True))
            for _ in range(depth))
        self.heads, self.head_dim = heads, head_dim

    def forward(self, streams: Sequence[torch.Tensor], experts: Sequence[int],
                mask: torch.Tensor, conds: Sequence[torch.Tensor | None] | None = None,
                past_kv: KVCache | None = None, return_kv: bool = False,
                detach_past: bool = False) -> tuple[list[torch.Tensor], KVCache | None]:
        """Run all layers.

        Args:
            streams: token tensors ``(B, L_i, width_of_expert)``, in sequence order.
            experts: which expert's weights each stream uses.
            mask: ``(L_new, L_past + L_new)`` boolean, ``True`` = may attend.
            conds: per stream, the AdaRMSNorm condition ``(B, cond_width)`` or ``None``.
            past_kv: cached keys/values of earlier tokens (one pair per layer).
            return_kv: also return this call's keys/values (to cache a prefix).
            detach_past: stop gradients into ``past_kv`` (π0.7 knowledge insulation).

        Returns:
            Updated streams and (optionally) the per-layer key/value cache of this call.
        """
        conds = list(conds or [None] * len(streams))
        streams = list(streams)
        lengths = [s.shape[1] for s in streams]
        new_kv: KVCache = []
        for layer_idx, layer in enumerate(self.layers):
            qs, ks, vs = [], [], []
            for x, e, c in zip(streams, experts, conds, strict=True):
                q, k, v = layer[e].qkv(x, c)
                qs.append(q), ks.append(k), vs.append(v)
            q, k, v = torch.cat(qs, 2), torch.cat(ks, 2), torch.cat(vs, 2)
            if return_kv:
                new_kv.append((k, v))
            if past_kv is not None:
                pk, pv = past_kv[layer_idx]
                if detach_past:
                    pk, pv = pk.detach(), pv.detach()
                k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
            scores = q @ k.transpose(-1, -2) / self.head_dim**0.5
            scores = scores.masked_fill(~mask, float("-inf"))
            out = scores.softmax(-1) @ v                                  # (B, H, L_new, hd)
            out = out.transpose(1, 2).flatten(2)
            chunks = out.split(lengths, dim=1)
            streams = [layer[e].mlp(x + layer[e].o(o), c)
                       for x, o, e, c in zip(streams, chunks, experts, conds, strict=True)]
        return streams, (new_kv if return_kv else None)
