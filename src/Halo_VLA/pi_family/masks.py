"""Block-causal attention masks, the way openpi builds them.

Each token carries an ``ar`` ("autoregressive") flag. ``ar = True`` starts a new block that *earlier* tokens cannot
see; ``ar = False`` continues the current block. With ``c = cumsum(ar)``, token ``i`` may attend token ``j`` iff
``c[j] <= c[i]`` (and ``j`` is a real, non-padding token).

* a run of ``[1, 0, 0, 0]``      -> one bidirectional block of 4 tokens
* a run of ``[1, 1, 1, 1]``      -> 4 causal tokens
* blocks concatenate: a later block sees all earlier ones, never the other way round.

That single rule produces every mask in the π family: π0's ``[prefix | state | actions]``, π0.5's
``[prefix | causal text/FAST | actions]`` and π0.7's ``[observation | subgoals | causal prompt | actions]``.
"""

from __future__ import annotations

import torch


def make_attention_mask(ar: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Boolean (B, L, L) mask, ``True`` = query (row) may attend key (column).

    Padding queries are allowed to attend themselves so softmax never sees an all-masked row.
    """
    if ar.shape != valid.shape:
        raise ValueError(f"ar {tuple(ar.shape)} and valid {tuple(valid.shape)} must match")
    cum = torch.cumsum(ar.long(), dim=-1)
    mask = cum[:, None, :] <= cum[:, :, None]
    mask = mask & valid[:, None, :]
    eye = torch.eye(ar.shape[-1], dtype=torch.bool, device=ar.device)
    return mask | (eye[None] & ~valid[:, :, None])


def split_prefix_suffix_mask(
    prefix_ar: torch.Tensor,
    prefix_valid: torch.Tensor,
    action_visible: torch.Tensor,
    suffix_ar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the joint mask and split it into the prefix block and the suffix rows.

    Returns ``(prefix_mask (B, Lp, Lp), suffix_mask (B, Ls, Lp + Ls))``. ``action_visible`` removes prefix keys the
    action expert must not read (π0.5 / π0.7: the FAST tokens, which would otherwise leak the answer).

    The first suffix token must have ``ar = True``; this guarantees that no prefix token attends a suffix token,
    which is what makes the prefix KV cache exact.
    """
    batch, lp = prefix_ar.shape
    if not bool(suffix_ar[..., 0].all()):
        raise ValueError("the first suffix token must start a new block (ar = True)")
    suffix_ar = (
        suffix_ar.expand(batch, -1) if suffix_ar.dim() == 2 else suffix_ar[None].expand(batch, -1)
    )
    ar = torch.cat([prefix_ar, suffix_ar], dim=1)
    valid = torch.cat([prefix_valid, torch.ones_like(suffix_ar, dtype=torch.bool)], dim=1)
    full = make_attention_mask(ar, valid)
    suffix_rows = full[:, lp:].clone()
    suffix_rows[:, :, :lp] &= action_visible[:, None, :]
    assert not full[:, :lp, lp:].any(), (
        "a prefix token attends the suffix - the KV cache would be wrong"
    )
    return full[:, :lp, :lp], suffix_rows
