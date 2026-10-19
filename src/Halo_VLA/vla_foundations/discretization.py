"""Discretising continuous actions into tokens (RT-2, OpenVLA).

RT-2 turns every continuous action dimension into one of **256 uniform bins** and writes the bin indices as text tokens, so a
vision-language model can *emit actions as words* with its ordinary next-token loss. OpenVLA keeps the 256 bins but sets the bin range
to the 1st-99th **quantile** of the training actions instead of the min-max range, so a few outlier actions cannot stretch the
intervals and waste resolution. :class:`ActionBinner` implements both.

Quantisation is lossy by construction: a bin is ``(high - low) / num_bins`` wide, so a decoded action is off by up to half a bin,
and the error accumulates when actions are integrated into a trajectory (see ``action_representations_step_by_step.ipynb``).
"""

from __future__ import annotations

import numpy as np
import torch


class ActionBinner:
    """Per-dimension uniform binning of continuous actions.

    Args:
        num_bins: bins per dimension (RT-2 / OpenVLA: 256).
        low / high: the interval that is divided; values outside are clipped to the first / last bin. Scalars or per-dimension arrays.
    """

    def __init__(self, num_bins: int = 256, low=-1.5, high=1.5):
        if num_bins < 2:
            raise ValueError("need at least two bins")
        self.num_bins = num_bins
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)

    @classmethod
    def from_quantiles(
        cls, actions: np.ndarray, num_bins: int = 256, q: float = 0.01
    ) -> ActionBinner:
        """OpenVLA's rule: divide the ``[q, 1 - q]`` quantile range of each dimension (actions: (N, D))."""
        flat = np.asarray(actions).reshape(-1, np.asarray(actions).shape[-1])
        return cls(num_bins, np.quantile(flat, q, axis=0), np.quantile(flat, 1 - q, axis=0))

    @property
    def bin_width(self) -> np.ndarray:
        return (self.high - self.low) / self.num_bins

    def encode(self, actions) -> torch.Tensor:
        """Continuous actions (..., D) → integer bin indices in ``[0, num_bins)`` (same shape)."""
        a = torch.as_tensor(actions, dtype=torch.float32)
        low, high = torch.as_tensor(self.low), torch.as_tensor(self.high)
        unit = ((a - low) / (high - low)).clamp(0.0, 1.0 - 1e-6)
        return (unit * self.num_bins).long()

    def decode(self, bins) -> torch.Tensor:
        """Bin indices → the centre of each bin (the best single guess for the original value)."""
        b = torch.as_tensor(bins).float()
        low, high = torch.as_tensor(self.low), torch.as_tensor(self.high)
        return low + (b + 0.5) / self.num_bins * (high - low)

    def roundtrip(self, actions) -> torch.Tensor:
        """``decode(encode(a))``: what the policy can at best output for the demonstrated action ``a``."""
        return self.decode(self.encode(actions))

    def max_error(self) -> np.ndarray:
        """Worst-case reconstruction error per dimension for in-range actions: half a bin."""
        return self.bin_width / 2
