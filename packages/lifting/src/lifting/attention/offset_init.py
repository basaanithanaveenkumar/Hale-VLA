"""Initialisation of deformable sampling offsets (shared by all deformable modules)."""

from __future__ import annotations

import math

import torch
from torch import nn


def init_sampling_offsets(
    linear: nn.Linear, num_heads: int, num_levels: int, num_points: int, dims: int = 2
) -> None:
    """Zero weights and a bias that spreads each head's points along its own direction.

    Point ``i`` of a head starts ``i + 1`` cells away from the reference point, which
    matches the Deformable-DETR initialisation in 2D and generalises it to 3D using
    Fibonacci-sphere directions.
    """
    nn.init.zeros_(linear.weight)
    if dims == 2:
        thetas = torch.arange(num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
        dirs = torch.stack([thetas.cos(), thetas.sin()], -1)
    elif dims == 3:
        idx = torch.arange(num_heads, dtype=torch.float32) + 0.5
        phi = torch.acos(1 - 2 * idx / num_heads)
        theta = math.pi * (1 + 5**0.5) * idx
        dirs = torch.stack([phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()], -1)
    else:
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    dirs = dirs / dirs.abs().max(-1, keepdim=True).values
    grid = dirs.view(num_heads, 1, 1, dims).repeat(1, num_levels, num_points, 1)
    grid = grid * torch.arange(1, num_points + 1, dtype=torch.float32).view(1, 1, -1, 1)
    with torch.no_grad():
        linear.bias.copy_(grid.flatten())
