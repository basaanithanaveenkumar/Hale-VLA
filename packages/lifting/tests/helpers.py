from __future__ import annotations

import torch


def rotation_z(deg: float) -> torch.Tensor:
    import math

    a = math.radians(deg)
    t = torch.eye(4)
    t[0, 0], t[0, 1], t[1, 0], t[1, 1] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
    return t
