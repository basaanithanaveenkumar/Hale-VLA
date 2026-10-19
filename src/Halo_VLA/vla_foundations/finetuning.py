"""Parameter-efficient fine-tuning for VLAs: LoRA, OFT (orthogonal fine-tuning) and freezing helpers.

* **LoRA** (Hu et al., 2021): ``W x → W x + (α/r) B A x`` with ``A (r, in)``, ``B (out, r)``, ``B = 0`` at the start.
* **OFT** (Qiu et al., 2023, *Controlling Text-to-Image Diffusion by Orthogonal Finetuning*): the frozen weight is multiplied by a learned
  block-diagonal **orthogonal** matrix ``R``, ``W' = W R``, so the angles between the neurons (rows of ``W``) - the "hyperspherical
  energy" - cannot change. ``R`` is parametrised with the **Cayley transform** ``R = (I + Q)(I - Q)^{-1}``, ``Q = A - A^T``.

  Not to be confused with **OpenVLA-OFT** (Kim et al., 2025, "Optimizing Speed and Success"), whose "OFT" is an *optimised fine-tuning
  recipe* (parallel decoding, action chunking, a continuous action head trained with L1) and has nothing to do with orthogonality.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ATTENTION_PROJECTIONS = ("q", "k", "v", "o")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base, self.rank, self.scaling = base, rank, alpha / rank
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features) / base.in_features**0.5)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_b @ self.lora_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * ((x @ self.lora_a.T) @ self.lora_b.T)

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.delta_weight()


class OFTLinear(nn.Module):
    """A frozen ``nn.Linear`` whose input is rotated by a learned block-diagonal orthogonal matrix.

    ``y = W (R x)`` with ``R = blockdiag(R_1 … R_k)``, each block ``R_i = (I + Q_i)(I - Q_i)^{-1}``. ``R = I`` at initialisation.
    """

    def __init__(self, base: nn.Linear, block_size: int = 8):
        super().__init__()
        if base.in_features % block_size:
            raise ValueError("block_size must divide in_features")
        self.base, self.block_size = base, block_size
        self.blocks = base.in_features // block_size
        self.skew = nn.Parameter(torch.zeros(self.blocks, block_size, block_size))
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def rotation(self) -> torch.Tensor:
        """(blocks, b, b) orthogonal matrices from the Cayley transform of the skew-symmetric parameters."""
        q = self.skew - self.skew.transpose(-1, -2)
        eye = torch.eye(self.block_size, device=q.device).expand_as(q)
        return (eye + q) @ torch.linalg.inv(eye - q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        z = x.reshape(-1, self.blocks, self.block_size)
        z = torch.einsum(
            "nkj,kij->nki", z, self.rotation()
        )  # R_k applied to each block of the input
        return self.base(z.reshape(shape))

    def merged_weight(self) -> torch.Tensor:
        """``W R`` as a dense matrix (what the fine-tuned layer is equivalent to)."""
        w = self.base.weight.reshape(self.base.out_features, self.blocks, self.block_size)
        return torch.einsum("oki,kij->okj", w, self.rotation()).reshape_as(self.base.weight)


def inject_adapters(
    module: nn.Module, kind: str, targets=ATTENTION_PROJECTIONS, **kwargs
) -> list[nn.Module]:
    """Replace every ``nn.Linear`` called one of ``targets`` (e.g. ``q``/``k``/``v``/``o``) below ``module`` by a LoRA / OFT wrapper.

    Returns the created wrappers. ``kind`` is ``"lora"`` or ``"oft"``; ``kwargs`` go to the wrapper (``rank``, ``alpha`` / ``block_size``).
    """
    if kind not in ("lora", "oft"):
        raise ValueError("kind must be 'lora' or 'oft'")
    made = []
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and type(child) is nn.Linear:
                wrapped = (
                    LoRALinear(child, **kwargs) if kind == "lora" else OFTLinear(child, **kwargs)
                )
                setattr(parent, name, wrapped)
                made.append(wrapped)
    return made


def freeze_all_but(model: nn.Module, trainable) -> None:
    """Freeze every parameter, then re-enable those whose name satisfies ``trainable(name)``."""
    for name, p in model.named_parameters():
        p.requires_grad_(bool(trainable(name)))


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable, fraction=trainable / max(total, 1))


def hyperspherical_energy(weight: torch.Tensor) -> float:
    """Mean pairwise ``1 / ||w_i - w_j||`` between the unit-normalised rows of ``weight`` (invariant under rotations of the rows' space)."""
    w = nn.functional.normalize(weight.detach().float(), dim=1)
    d = torch.cdist(w, w)
    iu = torch.triu_indices(len(w), len(w), 1)
    return float((1.0 / d[iu[0], iu[1]].clamp_min(1e-6)).mean())
