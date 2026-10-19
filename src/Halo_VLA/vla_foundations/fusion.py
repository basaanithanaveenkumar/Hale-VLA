"""Where do vision and language meet? Early, late and pooled fusion in one policy.

All three variants predict action chunks with the same flow-matching expert (alternating self- / cross-attention); they differ only in
**what the expert's cross-attention reads**, i.e. when the two modalities are allowed to interact:

* ``early`` (OpenVLA, RT-2, π0): the (projected) image tokens and the word tokens go into **one transformer together**; every layer lets
  every image token attend every word and vice versa. The expert reads the fused tokens.
* ``late``: images and words are encoded by **separate towers** that never see each other; the expert's cross-attention over the
  *concatenation* is the only place they meet (as in architectures that fuse at the action head).
* ``pooled``: each modality is mean-pooled to one vector and mixed by an MLP (CLIP-style global fusion): the expert reads a *single* token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from open_vlas.attention import RMSNorm, causal_mask, init_linear
from open_vlas.data import VLABatch
from open_vlas.embodiments import MAX_ACTION_DIM
from open_vlas.flow import ForwardFlow
from open_vlas.smolvla import ExpertBlock
from pi_family.layers import ExpertLayer
from pi_family.pi0 import ActionTimeMLP
from pi_family.policy import PiLosses
from pi_family.tokenizers import PiVocabulary


@dataclass
class FusionConfig:
    """Configuration of :class:`FusionPolicy`."""

    fusion: str = "early"  # "early" | "late" | "pooled"
    width: int = 64
    llm_depth: int = 3  # joint layers for early fusion
    side_depth: int = 2  # per-tower layers for late fusion
    heads: int = 4
    expert_depth: int = 4
    expert_ratio: float = 0.75
    chunk_size: int = 16
    action_dim: int = 2
    merge: int = 2  # 2x2 token merge in the projector (64 → 16 visual tokens)
    num_denoise_steps: int = 10
    flow_samples_per_example: int = 4
    train_vision: bool = False

    def __post_init__(self) -> None:
        if self.fusion not in ("early", "late", "pooled"):
            raise ValueError("fusion must be 'early', 'late' or 'pooled'")

    @property
    def expert_width(self) -> int:
        return max(self.heads, int(round(self.width * self.expert_ratio / self.heads)) * self.heads)


class FusionPolicy(nn.Module):
    """Vision encoder + text embedding + a chosen fusion + a flow-matching action expert.

    Args:
        vision: module mapping images ``(B, 3, S, S)`` to patch tokens ``(B, N, vision_width)`` (frozen unless ``cfg.train_vision``).
        vision_width: its token width (the sum of the widths for a fused encoder).
        cfg: :class:`FusionConfig`.
        vocab: word vocabulary of the instructions.
    """

    def __init__(
        self, vision: nn.Module, vision_width: int, cfg: FusionConfig, vocab: PiVocabulary
    ):
        super().__init__()
        self.cfg, self.vocab = cfg, vocab
        w = cfg.width
        self.vision = vision.requires_grad_(cfg.train_vision)
        merge_in = vision_width * cfg.merge**2
        self.projector = nn.Sequential(nn.Linear(merge_in, w), nn.GELU(), nn.Linear(w, w))
        self.text_emb = nn.Embedding(vocab.size, w)
        self.pos = nn.Embedding(96, w)

        def mk(n):
            return nn.ModuleList(ExpertLayer(w, cfg.heads, w // cfg.heads, 2.0) for _ in range(n))

        if cfg.fusion == "early":
            self.llm = mk(cfg.llm_depth)
        elif cfg.fusion == "late":
            self.vis_tower, self.txt_tower = mk(cfg.side_depth), mk(cfg.side_depth)
        else:
            self.mix = nn.Sequential(nn.Linear(2 * w, w), nn.GELU(), nn.Linear(w, w))
        self.feature_norm = RMSNorm(w)
        ew = cfg.expert_width
        self.action_time = ActionTimeMLP(cfg.action_dim, ew)
        self.action_position = nn.Parameter(torch.zeros(1, cfg.chunk_size, ew))
        self.blocks = nn.ModuleList(
            ExpertBlock("cross" if i % 2 == 0 else "self", ew, w, cfg.heads, 2.0)
            for i in range(cfg.expert_depth)
        )
        self.out_norm = RMSNorm(ew)
        self.action_out = nn.Linear(ew, cfg.action_dim)
        nn.init.trunc_normal_(self.text_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos.weight, std=0.02)
        nn.init.trunc_normal_(self.action_position, std=0.02)
        init_linear(self.action_out)
        self.flow = ForwardFlow()

    # ---------------------------------------------------------------------------------------- fusion
    def _visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.cfg.train_vision and torch.is_grad_enabled()):
            tok = self.vision(images)
        b, n, c = tok.shape
        g, m = int(n**0.5), self.cfg.merge
        tok = (
            tok.view(b, g // m, m, g // m, m, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, (g // m) ** 2, m * m * c)
        )
        return self.projector(tok)

    def _text_tokens(self, texts: list, device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocab.encode(t) for t in texts]
        n = max(len(i) for i in ids)
        tok = torch.zeros(len(ids), n, dtype=torch.long, device=device)
        valid = torch.zeros(len(ids), n, dtype=torch.bool, device=device)
        for k, i in enumerate(ids):
            tok[k, : len(i)] = torch.tensor(i, device=device)
            valid[k, : len(i)] = True
        return self.text_emb(tok), valid

    @staticmethod
    def _run(layers, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid[:, None, None, :]
        for layer in layers:
            q, k, v, gate = layer.project(x)
            x = layer.finish(x, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), gate)
        return x

    def encode(self, batch: VLABatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused features ``(B, L, width)`` and their validity mask ``(B, 1, 1, L)`` for the expert's cross-attention."""
        dev = self.action_position.device
        img = self._visual_tokens(batch.images.to(dev))
        txt, tvalid = self._text_tokens(batch.task, dev)
        ivalid = torch.ones(img.shape[:2], dtype=torch.bool, device=dev)
        if self.cfg.fusion == "early":
            x = torch.cat([img, txt], 1)
            x = x + self.pos(torch.arange(x.shape[1], device=dev))[None]
            valid = torch.cat([ivalid, tvalid], 1)
            x = self._run(self.llm, x, valid)  # one transformer: every image token sees every word
        elif self.cfg.fusion == "late":
            img = self._run(
                self.vis_tower, img + self.pos(torch.arange(img.shape[1], device=dev))[None], ivalid
            )
            txt = self._run(
                self.txt_tower, txt + self.pos(torch.arange(txt.shape[1], device=dev))[None], tvalid
            )
            x, valid = (
                torch.cat([img, txt], 1),
                torch.cat([ivalid, tvalid], 1),
            )  # the towers never interacted
        else:
            v = img.mean(1)
            t = (txt * tvalid[..., None]).sum(1) / tvalid.sum(1, keepdim=True)
            x, valid = (
                self.mix(torch.cat([v, t], -1))[:, None],
                torch.ones(len(img), 1, dtype=torch.bool, device=dev),
            )
        return self.feature_norm(x), valid[:, None, None, :]

    # ---------------------------------------------------------------------------------------- expert
    def velocity(
        self, feats: torch.Tensor, fmask: torch.Tensor, noisy: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        t = tau[:, None].expand(-1, noisy.shape[1])
        x = self.action_time(noisy, t) + self.action_position
        smask = causal_mask(x.shape[1], x.device)[None, None]
        for block in self.blocks:
            x = block(x, feats, fmask, smask)
        return self.action_out(self.out_norm(x))

    def compute_loss(self, batch: VLABatch) -> PiLosses:
        feats, fmask = self.encode(batch)
        actions = batch.actions[:, :, : self.cfg.action_dim].to(feats.device)
        k = self.cfg.flow_samples_per_example
        if k > 1:
            feats, fmask, actions = (
                feats.repeat_interleave(k, 0),
                fmask.repeat_interleave(k, 0),
                actions.repeat_interleave(k, 0),
            )
        noise = torch.randn_like(actions)
        tau = self.flow.sample_time(actions.shape[0], device=actions.device)
        pred = self.velocity(feats, fmask, self.flow.interpolate(actions, noise, tau), tau)
        loss = (pred - self.flow.target_velocity(actions, noise)).pow(2).mean()
        return PiLosses(loss, loss, loss.new_zeros(()), {"flow": loss.item()})

    @torch.no_grad()
    def sample_actions(self, batch: VLABatch, num_steps: int | None = None) -> torch.Tensor:
        feats, fmask = self.encode(batch)
        b, h, a = batch.size, self.cfg.chunk_size, self.cfg.action_dim
        x0 = torch.randn(b, h, a, device=feats.device)
        out = self.flow.integrate(
            lambda x, t: self.velocity(feats, fmask, x, t),
            x0,
            num_steps or self.cfg.num_denoise_steps,
        )
        return F.pad(out, (0, MAX_ACTION_DIM - a))

    def parameter_counts(self) -> dict:
        def count(m, trainable=False):
            return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)

        return dict(total=count(self), trainable=count(self, True))


class FusedVision(nn.Module):
    """OpenVLA-style *fused* vision backbone: concatenate the patch features of several encoders along the channel axis."""

    def __init__(self, *encoders: nn.Module):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.cat([e(images) for e in self.encoders], dim=-1)
