"""Configuration dataclasses for the π-family policies (π0, π0.5, π0.7).

Each config inherits from the previous generation, mirroring how Physical Intelligence built every model on top of
the last one: π0.5 = π0 + discrete state / FAST tokens / hierarchical text / knowledge insulation, and π0.7 = π0.5 +
MEM-style history, subgoal images, episode metadata, control mode, prompt dropout, CFG and training-time RTC.

The defaults are *toy* sizes so every notebook trains on a laptop CPU in about a minute. The published sizes are
recorded next to each field for reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pi0Config:
    """π0 (Black et al., 2024, arXiv:2410.24164): a VLM + flow-matching action expert.

    Published: PaliGemma-3B backbone (width 2048, 18 layers), 300M action expert (width 1024), H = 50 actions,
    10 Euler steps at inference, time sampled from a shifted Beta(1.5, 1).
    """

    # --- observation -------------------------------------------------------------------------------------------
    image_size: int = 32  # published: 224 (SigLIP)
    patch_size: int = 8  # -> (32 / 8)^2 = 16 image tokens per view
    vision_stem_channels: int = (
        0  # 0 = plain ViT patch embedding; >0 adds a 3x3 conv stem (~5x slower on CPU)
    )
    state_dim: int = 2  # proprioception q_t
    action_dim: int = 2  # one action a_t
    chunk_size: int = 16  # H, the action chunk length (published: 50)

    # --- transformer -------------------------------------------------------------------------------------------
    width: int = 64  # backbone ("VLM") width            (published: 2048)
    expert_width: int = 48  # action-expert width (narrower)    (published: 1024)
    depth: int = 2  # both experts have the same depth  (published: 18)
    num_heads: int = 4  # shared by both experts so their attention can be joined
    head_dim: int = 16
    mlp_ratio: float = 2.0
    max_prefix_len: int = 96  # learned positions for the prefix tokens
    time_embed_dim: int = 32  # sinusoidal width of the flow time τ
    # Final norm of the action expert. The velocity target ε - a contains the input noise ε, so the expert must pass
    # x_τ almost linearly to its output, and a plain RMSNorm divides by |h|, which varies with ε. In π0's 48-wide toy
    # expert, after 1200 steps, closed-loop success was 17 % with the norm and 44 % without (94 % after 2400 steps),
    # so the toy π0 drops it. π0.5 / π0.7 keep it: their *adaptive* norm gives the output a τ-dependent scale, and
    # there it helps (bare π0.5 on reach: first-action error 0.39 without, 0.24 with, at 1200 steps).
    expert_output_norm: bool = False

    # --- flow matching -----------------------------------------------------------------------------------------
    action_objective: str = (
        "flow_matching"  # "flow_matching" (π0) or "regression" (an L2 baseline for contrast)
    )
    num_denoise_steps: int = 10  # Euler steps from τ = 1 (noise) to τ = 0 (actions)
    time_beta_a: float = 1.5  # p(τ) ∝ Beta(1.5, 1): more weight on the noisy end
    time_beta_b: float = 1.0
    time_max: float = 0.999  # τ = 0.999 · Beta + 0.001, never exactly 0 or 1
    flow_samples_per_example: int = (
        4  # (ε, τ) draws per observation; they share the cached prefix pass
    )

    # --- loss weights ------------------------------------------------------------------------------------------
    flow_loss_weight: float = 1.0
    text_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.action_objective not in ("flow_matching", "regression"):
            raise ValueError(f"unknown action_objective {self.action_objective!r}")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class Pi05Config(Pi0Config):
    """π0.5 (Intelligence et al., 2025, arXiv:2504.16054) and its knowledge-insulated variant π0.5-KI
    (Driess et al., 2025, arXiv:2505.23705).

    Published: state discretised into 256 bins as text tokens, FAST action tokens, high-level subtask prediction,
    loss = CE(text + FAST) + α · flow-matching with α = 10, adaRMSNorm time conditioning in the action expert.
    The original π0.5 lets the flow gradient reach the VLM; π0.5-KI (and π0.7) stop it ("knowledge insulation"),
    which is the ``knowledge_insulation`` switch here.
    """

    state_bins: int = 32  # published: 256
    ordinal_state_init: bool = (
        True  # stand-in for a pre-trained VLM's knowledge that numbers are ordered
    )
    expert_output_norm: bool = (
        True  # adaptive final norm, as in openpi (see the comment in Pi0Config)
    )
    fast_scale: float = 6.0  # γ in round(γ · DCT(a)); larger = finer quantisation, more tokens
    fast_max_coefficient: int = 12  # quantised DCT values are clipped to [-K, K]
    fast_bpe_merges: int = 48  # FAST compresses the integer sequence with BPE
    knowledge_insulation: bool = (
        False  # π0.5-KI / π0.7: stop-gradient from the action expert into the backbone
    )
    flow_loss_weight: float = 10.0  # α in the π0.5 loss
    predict_subtask: bool = True  # train the high-level "what to do next" text head
    train_fast_tokens: bool = (
        True  # train the backbone on FAST tokens (needed for insulation to make sense)
    )
    max_prefix_len: int = 128


@dataclass
class Pi07Config(Pi05Config):
    """π0.7 (Physical Intelligence, Apr 2026, "π0.7: a Steerable Generalist Robotic Foundation Model").

    Published: Gemma3-4B backbone (incl. a 400M vision encoder), MEM-style history encoder (up to 6 frames per view,
    1 s stride, compressed to one frame's worth of tokens), 860M action expert with adaRMSNorm, H = 50, 5 denoising
    steps, multimodal prompt (subtask text, subgoal images, episode metadata, control mode) trained with per-component
    dropout, CFG on metadata with β ∈ {1.3, 1.7, 2.2} and training-time real-time chunking with 0..12 step delays.
    """

    history_frames: int = 4  # current frame + 3 past frames (published: up to 6)
    history_stride: int = 2  # steps between history frames (published: 1 s)
    history_dropout: float = 0.3  # drop the whole history with this probability
    # MEM feeds past robot states as tokens too. In a demo-trained policy those reveal the current velocity, so the model
    # learns "keep doing what the last steps did" instead of reading the scene (the imitation "copycat" problem): fine on
    # demos, brittle in closed loop. Off by default here; past *frames* still give the history.
    use_state_history: bool = False
    subgoal_probability: float = 0.25  # fraction of training examples that get a subgoal image
    subtask_dropout_with_subgoal: float = 0.3
    metadata_dropout: float = 0.15  # drop all metadata
    metadata_component_dropout: float = 0.05
    max_rtc_delay: int = 4  # simulated inference delay in steps (published: 0..12)
    high_level_fraction: float = (
        0.2  # examples trained as the high-level policy (predict the subtask text)
    )
    num_denoise_steps: int = 5  # published: 5
    max_prefix_len: int = 160
    knowledge_insulation: bool = True  # π0.7 trains with the KI recipe (§III of the report)
    cfg_scales: tuple = field(default_factory=lambda: (1.3, 1.7, 2.2))
