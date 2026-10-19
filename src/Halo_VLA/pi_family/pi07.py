"""π0.7: a steerable generalist with a multimodal prompt (Physical Intelligence, April 2026).

π0.7 keeps π0.5's backbone + action expert, FAST-token knowledge insulation and adaRMSNorm, and changes *what the
model is told*. The context C_t grows from a task string to::

    [ obs: MEM history tokens ] [ subgoal image tokens ] <bos> task … ; subtask … ; speed 20 ; quality 5 ;
      bidirectional               bidir., sees obs         mistake false ; control mode ee <q_t> <q_t-2> … <sep>
                                                           └──────────── causal prompt ─────────────┘
    [ <boa> FAST … <eoa> ]  (KI targets, hidden from the expert)   [ a_1 … a_H ]  action expert, adaRMS(τ per token)

* **MEM history** - several past frames compressed to one frame's worth of tokens; past *states* are extra tokens.
* **Subgoal images** - "what the scene should look like next", from a world model at run time.
* **Episode metadata** - speed / quality / mistake labels, so failed and slow data can be trained on *and* the model
  can be asked for fast, high-quality, mistake-free behaviour.
* **Control mode** - ``ee`` or ``joint``.
* **Prompt dropout** - every component is dropped at the paper's rates, so any subset works at test time, and
  classifier-free guidance on the metadata becomes possible.
* **Training-time RTC** - a random prefix of the chunk is given clean (τ = 0) and not supervised, so at run time the
  chunk can continue the actions that are already being executed during inference latency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pi_family.config import Pi07Config
from pi_family.layers import _init_linear
from pi_family.pi05 import TimeConditioner
from pi_family.policy import ImageBlock, PiPolicy, PrefixSequence, SuffixSequence, TokenStream
from pi_family.tokenizers import EpisodeMetadata, FastActionTokenizer, PiVocabulary
from pi_family.vision import HistoryVideoEncoder


class Pi07Policy(PiPolicy):
    """π0.7 at toy scale (single camera; the paper uses up to four views and three subgoal views)."""

    def __init__(
        self,
        config: Pi07Config,
        fast_tokenizer: FastActionTokenizer,
        vocab: PiVocabulary | None = None,
    ):
        vocab = vocab or PiVocabulary(fast_size=fast_tokenizer.vocab_size)
        super().__init__(config, vocab, expert_cond_dim=config.expert_width)
        c = config
        self.fast = fast_tokenizer
        self.history_encoder = HistoryVideoEncoder(
            self.image_encoder, c.width, c.history_frames, c.num_heads
        )
        self.subgoal_type = nn.Parameter(
            torch.zeros(1, 1, c.width)
        )  # "this is a goal, not the present"
        self.state_proj = nn.Linear(c.state_dim, c.width)  # MEM: linear state tokens
        self.state_age = nn.Parameter(torch.zeros(c.history_frames, c.width))
        self.action_in = nn.Linear(c.action_dim, c.expert_width)
        self.time_cond = TimeConditioner(c.time_embed_dim, c.expert_width)
        nn.init.trunc_normal_(self.subgoal_type, std=0.02)
        nn.init.trunc_normal_(self.state_age, std=0.02)
        _init_linear(self.state_proj)
        _init_linear(self.action_in)

    @property
    def insulate(self) -> bool:
        return self.config.knowledge_insulation

    def encode_observation(self, images: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(images[:, None])  # a history of one frame: the current view

    # ------------------------------------------------------------------------------------------ prompt
    @staticmethod
    def prompt_text(
        task: str, subtask: str | None, metadata: EpisodeMetadata, control_mode: str
    ) -> str:
        """The language part of π0.7's prompt, e.g.
        ``task go to red ; subtask go to red ; speed 20 ; quality 5 ; mistake false ; control mode ee``."""
        parts = [f"task {task}"]
        if subtask:
            parts.append(f"subtask {subtask}")
        if metadata is not None and not metadata.is_empty():
            parts.append(metadata.prompt())
        parts.append(f"control mode {control_mode}")
        return " ; ".join(parts)

    def _dropout(self, batch, train: bool):
        """Decide per example which prompt components are present (π0.7 §V-E rates during training)."""
        c, b = self.config, batch.size
        has_goal = batch.subgoal is not None and batch.subgoal_valid is not None
        goal_ok = batch.subgoal_valid.clone() if has_goal else torch.zeros(b, dtype=torch.bool)
        history_valid = (
            batch.history_valid.clone()
            if batch.history_valid is not None
            else torch.ones(b, 1, dtype=torch.bool)
        )
        metadata = list(batch.metadata) if batch.metadata else [EpisodeMetadata() for _ in range(b)]
        subtasks = list(batch.subtask)
        high_level = torch.zeros(b, dtype=torch.bool)
        if not train:
            return goal_ok, subtasks, metadata, history_valid, high_level
        use_goal = goal_ok & (torch.rand(b) < c.subgoal_probability)
        for i in range(b):
            if use_goal[i] and torch.rand(()) < c.subtask_dropout_with_subgoal:
                subtasks[i] = None
            m = metadata[i]
            if torch.rand(()) < c.metadata_dropout:
                metadata[i] = EpisodeMetadata()
            else:

                def keep(value):
                    return None if torch.rand(()) < c.metadata_component_dropout else value

                metadata[i] = EpisodeMetadata(keep(m.speed), keep(m.quality), keep(m.mistake))
        history_valid[torch.rand(b) < c.history_dropout, 1:] = False
        high_level = torch.rand(b) < c.high_level_fraction
        return use_goal, subtasks, metadata, history_valid, high_level

    def build_prefix(
        self, batch, *, train: bool, mode: str = "act", generated=None
    ) -> PrefixSequence:
        c, v = self.config, self.vocab
        device = self.prefix_position.weight.device
        use_goal, subtasks, metadata, history_valid, high_level = self._dropout(batch, train)
        frames = batch.history if batch.history is not None else batch.images[:, None]
        history_valid = history_valid[:, : frames.shape[1]]
        obs = ImageBlock(
            self.history_encoder(frames.to(device), history_valid.to(device)), name="observation"
        )
        blocks = [obs]
        if use_goal.any():
            goal = self.image_encoder(batch.subgoal.to(device)) + self.subgoal_type
            blocks.append(ImageBlock(goal, use_goal.to(device), name="subgoal"))
        states = batch.state_history if batch.state_history is not None else batch.state[:, None]
        streams = []
        for i in range(batch.size):
            s = TokenStream().add([v["<bos>"]], causal=True)
            if mode == "subtask" or bool(high_level[i]):
                # the high-level policy: same architecture, predicts the next subtask from the task and the scene
                s.add(v.encode(f"task {batch.task[i]} ; subtask"), causal=True)
                if mode == "subtask":
                    s.add(generated[i] if generated else [], causal=True)
                else:
                    s.add(v.encode(batch.subtask[i]) + [v["<eos>"]], causal=True, target=True)
                streams.append(s)
                continue
            s.add(
                v.encode(
                    self.prompt_text(batch.task[i], subtasks[i], metadata[i], batch.control_mode[i])
                ),
                causal=True,
            )
            n_states = (
                states.shape[1] if c.use_state_history else 1
            )  # current state only unless enabled
            valid_k = [
                k
                for k in range(n_states)
                if bool(history_valid[i, min(k, history_valid.shape[1] - 1)])
            ]
            proprio = self.state_proj(states[i, valid_k].to(device)) + self.state_age[valid_k]
            s.add_continuous(proprio, causal=True)
            s.add([v["<sep>"]], causal=True)
            if train and c.train_fast_tokens:
                s.add([v["<boa>"]], causal=True, visible=False)
                s.add(
                    v.fast_token(self.fast.encode(batch.actions[i].cpu())) + [v["<eoa>"]],
                    causal=True,
                    target=True,
                    visible=False,
                )
            streams.append(s)
        flow_weight = (~high_level).float().to(device)  # high-level examples have no action prompt
        return self.assemble_prefix(blocks, streams, flow_weight)

    def build_suffix(
        self, batch, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> SuffixSequence:
        tokens = self.action_in(noisy_actions) + self.action_position
        ar = torch.zeros(self.config.chunk_size, dtype=torch.bool, device=noisy_actions.device)
        ar[0] = True
        return SuffixSequence(tokens, ar, self.time_cond(time), action_start=0)  # τ per token (RTC)

    def prepare_flow_inputs(self, actions: torch.Tensor, time: torch.Tensor):
        """Training-time RTC: the first ``d ~ U{0..max_delay}`` actions are given clean and not supervised."""
        b, h = time.shape
        delay = torch.randint(0, self.config.max_rtc_delay + 1, (b,), device=time.device)
        committed = torch.arange(h, device=time.device)[None] < delay[:, None]
        return time.masked_fill(committed, 0.0), (~committed).float()

    # ------------------------------------------------------------------------------------------ inference
    def predict_subtask(self, batch) -> list:
        allowed = [self.vocab[w] for w in "go to red green blue <eos>".split()]
        return [
            self.vocab.decode(ids)
            for ids in self.generate(batch, "subtask", 5, allowed_ids=allowed)
        ]

    @torch.no_grad()
    def act(self, batch, committed=None, cfg_scale: float = 0.0, num_steps=None):
        """Sample a chunk; ``cfg_scale`` > 0 applies classifier-free guidance on the episode metadata."""
        return self.sample_actions(
            batch,
            num_steps=num_steps,
            committed=committed,
            cfg_scale=cfg_scale,
            uncond_batch=batch.without_metadata() if cfg_scale > 0 else None,
        )


class Pi07Runtime:
    """Algorithm 1 of the π0.7 report, as a policy function for :class:`pi_family.toy.ClosedLoopEvaluator`.

    Keeps, per environment, the current subtask (from the high-level policy, or the evaluator's oracle "coach") and
    the current subgoal image (from the world model). The subgoal is refreshed when the subtask changes or every
    ``refresh_steps`` steps. Every call returns an action chunk; with ``committed`` actions it continues them (RTC).
    The report runs the subtask / subgoal generation asynchronously; here it is synchronous for clarity.
    """

    def __init__(
        self,
        policy: Pi07Policy,
        metadata: EpisodeMetadata,
        world_model=None,
        high_level: bool = False,
        refresh_steps: int = 8,
        exec_horizon: int = 8,
        cfg_scale: float = 0.0,
        use_subtask: bool = True,
    ):
        self.policy, self.metadata, self.world_model = policy, metadata, world_model
        self.high_level, self.refresh_steps, self.exec_horizon = (
            high_level,
            refresh_steps,
            exec_horizon,
        )
        self.cfg_scale, self.use_subtask = cfg_scale, use_subtask
        self.reset()

    def reset(self) -> None:
        self.step, self.subtasks, self.subgoal, self.last_refresh = 0, None, None, None
        self.log = []

    def __call__(self, batch, committed=None) -> torch.Tensor:
        b = batch.size
        subtasks = self.policy.predict_subtask(batch) if self.high_level else list(batch.subtask)
        if self.world_model is not None:
            changed = self.subtasks is None or subtasks != self.subtasks
            stale = self.last_refresh is None or self.step - self.last_refresh >= self.refresh_steps
            if changed or stale:
                self.subgoal = self.world_model.sample(batch.images, subtasks)
                self.last_refresh = self.step
        self.subtasks = subtasks
        batch = batch.replace(
            subtask=subtasks if self.use_subtask else [None] * b, metadata=[self.metadata] * b
        )
        if self.subgoal is not None:
            batch = batch.replace(
                subgoal=self.subgoal, subgoal_valid=torch.ones(b, dtype=torch.bool)
            )
        self.log.append(dict(step=self.step, subtasks=subtasks))
        self.step += self.exec_horizon
        return self.policy.act(batch, committed=committed, cfg_scale=self.cfg_scale)
