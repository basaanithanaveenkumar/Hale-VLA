"""π-family policies (π0 / π0.5 / π0.7): masks, KV cache, flow matching, tokenisers, insulation, prompts, toy world."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pi_family import (
    AdaRMSNorm,
    EpisodeMetadata,
    FastActionTokenizer,
    FlowMatching,
    Pi0Config,
    Pi0Policy,
    Pi05Config,
    Pi05Policy,
    Pi07Config,
    Pi07Policy,
    PiTrainer,
    StateDiscretizer,
    SubgoalWorldModel,
    bin_speed,
    make_attention_mask,
    split_prefix_suffix_mask,
)
from pi_family.toy import (
    ChunkDataset,
    ClosedLoopEvaluator,
    DemoGenerator,
    PointWorld,
    make_scenarios,
)

SMALL = dict(
    width=32,
    expert_width=24,
    depth=1,
    num_heads=2,
    head_dim=8,
    chunk_size=8,
    flow_samples_per_example=2,
)


@pytest.fixture(scope="module")
def world():
    return PointWorld()


@pytest.fixture(scope="module")
def data(world):
    gen = DemoGenerator(world, idle_steps=8)
    eps = gen.dataset(
        ["reach", "collect", "marked"],
        40,
        np.random.default_rng(0),
        styles=("expert", "sloppy", "mistake"),
        joint_fraction=0.5,
    )
    return ChunkDataset(eps, world, chunk_size=8, history_frames=3, history_stride=2)


@pytest.fixture(scope="module")
def batch(data):
    return data.sample(5, np.random.default_rng(1))


@pytest.fixture(scope="module")
def fast(data):
    return FastActionTokenizer(8, 2, num_merges=8).fit(
        data.sample(64, np.random.default_rng(2)).actions
    )


def _joint_equals_cached(model, batch):
    prefix = model.build_prefix(batch, train=False)
    noisy, tau = torch.randn(batch.size, 8, 2), torch.rand(batch.size, 8)
    with torch.no_grad():
        _, cache = model.encode_prefix(prefix)
        cached = model.velocity(batch, prefix, cache, noisy, tau)
        suffix = model.build_suffix(batch, noisy, tau)
        pm, sm = split_prefix_suffix_mask(prefix.ar, prefix.valid, prefix.action_visible, suffix.ar)
        pos = model.prefix_position(torch.arange(prefix.length))[None]
        _, hy = model.transformer.forward_joint(
            prefix.embeds + pos, suffix.embeds, pm, sm, suffix.cond
        )
        joint = model.action_out(hy[:, suffix.action_start :])
    return (cached - joint).abs().max().item()


# --------------------------------------------------------------------------------------------- masks / flow
def test_block_causal_mask_rule():
    ar = torch.tensor([[1, 0, 0, 1, 1, 0]], dtype=torch.bool)
    m = make_attention_mask(ar, torch.ones_like(ar))[0]
    assert m[:3, :3].all() and not m[:3, 3:].any()  # first block: bidirectional, blind to the right
    assert m[3, :4].all() and not m[3, 4:].any()
    assert m[4:, :].all()  # last block sees everything


def test_padding_keys_masked_and_rows_non_empty():
    ar = torch.tensor([[1, 0, 1, 1]], dtype=torch.bool)
    valid = torch.tensor([[True, True, True, False]])
    m = make_attention_mask(ar, valid)[0]
    assert not m[:, 3][:3].any() and m.any(-1).all()


def test_flow_matching_straight_path_is_exact():
    flow = FlowMatching()
    a, eps = torch.randn(3, 8, 2), torch.randn(3, 8, 2)
    assert torch.allclose(
        flow.integrate(lambda x, t: flow.target_velocity(a, eps), eps, 4), a, atol=1e-6
    )
    t = flow.sample_time(10_000, generator=torch.Generator().manual_seed(0))
    assert 0.001 <= t.min() and t.max() <= 1.0 and t.mean() > 0.55  # Beta(1.5, 1) mean 0.6


# --------------------------------------------------------------------------------------------- tokenisers
def test_fast_roundtrip_and_compression(fast, data):
    chunks = data.sample(32, np.random.default_rng(3)).actions
    lengths = [len(fast.encode(c)) for c in chunks]
    recon = torch.stack([fast.decode(fast.encode(c)) for c in chunks])
    assert max(lengths) < 8 * 2  # BPE compresses the 16 integers
    assert (recon - chunks).abs().max() < 0.6 / fast.scale * 2  # error bounded by quantisation
    assert fast.decode([]).shape == (8, 2)  # robust to empty / truncated decodes


def test_state_discretizer_roundtrip():
    s = torch.rand(100, 2)
    d = StateDiscretizer(32)
    assert (d.decode(d.encode(s)) - s).abs().max() <= 0.5 / 32 + 1e-6


def test_metadata_prompt_and_speed_bins():
    assert EpisodeMetadata(20, 5, False).prompt() == "speed 20 ; quality 5 ; mistake false"
    assert EpisodeMetadata(None, 3, None).prompt() == "quality 3"
    assert bin_speed(23) == 20 and bin_speed(26) == 30 and bin_speed(3) == 10


# --------------------------------------------------------------------------------------------- policies
def test_pi0_cache_exact_and_loss(batch):
    model = Pi0Policy(Pi0Config(**SMALL))
    assert _joint_equals_cached(model, batch) < 1e-5
    losses = model.compute_loss(batch)
    losses.total.backward()
    assert torch.isfinite(losses.total) and model.image_encoder.patchify.weight.grad is not None
    assert model.sample_actions(batch).shape == (batch.size, 8, 2)


def test_pi0_prefix_never_attends_suffix(batch):
    model = Pi0Policy(Pi0Config(**SMALL))
    prefix = model.build_prefix(batch, train=False)
    suffix = model.build_suffix(batch, torch.zeros(batch.size, 8, 2), torch.ones(batch.size, 8))
    _, sm = split_prefix_suffix_mask(prefix.ar, prefix.valid, prefix.action_visible, suffix.ar)
    lp = prefix.length
    assert not sm[:, 0, lp + 1 :].any()  # state does not see the actions
    assert sm[:, 1:, lp:].all()  # actions see the state and each other


def _wake_gates(model):
    """adaRMS gates are zero at init (adaLN-Zero), which blocks every gradient path; give them a random start."""
    for m in model.modules():
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.1)
    return model


def test_pi05_insulation_and_hidden_fast_tokens(batch, fast):
    model = _wake_gates(Pi05Policy(Pi05Config(**SMALL, knowledge_insulation=True), fast))
    prefix = model.build_prefix(batch, train=True)
    is_fast = (
        model.vocab.is_fast(prefix.ids)
        | (prefix.ids == model.vocab["<boa>"])
        | (prefix.ids == model.vocab["<eoa>"])
    )
    assert is_fast.any() and not (prefix.action_visible & is_fast).any()
    assert _joint_equals_cached(model, batch) < 1e-5
    losses = model.compute_loss(batch)
    model.zero_grad()
    losses.flow.backward()
    assert all(
        p.grad is None or p.grad.abs().sum() == 0 for p in model.transformer.backbone.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer.expert.parameters()
    )
    open_model = _wake_gates(Pi05Policy(Pi05Config(**SMALL, knowledge_insulation=False), fast))
    open_model.compute_loss(batch).flow.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in open_model.transformer.backbone.parameters()
    )


def test_pi05_generation_and_fast_decoding(batch, fast):
    model = Pi05Policy(Pi05Config(**SMALL), fast)
    subtasks = model.predict_subtask(batch)
    assert len(subtasks) == batch.size and all(isinstance(s, str) for s in subtasks)
    assert model.decode_fast_actions(batch, max_new_tokens=6).shape == (batch.size, 8, 2)
    assert model.act(batch).shape == (batch.size, 8, 2)


def test_pi07_prompt_dropout_rtc_and_cfg(batch, fast):
    torch.manual_seed(0)
    model = Pi07Policy(Pi07Config(**SMALL, history_frames=3), fast)
    assert _joint_equals_cached(model, batch) < 1e-5
    text = Pi07Policy.prompt_text("go to red", "go to red", EpisodeMetadata(20, 5, False), "joint")
    assert (
        text
        == "task go to red ; subtask go to red ; speed 20 ; quality 5 ; mistake false ; control mode joint"
    )
    time, mask = model.prepare_flow_inputs(torch.zeros(64, 8, 2), torch.full((64, 8), 0.5))
    committed = time == 0
    assert committed.any() and (mask[committed] == 0).all() and (mask[~committed] == 1).all()
    assert (
        committed[:, :-1] | ~committed[:, 1:]
    ).all()  # committed actions form a prefix of the chunk
    losses = model.compute_loss(batch)
    assert torch.isfinite(losses.total)
    prior = torch.randn(batch.size, 3, 2)
    out = model.act(batch, committed=prior, cfg_scale=1.5)
    assert torch.allclose(out[:, :3], prior)  # RTC keeps the committed actions
    assert len(model.predict_subtask(batch)) == batch.size


def test_world_model_shapes(batch):
    wm = SubgoalWorldModel(width=32, depth=1)
    loss = wm.compute_loss(batch.images, batch.subtask, batch.subgoal)
    loss.backward()
    assert torch.isfinite(loss)
    assert wm.sample(batch.images[:2], ["go to red", "go to crimson"], num_steps=2).shape == (
        2,
        3,
        32,
        32,
    )


# --------------------------------------------------------------------------------------------- toy world
def test_demos_annotations(world):
    gen = DemoGenerator(world)
    rng = np.random.default_rng(4)
    ep = gen.demo("collect", rng)
    assert (
        ep.subtask[0] == f"go to {ep.targets[0]}"
        and ep.subtask[ep.active_steps - 1] == f"go to {ep.targets[1]}"
    )
    assert np.linalg.norm(ep.robot[ep.active_steps] - ep.scene.objects[ep.targets[1]]) < 1e-5
    mistake = gen.demo("reach", rng, style="mistake")
    assert mistake.mistake.any() and mistake.metadata.mistake and mistake.metadata.quality <= 2


def test_rollout_oracle_succeeds_with_delay(world):
    gen = DemoGenerator(world)
    scenarios = make_scenarios(gen, "collect", 8, seed=5)
    ev = ClosedLoopEvaluator(world, chunk_size=16, exec_horizon=8, max_steps=80)

    def oracle(batch, committed):
        out = []
        for k, s in enumerate(scenarios):
            p = batch.state[k].numpy()
            goal = s.scene.objects[batch.subtask[k].split()[-1]]
            acts = []
            if committed is not None:  # these run before the new chunk takes over
                for a in committed[k].numpy():
                    acts.append(a)
                    p = world.step(p, a)
            while len(acts) < 16:
                d = goal - p
                a = d / max(np.linalg.norm(d), 1e-8) * min(0.9, np.linalg.norm(d) / world.max_step)
                acts.append(a)
                p = world.step(p, a)
            out.append(acts)
        return torch.tensor(np.array(out), dtype=torch.float32)

    assert ev.run(oracle, scenarios).success.all()
    assert ev.run(oracle, scenarios, delay=3, pass_committed=True).success.all()


def test_trainer_reduces_loss(data):
    torch.manual_seed(0)
    model = Pi0Policy(Pi0Config(**SMALL, action_objective="regression"))
    hist = PiTrainer(model, data, batch_size=16, warmup=5).fit(60)
    assert np.mean(hist.losses[-10:]) < np.mean(hist.losses[:10])
