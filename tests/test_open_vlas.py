"""SmolVLA, GR00T N1 / N1.5 and their shared pieces: flow, VLM, attention, embodiments, async inference, latent actions."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from open_vlas import (
    EMBODIMENTS,
    Attention,
    EmbodiedEvaluator,
    EmbodiedMixture,
    EmbodiedSource,
    ForwardFlow,
    SmolVLAConfig,
    SmolVLAPolicy,
    ToyVLM,
    VLMConfig,
    make_embodied_scenarios,
    make_reach_demos,
)
from open_vlas.async_inference import ActionQueue, AsyncConfig, AsyncInferenceRunner
from open_vlas.data import VLABatch, collate
from open_vlas.embodiments import LATENT_DIM, MAX_ACTION_DIM, embodied_arrays
from open_vlas.groot import (
    AdaLayerNorm,
    CategorySpecificLinear,
    GR00TConfig,
    GR00TPolicy,
)
from open_vlas.latent_actions import (
    LatentActionModel,
    frame_pair_sampler,
    train_latent_action_model,
)
from open_vlas.vlm import PatchShuffleVisionEncoder
from pi_family.toy import DemoGenerator, PointWorld

TINY = VLMConfig(width=32, depth=4, num_heads=2, head_dim=16)


@pytest.fixture(scope="module")
def world():
    return PointWorld()


@pytest.fixture(scope="module")
def demos(world):
    return make_reach_demos(DemoGenerator(world), 24, np.random.default_rng(0))


@pytest.fixture(scope="module")
def mixture(world, demos):
    srcs = [
        EmbodiedSource(world, demos, EMBODIMENTS[n]) for n in ("point2d", "joint2d", "gripper3d")
    ]
    return EmbodiedMixture(srcs)


@pytest.fixture(scope="module")
def batch(mixture):
    return mixture.sample(6, np.random.default_rng(1))


@pytest.fixture(scope="module")
def vlm():
    torch.manual_seed(0)
    return ToyVLM(TINY)


# ------------------------------------------------------------------------------------------------ flow / attention
def test_forward_flow_straight_path_and_time_distribution():
    flow = ForwardFlow()
    a, eps = torch.randn(3, 8, 2), torch.randn(3, 8, 2)
    out = flow.integrate(lambda x, t: flow.target_velocity(a, eps), eps, 4)
    assert torch.allclose(out, a, atol=1e-5)  # τ = 0 is noise, τ = 1 is data
    t = flow.sample_time(20_000, generator=torch.Generator().manual_seed(0))
    assert (
        0.0 <= t.min() and t.max() <= 0.999 and t.mean() < 0.5
    )  # Beta((s-τ)/s;1.5,1) favours the noisy end


def test_attention_cross_matches_manual_and_weights_normalised():
    torch.manual_seed(0)
    att = Attention(16, 2, 8, context_dim=24)
    x, ctx = torch.randn(2, 5, 16), torch.randn(2, 7, 24)
    out, w = att(x, ctx, return_weights=True)
    assert torch.allclose(w.sum(-1), torch.ones(2, 2, 5), atol=1e-5)
    assert torch.allclose(out, att(x, ctx), atol=1e-5)  # the fused and the explicit paths agree
    mask = torch.zeros(2, 1, 5, 7, dtype=torch.bool)
    mask[..., :3] = True
    _, wm = att(x, ctx, mask, return_weights=True)
    assert wm[..., 3:].abs().max() == 0


# ------------------------------------------------------------------------------------------------ VLM
def test_pixel_shuffle_is_a_lossless_rearrangement():
    cfg = VLMConfig(width=8, depth=1, num_heads=1, head_dim=8)
    enc = PatchShuffleVisionEncoder(cfg).eval()
    img = torch.rand(2, 3, 32, 32)
    assert enc(img).shape == (2, 16, 8) and enc.patch_tokens(img).shape == (2, 64, 8)
    # manual 2x2 merge of the patch grid must equal the module's reshape
    tok = enc.patch_tokens(img).view(2, 8, 8, 8)
    blocks = [
        tok[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2].reshape(2, -1)
        for i in range(4)
        for j in range(4)
    ]
    assert torch.allclose(enc(img), enc.merge(torch.stack(blocks, 1)), atol=1e-5)


def test_vlm_inputs_masks_and_truncation(vlm):
    imgs = torch.rand(2, 3, 32, 32)
    inp = vlm.build_inputs(
        imgs, ["go to red", "go to blue"], extra=torch.randn(2, 1, 32), answers=[[5], []]
    )
    assert inp.embeds.shape[:2] == inp.valid.shape and inp.num_image_tokens == 16
    assert inp.ar[0].sum() == 2 and inp.ar[1].sum() == 1  # image-start bit + one answer token
    hs = vlm.forward_layers(inp)
    assert len(hs) == 4
    small = ToyVLM(TINY).truncate(2)
    assert len(small.forward_layers(inp)) == 2
    with pytest.raises(RuntimeError):
        small.logits(hs[-1])  # a truncated VLM has no LM head any more
    frozen = ToyVLM(TINY).freeze(vision=True, language=True)
    assert not any(p.requires_grad for p in frozen.parameters())


# ------------------------------------------------------------------------------------------------ embodiments / data
def test_embodiment_roundtrip_and_arrays(demos):
    ep = demos[0]
    for name in ("point2d", "joint2d"):
        emb = EMBODIMENTS[name]
        a = emb.action_from_ee(ep.actions[3], ep.robot[3], ep.scene.objects[ep.targets[0]])
        assert np.allclose(emb.ee_from_action(a), ep.actions[3], atol=1e-5)
    states, actions = embodied_arrays(ep, EMBODIMENTS["gripper3d"])
    assert states.shape == (ep.length + 1, 3) and actions.shape == (ep.length, 3)
    assert (
        actions[0, 2] == -1.0 and actions[ep.active_steps - 1, 2] == 1.0
    )  # closes near the target
    assert states[1, 2] == actions[0, 2]  # the state reports the previous command


def test_collate_pads_and_masks(batch):
    assert batch.state.shape[1] == 3 and batch.actions.shape[2] == MAX_ACTION_DIM
    for i in range(batch.size):
        d = EMBODIMENTS_BY_ID[int(batch.embodiment[i])].action_dim
        assert batch.action_mask[i].sum() == d and batch.actions[i, :, d:].abs().sum() == 0


EMBODIMENTS_BY_ID = {e.id: e for e in EMBODIMENTS.values()}


def test_lapa_batches_need_a_latent_function(world, demos):
    src = EmbodiedSource(world, demos, EMBODIMENTS["lapa"])
    items = src.sample_items(4, np.random.default_rng(0))
    with pytest.raises(ValueError):
        collate(items)
    b = collate(items, lambda x, f: torch.ones(x.shape[0], LATENT_DIM))
    assert b.actions[:, :, :LATENT_DIM].min() == 1.0 and b.action_mask.sum(1).tolist() == [4.0] * 4


# ------------------------------------------------------------------------------------------------ SmolVLA
def test_smolvla_causality_loss_and_frozen_vlm(vlm, batch):
    torch.manual_seed(0)
    cfg = SmolVLAConfig(vlm=TINY)
    model = SmolVLAPolicy(vlm, cfg)
    assert cfg.block_kinds() == ["self", "cross"] and cfg.expert_width == 24
    assert not any(p.requires_grad for p in model.vlm.parameters())
    assert len(model.vlm.layers) == 2  # layer skipping: N = L / 2
    ctx = model.encode(batch)
    noisy, tau = torch.randn(batch.size, 16, 2), torch.rand(batch.size)
    base = model.velocity(ctx, noisy, tau)
    changed = noisy.clone()
    changed[:, 8:] += 1.0
    assert torch.allclose(
        model.velocity(ctx, changed, tau)[:, :8], base[:, :8], atol=1e-5
    )  # causal: the past ignores the future
    assert not torch.allclose(model.velocity(ctx, changed, tau)[:, 8:], base[:, 8:], atol=1e-4)
    loss = model.compute_loss(batch)
    loss.total.backward()
    assert torch.isfinite(loss.total) and model.blocks[0].attn.q.weight.grad is not None
    assert model.vlm.vision.patchify.weight.grad is None


@pytest.mark.parametrize("layout", ["interleaved", "cross", "joint"])
def test_smolvla_layouts_and_sampling(vlm, batch, layout):
    model = SmolVLAPolicy(vlm, SmolVLAConfig(vlm=TINY, layout=layout))
    out = model.sample_actions(batch)
    assert out.shape == (batch.size, 16, MAX_ACTION_DIM)
    ctx = model.encode(batch)
    _, weights = model.velocity(
        ctx, torch.randn(batch.size, 16, 2), torch.rand(batch.size), return_weights=True
    )
    assert len(weights) == len(model.blocks)


def test_smolvla_state_placement_and_objective(vlm, batch):
    for kw in (
        dict(state_in="suffix"),
        dict(objective="regression"),
        dict(self_attention="bidirectional"),
    ):
        m = SmolVLAPolicy(vlm, SmolVLAConfig(vlm=TINY, **kw))
        assert torch.isfinite(m.compute_loss(batch).total)
        assert m.sample_actions(batch).shape[0] == batch.size
    shallow = SmolVLAPolicy(vlm, SmolVLAConfig(vlm=TINY, skip_to=1)).parameter_counts()
    deep = SmolVLAPolicy(vlm, SmolVLAConfig(vlm=TINY, skip_to=4)).parameter_counts()
    assert shallow["vlm"] < deep["vlm"] and shallow["expert"] < deep["expert"]


# ------------------------------------------------------------------------------------------------ GR00T
def test_category_specific_linear_routes_each_sample():
    torch.manual_seed(0)
    layer = CategorySpecificLinear(3, 4, 5)
    x, cat = torch.randn(6, 4), torch.tensor([0, 1, 2, 0, 1, 2])
    manual = torch.stack([x[i] @ layer.weight[cat[i]] + layer.bias[cat[i]] for i in range(6)])
    assert torch.allclose(layer(x, cat), manual, atol=1e-6)
    assert torch.allclose(layer(x[:, None].expand(-1, 3, -1), cat)[:, 1], manual, atol=1e-6)


def test_adaln_starts_as_plain_layernorm():
    norm = AdaLayerNorm(16)
    x, temb = torch.randn(2, 5, 16), torch.randn(2, 16)
    assert torch.allclose(norm(x, temb), torch.nn.functional.layer_norm(x, (16,)), atol=1e-5)


def test_groot_n1_vs_n15_configuration(vlm):
    n1, n15 = GR00TConfig.n1(vlm=TINY), GR00TConfig.n1_5(vlm=TINY)
    assert (
        n1.vlm_layer == 2
        and n1.freeze == "language"
        and not n1.vl_layernorm
        and n1.flare_weight == 0
    )
    assert n15.freeze == "all" and n15.vl_layernorm and n15.flare_weight == 0.2
    m1, m15 = GR00TPolicy(vlm, n1), GR00TPolicy(vlm, n15)
    assert any(p.requires_grad for p in m1.vlm.vision.parameters())  # N1 trains the vision encoder
    assert not any(p.requires_grad for p in m15.vlm.parameters())  # N1.5 freezes the whole VLM
    assert m15.future_tokens is not None and m1.future_tokens is None
    with pytest.raises(ValueError):
        GR00TConfig(vlm=TINY, freeze="sometimes")


@pytest.mark.parametrize("n15", [False, True])
def test_groot_loss_sampling_and_masking(vlm, batch, n15):
    torch.manual_seed(0)
    cfg = GR00TConfig.n1_5(vlm=TINY) if n15 else GR00TConfig.n1(vlm=TINY)
    model = GR00TPolicy(vlm, cfg)
    loss = model.compute_loss(batch)
    loss.total.backward()
    assert torch.isfinite(loss.total) and ("flare" in loss.metrics) == n15
    out = model.sample_actions(batch)
    assert out.shape == (batch.size, 16, MAX_ACTION_DIM)
    pad = (1 - batch.action_mask)[:, None, :].expand_as(out).bool()
    assert out[pad].abs().max() == 0  # padded action dimensions stay exactly zero
    if n15:
        assert model.future_tokens.grad is not None
        t = model.future_targets(batch.future_images)
        assert t.shape == (batch.size, cfg.num_future_tokens, TINY.width)


def test_groot_video_only_samples_do_not_contribute_flow_loss(vlm, batch):
    model = GR00TPolicy(vlm, GR00TConfig.n1_5(vlm=TINY))
    video = batch.replace(flow_weight=torch.zeros(batch.size))
    loss = model.compute_loss(video)
    assert (
        loss.flow.item() == 0.0 and loss.metrics["flare"] > 0
    )  # only FLARE trains from action-less video


# ------------------------------------------------------------------------------------------------ evaluation / async
def test_embodied_evaluator_with_oracle_policy(world):
    gen = DemoGenerator(world)
    scen = make_embodied_scenarios(gen, "gripper3d", 8, seed=3)
    emb = EMBODIMENTS["gripper3d"]

    def oracle(b: VLABatch):
        out = np.zeros((b.size, 16, MAX_ACTION_DIM), np.float32)
        for k, s in enumerate(scen):
            p, g = b.state[k, :2].numpy().copy(), s.scene.objects[s.target]
            for h in range(16):
                d = g - p
                a = d / max(np.linalg.norm(d), 1e-8) * min(0.9, np.linalg.norm(d) / world.max_step)
                near = np.linalg.norm(d) < 0.15
                out[k, h] = emb.action_from_ee(a, p, g).tolist() + [0.0]
                out[k, h, 2] = 1.0 if near else -1.0
                p = world.step(p, a)
        return torch.from_numpy(out)

    res = EmbodiedEvaluator(world, 16, 8, 40).run(oracle, scen)
    assert res.success.all()


def test_action_queue_merge_modes():
    q = ActionQueue()
    q.merge(np.arange(4, dtype=float)[:, None], "replace")
    assert len(q) == 4
    q.merge(np.full((6, 1), 9.0), "keep_old")
    assert [float(q.pop()[0]) for _ in range(4)] == [0.0, 1.0, 2.0, 3.0] and len(q) == 2
    q2 = ActionQueue()
    q2.merge(np.zeros((2, 1)), "replace")
    q2.merge(np.full((3, 1), 2.0), "average")
    assert [float(q2.pop()[0]) for _ in range(3)] == [1.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        q2.merge(np.zeros((1, 1)), "nonsense")


def test_async_beats_sync_idle_time_with_slow_robot():
    world = PointWorld(max_step=0.02)
    scen = make_embodied_scenarios(DemoGenerator(world), "point2d", 6, seed=5)
    cur = {}

    def oracle(image, state, task):
        s = cur["s"]
        p, g, out = state[:2].copy(), s.scene.objects[s.target], []
        for _ in range(16):
            d = g - p
            a = d / max(np.linalg.norm(d), 1e-8) * min(0.9, np.linalg.norm(d) / world.max_step)
            out.append(a)
            p = world.step(p, a)
        return np.array(out)

    def run(g):
        res = []
        for s in scen:
            cur["s"] = s
            cfg = AsyncConfig(threshold=g, latency=4, max_steps=100)
            res.append(
                AsyncInferenceRunner(world, oracle, cfg, rng=np.random.default_rng(0)).run(
                    s.scene, s.task, s.target
                )
            )
        return res

    sync, asyn = run(0.0), run(0.7)
    assert all(r.success for r in sync + asyn)
    assert np.mean([r.idle_steps for r in asyn]) < np.mean([r.idle_steps for r in sync])
    assert np.mean([r.steps for r in asyn]) < np.mean([r.steps for r in sync])
    assert np.mean([r.inferences for r in asyn]) > np.mean(
        [r.inferences for r in sync]
    )  # reactivity costs compute


# ------------------------------------------------------------------------------------------------ latent actions
def test_latent_action_model_trains_and_standardises(world, demos):
    torch.manual_seed(0)
    lam = LatentActionModel(hidden=8, codebook_size=4)
    sampler = frame_pair_sampler(EmbodiedSource(world, demos, EMBODIMENTS["lapa"]), 16)
    hist = train_latent_action_model(lam, sampler, steps=30, batch_size=16)
    assert len(hist.losses) == 30
    x, f, extras = sampler(64, np.random.default_rng(0))
    assert extras.shape == (64, 4)
    z = lam.latent(x, f)
    assert z.shape == (64, LATENT_DIM) and abs(float(z.mean())) < 0.5
    img = lam.decode(x, lam.codebook.weight[0][None].expand(64, -1))
    assert img.shape == x.shape
    assert lam.reset_dead_codes(torch.zeros(4)) in (0, 4)


def test_video_only_lapa_samples_need_no_latent_function(world, demos):
    src = EmbodiedSource(world, demos, EMBODIMENTS["lapa"], flow_weight=0.0)
    b = collate(
        src.sample_items(4, np.random.default_rng(0))
    )  # no latent_fn: they only train FLARE
    assert b.flow_weight.sum() == 0 and b.future_images is not None
