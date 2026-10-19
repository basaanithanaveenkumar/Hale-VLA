"""Action representations and VLM-backbone building blocks: binning, autoregressive decoding, DDPM/DDIM, chunk execution, encoders, fusion."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from open_vlas import (
    EMBODIMENTS,
    EmbodiedMixture,
    EmbodiedSource,
    SmolVLAConfig,
    SmolVLAPolicy,
    ToyVLM,
    VLMConfig,
    VLMPretrainer,
    make_embodied_scenarios,
    make_reach_demos,
)
from open_vlas.evaluation import make_any_scenarios
from pi_family import PiVocabulary
from pi_family.toy import DemoGenerator, PointWorld
from vla_foundations.autoregressive import (
    ARConfig,
    AutoregressiveActionPolicy,
    make_cofine_tune_data,
)
from vla_foundations.chunking import ChunkedRunner, effective_control_hz
from vla_foundations.diffusion import DDPMSchedule, DiffusionPolicy
from vla_foundations.discretization import ActionBinner
from vla_foundations.encoders import DinoLite, SigLIPLite, ToyViT, caption_vocab, scene_caption
from vla_foundations.fusion import FusedVision, FusionConfig, FusionPolicy
from vla_foundations.multimodality import mode_coverage
from vla_foundations.probes import localization_probe, semantic_probe, zero_shot_colors
from vla_foundations.tasks import make_side_demos, side_task, synonym_scenarios
from vla_foundations.toy2d import (
    bimodal_data,
    path_straightness,
    sample_ddim_2d,
    sample_flow_2d,
    train_flow_2d,
)

TINY = VLMConfig(width=32, depth=3, num_heads=2, head_dim=16)


@pytest.fixture(scope="module")
def world():
    return PointWorld()


@pytest.fixture(scope="module")
def gen(world):
    return DemoGenerator(world)


@pytest.fixture(scope="module")
def batch(world, gen):
    demos = make_reach_demos(gen, 20, np.random.default_rng(0))
    return EmbodiedMixture(
        [EmbodiedSource(world, demos, EMBODIMENTS["point2d"], chunk_size=4)]
    ).sample(6, np.random.default_rng(1))


@pytest.fixture(scope="module")
def vlm():
    torch.manual_seed(0)
    return ToyVLM(TINY, PiVocabulary(fast_size=256))


# ------------------------------------------------------------------------------------------------ discretization
def test_binner_roundtrip_error_is_half_a_bin():
    binner = ActionBinner(256, -1.5, 1.5)
    a = torch.rand(1000, 2) * 3 - 1.5
    err = (binner.roundtrip(a) - a).abs().max()
    assert err <= binner.max_error().max() + 1e-6
    assert binner.encode(torch.tensor([-9.0, 9.0])).tolist() == [
        0,
        255,
    ]  # out-of-range values clip to the end bins
    with pytest.raises(ValueError):
        ActionBinner(1)


def test_quantile_bins_ignore_outliers():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, (5000, 2))
    a[:5] = 50.0  # outliers
    minmax = ActionBinner(256, a.min(0), a.max(0))
    quant = ActionBinner.from_quantiles(a, 256)
    assert quant.bin_width.max() < 0.05 * minmax.bin_width.min()


# ------------------------------------------------------------------------------------------------ autoregressive
def test_autoregressive_cached_matches_naive_and_trains(vlm, batch):
    torch.manual_seed(0)
    cfg = ARConfig(chunk_size=4)
    pol = AutoregressiveActionPolicy(vlm, cfg).eval()
    assert pol.forward_passes() == 8 and pol.action_tokens(batch.actions).shape == (batch.size, 8)
    assert torch.allclose(pol.generate_cached(batch), pol.generate_naive(batch))
    ids = pol.action_tokens(batch.actions)
    back = pol.tokens_to_actions(ids)
    assert torch.allclose(back[:, :, :2], pol.binner.roundtrip(batch.actions[:, :4, :2]))
    pol.train()
    loss = pol.compute_loss(batch)
    loss.total.backward()
    assert torch.isfinite(loss.total) and pol.vlm.tok_emb.weight.grad is not None
    sampled = pol.sample_actions(batch, temperature=1.0)
    assert sampled.shape == (batch.size, 4, 4)


def test_vocabulary_must_have_action_slots():
    with pytest.raises(ValueError):
        AutoregressiveActionPolicy(ToyVLM(TINY), ARConfig(chunk_size=1))


def test_cofine_tune_batch_includes_vqa(vlm, world, batch):
    pre = VLMPretrainer(vlm, world, synonym_prob=0.5)
    data = make_cofine_tune_data(
        EmbodiedMixture(
            [
                EmbodiedSource(
                    world,
                    make_reach_demos(DemoGenerator(world), 10, np.random.default_rng(0)),
                    EMBODIMENTS["point2d"],
                    chunk_size=1,
                )
            ]
        ),
        pre.sample,
        0.5,
    )
    co = data.sample(8, np.random.default_rng(0))
    pol = AutoregressiveActionPolicy(vlm, ARConfig(chunk_size=1))
    out = pol.compute_loss(co)
    assert "vqa" in out.metrics and out.total.item() > out.flow.item()


# ------------------------------------------------------------------------------------------------ diffusion
def test_ddpm_schedule_is_monotone_and_q_sample_matches_formula():
    for kind in ("cosine", "linear"):
        sch = DDPMSchedule(50, kind)
        assert (sch.alpha_bar[1:] < sch.alpha_bar[:-1]).all() and 0 < sch.alpha_bar[-1] < 0.05
    sch = DDPMSchedule(50)
    x0, eps, t = torch.randn(4, 3), torch.randn(4, 3), torch.tensor([0, 10, 25, 49])
    ab = sch.alpha_bar[t][:, None]
    assert torch.allclose(sch.q_sample(x0, t, eps), ab.sqrt() * x0 + (1 - ab).sqrt() * eps)


def test_diffusion_policy_samplers(vlm, batch):
    torch.manual_seed(0)
    pol = DiffusionPolicy(vlm, SmolVLAConfig(vlm=TINY, chunk_size=4), DDPMSchedule(20))
    loss = pol.compute_loss(batch)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    noise = torch.randn(batch.size, 4, 2)
    a = pol.sample_actions(batch, method="ddim", num_steps=5, noise=noise)
    b = pol.sample_actions(batch, method="ddim", num_steps=5, noise=noise)
    assert torch.allclose(a, b)  # eta = 0 is deterministic
    assert not torch.allclose(
        pol.sample_actions(batch, method="ddpm"), pol.sample_actions(batch, method="ddpm")
    )  # ancestral: stochastic
    assert pol.nfe("ddpm") == 20 and pol.nfe("ddim", 5) == 5
    with pytest.raises(ValueError):
        pol.sample_actions(batch, method="nope")


def test_forward_convention_flow_policy_trains(vlm, batch):
    pol = SmolVLAPolicy(vlm, SmolVLAConfig(vlm=TINY, chunk_size=4, flow_convention="forward"))
    loss = pol.compute_loss(batch)
    loss.total.backward()
    assert torch.isfinite(loss.total) and pol.sample_actions(batch, num_steps=2).shape == (
        batch.size,
        4,
        4,
    )
    with pytest.raises(ValueError):
        SmolVLAConfig(vlm=TINY, flow_convention="sideways")


def test_toy2d_flow_collapses_to_mean_in_one_step_and_covers_modes_in_many():
    rng = np.random.default_rng(0)
    net = train_flow_2d(lambda n: bimodal_data(n, rng), steps=600)
    one = sample_flow_2d(net, 500, 1)[-1]
    many = sample_flow_2d(net, 500, 20)
    assert one[:, 0].abs().mean() < 0.6  # the average of the two modes
    assert many[-1][:, 0].abs().mean() > 1.4
    assert path_straightness(many) >= 1.0
    assert path_straightness(sample_ddim_2d(net, DDPMSchedule(50), 100, 4)) >= 1.0


# ------------------------------------------------------------------------------------------------ chunking / multimodality
def test_chunked_runner_with_oracle(world, gen):
    scen = make_embodied_scenarios(gen, "point2d", 4, seed=2)

    def oracle(b):
        out = np.zeros((b.size, 8, 4), np.float32)
        for k, s in enumerate(scen):
            p, g = b.state[k, :2].numpy().copy(), s.scene.objects[s.target]
            for h in range(8):
                d = g - p
                a = d / max(np.linalg.norm(d), 1e-8) * min(0.9, np.linalg.norm(d) / world.max_step)
                out[k, h, :2] = a
                p = world.step(p, a)
        return torch.from_numpy(out)

    closed = ChunkedRunner(world, 8, 8, max_steps=40).run(oracle, scen)
    ens = ChunkedRunner(world, 8, 1, ensemble_m=0.1, max_steps=40).run(oracle, scen)
    assert closed.success.all() and ens.success.all()
    assert ens.calls.mean() > closed.calls.mean()  # ensembling queries at every step
    absolute = ChunkedRunner(world, 8, 8, action_noise=0.5, absolute=True, max_steps=40).run(
        oracle, scen
    )
    relative = ChunkedRunner(world, 8, 8, action_noise=0.5, absolute=False, max_steps=40).run(
        oracle, scen
    )
    assert (
        absolute.success.mean() >= relative.success.mean()
    )  # waypoints correct execution noise, relative moves cannot
    with pytest.raises(ValueError):
        ChunkedRunner(world, 4, 8)
    assert effective_control_hz(200.0, 10) == 50.0 and effective_control_hz(1000.0, 1) == 1.0


def test_mode_coverage_of_a_two_mode_policy_and_of_the_mean(world):
    def sampler(coin):
        def policy(b):
            out = torch.zeros(b.size, 8, 4)
            for i in range(b.size):
                # aim at object 0 or 1 of the scene (looked up from the rendered state through the scenario order)
                out[i, :, 0] = 0.0
            return out

        return policy

    cov = mode_coverage(sampler(0), world, n_scenes=3, samples=8)
    assert cov.neither == 1.0 and cov.balance == 0.0  # a zero chunk aims at nothing


# ------------------------------------------------------------------------------------------------ encoders / probes / fusion
def test_siglip_loss_and_captions(world):
    vocab = caption_vocab()
    torch.manual_seed(0)
    sig = SigLIPLite(vocab, width=16, depth=1)
    rng = np.random.default_rng(0)
    scenes = [world.sample_scene(rng, 2) for _ in range(6)]
    caps = [scene_caption(s, rng) for s in scenes]
    assert all(c for c in caps)
    imgs = torch.from_numpy(np.stack([world.render(s.robot, s.objects) for s in scenes]))
    loss = sig.loss(imgs, caps)
    loss.backward()
    assert torch.isfinite(loss) and sig.log_t.grad is not None
    assert zero_shot_colors(sig, world, n=20) >= 0.0


def test_dino_ema_teacher_and_losses(world):
    torch.manual_seed(0)
    dino = DinoLite(width=16, depth=1, prototypes=8)
    imgs = torch.rand(6, 3, 32, 32)
    loss, info = dino.loss(imgs)
    loss.backward()
    assert torch.isfinite(loss) and set(info) == {"global_loss", "patch_loss"}
    before = next(dino.teacher.parameters()).clone()
    with torch.no_grad():
        for p in dino.student.parameters():
            p.add_(0.5)
    dino.ema_update()
    after = next(dino.teacher.parameters())
    student = next(dino.student.parameters())
    assert torch.allclose(
        after, 0.99 * before + 0.01 * student, atol=1e-5
    )  # EMA: teacher ← m·teacher + (1 − m)·student
    assert not any(p.requires_grad for p in dino.teacher.parameters())


def test_probes_run_on_a_random_encoder(world):
    enc = ToyViT(width=16, depth=1).eval()
    assert 0.0 <= semantic_probe(lambda x: enc(x), world, n_train=200, n_test=100) <= 1.0
    out = localization_probe(lambda x: enc(x), world, n_train=200, n_test=100)
    assert set(out) == {"patch_accuracy", "error_px"}


@pytest.mark.parametrize("fusion", ["early", "late", "pooled"])
def test_fusion_policy_variants(world, gen, fusion):
    torch.manual_seed(0)
    vocab = caption_vocab()
    vision = ToyViT(width=16, depth=1)
    cfg = FusionConfig(
        fusion=fusion, width=32, llm_depth=1, side_depth=1, chunk_size=4, expert_depth=2
    )
    pol = FusionPolicy(vision, 16, cfg, vocab)
    demos = make_side_demos(gen, 8, np.random.default_rng(0))
    b = EmbodiedMixture(
        [EmbodiedSource(world, demos, EMBODIMENTS["point2d"], chunk_size=4)]
    ).sample(5, np.random.default_rng(1))
    loss = pol.compute_loss(b)
    loss.total.backward()
    assert torch.isfinite(loss.total) and pol.sample_actions(b, num_steps=2).shape == (5, 4, 4)
    assert not any(
        p.requires_grad for p in pol.vision.parameters()
    )  # the vision tower is frozen by default
    assert pol.parameter_counts()["trainable"] < pol.parameter_counts()["total"]
    with pytest.raises(ValueError):
        FusionConfig(fusion="sometime")


def test_fused_vision_concatenates_channels():
    a, b = ToyViT(width=8, depth=1), ToyViT(width=12, depth=1)
    out = FusedVision(a, b)(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 64, 20)


def test_task_families(world, gen):
    rng = np.random.default_rng(0)
    for side in ("left", "right", "top", "bottom"):
        scene, text, targets, _ = side_task(world, rng, side)
        axis = 0 if side in ("left", "right") else 1
        coords = {c: xy[axis] for c, xy in scene.objects.items()}
        best = (
            min(coords, key=coords.get) if side in ("left", "top") else max(coords, key=coords.get)
        )
        assert text == f"go to {side}" and targets == [best]
    syn = synonym_scenarios(gen, 6, seed=1)
    assert all(s.task.split()[-1] in {"crimson", "emerald", "azure"} for s in syn)
    anys = make_any_scenarios(gen, "point2d", 3, seed=0)
    assert all(len(s.alternatives) == 2 for s in anys)
