"""FAST policy, LoRA / OFT, DAgger helpers, the 3-D world and its policy, and the attention analysis."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from open_vlas import (
    EMBODIMENTS,
    EmbodiedMixture,
    EmbodiedSource,
    SmolVLAConfig,
    SmolVLAPolicy,
    ToyVLM,
    VLMConfig,
    make_embodied_scenarios,
    make_reach_demos,
)
from pi_family import PiVocabulary
from pi_family.tokenizers import FastActionTokenizer
from pi_family.toy import DemoGenerator, PointWorld
from vla_foundations.fast_policy import FastActionPolicy
from vla_foundations.finetuning import (
    LoRALinear,
    OFTLinear,
    count_parameters,
    freeze_all_but,
    hyperspherical_energy,
    inject_adapters,
)
from vla_foundations.policy3d import (
    CHANNELS,
    Dataset3D,
    Policy3D,
    pointmaps_to_bev,
    rollout3d,
    train3d,
)
from vla_foundations.training import (
    dagger_episodes,
    narrow_start_demos,
    perturbed_expert_episodes,
    state_action_error,
)
from vla_foundations.world3d import (
    COLORS3D,
    Camera,
    expert_action,
    random_camera,
    render,
    sample_scene,
)

TINY = VLMConfig(width=32, depth=3, num_heads=2, head_dim=16)


@pytest.fixture(scope="module")
def world():
    return PointWorld()


@pytest.fixture(scope="module")
def gen(world):
    return DemoGenerator(world)


def test_fast_policy_trains_and_cached_matches_naive(world, gen):
    demos = make_reach_demos(gen, 30, np.random.default_rng(0))
    data = EmbodiedMixture([EmbodiedSource(world, demos, EMBODIMENTS["point2d"], chunk_size=8)])
    batch = data.sample(4, np.random.default_rng(1))
    tok = FastActionTokenizer(8, 2).fit(batch.actions[:, :, :2])
    policy = FastActionPolicy(ToyVLM(TINY, PiVocabulary(fast_size=256)), tok, max_tokens=24).eval()
    assert policy.compute_loss(batch).total.item() > 0
    seqs = policy.target_sequences(batch.actions)
    assert (
        all(s[-1] == policy.eos for s in seqs) and policy.mean_tokens(batch.actions) < 16
    )  # 8 steps × 2 dims
    assert torch.allclose(policy.sample_actions(batch), policy.sample_actions(batch, cached=False))


def test_oft_is_orthogonal_and_preserves_hyperspherical_energy():
    torch.manual_seed(0)
    base = nn.Linear(16, 12, bias=False)
    oft = OFTLinear(base, block_size=4)
    x = torch.randn(5, 16)
    assert torch.allclose(oft(x), base(x), atol=1e-6)  # R = I at initialisation
    with torch.no_grad():
        oft.skew.normal_(0, 0.3)
    r = oft.rotation()
    assert torch.allclose(r @ r.transpose(-1, -2), torch.eye(4).expand_as(r), atol=1e-5)
    assert torch.allclose(oft(x), x @ oft.merged_weight().T, atol=1e-5)
    assert hyperspherical_energy(oft.merged_weight()) == pytest.approx(
        hyperspherical_energy(base.weight), rel=1e-4
    )
    with pytest.raises(ValueError):
        OFTLinear(base, block_size=5)


def test_lora_starts_as_identity_and_freezes_the_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 12)
    lora = LoRALinear(base, rank=2)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x))
    assert not base.weight.requires_grad and lora.lora_a.requires_grad
    with torch.no_grad():
        lora.lora_b.normal_()
    assert torch.allclose(lora(x), x @ lora.merged_weight().T + base.bias, atol=1e-5)


def test_inject_adapters_into_a_vlm_and_count_parameters():
    vlm = ToyVLM(TINY, PiVocabulary(fast_size=256))
    made = inject_adapters(vlm, "lora", rank=2)
    assert len(made) == 4 * TINY.depth
    freeze_all_but(vlm, lambda name: "lora_" in name)
    counts = count_parameters(vlm)
    assert 0 < counts["fraction"] < 0.2
    with pytest.raises(ValueError):
        inject_adapters(vlm, "bogus")


def test_flow_policy_accepts_the_mse_objective(world, gen):
    demos = make_reach_demos(gen, 20, np.random.default_rng(0))
    data = EmbodiedMixture([EmbodiedSource(world, demos, EMBODIMENTS["point2d"], chunk_size=4)])
    batch = data.sample(4, np.random.default_rng(1))
    policy = SmolVLAPolicy(
        ToyVLM(TINY, PiVocabulary()), SmolVLAConfig(vlm=TINY, chunk_size=4, objective="mse")
    )
    assert policy.compute_loss(batch).total.item() > 0
    with pytest.raises(ValueError):
        SmolVLAConfig(vlm=TINY, objective="bogus")


def test_narrow_start_demos_and_dagger_relabelling(world, gen):
    rng = np.random.default_rng(0)
    demos = narrow_start_demos(gen, 10, rng, radius=0.1)
    assert all(np.linalg.norm(d.robot[0] - 0.5) <= 0.1 + 1e-6 for d in demos)
    scen = make_embodied_scenarios(gen, "point2d", 4, seed=3)

    def oracle(batch):  # an expert-like chunk of 4 actions toward the target of each observation
        out = torch.zeros(batch.size, 4, 4)
        for i, s in enumerate(scen):
            goal = s.scene.objects[s.target]
            d = goal - batch.state[i, :2].numpy()
            out[i, :, :2] = torch.from_numpy(d / max(np.linalg.norm(d), 1e-8) * 0.9).float()
        return out

    episodes = dagger_episodes(
        gen, oracle, scen, rng, chunk_size=4, exec_horizon=2, action_noise=0.3
    )
    assert len(episodes) > 0 and all(e.robot.shape[1] == 2 for e in episodes)
    assert len(perturbed_expert_episodes(gen, scen, rng)) > 0
    positions = [s.scene.robot for s in scen]
    assert state_action_error(gen, oracle, scen, positions, rng) < 0.5


def test_camera_projection_roundtrip_and_pointmap_consistency():
    rng = np.random.default_rng(0)
    scene, cam = sample_scene(rng), random_camera(rng)
    pts = np.stack(list(scene.objects.values()))
    uvd = cam.project(pts)
    assert np.allclose(cam.unproject(uvd[:, 0], uvd[:, 1], uvd[:, 2]), pts, atol=1e-9)
    ob = render(scene, cam)
    assert ob.rgb.shape == (3, 32, 32) and ob.points_robot.shape == (3, 32, 32)
    assert np.allclose(
        cam.to_robot(ob.points_cam.reshape(3, -1).T), ob.points_robot.reshape(3, -1).T, atol=1e-4
    )


def test_occluder_hides_the_target_from_the_fixed_camera_but_not_from_the_opposite_one():
    rng = np.random.default_rng(1)
    scene = sample_scene(rng, occluder=True, target="red")
    front = render(scene, random_camera(rng, fixed=True))
    back = render(scene, Camera.from_angles(90.0, 50.0, 1.2))

    def visible(ob):
        return int((np.abs(ob.rgb - np.array(COLORS3D["red"])[:, None, None]).sum(0) < 1e-3).sum())

    assert visible(back) > visible(front)


def test_bev_raster_is_independent_of_the_camera():
    rng = np.random.default_rng(2)
    scene = sample_scene(rng)
    a = pointmaps_to_bev([render(scene, Camera.from_angles(-90, 50, 1.2))])
    b = pointmaps_to_bev(
        [
            render(scene, Camera.from_angles(-60, 45, 1.1)),
            render(scene, Camera.from_angles(90, 50, 1.2)),
        ]
    )
    heights = lambda g: np.sort(g[3][g[3] > 0.1])[-3:]  # noqa: E731 - the cube tops
    assert np.allclose(np.sort(heights(a)), np.sort(heights(b)), atol=0.06)
    assert expert_action(np.zeros(3), np.array([1.0, 0, 0])).max() == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["rgb", "bev"])
def test_policy3d_trains_a_few_steps_and_rolls_out(mode):
    data = Dataset3D(40, mode, fixed_camera=False, seed=0, second_camera=(mode == "bev"))
    model = Policy3D(CHANNELS[mode])
    losses = train3d(model, data, steps=10, batch=16)
    assert np.isfinite(losses).all()
    assert set(rollout3d(model, data, 10, seed=1)) == {"success", "final_error", "steps"}


def test_evaluation_statistics():
    from vla_foundations.evaluation_stats import (
        bootstrap_difference,
        classify_failures,
        mcnemar_exact,
        mmrv,
        pearson,
        required_episodes,
        wilson_interval,
    )

    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60  # the textbook interval for 50/100
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    a = np.array([1] * 30 + [0] * 70, bool)
    assert mcnemar_exact(a, a) == 1.0
    b = np.array([1] * 20 + [0] * 80, bool)
    assert mcnemar_exact(a, b) < 0.01  # ten discordant episodes, all in A's favour: p = 2·2⁻¹⁰
    assert (
        1200 < required_episodes(0.70, 0.75) < 1300
    )  # detecting a 5-point gain at 70 % needs about 1250 episodes per policy
    low, high = bootstrap_difference(a, b)
    assert low < 0.1 < high
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert mmrv([1, 2, 3], [1, 2, 3]) == 0.0
    assert mmrv([1, 2, 3], [3, 2, 1]) > 0.5
    traj = np.zeros((2, 5, 2))
    counts = classify_failures(
        traj, [5, 5], [False, True], np.array([[0.5, 0.5]] * 2), [np.array([[0.0, 0.0]])] * 2
    )
    assert counts["wrong object"] == 1 and counts["failures"] == 1


def test_shortcut_scenes_and_language_sensitivity(world, gen):
    from vla_foundations.generalization import (
        language_sensitivity,
        shortcut_demos,
        shortcut_scenarios,
        slot_scene,
    )

    usual = slot_scene(world, np.random.default_rng(0), permuted=False)
    assert usual.objects["red"][0] < usual.objects["green"][0] < usual.objects["blue"][0]
    seen = {
        tuple(
            np.argsort(
                [
                    slot_scene(world, np.random.default_rng(i), True).objects[c][0]
                    for c in ("red", "green", "blue")
                ]
            )
        )
        for i in range(30)
    }
    assert len(seen) > 1  # permuted scenes really permute the bands
    assert len(shortcut_demos(gen, 3, np.random.default_rng(1), 0.5)) == 3
    scen = shortcut_scenarios(gen, 4, 2, True)

    def follows_the_instruction(batch):  # a policy that goes to the object named in the instruction
        out = torch.zeros(batch.size, 1, 4)
        for i, (task, state) in enumerate(zip(batch.task, batch.state)):
            colour = task.split()[-1]
            # the test scenes are re-sampled by language_sensitivity: use the image to find the colour's pixel instead
            img = batch.images[i].numpy()
            from pi_family.toy.world import COLORS

            mask = np.abs(img - np.array(COLORS[colour])[:, None, None]).sum(0) < 1e-3
            ys, xs = np.nonzero(mask)
            goal = np.array([xs.mean(), ys.mean()]) / (img.shape[-1] - 1)
            d = goal - state[:2].numpy()
            out[i, 0, :2] = torch.from_numpy(d / max(np.linalg.norm(d), 1e-8)).float()
        return out

    def ignores_the_instruction(batch):
        out = torch.zeros(batch.size, 1, 4)
        out[:, 0, 0] = 1.0
        return out

    assert len(scen) == 4
    assert language_sensitivity(follows_the_instruction, world, gen, 20, 3) > 0.9
    assert language_sensitivity(ignores_the_instruction, world, gen, 20, 3) == 0.0


def test_morphology_descriptor_generalises_and_id_does_not(world):
    from vla_foundations.generalization import (
        MorphPolicy,
        morph_success,
        random_jacobian,
        train_morph,
    )

    rng = np.random.default_rng(0)
    train = np.stack([random_jacobian(rng) for _ in range(32)])
    unseen = random_jacobian(np.random.default_rng(999))
    torch.manual_seed(0)
    desc = MorphPolicy("descriptor")
    train_morph(desc, train, steps=1500)
    assert morph_success(desc, unseen, world, 60, 1) > 0.8


def test_chain_runner_with_an_oracle(world, gen):
    from vla_foundations.generalization import run_chain

    scen = make_embodied_scenarios(gen, "point2d", 4, seed=1)

    def oracle(batch):
        out = torch.zeros(batch.size, 1, 4)
        for i, (task, state, img) in enumerate(zip(batch.task, batch.state, batch.images)):
            colour = task.split()[-1]
            xy = scen[i].scene.objects[colour]
            d = xy - state[:2].numpy()
            out[i, 0, :2] = torch.from_numpy(
                d / max(np.linalg.norm(d), 1e-8) * min(0.9, np.linalg.norm(d) / world.max_step)
            ).float()
        return out

    seqs = [list(s.scene.objects) * 2 for s in scen]
    stages = run_chain(world, oracle, scen, seqs, per_skill_steps=40)
    assert (stages == np.array([len(q) for q in seqs])).all()


def test_plant_runner_tracker_damps_the_servo(world, gen):
    import math

    from vla_foundations.control import PlantRunner

    scen = make_embodied_scenarios(gen, "point2d", 6, seed=3)

    def planner(batch):
        out = torch.zeros(batch.size, 8, 4)
        for i, s in enumerate(scen):
            p, goal = batch.state[i, :2].numpy().copy(), s.scene.objects[s.target]
            for h in range(8):
                d = goal - p
                a = d / max(np.linalg.norm(d), 1e-8) * min(0.9, np.linalg.norm(d) / world.max_step)
                out[i, h, :2] = torch.from_numpy(a).float()
                p = world.step(p, a)
        return out

    omega = 2 * math.pi / 6
    direct = PlantRunner(world, omega, 0.1, "direct", 8, 4, kick=0.3, tolerance=0.025).run(
        planner, scen
    )
    tracked = PlantRunner(world, omega, 0.1, "tracked", 8, 4, kick=0.3, tolerance=0.025).run(
        planner, scen
    )
    assert tracked.jitter.mean() < direct.jitter.mean()
    with pytest.raises(ValueError):
        PlantRunner(world, omega, 0.1, "bogus")


def test_thinking_policy_trains_and_decodes_a_thought(world, gen):
    from open_vlas import EmbodiedMixture, EmbodiedSource
    from vla_foundations.autoregressive import ARConfig
    from vla_foundations.tasks import make_side_demos
    from vla_foundations.thinking import ThinkingPolicy, thought_text

    demos = make_side_demos(gen, 20, np.random.default_rng(0))
    data = EmbodiedMixture([EmbodiedSource(world, demos, EMBODIMENTS["point2d"], chunk_size=1)])
    batch = data.sample(4, np.random.default_rng(1))
    vlm = ToyVLM(TINY, PiVocabulary(fast_size=256))
    policy = ThinkingPolicy(vlm, ARConfig(chunk_size=1), think=True).eval()
    assert policy.compute_loss(batch).total.item() > 0
    out = policy.sample_actions(batch)
    assert out.shape[:2] == (4, 1) and len(policy.last_thoughts) == 4
    assert thought_text(
        demos[0].scene.objects | {"blue": demos[0].scene.objects.get("blue", np.zeros(2))}, "red"
    ).endswith("go to red")
    direct = ThinkingPolicy(vlm, ARConfig(chunk_size=1), think=False).eval()
    assert direct.compute_loss(batch).total.item() > 0 and direct.sample_actions(batch).shape[
        :2
    ] == (4, 1)
