# Chapter 9 — Race Day: Training, Watching the Curves, and Debugging

*[← Teaching Style with a GAN: AMP](08-amp.md) · [Contents](README.md) · [Next: From Checkpoint to Concrete Floor →](10-play-export-deploy.md)*

---

You know every part of the machine now. Time to turn the key.

## 9.1 The pre-flight check: the quick test

The README insists on a short run first, and it's right to. A full run takes
hours; discovering a broken install at iteration 3 is much better than at
iteration 3,000.

```bash
./isaac_asimov.sh --train \
    --task Asimov1-Velocity-AMP-v0 --num_envs 128 --headless --max_iterations 100
```

The README estimates about ten minutes on an RTX 4090. With 128 robots and 100
iterations, you won't get a walking policy; the point is that every part of
the pipeline (Isaac Sim startup, URDF import, motion-clip loading, the AMP
dimension checks, logging, checkpointing) runs end to end.

What you should see in the terminal:

1. Isaac Sim starting (a lot of extension-loading output; it's normal).
2. `[INFO] Logging experiment in directory: .../logs/rsl_rl/asimov_velocity_amp`.
3. Isaac Lab's manager summaries: tables listing every observation term with
   its shape, every reward term with its weight, every event. **Read these
   tables once.** They're the ground truth of what your config produced.
4. Per-iteration training logs from RSL‑RL.
5. `Training time: ... seconds` at the end.

> ⚠️ **Pothole: skipping the manager tables**
> The observation manager prints each group's terms and dimensions at
> startup. If you've changed the config and the `policy` group isn't 78-d
> anymore, you've just changed the deployment interface. This table is where
> you'd notice.

## 9.2 The full runs

**AMP (recommended by the README):**

```bash
./isaac_asimov.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless
```

**Plain PPO baseline:**

```bash
./isaac_asimov.sh --train --task Asimov1-Velocity-v0 --num_envs 4096 --headless
```

The README says the baseline was trained with 4,096 environments on an A6000
or RTX PRO 6000, and has been tested on A6000, PRO 6000, 4090 and 3090. If you
run out of GPU memory, lower `--num_envs`, knowing that the policy may take
longer to converge or be less stable for the same number of iterations.

> 🔧 **Under the Hood: what `--num_envs` really changes**
> Halving the environments halves the batch per iteration (and the AMP
> replay buffer fills with fewer, less diverse transitions). But
> `num_mini_batches` stays 4, so each gradient step also sees half as many
> samples, with noisier gradients. If you have to drop to 1,024 envs,
> consider raising `num_steps_per_env` to keep the batch size up. The
> trade-off is exactly the one you know from supervised learning: smaller
> batches, noisier steps.

**Multi-GPU** (two GPUs, 4,096 envs each):

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/rsl_rl/train.py \
    --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless --distributed
```

In `train.py`, `--distributed` pins each process to `cuda:<local_rank>` and
offsets the seed by the rank, so each GPU's environments explore differently:

```python
if args_cli.distributed:
    env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    agent_cfg.device = f"cuda:{app_launcher.local_rank}"
    seed = agent_cfg.seed + app_launcher.local_rank
    env_cfg.seed = seed
    agent_cfg.seed = seed
```

Distributed training on CPU is explicitly rejected with an error.

### Useful flags

From `train.py` and `cli_args.py`:

| Flag | What it does |
|------|--------------|
| `--task <id>` | Which registered task |
| `--num_envs <n>` | Override the number of environments |
| `--max_iterations <n>` | Override the iteration count |
| `--seed <n>` | Seed (`-1` picks a random one) |
| `--headless` | No GUI (much faster) |
| `--video`, `--video_length`, `--video_interval` | Record rollout clips during training (turns on cameras) |
| `--resume`, `--load_run <dir>`, `--checkpoint <file>` | Resume training from a previous run |
| `--experiment_name`, `--run_name` | Control the log folder names |
| `--logger {tensorboard,wandb,neptune}`, `--log_project_name` | Logging backend |
| `--export_io_descriptors` | Export a description of the env's inputs/outputs |
| `--device` | Simulation device (from Isaac Lab's `AppLauncher`) |

Plus Hydra overrides for any config field (Chapter 3).

### Where things land

```
logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>_<run_name>/
├── params/
│   ├── env.yaml         ← the full, resolved environment config
│   └── agent.yaml       ← the full, resolved agent config
├── model_0.pt, model_500.pt, ...   ← checkpoints every 500 iterations
├── events.out.tfevents...          ← TensorBoard logs (default logger)
├── (git state/diff files, via runner.add_git_repo_to_log)
└── videos/train/ (if --video)
```

For the AMP task, `<experiment_name>` is `asimov_velocity_amp` and
`<run_name>` is `amp`; for PPO they're `asimov1_velocity` and `ppo`.

`params/env.yaml` and `params/agent.yaml` are dumped by `train.py` after all
overrides are applied. **They're your experiment's source of truth.** When
someone asks "what reward weights did run X use?", the answer is in that
folder, not in whatever the code says today.

> 🚗 **Driving Déjà Vu**
> This is the same discipline as versioning your model configs and dataset
> manifests with every training run. RSL‑RL even snapshots the git state of
> the code (`add_git_repo_to_log`). Treat these folders like experiment
> records, not scratch space.

Also note `init_at_random_ep_len=True` in the `runner.learn(...)` call. At the
start, each environment's episode counter is randomized, so resets are
staggered from the beginning instead of all 4,096 robots timing out on the
same step (which would make the data very non-stationary).

## 9.3 Reading the dashboard

Launch TensorBoard on the logs directory:

```bash
tensorboard --logdir logs/rsl_rl
```

You'll see several families of curves. Exact names can vary a little between
RSL‑RL and Isaac Lab versions; here's what to look for and what it means.

### The headline metrics

- **Mean reward** and **mean episode length**. Episode length is the easiest
  "is it alive?" signal. Early on, robots fall within a second or two. As
  training progresses, episode length climbs toward the 1,000-step
  maximum (20 s). If it plateaus low, the robot is falling.
- **`Episode_Termination/fell_over`** vs **`Episode_Termination/time_out`**:
  the share of episodes ending in a fall vs a timeout. You want the fall rate
  to trend toward zero.

### The reward breakdown

Isaac Lab logs `Episode_Reward/<term>` for each of the 18 terms. This is
where you diagnose *what* the policy is doing:

- Is `track_linear_velocity` rising? It's learning to walk where it's told.
- Is a penalty growing? Something undesirable is increasing. Maybe
  `self_collisions` is up because the arms swing too wide.
- Is one penalty much larger than everything else? It's likely dominating the
  gradient, and the policy is optimizing *that* instead of walking.

### The custom metrics

The repo's reward functions log extra physical quantities under `Metrics/`:

| Metric | Healthy trend |
|--------|---------------|
| `Metrics/air_time_mean` | Settles around a plausible swing duration (a few tenths of a second) |
| `Metrics/peak_height_mean` | Approaches the 0.10 m target |
| `Metrics/slip_velocity_mean` | Low and falling |
| `Metrics/landing_force_mean` | Stable, not climbing |
| `Metrics/max_contact_force` | Mostly below the 350 N limit |
| `Metrics/stumble_rate` | Low |
| `Metrics/angular_momentum_mean` | Low and stable |
| `Metrics/pose_disturbance_std_scale` | Near 1.0 when undisturbed, spikes after pushes |

These are physical units you can reason about. "Peak swing height 3 cm" is
immediately meaningful in a way that "reward 0.013" is not.

### The PPO internals

- **Value loss**: should drop and then stay moderate. A value loss that
  explodes usually means rewards are huge or badly scaled.
- **Surrogate loss**: small and noisy; not very informative alone.
- **Entropy / action noise std**: should *slowly* decline. If it collapses
  quickly, exploration died early (consider more `entropy_coef`). If it
  stays high forever, the policy never became confident.
- **Learning rate**: with the adaptive schedule, it moves up and down. If
  it's pinned at the 1e‑5 floor, the policy's KL is constantly too high,
  meaning something is unstable.

### The AMP internals

- **`Train/mean_amp_reward`** and **`Train/mean_task_reward`**: how the two
  sources of reward evolve.
- **`Train/mean_discriminator_prediction`**: the average discriminator
  output on the policy's own transitions during rollouts.
- **`amp_expert_pred`** and **`amp_policy_pred`** (returned in the loss
  dict): the average discriminator output on expert vs policy transitions
  during updates.

The discriminator is trained toward +1 for expert and −1 for policy. How to
read the gap:

| `amp_expert_pred` | `amp_policy_pred` | Interpretation |
|---:|---:|---|
| ≈ +1 | ≈ −1 | Discriminator wins easily. AMP reward ≈ 0. The policy's style is far from the clip, or the discriminator is too strong. |
| ≈ +0.5 | ≈ 0 | Healthy competition. The policy gets meaningful, graded style reward. |
| ≈ 0 | ≈ 0 | The discriminator can't tell them apart: either the policy matches the style well, or the discriminator is too weak/regularized. |

If the discriminator dominates for a long time, levers include a higher
`amp_grad_pen_lambda`, more head weight decay, or a larger
`amp_update_interval` (train the discriminator less often).

## 9.4 A typical story of a training run

Every run is different, but locomotion runs commonly go through recognizable
phases. Knowing them saves you from killing a healthy run too early.

1. **The flailing (first few dozen iterations).** Robots start in random
   poses (±0.5 rad joint noise!) and collapse. Episode length is short. The
   PD controllers hold *something*, so it's not total chaos, but most robots
   fall within a second or two.
2. **The statue.** The policy discovers that not falling is worth a lot
   (every surviving step earns tracking and upright reward). It learns to stand
   and recover from the reset noise. Episode length jumps. Velocity tracking is
   still poor for moving commands.
3. **The shuffle.** It starts moving toward commanded velocities, often with
   tiny, sliding steps. `foot_clearance` and `foot_slip` penalties are high;
   `air_time` is low.
4. **The stepping.** Gait rewards kick in. Feet lift, swing height climbs
   toward 10 cm, tracking reward rises steeply.
5. **The polish (the long tail).** Thousands of iterations of slow
   improvement: smoother actions, better push recovery, more natural arm
   swing (angular momentum and AMP at work), cleaner standing.

> 🚗 **Driving Déjà Vu**
> Remember the long tail in AD, where the last 1% of scenarios took 90% of
> the effort? Same shape here. The robot "walks" early; it walks *well*,
> robustly enough to trust on hardware, much later. Don't judge a run at 10%
> of its budget.

## 9.5 The debugging field guide

| Symptom | Likely causes | Where to look / what to try |
|---------|---------------|-----------------------------|
| Crash at start: `Invalid motion file` | Clip missing (e.g., non-editable install moved paths) | `ASIMOV_1_MOTION_FILES`, `motions/` folder |
| Crash: `AMP observation group ... has N features but the motion dataset produces frames of M` | `AmpObsCfg` and `ASIMOV_1_AMP_OBS_TERMS` out of sync, or wrong joint list | `amp_env_cfg.py`, `MotionDatasetCfg.joint_names` |
| Crash: `AMPPPO expects an observation group 'amp'` | Using the AMP agent config with the non-AMP env | Task registration pairs (Chapter 3) |
| Crash: URDF not found | `asimov-1` submodule not fetched, or package installed non-editable | `INSTALL.md`, `ASIMOV_1_MODEL_DIR` |
| "Please install the correct version of RSL-RL" | RSL‑RL older than 5.0.1 | Reinstall pinned version |
| CUDA out of memory | Too many envs for your GPU | Lower `--num_envs` |
| Episode length never rises | Reward scale broken, termination too strict, or robot can't physically stand | Check `Episode_Termination/*`; play a checkpoint and *look* |
| Robot stands but won't walk | Tracking reward too weak vs posture/penalties; standing is a local optimum | Reward breakdown; check `pose` and `joint_deviation_l1` are gated as intended |
| Shuffling gait | Gait terms too weak, or terrain too easy | `foot_clearance`, `air_time`, `Metrics/peak_height_mean` |
| Jittery, buzzing joints | Action smoothness too weak | `action_rate_l2` weight; check action std isn't stuck high |
| Arms flailing or hitting the body | Arm posture too loose, self-collision too weak | `pose` σ for arm joints, `self_collisions` |
| Robot hops | Vertical velocity not penalized enough | `track_linear_velocity` includes `v_z`; `air_time` upper bound |
| NaNs | Physics blow-up from extreme actions or penetration | `max_depenetration_velocity`, observation clip (`amp_rollout_obs_clip`), lower learning rate |
| Great in Play, bad under randomization | Overfitting to nominal physics | Evaluate on the training config; sim-to-sim test |

The golden rule: **when in doubt, watch the robot.** Curves tell you *that*
something is wrong; video tells you *what*. Run `play.py` on the latest
checkpoint, or train with `--video`.

## 9.6 Running the unit tests

The repo includes `tests/test_amp_components.py`, five focused tests for the
AMP machinery:

| Test | What it pins down |
|------|-------------------|
| `test_feature_normalizer_matches_baseline_numpy_equations` | The normalizer's forward pass and running-stat update match hand-written NumPy |
| `test_discriminator_reward_and_raw_gradient_penalty_match_baseline_formulas` | The reward formula and the (raw-input) gradient penalty |
| `test_replay_buffer_keeps_state_pairs_aligned_across_wraparound` | State/next-state pairing survives ring-buffer wraparound |
| `test_amp_ppo_pairs_frames_and_blends_rewards_without_custom_runner` | `process_env_step` pairs frames correctly, applies the gate, blends rewards, and clips observations |
| `test_amp_ppo_clips_one_combined_actor_critic_gradient_norm` | Gradient clipping treats actor + critic as one vector |

These tests import only `torch`, `numpy`, `tensordict` and `rsl_rl`, not
Isaac Sim, so they run in seconds without a GPU simulator:

```bash
pytest          # pyproject.toml sets pythonpath=source/isaac_asimov, testpaths=tests
```

Notice what they test: not "does the robot walk?" (that's an expensive,
statistical question) but **the invariants that would silently corrupt
training if broken**: pairing, reward formulas, normalization math, clipping.
That's the right testing strategy for RL code, and it transfers directly to
any ML codebase.

## 9.7 Code hygiene

The repo uses `black` and `isort` (line length 120), `flake8`, and a
pre-commit config with basic checks (trailing whitespace, YAML/TOML validity,
merge conflicts, private-key detection). `pyright` runs in basic mode.
Run `pre-commit install` once, and your commits will be formatted to match.

## 🏁 Pit Stop

1. Why run the 128-env, 100-iteration test first?
2. Where do you find the exact config a past run used?
3. Which single curve is the quickest "is it alive?" check?
4. `amp_expert_pred ≈ +1` and `amp_policy_pred ≈ −1` for a long time: what's
   happening, and name one lever to change it.
5. Why don't the unit tests need Isaac Sim?

<details>
<summary>Answers</summary>

1. To validate the whole pipeline (install, assets, AMP checks, logging) in
   minutes rather than discovering failures hours into a run.
2. `logs/rsl_rl/<experiment>/<run>/params/env.yaml` and `agent.yaml`.
3. Mean episode length (and the fall vs timeout termination split).
4. The discriminator wins easily, so the AMP reward is near zero. Raise
   `amp_grad_pen_lambda`, raise head weight decay, or train the discriminator
   less often with `amp_update_interval`.
5. They test the algorithm components directly (`AMPPPO`, discriminator,
   replay buffer, normalizer), which depend only on PyTorch, NumPy, TensorDict
   and RSL‑RL.

</details>

---

*[← Teaching Style with a GAN: AMP](08-amp.md) · [Contents](README.md) · [Next: From Checkpoint to Concrete Floor →](10-play-export-deploy.md)*
