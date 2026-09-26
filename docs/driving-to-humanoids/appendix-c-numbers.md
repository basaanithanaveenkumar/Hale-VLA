# Appendix C — The Numbers Card

*[← Appendix B](appendix-b-file-reference.md) · [Contents](README.md)*

---

Every important constant in `isaac_asimov`, on one page. Print it, pin it
next to your monitor.

## Robot

| Quantity | Value | Source |
|----------|-------|--------|
| Actuated joints | 23 (6 per leg, 1 waist, 5 per arm) | `ASIMOV_1_JOINT_NAMES` |
| Bodies in motion clip | 26 | `.npz` `body_names` |
| Spawn height (pelvis) | 0.639 m | `ASIMOV_1_STANDING_INIT_STATE` |
| Soft joint limit factor | 0.9 | `ASIMOV_1_DELAYED_CFG` |
| Action scale | 0.25 rad per unit | `ASIMOV_1_ACTION_SCALE` |
| Actuator delay | 0–5 physics steps (0–25 ms) | `DELAY_MIN_LAG`, `DELAY_MAX_LAG` |
| PhysX solver iterations | 8 position / 4 velocity | `articulation_props` |
| Foot site offset | (0.05, 0.0, −0.025) m from ankle-roll link | `FOOT_SITE_OFFSET` |

### Actuators

| Group | Kp | Kd | Effort (N·m) | Armature | Friction |
|-------|---:|---:|---:|---:|---:|
| hip pitch | 150 | 5 | 45 | 0.0698 | 0.70 |
| hip roll | 150 | 5 | 45 | 0.1400 | 0.20 |
| hip yaw | 150 | 5 | 28 | 0.0687 | 0.70 |
| knee | 150 | 5 | 45 | 0.0330 | 0.70 |
| ankle pitch | 110 | 5 | 40 | 0.0484 | 0.40 |
| ankle roll | 110 | 5 | 17 | 0.0484 | 0.40 |
| waist | 65 | 5 | 40 | 0.0698 | 0.70 |
| shoulder pitch | 57 | 5 | 30 | 0.1400 | 0.20 |
| shoulder roll | 86 | 5 | 25 | 0.0330 | 0.70 |
| shoulder yaw | 96 | 5 | 20 | 0.0687 | 0.70 |
| elbow + wrist | 40 | 2 | 12 | 0.0242 | 0.40 |

## Simulation and timing

| Quantity | Value |
|----------|-------|
| Physics step | 0.005 s (200 Hz) |
| Decimation | 4 |
| Policy step | 0.02 s (50 Hz) |
| Episode length | 20 s = 1,000 policy steps |
| Training envs | 4,096 |
| Play envs | 32 |
| Terrain | 9 × 21 tiles of 8 × 8 m, 20 m border; 50% flat, 50% rough (2/4/6 cm heights on a 10 cm grid) |
| Play terrain | 5 × 5 tiles, 10 m border |

## Commands (`twist`)

| Quantity | Training | Play |
|----------|----------|------|
| `vx` | −0.6 to 0.8 m/s | 0.6 to 0.8 m/s |
| `vy` | −0.5 to 0.5 m/s | −0.5 to 0.5 m/s |
| `ωz` | −0.8 to 0.8 rad/s | −0.6 to 0.6 rad/s |
| Resample | every 3–8 s | every 3–8 s |
| Standing envs | 20% | 20% |
| Heading envs | 30% (target heading 0, stiffness 0.5) | same |

## Observations

| Group | Dim | Noise | Notes |
|-------|----:|-------|-------|
| `policy` | 78 | on (off in Play) | gyro ×0.25 (delay 0–1), gravity (delay 0–2), command, joint pos (slot order), joint vel ×0.1 (slot order), last action (action order) |
| `critic` | 93 | off | policy terms clean (joint vel ×1.0) + base lin vel (3), foot height (2), air time (2), contact (2), contact forces (6) |
| `amp` | 46 | off | joint pos rel + joint vel, action order |

## Rewards (all multiplied by `dt = 0.02` inside Isaac Lab)

| Term | Weight | Key params |
|------|-------:|------------|
| track_linear_velocity | +5.0 | σ 0.5 |
| track_angular_velocity | +3.0 | σ 0.7071 |
| upright | +1.0 | σ √0.2, torso |
| pose | +1.0 | walk/run thresholds 0.1/1.5; slow weight 2.0 below 0.3 m/s; disturbance scale up to 2.5 |
| air_time | +0.5 | 0.05–0.5 s; gate 0.5 |
| foot_clearance | −2.0 | target 0.10 m; gate 0.05 |
| foot_swing_height | −0.25 | target 0.10 m; gate 0.05 |
| foot_slip | −0.1 | gate 0.05 |
| feet_orientation | −1.0 | |
| action_rate_l2 | −0.5 | |
| body_ang_vel | −0.08 | torso |
| angular_momentum | −0.03 | |
| dof_pos_limits | −1.0 | soft limits |
| soft_landing | −1e‑5 | gate 0.05 |
| joint_deviation_l1 | −1.0 | only when command < 0.1 |
| feet_stumble | −1.25 | horizontal > 4 × vertical |
| feet_contact_force_limit | −5e‑4 | 350 N |
| self_collisions | −1.0 | 10 N threshold |

## Randomization

| Event | Mode | Range |
|-------|------|-------|
| foot_friction | startup | 0.8–1.2 (64 buckets) |
| qpos0_rand | startup | ±0.02 rad per joint |
| base_com (torso) | startup | x 0–0.05 m, y 0, z 0.03–0.07 m |
| reset_base | reset | x, y ±0.5 m; yaw ±π; pitch ±0.15; roll ±0.1; pitch rate ±0.5, roll rate ±0.3 |
| reset_robot_joints | reset | pos ±0.5 rad, vel ±0.5 rad/s |
| pd_gains_rand | reset | Kp, Kd × 0.8–1.2 |
| push_robot | every 1–3 s | lin ±0.5 (z ±0.3) m/s; roll/pitch ±0.4, yaw ±0.5 rad/s |

## Terminations

| Term | Condition |
|------|-----------|
| time_out | 20 s (bootstrapped) |
| fell_over | base tilt > 70° |

## PPO

| Param | Value |
|-------|-------|
| Actor / critic | MLP 512‑256‑128, ELU, no obs normalization |
| Initial action std | 1.0 (`std_type="scalar"`) |
| Steps per env per iteration | 24 |
| Batch per iteration | 98,304 (at 4,096 envs) |
| Epochs × minibatches | 5 × 4 (minibatch 24,576) |
| Clip | 0.2 (policy and value) |
| Entropy coef | 0.005 |
| Value loss coef | 1.0 |
| Learning rate | 1e‑3, adaptive to KL 0.01 (bounds 1e‑5 to 1e‑2, factor 1.5) |
| γ / λ | 0.99 / 0.95 |
| Max grad norm | 1.0 (actor + critic) |
| Max iterations | 10,000 |
| Save interval | 500 |
| Experiment names | `asimov1_velocity` (PPO), `asimov_velocity_amp` (AMP) |

## AMP

| Param | Value |
|-------|-------|
| Motion clip | `policy_delay_walk_slow.npz`: 2,472 frames, 50 fps, 49.4 s, ~0.36 m/s |
| Discriminator | MLP 92‑256‑256‑1, ReLU |
| Loss | LSGAN (expert → +1, policy → −1) |
| Reward | `max(0, 1 − 0.25(d − 1)²)` |
| Gradient penalty λ | 10 |
| Reward coef / task lerp | 0.3 / 0.7 → `r = 0.09·r_amp·gate + 0.7·r_task` |
| Command gate | ‖[vx, vy, ωz]‖ > 0.1 |
| Replay buffer | 100,000 |
| Weight decay | trunk 1e‑3, head 1e‑1 |
| Update interval | every iteration |
| Rollout obs clip | ±500 |

## Software versions

| Component | Version |
|-----------|---------|
| Python | 3.11 (install scripts); ≥ 3.10 (`setup.py`) |
| Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0 (CUDA 12.8) |
| RSL‑RL | 5.0.1 |
| Isaac Lab | submodule pinned at `b0542fe` |
| NumPy | < 2 |
| OS | Ubuntu 22.04+ x86_64 |
| Tested GPUs (README) | RTX A6000, RTX PRO 6000, RTX 4090, RTX 3090 |

---

*[← Appendix B](appendix-b-file-reference.md) · [Contents](README.md)*
