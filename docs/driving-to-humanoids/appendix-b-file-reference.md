# Appendix B — File-by-File Reference

*[← Appendix A](appendix-a-glossary.md) · [Contents](README.md) · [Appendix C: The Numbers Card →](appendix-c-numbers.md)*

---

Every file in `menloresearch/isaac_asimov` (at commit `bdf28f5`), what it
does, and which chapter covers it. Paths are relative to the repo root.

## Top level

| File | Purpose | Chapter |
|------|---------|---------|
| `README.md` | Overview, quick install, train/play commands, tested GPUs, acknowledgements, community link | 0, 3, 9, 10 |
| `INSTALL.md` | Advanced install: reuse your own Isaac Lab, uv or conda, sparse checkout of the robot model, `ASIMOV_1_MODEL_DIR` | 3 |
| `LICENSE` | License text (the package is declared BSD-3-Clause in `setup.py`) | |
| `isaac_asimov.sh` | Launcher: `--install`, `--list`, `--train`, `--play` → the matching Python script; honors `PYTHON_EXE` | 3 |
| `quick_install.sh` | One-shot setup: apt packages, submodules (sparse for `asimov-1`), uv venv (Python 3.11), Isaac Sim 5.1.0, PyTorch 2.7.0/cu128, Isaac Lab + RSL‑RL, this extension | 3 |
| `pyproject.toml` | Tooling config: black/isort (120 cols, custom Isaac Lab import section), pytest (`pythonpath`, `testpaths`), pyright (basic) | 9 |
| `.pre-commit-config.yaml` | black, isort, whitespace/EOF fixers, YAML/TOML checks, merge-conflict and private-key detection | 9 |
| `.flake8` | Lint config (120 cols, ignores for star imports in `__init__.py`) | 9 |
| `.gitignore` | Ignores logs, checkpoints (`*.pt`, `*.onnx`), USD files, and `*.npz` except the one shipped clip | 9, 11 |
| `.gitattributes` | Line endings; `*.npz` as binary | |
| `.gitmodules` | Submodules: `third_party/asimov-1`, `third_party/IsaacLab` | 3 |

## `scripts/`

| File | Purpose | Chapter |
|------|---------|---------|
| `list_envs.py` | Launch Isaac Sim headless, import tasks, print registered `Asimov1-*` IDs and config entry points; optional `--keyword` filter | 3 |
| `rsl_rl/cli_args.py` | Adds RSL‑RL CLI args (`--experiment_name`, `--run_name`, `--resume`, `--load_run`, `--checkpoint`, `--logger`, `--log_project_name`) and applies them to the agent config | 9 |
| `rsl_rl/train.py` | Parse args → launch Isaac Sim → check RSL‑RL ≥ 5.0.1 → Hydra-resolve configs → set device/seed (distributed-aware) → make env → wrap for RSL‑RL → `OnPolicyRunner`/`DistillationRunner` → optional resume → dump `params/*.yaml` → `learn()` | 9 |
| `rsl_rl/play.py` | Resolve checkpoint (pretrained / `--checkpoint`/`--target` / latest) → make env → load → export `policy.pt` + `policy.onnx` (+ optional `--onnx-output`) → inference loop with optional `--real-time` and `--video` | 10 |

## `source/isaac_asimov/` (the installable extension)

| File | Purpose | Chapter |
|------|---------|---------|
| `setup.py` | Package metadata from `config/extension.toml`; installs `numpy<2` and `rsl-rl-lib==5.0.1`; ships the motion `.npz` via `package_data` | 3 |
| `pyproject.toml` | Build-system metadata for the extension | 3 |
| `config/extension.toml` | Isaac Lab extension metadata: version 0.1.0, depends on `isaaclab`, `isaaclab_rl`, `isaaclab_tasks` | 3 |
| `docs/README.md`, `docs/CHANGELOG.rst` | Extension readme; changelog (0.1.0, 2026‑09‑15: velocity and AMP tasks, packaged as a standalone extension) | |

### `isaac_asimov/assets/robots/`

| File | Key contents | Chapter |
|------|--------------|---------|
| `asimov_1.py` | `ASIMOV_1_URDF_PATH` (with `ASIMOV_1_MODEL_DIR` override); `ASIMOV_1_JOINT_NAMES` (23, action order); `ASIMOV_1_ACTION_SCALE = 0.25`; `ASIMOV_1_ACTUATORS` (11 `DelayedPDActuatorCfg` groups, delay 0–5); `ASIMOV_1_STANDING_INIT_STATE`; `ASIMOV_1_DELAYED_CFG` | 1, 2 |
| `__init__.py` | Re-exports `ASIMOV_1_ACTION_SCALE`, `ASIMOV_1_DELAYED_CFG`, `ASIMOV_1_JOINT_NAMES` | |

### `isaac_asimov/tasks/`

| File | Key contents | Chapter |
|------|--------------|---------|
| `__init__.py` | `import_packages(__name__)`: imports all subpackages so their `gym.register` calls run | 3 |
| `locomotion/__init__.py` | Registers 4 tasks: `Asimov1-Velocity-v0`, `-Play-v0`, `-AMP-v0`, `-AMP-Play-v0` | 3 |
| `locomotion/velocity_env_cfg.py` | `COBBLESTONE_ROAD_CFG`; joint slots; scene (terrain, robot, 2 contact sensors, light); commands; actions; observations (policy/critic); events; 18 rewards; terminations; `Asimov1VelocityEnvCfg` and `_PLAY` | 4, 5, 6 |
| `locomotion/amp_env_cfg.py` | `AmpObsCfg` (joint pos + vel, with drift assert); motion file list; key bodies; anchor; `Asimov1AmpEnvCfg` and `_PLAY` | 8 |
| `locomotion/motion_dataset.py` | `MotionDataset` (load/validate/reorder clips, body-relative feature properties, transition indices, sampling generator) and `MotionDatasetCfg` | 8 |
| `locomotion/motions/policy_delay_walk_slow.npz` | The reference clip: 2,472 frames at 50 fps (49.4 s), 23 joints, 26 bodies | 8 |
| `locomotion/agents/rsl_rl_ppo_cfg.py` | `AMPPPOAlgorithmCfg` (AMP hyperparameters, `class_name` → `AMPPPO`); `Asimov1PPORunnerCfg`; `Asimov1AMPRunnerCfg` | 7, 8 |
| `locomotion/mdp/__init__.py` | Star-imports Isaac Lab's MDP terms plus this repo's events, observations and rewards into one `mdp` namespace | 4 |
| `locomotion/mdp/observations.py` | `FOOT_SITE_OFFSET`; `foot_pos_w`, `foot_vel_w`, `foot_height`, `foot_air_time`, `foot_contact`, `foot_contact_forces`; `delayed_obs` class | 4 |
| `locomotion/mdp/rewards.py` | Command gate; tracking, orientation, posture, gait, smoothness and safety terms (16 custom functions/classes; the other 2 of the 18 rewards are Isaac Lab built-ins) | 5 |
| `locomotion/mdp/events.py` | `randomize_joint_default_pos` (calibration randomization that also syncs the action offset and saves `nominal_default_joint_pos`) | 6 |

### `isaac_asimov/algorithms/`

| File | Key contents | Chapter |
|------|--------------|---------|
| `amp_ppo.py` | `AMPPPO(PPO)`: `construct_algorithm` (build dataset, check dims), `process_env_step` (AMP reward, gate, replay insert, blend, obs clip), `_joint_update` (PPO + LSGAN + grad penalty), multi-GPU sync, save/load with split-optimizer compatibility | 7, 8 |
| `discriminator.py` | `AMPFeatureNormalizer` (running mean/var, float64), `AMPDiscriminator` (MLP, reward, gradient penalty, normalization update) | 8 |
| `replay_buffer.py` | `AMPReplayBuffer`: GPU ring buffer of aligned `(state, next_state)` pairs | 8 |

## `tests/`

| File | Purpose | Chapter |
|------|---------|---------|
| `test_amp_components.py` | 5 unit tests: normalizer math, reward and gradient-penalty formulas, replay-buffer alignment across wraparound, AMP-PPO frame pairing/gating/blending/clipping, combined actor+critic gradient clipping | 9 |

## `third_party/` (submodules)

| Path | What it is |
|------|------------|
| `IsaacLab/` | Isaac Lab at a pinned commit (`b0542fe`) |
| `asimov-1/` | Menlo's hardware repo; only `sim-model/` is checked out (URDF + meshes at `sim-model/urdf/asimov_1.urdf`) |

---

*[← Appendix A](appendix-a-glossary.md) · [Contents](README.md) · [Appendix C: The Numbers Card →](appendix-c-numbers.md)*
