# Chapter 3 — A Thousand Worlds per Second: Simulation, Installation, and the Repo Tour

*[← Muscles Made of Math](02-actuators-and-pd.md) · [Contents](README.md) · [Next: The MDP Is a Config File →](04-the-mdp-as-config.md)*

---

Picture the data engine you worked with in AD: a fleet of cars logging
sensor data, an ingestion pipeline, a labeling vendor, a curation team, and a
training cluster at the end. Months between "we need more night-time
cut-ins" and "the model has seen more night-time cut-ins".

Now picture this: you type one command, and within seconds **4,096 robots**
are walking, falling, and getting back up on a single GPU. Each one lives in
its own 2.5 m square of the world. Every 20 milliseconds of simulated time, all
of them produce a fresh, perfectly labeled training sample. In a typical
full training run, the robot population experiences on the order of a
**billion** control steps, which works out to over half a year of simulated
walking.

That's the data engine of legged robotics. This chapter explains how it
works, how to install it, and how the repo is organized around it.

## 3.1 The simulation stack

There are four layers, from bottom to top:

```
┌──────────────────────────────────────────────────────────────────┐
│ isaac_asimov  (this repo)                                        │
│   robot config · task configs · rewards · AMP algorithm          │
├──────────────────────────────────────────────────────────────────┤
│ RSL-RL 5.0.1                     │ Isaac Lab (isaaclab,          │
│   PPO, runners, actor/critic     │ isaaclab_rl, isaaclab_tasks)  │
│   models, rollout storage        │   managers, MDP terms,        │
│                                  │   sensors, terrain, wrappers  │
├──────────────────────────────────┴───────────────────────────────┤
│ Isaac Sim 5.1.0  (Omniverse Kit app, USD scenes, rendering)      │
├──────────────────────────────────────────────────────────────────┤
│ PhysX 5 on the GPU  (rigid bodies, articulations, contacts)      │
└──────────────────────────────────────────────────────────────────┘
```

- **PhysX** does the physics: integrating velocities, resolving contacts,
  enforcing joint limits. Crucially, it runs *on the GPU* and exposes state as
  GPU tensors, so observations and rewards never need to leave the device.
- **Isaac Sim** is NVIDIA's robotics simulator. It hosts the scene (in USD,
  the Universal Scene Description format from Pixar), handles rendering, and
  converts the URDF into USD.
- **Isaac Lab** is the robot-learning framework. It gives you
  `ManagerBasedRLEnv`: an environment you describe entirely with config
  classes, where each piece (observations, rewards, events) is a "manager".
- **RSL‑RL** is the learning library: PPO, the runner that alternates between
  collecting data and updating, and the actor/critic network classes.

> 🚗 **Driving Déjà Vu**
> Map it onto your AD simulation stack. PhysX is your vehicle dynamics model
> plus collision checker. Isaac Sim is CARLA or your in-house renderer.
> Isaac Lab is the scenario framework (the thing where you define "a
> pedestrian crosses at t = 3 s" and "measure time-to-collision"). RSL‑RL is
> your training framework. The big difference is that here, *everything runs
> batched on one GPU*, and the scenario framework directly produces training
> tensors.

## 3.2 The clocks: `dt`, `decimation`, and episode length

At the end of `velocity_env_cfg.py`:

```python
@configclass
class Asimov1VelocityEnvCfg(ManagerBasedRLEnvCfg):
    scene: Asimov1SceneCfg = Asimov1SceneCfg(num_envs=4096, env_spacing=2.5)
    ...
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
```

| Setting | Value | Meaning |
|---------|-------|---------|
| `sim.dt` | 0.005 s | Physics step: 200 Hz |
| `decimation` | 4 | The policy acts every 4 physics steps |
| policy period | 4 × 0.005 = 0.02 s | Policy: **50 Hz** (Isaac Lab calls this `step_dt`) |
| `episode_length_s` | 20 s | Max episode length: 20 / 0.02 = **1,000 policy steps** |
| `render_interval` | 4 | Render once per policy step (only matters when not headless) |
| `num_envs` | 4096 | Parallel robots |
| `env_spacing` | 2.5 m | Distance between robots' origins (overridden by the terrain grid, as we'll see) |

Why 200 Hz physics but 50 Hz policy? Contacts are stiff. A foot hitting the
ground at 1 m/s needs small time steps to resolve without jitter or
penetration. But the policy doesn't need to decide that often, and every
policy step costs a network forward pass plus bookkeeping. Decimation lets you
pay for accurate physics without paying for more decisions.

> 🔧 **Under the Hood: why 50 Hz?**
> It's a sweet spot seen across many legged-robot projects. Much slower
> (say 10 Hz, a typical AD planning rate) and the robot can't react within a
> single stride. Much faster (500 Hz) and each step changes so little that
> credit assignment gets harder, and onboard compute gets tight. With
> `gamma = 0.99` (Chapter 7) at 50 Hz, the policy's effective planning
> horizon is roughly 1 / (1 − 0.99) = 100 steps = 2 seconds: about two or
> three strides.

## 3.3 Installing it

The README offers a one-shot script for a fresh Ubuntu 22.04+ x86_64 machine
with an NVIDIA driver, [uv](https://docs.astral.sh/uv/) and `sudo`:

```bash
git clone https://github.com/menloresearch/isaac_asimov.git
cd isaac_asimov
./quick_install.sh
```

Let's read `quick_install.sh` so it's not magic:

```bash
set -euo pipefail                             # stop on any error

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

# 1. System packages: compilers for some Isaac Lab deps, and a GL library
#    that Isaac Sim's omni.iray extension needs at runtime.
sudo apt-get update && sudo apt-get install -y cmake build-essential libglu1-mesa

# 2. Submodules: Isaac Lab (full), and only the sim-model folder of asimov-1.
git submodule update --init third_party/IsaacLab
git submodule update --init --filter=blob:none third_party/asimov-1
git -C third_party/asimov-1 sparse-checkout set sim-model

# 3. A Python 3.11 venv. --seed installs pip, which Isaac Lab's installer needs.
uv venv --seed --python 3.11
source .venv/bin/activate

# 4. Isaac Sim (from NVIDIA's index) and CUDA 12.8 PyTorch.
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 5. Isaac Lab + its RSL-RL integration, in editable mode.
(cd third_party/IsaacLab && ./isaaclab.sh --install rsl_rl)

# 6. This repo's extension, in editable mode.
uv pip install -e "${ROOT}/source/isaac_asimov"
```

The versions are pinned deliberately: Isaac Sim 5.1.0, PyTorch 2.7.0 with
CUDA 12.8, RSL‑RL 5.0.1 (via `setup.py`), and Isaac Lab at the submodule's
pinned commit. `INSTALL.md` notes the project expects Isaac Lab `main`
(reporting 2.3.2 at the time) with RSL‑RL 5 support, and that other versions
may not be compatible.

`INSTALL.md` also documents two alternatives: reusing your own Isaac Lab
checkout, and a conda + pip setup. Both end with:

```bash
./isaac_asimov.sh --install     # = pip install -e source/isaac_asimov
```

> ⚠️ **Pothole: version drift**
> Robot-learning stacks are notoriously sensitive to version mismatches. Isaac
> Lab's APIs change between releases, and RSL‑RL 5 changed its config
> format significantly from 2.x/3.x. `train.py` checks the installed RSL‑RL
> version at startup and exits with an install command if it's older than
> 5.0.1. Treat the pins as load-bearing.

## 3.4 The launcher: `isaac_asimov.sh`

A thin wrapper. Here's the whole logic:

```bash
PYTHON_EXE="${PYTHON_EXE:-python}"

case "${1:-}" in
    -i|--install) "${PYTHON_EXE}" -m pip install -e "${ISAAC_ASIMOV_ROOT}/source/isaac_asimov" ;;
    -l|--list)    shift; "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/list_envs.py" "$@" ;;
    -t|--train)   shift; "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/rsl_rl/train.py" "$@" ;;
    -p|--play)    shift; "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/rsl_rl/play.py" "$@" ;;
    *)            usage; exit 2 ;;
esac
```

So `./isaac_asimov.sh --train --task X --num_envs 128` is exactly
`python scripts/rsl_rl/train.py --task X --num_envs 128`. You can set
`PYTHON_EXE` to use a different interpreter.

`--list` runs `scripts/list_envs.py`, which starts Isaac Sim headless, imports
the tasks, and prints every registered gym ID starting with `Asimov1-`
alongside its config entry point. An optional `--keyword` filters the list.

## 3.5 The repo map

```
isaac_asimov/
├── README.md, INSTALL.md, LICENSE
├── isaac_asimov.sh              ← launcher (install | list | train | play)
├── quick_install.sh             ← one-shot environment setup
├── pyproject.toml               ← black/isort/pytest/pyright settings
├── .pre-commit-config.yaml, .flake8
├── scripts/
│   ├── list_envs.py             ← print registered Asimov1-* tasks
│   └── rsl_rl/
│       ├── cli_args.py          ← --resume, --load_run, --checkpoint, --logger...
│       ├── train.py             ← build env + runner, learn, log
│       └── play.py              ← load checkpoint, export JIT/ONNX, run loop
├── source/isaac_asimov/         ← the installable Isaac Lab extension
│   ├── setup.py                 ← reads version/metadata from config/extension.toml
│   ├── config/extension.toml    ← extension metadata (v0.1.0)
│   ├── docs/                    ← README, CHANGELOG
│   └── isaac_asimov/
│       ├── assets/robots/asimov_1.py         ← robot + actuators   (Ch. 1–2)
│       ├── tasks/locomotion/
│       │   ├── __init__.py                   ← gym registration   (this chapter)
│       │   ├── velocity_env_cfg.py           ← the whole MDP       (Ch. 4–6)
│       │   ├── amp_env_cfg.py                ← adds the "amp" obs group (Ch. 8)
│       │   ├── motion_dataset.py             ← loads the expert clip    (Ch. 8)
│       │   ├── motions/policy_delay_walk_slow.npz
│       │   ├── mdp/observations.py, rewards.py, events.py   (Ch. 4–6)
│       │   └── agents/rsl_rl_ppo_cfg.py      ← PPO + AMP hyperparameters (Ch. 7–8)
│       └── algorithms/
│           ├── amp_ppo.py                    ← PPO subclass with AMP   (Ch. 8)
│           ├── discriminator.py              ← discriminator + normalizer
│           └── replay_buffer.py              ← policy transition buffer
├── tests/test_amp_components.py              ← unit tests for AMP pieces (Ch. 9)
└── third_party/
    ├── IsaacLab/     (submodule)
    └── asimov-1/     (submodule, sparse: sim-model only)
```

## 3.6 How a task gets registered

When you pass `--task Asimov1-Velocity-AMP-v0`, how does Isaac Lab find the
right configs? Through [Gymnasium](https://gymnasium.farama.org/)'s registry.

`tasks/locomotion/__init__.py`:

```python
gym.register(
    id="Asimov1-Velocity-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Asimov1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1AMPRunnerCfg",
    },
)
```

Each task ID binds two things:

1. **The environment config** (`env_cfg_entry_point`): what the world,
   robot, observations and rewards look like.
2. **The agent config** (`rsl_rl_cfg_entry_point`): the algorithm and its
   hyperparameters.

The four registered tasks form a neat 2 × 2:

|  | Training config | Play config |
|--|-----------------|-------------|
| **PPO** | `Asimov1-Velocity-v0` → `Asimov1VelocityEnvCfg` + `Asimov1PPORunnerCfg` | `Asimov1-Velocity-Play-v0` → `Asimov1VelocityEnvCfg_PLAY` + `Asimov1PPORunnerCfg` |
| **AMP** | `Asimov1-Velocity-AMP-v0` → `Asimov1AmpEnvCfg` + `Asimov1AMPRunnerCfg` | `Asimov1-Velocity-AMP-Play-v0` → `Asimov1AmpEnvCfg_PLAY` + `Asimov1AMPRunnerCfg` |

And who imports `tasks/locomotion/__init__.py`? The parent package,
`tasks/__init__.py`:

```python
from isaaclab_tasks.utils import import_packages
import_packages(__name__, blacklist_pkgs=[])
```

`import_packages` recursively imports every subpackage, which runs every
`gym.register` call. Then `train.py` and `play.py` do:

```python
import isaaclab_tasks  # noqa: F401     ← registers Isaac Lab's own tasks
import isaac_asimov.tasks  # noqa: F401 ← registers ours
```

Those "unused" imports are load-bearing. Delete one, and your task "doesn't
exist".

> 🚗 **Driving Déjà Vu**
> This is the same pattern as a model or scenario registry in your AD
> codebase: a string key maps to a config and a class, and a decorator or an
> import side effect registers it. Same benefits (CLI-friendly, easy to add
> variants), same pitfalls (import-order bugs, "why isn't my variant found?").

## 3.7 Configs as code: `@configclass`

Everything in this repo is a `@configclass`: Isaac Lab's flavor of Python
dataclass. Configs are ordinary Python, so:

- **Inheritance is how variants are made.** `Asimov1AmpEnvCfg` inherits
  `Asimov1VelocityEnvCfg` and adds one observation group. `_PLAY` variants
  inherit the training configs and turn things off.
- **`__post_init__` is where derived settings go.** For example, the physics
  material is copied from the terrain.
- **`.replace(...)` makes modified copies.** `ASIMOV_1_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`
  gives a copy with a new prim path.
- **Setting a term to `None` removes it.** The play config does
  `self.events.push_robot = None` to disable pushes.

And because `train.py` uses Isaac Lab's `@hydra_task_config` decorator, you can
override config values from the command line using Hydra syntax, without
editing code. Anything `argparse` doesn't recognize is forwarded to Hydra
(`args_cli, hydra_args = parser.parse_known_args()`). For example:

```bash
./isaac_asimov.sh --train --task Asimov1-Velocity-v0 --headless \
    env.rewards.air_time.weight=1.0 \
    agent.algorithm.entropy_coef=0.01
```

(The `env.` and `agent.` prefixes address the environment and agent configs.
Double-check the exact override syntax against the Isaac Lab version you have,
since it has evolved across releases.)

## 3.8 `{ENV_REGEX_NS}`: how one robot becomes 4,096

You'll see prim paths like `"{ENV_REGEX_NS}/Robot"` everywhere. Isaac Lab
builds the scene for one environment, then clones it `num_envs` times under
`/World/envs/env_0`, `/World/envs/env_1`, and so on. `{ENV_REGEX_NS}` is a
placeholder that expands to the regex `/World/envs/env_.*`, so a single config
line addresses the robot in *every* environment.

The cloned environments share one physics scene but don't interact
(the robots don't collide with each other). Physics state, sensor data and
command buffers are all tensors with a leading `num_envs` dimension. When a
reward function computes `asset.data.root_lin_vel_b`, it gets a tensor of
shape `[4096, 3]`, for every robot at once. **Never write a loop over
environments.** (There is a loop over *feet* in `feet_orientation_penalty`,
but it's over two feet, not 4,096 robots.)

## 🏁 Pit Stop

1. What are the physics rate, the policy rate, and the maximum episode length in
   policy steps?
2. Which two configs does a gym task ID bind together?
3. What happens if you remove `import isaac_asimov.tasks` from `train.py`?
4. How does the Play config disable the random pushes?
5. What does `{ENV_REGEX_NS}` expand to, and why is it needed?

<details>
<summary>Answers</summary>

1. 200 Hz physics, 50 Hz policy, 1,000 policy steps (20 s).
2. The environment config (`env_cfg_entry_point`) and the RSL‑RL agent config
   (`rsl_rl_cfg_entry_point`).
3. The `gym.register` calls never run, so `Asimov1-*` tasks aren't found.
4. `self.events.push_robot = None`.
5. A regex matching every cloned environment's namespace
   (`/World/envs/env_.*`), so one config line applies to all robots.

</details>

---

*[← Muscles Made of Math](02-actuators-and-pd.md) · [Contents](README.md) · [Next: The MDP Is a Config File →](04-the-mdp-as-config.md)*
