# Chapter 11 — Your First Five Experiments

*[← From Checkpoint to Concrete Floor](10-play-export-deploy.md) · [Contents](README.md) · [Next: Epilogue: The New Road →](12-epilogue-roadmap.md)*

---

Reading code gives you a map. Changing code teaches you the territory. Here
are five experiments, ordered from a 10-minute tweak to a multi-day project.
Each one teaches a different part of the system.

A word of method before you start, borrowed from AD: **change one thing, keep
a baseline, and write down your hypothesis before you look at the curves.**
RL results are noisy. Run at least two seeds before believing a difference.

> 🔧 **Under the Hood: the clean way to make a variant**
> You *can* edit `velocity_env_cfg.py` directly, or use Hydra overrides on
> the command line. For anything you'll keep, the cleanest pattern is the one
> the repo itself uses: **subclass the config and register a new task ID**.
> Your baseline stays untouched, and the variant is reproducible by name.
>
> ```python
> # tasks/locomotion/my_variants.py
> from isaaclab.utils import configclass
> from .amp_env_cfg import Asimov1AmpEnvCfg
>
> @configclass
> class Asimov1FastAmpEnvCfg(Asimov1AmpEnvCfg):
>     def __post_init__(self):
>         super().__post_init__()
>         self.commands.twist.ranges.lin_vel_x = (-0.6, 1.2)
> ```
>
> ```python
> # tasks/locomotion/__init__.py (add)
> gym.register(
>     id="Asimov1-Velocity-AMP-Fast-v0",
>     entry_point="isaaclab.envs:ManagerBasedRLEnv",
>     disable_env_checker=True,
>     kwargs={
>         "env_cfg_entry_point": f"{__name__}.my_variants:Asimov1FastAmpEnvCfg",
>         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1AMPRunnerCfg",
>     },
> )
> ```
>
> Also give the variant its own `experiment_name` (with `--experiment_name`
> or a subclassed runner config) so its logs don't mix with the baseline's,
> and so `play.py` finds the right checkpoints.

---

## Experiment 1: Walk faster (difficulty: ★☆☆☆☆)

**Goal:** Widen the forward speed range from 0.8 m/s to 1.2 m/s.

**Change:** The variant above (`lin_vel_x = (-0.6, 1.2)`).

**Hypotheses to check:**

- Tracking at high speed will be worse at first. Does the robot eventually
  learn a longer stride, a faster cadence, or both? Watch
  `Metrics/air_time_mean` (swing duration) and `track_linear_velocity`.
- With faster commands, `total_speed` exceeds the `running_threshold` of 1.5
  more often, so the looser `std_running` posture table kicks in more
  (Chapter 5). Does the gait look different at those speeds?
- **The AMP tension.** The motion clip is a *slow* walk (0.36 m/s average).
  The discriminator will pull high-speed motion toward slow-walk joint
  velocities. Compare against the PPO-only task with the same change. You may
  find AMP helps style at low speed but limits top speed. That's a real
  trade-off, and the usual fix is more (and faster) reference clips.

**What you'll learn:** how command distributions, posture scheduling and style
priors interact.

---

## Experiment 2: Add a reward term (difficulty: ★★☆☆☆)

**Goal:** Stop the feet from getting too close to each other, which causes
self-collisions and tripping.

**Step 1: write the term** in `mdp/rewards.py`:

```python
def feet_too_close(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    feet_xy = foot_pos_w(env, asset_cfg)[:, :, :2]            # [N, 2, 2]
    distance = torch.norm(feet_xy[:, 0] - feet_xy[:, 1], dim=-1)
    env.extras.setdefault("log", {})["Metrics/feet_distance_mean"] = torch.mean(distance)
    return torch.clamp(min_distance - distance, min=0.0)      # hinge: 0 when far enough
```

Everything here follows the repo's conventions: vectorized over
environments, reuses `foot_pos_w`, returns a non-negative cost, and logs a
physical metric.

Because `mdp/__init__.py` does `from .rewards import *`, the new function is
available as `mdp.feet_too_close` automatically.

**Step 2: add it to a config variant:**

```python
from isaaclab.managers import RewardTermCfg as RewTerm
from .velocity_env_cfg import _feet_cfg

@configclass
class Asimov1FeetSpacingEnvCfg(Asimov1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_too_close = RewTerm(
            func=mdp.feet_too_close,
            weight=-10.0,
            params={"min_distance": 0.15, "asset_cfg": _feet_cfg()},
        )
```

**Step 3: pick the weight.** Do the arithmetic before training. If feet are
5 cm too close, the cost is 0.05; times the weight 10 and `dt` 0.02 gives
0.01 per step, which is about 10% of a perfect velocity-tracking step. That's
noticeable but not dominant. Reason like this for every term you add.

**What you'll learn:** the full loop of reward design: hypothesis, term,
weight arithmetic, metric, verification.

---

## Experiment 3: Harder terrain (difficulty: ★★★☆☆)

**Goal:** Teach the robot slopes and small stairs.

**Change:** add sub-terrains to a copy of the terrain config. Isaac Lab ships
generators for these:

```python
import isaaclab.terrains as terrain_gen
from .velocity_env_cfg import COBBLESTONE_ROAD_CFG

HARDER_TERRAIN_CFG = COBBLESTONE_ROAD_CFG.replace(
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02,
            horizontal_scale=0.1, vertical_scale=0.005, border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.02, 0.08), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
    },
)
```

(Check the parameter names against your Isaac Lab version; these generator
classes are used in Isaac Lab's own rough-terrain configs.)

With `curriculum=True`, rows get progressively harder (`difficulty_range`
maps row index to difficulty). For robots to actually *move* between rows,
you also need a curriculum term that promotes robots that walk far and
demotes those that fail. Isaac Lab's locomotion tasks provide one
(`terrain_levels_vel`, in `isaaclab_tasks`' velocity-task MDP), wired up via
a `CurriculumCfg` in the environment config. You'd also want to lower
`max_init_terrain_level` so robots start on easy rows.

**The deeper problem.** This policy is **blind**: it has no idea what the
terrain ahead looks like. On 2 to 6 cm cobblestones, a robust gait (high
swing, quick recovery) is enough. On stairs, a blind robot can only discover
a step by hitting it. Blind stair climbing *is* possible with good
proprioception and robust training, but it's much harder, and there's a
second catch: `foot_height` and the clearance rewards use world z, not height
above terrain (Chapter 4), so they become misleading on slopes and stairs.

The standard next step is a **height scan**: a grid of terrain heights around
the robot (Isaac Lab's `RayCaster` sensor), fed to the critic first (privileged)
and then, if the real robot has a depth sensor, to the policy.

**What you'll learn:** curricula, the limits of blind locomotion, and the
first step toward perceptive locomotion (which is where your AD perception
skills come roaring back).

---

## Experiment 4: A new style for AMP (difficulty: ★★★★☆)

**Goal:** Change the robot's walking style by changing the reference motion.

**The file format.** `MotionDataset.load_motions` reads these keys from each
`.npz`. All but `body_names` are required, even the body arrays the current
config doesn't use as features:

| Key | Shape | Notes |
|-----|-------|-------|
| `fps` | scalar | Should match the 50 Hz policy rate (the loader doesn't resample) |
| `joint_pos` | `[T, J]` | **Absolute** angles, radians |
| `joint_vel` | `[T, J]` | rad/s |
| `joint_names` | `[J]` | Required when the config sets `joint_names` (it does) |
| `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` | `[T, B, 3]` | World frame |
| `body_quat_w` | `[T, B, 4]` | `wxyz` |
| `body_names` | `[B]` | Optional but strongly recommended: when present, it must contain the key bodies and the `pelvis_link` anchor, and it's used to index them. Without it, the loader assumes the file's body order matches the simulator's. |

**Where to get a clip:**

- **Record it from a policy in sim.** Train a policy with rewards shaped for
  the style you want (for instance, a higher target swing height), roll it
  out, and save these arrays at 50 Hz. The existing clip's name suggests this
  is how it was made (Chapter 8).
- **Retarget human motion capture** to Asimov‑1's joints. This is a project
  of its own: human and robot proportions and joint limits differ, and naive
  retargeting produces foot sliding and self-collisions. The README credits
  `whole_body_tracking` and `beyondAMP`, which work in this space.

**Using it:**

```python
ASIMOV_1_MOTION_FILES = [
    str(_MOTIONS_DIR / "policy_delay_walk_slow.npz"),
    str(_MOTIONS_DIR / "my_brisk_walk.npz"),
]
```

Multiple clips are concatenated, and `_build_transition_indices` guarantees no
transition crosses a clip boundary. All clips must share one body ordering.

> ⚠️ **Pothole: `.gitignore`**
> The repo's `.gitignore` ignores `*.npz` except the one shipped clip. Your
> new clip won't be committed unless you add an exception line for it. Also
> check `setup.py`'s `package_data` (`tasks/locomotion/motions/*.npz`), which
> already covers new files in that folder.

**What you'll learn:** how much of "gait quality" lives in data rather than in
reward terms, and the practical headaches of motion data pipelines, which
will feel very familiar from AD data work.

---

## Experiment 5: Left-right symmetry (difficulty: ★★★★★)

**Goal:** Make the gait symmetric by teaching the policy that the mirror image
of a good action is also a good action.

Humanoid policies trained from scratch are often slightly asymmetric: a
favorite leg, a longer stride on one side. RSL‑RL supports **symmetry**
through `symmetry_cfg`, currently `None` in this repo. `AMPPPO._joint_update`
already carries RSL‑RL's symmetry code paths (data augmentation and/or a
mirror loss), so wiring it up is a config-and-function job:

```python
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

algorithm = AMPPPOAlgorithmCfg(
    ...,
    symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=mirror_obs_and_actions,   # you write this
    ),
)
```

The hard part is `mirror_obs_and_actions(env, obs, actions)`. It must return
the original batch *plus* its mirror image, for every observation group the
actor and critic use, and for the actions. Mirroring means, for each term:

- **Swap** left and right joints (in the correct *order* for that block:
  slot order for joint observations, action order for actions).
- **Flip signs** where the mirror transformation requires it. For a mirror
  across the sagittal (x–z) plane: lateral velocity `vy`, turning rate `ωz`,
  roll and yaw angular velocity components, and the y component of projected
  gravity flip; so do roll and yaw joint angles. And then there are the
  *URDF axis conventions*, which from Chapter 1 we know differ between the
  left and right sides for some joints. Every joint needs checking.
- **Handle the non-joint terms** in the critic group: foot heights, air
  times, contacts and contact forces (swap feet; flip force y).

> ⚠️ **Pothole**
> A single wrong sign in a mirror function doesn't crash. It silently
> teaches the policy that a *wrong* action is symmetric to a right one, and
> the gait degrades in ways that are very hard to trace. Write a unit test:
> mirror twice and check you get the original back; mirror a recorded
> standing pose and check it's still a valid standing pose.

**What you'll learn:** robot kinematics conventions at the deepest level, and
how to inject prior knowledge (symmetry) into RL. Very analogous to
geometry-aware augmentation in AD (flipping images *and* their labels
correctly: left-hand traffic becomes right-hand traffic).

---

## Bonus ideas

- **Robustness sweep.** Write a script that loads a checkpoint and evaluates
  it on the training config with, for example, foot friction fixed at 0.5,
  1.0 and 1.5, or push magnitudes doubled. Plot survival rate vs
  perturbation. This is your "operational design domain" analysis, and it's
  exactly the kind of evaluation discipline AD engineers bring that many RL
  projects lack.
- **Velocity estimator.** Train a small network to predict base linear
  velocity (critic-only today) from the policy's observations, and feed its
  output to the policy. This is one well-known form of *regularized online
  adaptation* / estimator-based locomotion.
- **A recurrent policy.** Replace the MLP with an LSTM or GRU actor so the
  policy can infer things like friction and payload from history. RSL‑RL
  supports recurrent models; `play.py` already calls `policy.reset(dones)`.
- **Teacher–student distillation.** Train a teacher with privileged
  observations (like the critic's), then distill it into a deployable student.
  `train.py` already supports RSL‑RL's `DistillationRunner` code path.

## 🏁 Pit Stop

1. What's the cleanest way to create an experiment variant without disturbing
   the baseline?
2. For a new penalty, how do you sanity-check its weight before training?
3. Why do stairs expose a weakness in the current observations *and* rewards?
4. Which `.npz` keys does the motion loader require?
5. Name two kinds of sign flips a mirror function must handle.

<details>
<summary>Answers</summary>

1. Subclass the config, register a new gym task ID, and give it its own
   experiment name.
2. Multiply a typical cost value by the weight and by `dt` (0.02), and compare
   to the per-step bonus (for example, 0.1 for perfect velocity tracking).
3. The policy is blind (no terrain perception), and foot height is measured
   in world z rather than height above terrain.
4. `fps`, `joint_pos`, `joint_vel`, `joint_names`, `body_pos_w`,
   `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`; `body_names` is
   optional but strongly recommended.
5. Physical mirror flips (lateral and yaw quantities, roll/yaw joints) and
   URDF axis-convention flips between left and right joints.

</details>

---

*[← From Checkpoint to Concrete Floor](10-play-export-deploy.md) · [Contents](README.md) · [Next: Epilogue: The New Road →](12-epilogue-roadmap.md)*
