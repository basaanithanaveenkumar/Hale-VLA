# Chapter 6 — Lying to Your Robot on Purpose: Randomization, Terminations, and Sim-to-Real

*[← Speaking Robot: Rewards](05-rewards.md) · [Contents](README.md) · [Next: PPO for People Who Love Supervised Learning →](07-ppo.md)*

---

Every AD engineer has lived this moment. The model hits its metrics on the
validation set. Then it drives in a new city, at dusk, in the rain, and the
metrics fall off a cliff. Distribution shift.

Now imagine your *entire training set* came from a video game, and you had to
deploy on real streets with zero real examples. That's sim-to-real for
legged robots. The simulator is always wrong: the friction is a bit off, the
motors are a bit weaker, the torso is a bit heavier than the CAD model says,
and the sensors lie in ways no one has fully characterized.

The solution sounds absurd until you think about it: **make the simulator
wrong in many different ways, on purpose, so the real world looks like just
another variation**. That's *domain randomization*, and it lives in the
`EventCfg` section of `velocity_env_cfg.py`.

> 🚗 **Driving Déjà Vu**
> You've done this with images: color jitter, blur, random crops, weather
> augmentation, synthetic rain. Domain randomization is data augmentation for
> *physics*. The principle is identical: if the model has seen enough variety,
> the test distribution falls inside the training distribution.

## 6.1 Three kinds of events

Isaac Lab events run in one of three **modes**:

| Mode | When it runs | Used for |
|------|--------------|----------|
| `startup` | Once, when the simulation starts | Properties that are expensive to change or represent a fixed "robot identity": friction, mass distribution, joint calibration |
| `reset` | Each time an environment resets | Initial conditions and per-episode properties: pose, joint state, PD gains |
| `interval` | Periodically during an episode | Disturbances: pushes |

Because each of the 4,096 environments samples independently, a single training
run sees 4,096 slightly different robots, each one living through its own
sequence of randomized episodes.

## 6.2 The seven events

### Startup: giving each robot a unique identity

**`foot_friction`**

```python
foot_friction = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=list(FEET_BODIES)),
        "static_friction_range": (0.8, 1.2),
        "dynamic_friction_range": (0.8, 1.2),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
        "make_consistent": True,
    },
)
```

Each robot's feet get a friction coefficient between 0.8 and 1.2. Since the
ground is 1.0 and the combine mode is `multiply` (Chapter 4), the effective
foot-ground friction is 0.8 to 1.2. `num_buckets=64` means PhysX creates 64
distinct materials and assigns them randomly (materials are a limited resource
in PhysX, so you don't get one per robot). `make_consistent=True` ensures
dynamic friction never exceeds static friction, as in real physics.
Restitution is 0: feet don't bounce.

*Real-world gap addressed:* different floors (tile, concrete, carpet), worn
soles, dust.

**`qpos0_rand`**: a custom event

```python
qpos0_rand = EventTerm(
    func=mdp.randomize_joint_default_pos,
    mode="startup",
    params={"ranges": (-0.02, 0.02), "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
)
```

This one is defined in `mdp/events.py`, and it's clever:

```python
def randomize_joint_default_pos(env, env_ids, ranges, asset_cfg=SceneEntityCfg("robot"),
                                action_name="joint_pos"):
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # Remember the true default once, before perturbing it.
    if not hasattr(asset.data, "nominal_default_joint_pos"):
        asset.data.nominal_default_joint_pos = asset.data.default_joint_pos.clone()

    ...
    offsets = torch.empty((len(env_ids), len(joint_ids)), device=asset.device).uniform_(ranges[0], ranges[1])
    asset.data.default_joint_pos[env_ids[:, None], joint_ids] += offsets

    # Keep the action term's offset in sync with the new defaults.
    if action_name is not None and hasattr(env, "action_manager"):
        try:
            term = env.action_manager.get_term(action_name)
        except (KeyError, ValueError):
            return
        offset = getattr(term, "_offset", None)
        if isinstance(offset, torch.Tensor):
            term._offset = asset.data.default_joint_pos[:, term._joint_ids].clone()
```

It shifts each robot's "default" joint angles by up to ±0.02 rad (about 1.1°).
Then it updates the action term's internal offset so that
`q_target = q_default + 0.25·a` uses the *shifted* default.

What does that model? Think about what the policy experiences. Observations
are `q − q_default_shifted`, and actions are relative to `q_default_shifted`.
From the policy's perspective, the robot looks normal. But physically, the
"zero" it believes in is off by up to a degree. That's exactly **joint
calibration error**: on a real robot, encoder zero offsets are never
perfect. A policy that has only seen perfect calibration may lean or limp
when it's off by a degree.

It also saves `nominal_default_joint_pos` (the unperturbed default), which the
AMP motion dataset uses later so that expert motion is expressed relative
to the *true* default (Chapter 8).

> ⚠️ **Pothole: touching private attributes**
> `term._offset` and `term._joint_ids` are private attributes of Isaac Lab's
> action term. The code guards against a missing term with `try/except`, but
> an Isaac Lab update that renames these attributes would silently break the
> sync. This is one reason the repo pins its Isaac Lab version.

**`base_com`**

```python
base_com = EventTerm(
    func=mdp.randomize_rigid_body_com,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
        "com_range": {"x": (0.0, 0.05), "y": (0.0, 0.0), "z": (0.03, 0.07)},
    },
)
```

Shifts the torso's center of mass by 0 to 5 cm **forward** and 3 to 7 cm
**up**. Notice the ranges don't include zero on the z axis, and x is only
positive. This isn't symmetric noise around the URDF; it's a *systematic
bias* plus noise.

The code doesn't explain why. A reasonable inference is that the real
robot's torso carries mass the URDF doesn't fully capture (batteries,
compute, wiring, a head), and those sit higher and further forward than the
CAD model says. Whatever the source, the lesson is general: **if you know
the direction of a sim-to-real error, randomize around the corrected value,
not around the nominal one.**

### Reset: a fresh start, but never the same one

**`reset_base`**

```python
reset_base = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),
                       "pitch": (-0.15, 0.15), "roll": (-0.1, 0.1)},
        "velocity_range": {"pitch": (-0.5, 0.5), "roll": (-0.3, 0.3)},
    },
)
```

Each episode starts from a random position within ±0.5 m of the env origin,
any heading, a slight tilt (up to ±8.6° pitch, ±5.7° roll), and some initial
tipping velocity. The robot is born slightly off-balance, so it must recover
before it can do anything else.

**`reset_robot_joints`**

```python
reset_robot_joints = EventTerm(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={"position_range": (-0.5, 0.5), "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
)
```

Every joint starts up to **±0.5 rad** (about ±29°) from its default, with up to
±0.5 rad/s of velocity (clamped to the joint limits). That's a *big*
perturbation: some robots start with a leg halfway lifted or an arm flung
out. It forces the policy to learn to recover from awkward configurations,
which is exactly what happens after a real stumble.

**`pd_gains_rand`**

```python
pd_gains_rand = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "stiffness_distribution_params": (0.8, 1.2),
        "damping_distribution_params": (0.8, 1.2),
        "operation": "scale",
        "distribution": "uniform",
    },
)
```

Each episode, every joint's Kp and Kd are multiplied by a random factor in
[0.8, 1.2]. This models motor-to-motor variation, temperature effects (motors
get weaker as they heat up), and imperfect gain tracking on the motor drivers.

### Interval: the invisible bully

**`push_robot`**

```python
push_robot = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
                               "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.5, 0.5)}},
)
```

Every 1 to 3 seconds, each robot's base gets a random **velocity kick**: up
to 0.5 m/s horizontally, 0.3 m/s vertically, and up to 0.4 to 0.5 rad/s of
rotation, *added* to its current velocity. That's roughly what happens when
someone bumps into the robot or it steps on something unexpected. This is the
source of those "robot gets shoved and recovers" videos.

Remember the pose reward's disturbance relaxation from Chapter 5? This is
what it's for. When a push knocks the robot off its commanded velocity, the
pose reward loosens so the robot is free to take a big recovery step.

## 6.3 The sim-to-real ledger

Let's be systematic. Here's every mechanism in the repo that fights
sim-to-real gaps, and the gap it targets:

| Real-world gap | Mechanism in this repo | Where |
|----------------|------------------------|-------|
| Actuation latency | Random 0–25 ms actuator delay | `DelayedPDActuatorCfg` (Ch. 2) |
| Sensor latency | Random 0–1 / 0–2 policy-step obs delay | `delayed_obs` (Ch. 4) |
| Sensor noise | Uniform noise on all policy observations | `ObsTerm(noise=...)` (Ch. 4) |
| Motor dynamics | Armature and joint friction | Actuator config (Ch. 2) |
| Motor strength variation | Kp/Kd scaled 0.8–1.2 | `pd_gains_rand` |
| Encoder calibration | Default pose shifted ±0.02 rad | `qpos0_rand` |
| Mass distribution | Torso CoM biased forward/up | `base_com` |
| Floor friction | Foot friction 0.8–1.2 | `foot_friction` |
| Uneven ground | Half the tiles rough (2–6 cm) | Terrain (Ch. 4) |
| External disturbances | Velocity pushes every 1–3 s | `push_robot` |
| Awkward states | Large joint and base reset noise | `reset_*` events |
| Unobservable state | Asymmetric critic; no linear velocity for the policy | Obs groups (Ch. 4) |
| Jittery commands | Action rate penalty | Rewards (Ch. 5) |

And, just as important, what's **not** randomized in this repo, which you
might add as you push the robot harder:

- Link masses and inertias (only the torso CoM moves)
- Motor torque–speed curves (the effort limit is a flat cap here, while real
  motors lose torque at high speed)
- Gear backlash and joint compliance
- Ground restitution and ground friction variation beyond the feet
- IMU mounting misalignment or bias drift
- Terrain beyond small bumps (stairs, slopes, gaps)

> 🔧 **Under the Hood: the randomization trade-off**
> More randomization means more robustness but a more *conservative* policy.
> A robot that has to handle feet with 0.8 friction and 20% weaker motors in
> every episode walks more cautiously than one trained for exactly the real
> values. Too much, and the policy can't learn at all. The art is to
> randomize **only as wide as your real uncertainty**. That's why system
> identification (measuring the real robot) and randomization go hand in hand.

## 6.4 Terminations: when does an episode end?

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
```

Only two ways to end:

1. **Time out** after 20 s (1,000 steps). The `time_out=True` flag is
   important. It tells RSL‑RL this ending is *artificial* (the robot didn't do
   anything wrong; we just stopped watching), so the value function
   should **bootstrap**: estimate that the future would have continued to be
   good, instead of treating the end as "no more reward, ever". Getting this
   wrong biases the value function and makes robots behave strangely near the
   end of episodes.
2. **Fell over**: the angle between the base's "up" and world up exceeds 70°.
   Isaac Lab computes it as `acos(−projected_gravity_z)`. This is a *real*
   termination: no bootstrapping, and the robot forfeits all the reward it
   would have earned in the rest of the episode.

That second point is the implicit "don't fall" incentive. There's no explicit
fall penalty in the reward list. Falling is punished by *losing the future*:
with ~0.1 to 0.2 reward per step, falling at step 100 of 1,000 costs the
robot a large chunk of its potential return.

> 🚗 **Driving Déjà Vu**
> Timeouts vs. failures is the difference between "the scenario's
> simulation budget ran out" and "the car collided". In scenario-based
> evaluation you'd never score a clean timeout as a failure. Same idea here,
> but for *training* targets rather than evaluation metrics.

> ⚠️ **Pothole: 70° is generous**
> A robot lying on the ground on its side is at 90°, so it terminates. But a
> robot that has collapsed into a kneeling crouch with an upright torso never
> hits 70°. Nothing ends the episode, and it can sit there for the rest of the
> episode collecting what little reward it can. The `pose`, `upright` and
> tracking rewards make that a poor strategy, but if you ever see policies
> "sitting", consider adding a base-height termination
> (`mdp.root_height_below_minimum` exists in Isaac Lab).

## 6.5 The Play config: turning the lies off

When you *watch* a policy, you want to see its normal behavior, not a
randomized stress test. `Asimov1VelocityEnvCfg_PLAY` inherits the training
config and switches things off:

```python
@configclass
class Asimov1VelocityEnvCfg_PLAY(Asimov1VelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = int(1e9)                          # effectively infinite
        self.observations.policy.enable_corruption = False        # no obs noise
        self.events.push_robot = None                             # no pushes
        self.events.qpos0_rand = None                             # perfect calibration
        self.events.pd_gains_rand = None                          # nominal gains
        self.events.base_com = None                               # nominal CoM
        self.events.foot_friction = None                          # nominal friction
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ...}
        self.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ...}
        self.commands.twist.ranges.lin_vel_x = (0.6, 0.8)         # always walking forward
        self.commands.twist.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.twist.ranges.ang_vel_z = (-0.6, 0.6)
        self.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG.replace(
            num_rows=5, num_cols=5, border_width=10.0
        )
        self.scene.terrain.max_init_terrain_level = 4
```

Three details worth noticing:

- **Not everything is turned off.** The actuator *delay* is part of the robot
  config, not an event, so it stays on in Play. The observation delays in
  `delayed_obs` also stay on; `enable_corruption = False` only disables the
  *noise*. Play is "nominal robot", not "ideal robot".
- **The forward command is always 0.6 to 0.8 m/s** so you see the robot walk,
  not stand around. The 20% standing envs and 30% heading envs still apply.
- **Smaller terrain** (5 × 5 tiles) for 32 robots, which loads faster.

> ⚠️ **Pothole: "It works in Play" is not evidence**
> Play is the *easiest* version of the environment. A policy that walks
> beautifully in Play might fall constantly with randomization on. To judge
> robustness, evaluate on the *training* config (or a harder one). The best
> check before hardware is **sim-to-sim**: run the exported policy in a
> *different* simulator (the README acknowledges MuJoCo and mjlab, which are
> commonly used for this). If it survives a different physics engine, it has a
> better chance of surviving reality.

## 🏁 Pit Stop

1. Name the three event modes and one example of each from this repo.
2. What real-world problem does `qpos0_rand` model, and why does it also
   update the action term's offset?
3. Why is the torso CoM randomization range not centered on zero?
4. Why does a time-out need to be flagged differently from a fall?
5. Which sim-to-real mechanisms remain active in the Play config?
6. Name two physical effects this repo does *not* randomize.

<details>
<summary>Answers</summary>

1. `startup` (`foot_friction`, `qpos0_rand`, `base_com`), `reset`
   (`reset_base`, `reset_robot_joints`, `pd_gains_rand`), `interval`
   (`push_robot`).
2. Encoder zero-offset (calibration) error. The action offset must match the
   shifted default so that observations and actions share the same
   (miscalibrated) reference frame, as they would on a real robot.
3. It appears to model a systematic difference between the URDF and the real
   torso's mass distribution (an inference; the code doesn't say), so the
   randomization is centered on the believed-true value.
4. So the value function bootstraps at time-outs (the future would have
   continued) but not at falls (the future is lost).
5. The actuator delay and the observation delays.
6. Any two of: link masses/inertias, torque–speed curves, backlash/compliance,
   IMU misalignment/bias, restitution, non-foot friction, larger terrain
   features.

</details>

---

*[← Speaking Robot: Rewards](05-rewards.md) · [Contents](README.md) · [Next: PPO for People Who Love Supervised Learning →](07-ppo.md)*
