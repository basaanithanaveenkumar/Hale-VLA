# Chapter 5 — Speaking Robot: The Art and Science of Rewards

*[← The MDP Is a Config File](04-the-mdp-as-config.md) · [Contents](README.md) · [Next: Lying to Your Robot on Purpose →](06-randomization-and-sim2real.md)*

---

A story that circulates in every RL lab, in one form or another:

> An engineer adds a reward for "feet in the air" to encourage stepping.
> The next morning, the robot has learned to lie on its back and kick both
> legs at the sky. Maximum air time. Zero walking.

Rewards are the only language you have to tell the robot what you want, and
the robot is the most literal-minded listener you'll ever meet. It doesn't
know what you *meant*. It knows what you *measured*.

This chapter walks through every one of the 18 reward terms in
`velocity_env_cfg.py`, with the math from `mdp/rewards.py`, and explains what
each is protecting against.

## 5.1 How rewards are combined

The reward manager computes, every policy step, for every environment:

```
r_t = Σ_i  weight_i · term_i(s_t) · dt
```

where `dt` is the policy step (0.02 s). That multiplication by `dt` is done by
Isaac Lab's `RewardManager` itself, and it's easy to miss. It makes the
weights roughly "per second" quantities, so the total return doesn't blow up
if you change the control rate. It also means the per-step reward is small:
a perfect velocity-tracking step earns `5.0 × 1.0 × 0.02 = 0.1`.

> 🚗 **Driving Déjà Vu**
> A reward is a **cost function with the sign flipped**, and you've tuned
> plenty of those. Planner costs trade progress against comfort against
> safety margins, and every weight is a policy decision about priorities.
> The same is true here. The main difference: a planner optimizes the cost
> exactly at run time, while RL optimizes it *approximately*, *in
> expectation*, and *through a neural network that will exploit any loophole*.

### Two families of terms

Every term in this file falls into one of two shapes:

**1. Gaussian-kernel "bonuses" (positive weight):**

```
term = exp(−error² / σ²)
```

This is 1.0 when the error is zero and decays smoothly toward 0. It's bounded,
so a single huge error can't produce a huge negative number, and σ sets how
forgiving it is. At `error = σ`, the term is `e⁻¹ ≈ 0.37`.

**2. Penalties (negative weight):**

A non-negative cost like a squared velocity or a count of bad events, which
the negative weight turns into a punishment.

A healthy locomotion reward has a few strong, bounded bonuses that define the
*goal*, and many small penalties that define the *style* and the *safety
envelope*.

### The command gate

Many terms only apply when the robot is actually asked to move. That's
`_command_active`:

```python
def _command_active(env, command_name, threshold):
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (cmd_norm > threshold).float()
```

It combines the linear speed and the turning rate into one "how much motion
is requested" scalar (`|v_xy| + |ωz|`) and returns 1.0 above the threshold,
0.0 below. When the robot is told to stand still, gait rewards like "lift
your feet" switch off, and posture terms switch on.

## 5.2 The full reward table

Here's everything at a glance. "Gate" is the command threshold, if any.

| # | Name | Weight | Shape | Gate | In one sentence |
|---|------|-------:|-------|:----:|-----------------|
| 1 | `track_linear_velocity` | +5.0 | Gaussian, σ = 0.5 | | Move at the commanded `vx, vy`, without bouncing |
| 2 | `track_angular_velocity` | +3.0 | Gaussian, σ = 0.707 | | Turn at the commanded `ωz`, without wobbling |
| 3 | `upright` | +1.0 | Gaussian, σ = √0.2 | | Keep the torso vertical |
| 4 | `pose` | +1.0 | Gaussian, speed-dependent σ | | Stay near the default pose, loosely when walking |
| 5 | `air_time` | +0.5 | count | 0.5 | Take real steps (feet airborne 0.05–0.5 s) |
| 6 | `foot_clearance` | −2.0 | cost | 0.05 | Swinging feet should be 10 cm high |
| 7 | `foot_swing_height` | −0.25 | cost at touchdown | 0.05 | Each step's peak height should be 10 cm |
| 8 | `foot_slip` | −0.1 | cost | 0.05 | Planted feet shouldn't slide |
| 9 | `feet_orientation` | −1.0 | cost | | Keep feet flat |
| 10 | `action_rate_l2` | −0.5 | cost | | Don't change actions abruptly |
| 11 | `body_ang_vel` | −0.08 | cost | | Don't let the torso roll/pitch quickly |
| 12 | `angular_momentum` | −0.03 | cost | | Don't swing the whole body around |
| 13 | `dof_pos_limits` | −1.0 | cost | | Stay inside the soft joint limits |
| 14 | `soft_landing` | −1e‑5 | cost at touchdown | 0.05 | Land softly |
| 15 | `joint_deviation_l1` | −1.0 | cost | < 0.1 | When standing, really hold the default pose |
| 16 | `feet_stumble` | −1.25 | event | | Don't kick into obstacles |
| 17 | `feet_contact_force_limit` | −5e‑4 | cost | | Don't stomp above 350 N |
| 18 | `self_collisions` | −1.0 | count | | Don't hit yourself |

Now, one by one.

## 5.3 The goal: tracking the command

### 1. `track_linear_velocity` (+5.0)

```python
def track_linear_velocity(env, std, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)
```

```
term = exp( −( |v_cmd,xy − v_xy|² + v_z² ) / 0.25 )
```

This is the main objective, and it has the biggest weight. Note two details:

- The velocity is in the **base frame** (`root_lin_vel_b`), matching the
  command's frame. "Forward" means "the way the pelvis faces".
- The **vertical velocity** `v_z` is folded into the error. Bouncing up and
  down costs tracking reward. A simple way to discourage hopping gaits
  without a separate term.

With σ = 0.5 m/s, a tracking error of 0.25 m/s still earns
`exp(−0.0625/0.25) ≈ 0.78`. That's forgiving, which is good early in training
when the robot can barely stand.

### 2. `track_angular_velocity` (+3.0)

```
term = exp( −( (ωz,cmd − ωz)² + ωx² + ωy² ) / 0.5 )
```

The same idea for turning. And the same trick: the roll and pitch rates of
the base (`ωx`, `ωy`) are folded in as error, so wobbling costs reward.

### 3. `upright` (+1.0)

```python
def flat_orientation(env, std, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids and asset_cfg.body_ids != slice(None):
        body_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        gravity_dir = asset.data.GRAVITY_VEC_W
        projected = math_utils.quat_apply_inverse(body_quat, gravity_dir)
        xy_squared = torch.sum(torch.square(projected[:, :2]), dim=1)
    else:
        xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-xy_squared / std**2)
```

Gravity projected into the **torso** frame (`waist_yaw_link`, configured via
`TORSO_BODY`). When upright, the x and y components are zero. The function
falls back to the base (pelvis) if no body is specified. Using the torso
rather than the pelvis means the *upper body* must stay vertical, which is what
makes a walker look composed.

## 5.4 Posture: the variable-stiffness pose reward

### 4. `pose` (+1.0): `variable_posture`

This is the most sophisticated term in the file, so let's take it slowly.

The idea: reward being close to the default pose, but **how close** depends
on what the robot is doing.

```
term = exp( −mean_j( (q_j − q_default,j)² / σ_j² ) ) × pose_weight
```

The σ per joint comes from three tables, selected by the commanded motion
`total_speed = |v_cmd,xy| + |ωz,cmd|`:

| Mode | Condition | σ values (rad) |
|------|-----------|----------------|
| standing | `total_speed < 0.1` | 0.05 for **every** joint (very strict) |
| walking | `0.1 ≤ total_speed < 1.5` | hip pitch 0.5, knee 0.5, hip roll/yaw 0.15, ankle pitch 0.15, ankle roll 0.1, waist 0.15, shoulder pitch 0.15, other arm joints 0.1 |
| running | `total_speed ≥ 1.5` | as walking, but hip roll/yaw, ankle pitch, waist and shoulder pitch loosened to 0.25 |

Look at the walking column. Hip pitch and knee get σ = 0.5: *these are the
joints that make walking happen*, so they're allowed to swing freely. Hip
roll, hip yaw and the ankles get 0.15 or less: walking shouldn't need much
of them. The arms get 0.1 to 0.15: keep them relaxed near their default.

> 🔧 **Under the Hood: can the robot actually "run"?**
> The maximum possible `total_speed` from the command ranges is
> `√(0.8² + 0.5²) + 0.8 ≈ 1.74`, so the running table does kick in, but only
> for commands that combine fast diagonal motion with fast turning. Most of the
> time the walking table applies. The three-mode structure is there for
> when you widen the command ranges.

Two extra mechanisms make this term clever:

**Disturbance relaxation.** When the robot is being knocked around, forcing
it to hold a pose is counterproductive; it needs to move its limbs to recover.
So σ is *widened* by up to `disturbance_std_scale = 2.5×` based on a severity
score:

```python
lin_vel_error = |v_actual − v_cmd|              # how far off the commanded velocity
velocity_severity = clamp((lin_vel_error − 0.5) / (1.5 − 0.5), 0, 1)
tilt = |projected_gravity_xy|                     # how tilted the base is
tilt_severity   = clamp((tilt − 0.15) / (0.6 − 0.15), 0, 1)
severity = max(velocity_severity, tilt_severity)
std_scale = 1 + severity × (2.5 − 1)             # 1.0 … 2.5
```

A shove that makes the robot drift 1.5 m/s off its command, or tilt its
base so that the horizontal part of the gravity vector reaches 0.6 (about 37°),
relaxes the pose reward to its loosest. The metric
`Metrics/pose_disturbance_std_scale` logs the average.

**Speed-dependent weight.** When the commanded *linear* speed is below 0.3 m/s,
the term's value is multiplied by `pose_weight_slow = 2.0`; otherwise by
`pose_weight_fast = 1.0`. Slow and standing robots are held to their posture
twice as strongly.

> 🚗 **Driving Déjà Vu**
> This is **gain scheduling**, familiar from vehicle control: a
> lateral controller uses different gains at parking speed than on the
> highway. Here, the "gain" is how tightly the posture is regularized, and it's
> scheduled by the commanded speed and by a disturbance detector.

### 15. `joint_deviation_l1` (−1.0)

```python
def joint_deviation_l1(env, command_name="twist", command_threshold=0.1, asset_cfg=...):
    angle = q[:, joints] − q_default[:, joints]
    standing = 1.0 − _command_active(env, command_name, command_threshold)
    return torch.sum(torch.abs(angle), dim=1) * standing
```

Only active when **standing** (the gate is inverted). An L1 penalty, which
pushes small deviations all the way to zero harder than L2 would. The pose
reward's Gaussian barely pushes on tiny errors (its slope at zero is
flat). This L1 term makes the robot actually *settle* into its home pose
instead of hovering nearby, fidgeting.

## 5.5 Gait shaping: making it step like a walker

Without gait terms, a velocity-tracking policy often discovers shuffling:
feet barely leaving the ground, sliding forward in tiny increments. On
flat ground in sim, that's efficient. On real cobblestones, it trips.

### 5. `air_time` (+0.5)

```python
def feet_air_time(env, sensor_name, threshold_min=0.05, threshold_max=0.5,
                  command_name=None, command_threshold=0.5):
    current_air_time = sensor.data.current_air_time          # [N, 2]
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)               # 0, 1 or 2
    ...
    if command_name is not None:
        reward = reward * _command_active(env, command_name, command_threshold)
    return reward
```

Every step, each foot that has been in the air for between 0.05 s and 0.5 s
earns +1. So a normal step (foot lifts, swings for ~0.3 s, lands) earns
reward throughout its swing. A foot that stays up longer than 0.5 s stops
earning (no kicking at the sky, and no hopping on one leg). The gate at 0.5
means this only applies when the commanded motion is substantial.

`Metrics/air_time_mean` logs the average airborne time of airborne feet.

### 6. `foot_clearance` (−2.0)

```python
def feet_clearance(env, target_height, command_name=None, command_threshold=0.01, asset_cfg=...):
    foot_z = foot_pos_w(env, asset_cfg)[:, :, 2]
    vel_norm = torch.norm(foot_vel_w(env, asset_cfg)[:, :, :2], dim=-1)
    cost = torch.sum(torch.abs(foot_z - target_height) * vel_norm, dim=1)
    ...
```

```
cost = Σ_feet |z_foot − 0.10| · |v_foot,xy|
```

An elegant formulation. The cost is the height error *weighted by horizontal
foot speed*. A planted foot (speed ≈ 0) costs nothing no matter its height.
A foot moving forward fast should be at about 10 cm. So it effectively says:
**"when your foot moves, it should be 10 cm up."** That kills shuffling
directly.

### 7. `foot_swing_height` (−0.25)

```python
class feet_swing_height(ManagerTermBase):
    def __init__(self, cfg, env):
        ...
        self._peak_heights = torch.zeros((env.num_envs, n_feet), device=env.device)

    def __call__(self, env, sensor_name, target_height, command_name, command_threshold, asset_cfg):
        foot_heights = foot_pos_w(env, asset_cfg)[:, :, 2]
        in_air = ~_contact_flags(sensor)
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, foot_heights), self._peak_heights)
        first_contact = sensor.compute_first_contact(dt=env.step_dt)
        active = _command_active(env, command_name, command_threshold)
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        ...
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return cost
```

This one is a stateful class that tracks each foot's **peak height during
its swing**. At the moment of touchdown (`first_contact`), it charges
`(peak/0.10 − 1)²`, then resets the peak. So each step is graded once, on
whether it reached 10 cm. A step that peaks at 5 cm costs 0.25; one at 15 cm
costs 0.25 too. It's symmetric around the target.

`foot_clearance` shapes the *whole swing*; `foot_swing_height` grades each
*step's peak*. Together they produce a consistent, deliberate step height.

### 8. `foot_slip` (−0.1)

```
cost = Σ_feet |v_foot,xy|² · in_contact
```

A foot touching the ground shouldn't be sliding. Slipping wastes energy,
destroys tracking and, on real floors, is how robots fall. `Metrics/slip_velocity_mean`
logs the average sliding speed of feet in contact.

### 9. `feet_orientation` (−1.0)

```python
def feet_orientation_penalty(env, asset_cfg=...):
    body_quats = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(body_quats.shape[1]):
        grav_local = math_utils.quat_apply_inverse(body_quats[:, i], gravity.expand(env.num_envs, -1))
        penalty += torch.norm(grav_local[:, :2], dim=-1)
    return penalty
```

For each foot, gravity in the foot frame should be straight down. Any
horizontal component means the foot is tilted. Note this uses the **norm**,
not its square, so even small tilts are penalized linearly. Flat feet mean
full sole contact, which is better grip and a larger support polygon.

(The `for` loop is over **two feet**, not over environments, so it's still
fully vectorized over the 4,096 robots.)

### 14. `soft_landing` (−1e‑5)

```
cost = Σ_feet |F_contact| · first_contact
```

The contact force magnitude at the moment of touchdown. Forces are in newtons
(hundreds of them for a humanoid landing), hence the tiny weight. Soft
landings are quieter, gentler on gearboxes, and less likely to bounce.
`Metrics/landing_force_mean` logs the average impact.

### 16. `feet_stumble` (−1.25)

```python
def feet_stumble(env, sensor_name, ratio_threshold=4.0):
    forces = sensor.data.net_forces_w
    force_horizontal = torch.norm(forces[:, :, :2], dim=-1)
    force_vertical = torch.abs(forces[:, :, 2])
    stumble = force_horizontal > (ratio_threshold * force_vertical)
    stumble &= _contact_flags(sensor)
    any_stumble = torch.any(stumble, dim=1).float()
    ...
    return any_stumble
```

When a foot hits the *side* of a bump, the contact force is mostly
horizontal. If horizontal force exceeds 4× vertical, that's a toe catching
on an obstacle: a stumble. A binary penalty per step. This term and the
cobblestone terrain work together; on flat ground it would almost never fire.
`Metrics/stumble_rate` logs the fraction of robots stumbling.

### 17. `feet_contact_force_limit` (−5e‑4)

```
cost = Σ_feet max(0, |F_z| − 350 N)
```

A hinge penalty on vertical force above 350 N. Normal walking is allowed;
stomping isn't. `Metrics/max_contact_force` logs the maximum.

## 5.6 Smoothness and safety

### 10. `action_rate_l2` (−0.5)

```
cost = Σ_j (a_t,j − a_{t−1},j)²
```

Isaac Lab's built-in. Penalizes the change in raw action between consecutive
steps. This is **the** most important term for sim-to-real. Without it,
policies learn to vibrate: bang-bang motor commands at 50 Hz, which the
simulator tolerates and real gearboxes do not. It's also why the last action
is in the observation; the policy needs it to know what "smooth" means.

> 🚗 **Driving Déjà Vu**
> This is **jerk minimization** in your trajectory planner. Passengers hate
> jerky rides; gearboxes hate jerky torques. Same math, same reason.

### 11. `body_ang_vel` (−0.08)

```
cost = ωx² + ωy²  (torso, world frame)
```

Discourages the torso from rolling and pitching quickly. Combined with
`upright`, it makes the upper body calm.

### 12. `angular_momentum` (−0.03)

```python
class angular_momentum_penalty(ManagerTermBase):
    def __init__(self, cfg, env):
        ...
        self._masses = asset.data.default_mass.to(env.device)
        self._total_mass = self._masses.sum(dim=1, keepdim=True)

    def __call__(self, env, asset_cfg=_DEFAULT_ASSET_CFG):
        pos = asset.data.body_com_pos_w                   # [N, B, 3]
        vel = asset.data.body_com_lin_vel_w               # [N, B, 3]
        m = self._masses.unsqueeze(-1)                    # [N, B, 1]
        com = (m * pos).sum(dim=1) / self._total_mass     # whole-body CoM
        com_vel = (m * vel).sum(dim=1) / self._total_mass
        r = pos - com.unsqueeze(1)
        v = vel - com_vel.unsqueeze(1)
        angmom = (m * torch.cross(r, v, dim=-1)).sum(dim=1)
        return torch.sum(torch.square(angmom), dim=-1)
```

The **whole-body angular momentum about the center of mass**:
`L = Σ_bodies m_b · (r_b × v_b)`, with positions and velocities relative to
the CoM. Human walking keeps whole-body angular momentum small. Arms swing
opposite to legs precisely to cancel it. Penalizing `|L|²` nudges the policy
toward that natural counter-swing. (It approximates each body as a point mass
and ignores the bodies' own spin, which is a reasonable simplification for a
regularizer.)

It's a class because it caches the masses once at startup. It logs
`Metrics/angular_momentum_mean`.

> 🔧 **Under the Hood: why angular momentum matters**
> For a legged robot, the rate of change of angular momentum about the CoM
> is determined entirely by contact forces (gravity acts *at* the CoM and
> contributes no moment about it). A robot that builds up large angular momentum
> needs large, well-placed contact forces to stop it. That's hard with small
> feet and a 17 N·m ankle roll. Keeping `L` small keeps the robot "easy to
> balance". Classical humanoid control reasons explicitly about this
> ("centroidal dynamics"); here, it's just a penalty.

### 13. `dof_pos_limits` (−1.0)

Isaac Lab's built-in `joint_pos_limits`: the sum over joints of how far each
is outside its **soft** limits. Remember `soft_joint_pos_limit_factor=0.9`
from Chapter 1: the soft range is the middle 90% of the URDF range. Hitting
hard stops on real hardware is loud, damaging, and exactly the kind of thing
a policy trained without this term will do casually.

### 18. `self_collisions` (−1.0)

```python
def self_collision_cost(env, sensor_name, force_threshold=10.0):
    force_matrix = sensor.data.force_matrix_w          # [N, watched, filtered, 3]
    hit = force_matrix.norm(dim=-1) > force_threshold
    return hit.sum(dim=(1, 2)).float()
```

Counts the (extremity, body) pairs pressing with more than 10 N: wrists
into the pelvis, knees into the other hip, and so on. Each pair costs 1 per
step.

## 5.7 Reading the reward budget

It helps to know the *scale* of each term to understand which one dominates.
Per policy step (including the `dt = 0.02` factor):

- **Max positive reward per step:** tracking lin (5) + tracking ang (3) +
  upright (1) + pose (up to 2 when slow) + air time (0.5 × 2 feet = 1)
  = 12, times 0.02 = **0.24 per step**, or about **240 over a perfect
  1,000-step episode**.
- **Typical penalties** in a decent policy are a small fraction of that. If
  your logs show one penalty eating most of the bonus, that's your
  bottleneck.

Isaac Lab logs each term's episode average under `Episode_Reward/<name>` so
you can do this analysis on real training runs (Chapter 9).

## 5.8 Reward design lessons you can take anywhere

1. **Define the goal with bounded bonuses.** Gaussian kernels are stable and
   interpretable.
2. **Fold "don't do that" into the goal when you can.** Vertical velocity
   inside the linear tracking term, roll/pitch rate inside the angular tracking
   term. Fewer terms, fewer weights.
3. **Gate by context.** Gait rewards off when standing, posture penalties on.
4. **Make it stateful when the concept is stateful.** "Peak swing height" is
   a property of a *step*, not a single frame, so it gets a class with a
   buffer.
5. **Log everything you shape.** Most custom terms here write a `Metrics/*`
   value into `env.extras["log"]`. When the gait looks odd, those metrics tell
   you why.
6. **Every penalty you add is a bet.** It encodes an assumption about what good
   motion looks like. Hand-writing *all* of good motion as penalties is
   exhausting, and it's never quite right. Which is exactly why Chapter 8's
   AMP exists: it lets a *discriminator* learn what "natural" means from
   example motion, instead of you writing it down term by term.

## 🏁 Pit Stop

1. What does Isaac Lab multiply each reward term by, besides its weight?
2. Why does `track_linear_velocity` include the vertical velocity in its
   error?
3. What does `foot_clearance` cost for a planted foot? Why?
4. When is `joint_deviation_l1` active? Why L1 instead of L2?
5. What physical behavior does the `angular_momentum` penalty encourage in the
   arms?
6. Which term would you look at first if the real robot's motors buzzed and
   overheated?

<details>
<summary>Answers</summary>

1. The policy step `dt` (0.02 s).
2. To discourage bouncing/hopping without a separate term.
3. Zero, because the height error is multiplied by the foot's horizontal
   speed, which is about zero when planted.
4. When the command is below 0.1 (standing). L1 keeps a constant pull on
   small deviations, so the robot actually settles rather than hovering near
   the pose.
5. Counter-swinging the arms against the legs, which cancels whole-body
   angular momentum.
6. `action_rate_l2` (and the related smoothness terms): buzzing usually means
   high-frequency action changes.

</details>

---

*[← The MDP Is a Config File](04-the-mdp-as-config.md) · [Contents](README.md) · [Next: Lying to Your Robot on Purpose →](06-randomization-and-sim2real.md)*
