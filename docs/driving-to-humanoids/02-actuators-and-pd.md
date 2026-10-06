# Chapter 2 — Muscles Made of Math: Actuators, PD Control, and Delay

*[← Meet Asimov‑1](01-meet-asimov.md) · [Contents](README.md) · [Next: A Thousand Worlds per Second →](03-simulation-and-repo-tour.md)*

---

Here's a question that trips up almost everyone coming from ML:

> *The policy outputs 23 numbers. Are those torques?*

No. And the reason they're not is one of the most important design
decisions in all of legged robot learning.

## 2.1 What the network actually outputs

Look at the action config in `tasks/locomotion/velocity_env_cfg.py`:

```python
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        preserve_order=True,
        scale=ASIMOV_1_ACTION_SCALE,      # 0.25
        use_default_offset=True,
    )
```

This tells Isaac Lab: *"interpret the policy's 23 outputs as joint position
targets, in this exact joint order, scaled by 0.25, offset by the default
pose."* In an equation, for each joint *i*:

```
q_target[i] = q_default[i] + 0.25 * a[i]
```

where `a` is the raw network output and `q_default` is the standing pose from
Chapter 1.

So the network says, in effect: *"left knee, I'd like you to be 0.1 radians
more bent than your default."* It doesn't say *how hard* to push. That's the
actuator's job.

`preserve_order=True` is subtle but critical: it forces the action vector to
follow `ASIMOV_1_JOINT_NAMES` order, instead of whatever order the simulator
happened to parse joints from the URDF. (Isaac Lab, like PhysX, typically
orders joints breadth-first through the kinematic tree, which interleaves left
and right.) With `preserve_order=True`, action index 3 is always
`left_knee_joint`. Your deployment code will thank you.

## 2.2 From target to torque: the PD controller

Once there's a target, something must convert it into torque. That's the
**PD controller** (proportional-derivative):

```
τ = Kp · (q_target − q) + Kd · (0 − q̇)
τ = clip(τ, −effort_limit, +effort_limit)
```

- `Kp` (**stiffness**) pulls the joint toward the target like a spring. Twice
  the error, twice the torque.
- `Kd` (**damping**) resists motion like a shock absorber. It opposes
  velocity, which prevents oscillation.
- The target velocity is zero; the policy only sets positions.
- `effort_limit` is the maximum torque the motor can produce, in N·m.

> 🚗 **Driving Déjà Vu**
> You've seen this exact structure in lateral control: steering command =
> `Kp · lateral_error + Kd · lateral_error_rate`. The difference is that in
> AD, a *planner* sets the target path and a hand-tuned PD tracks it. Here,
> a *neural network* sets the target 50 times a second, and the PD runs
> underneath at 200 Hz. The network is the planner; the PD is the low-level
> controller. It's the same two-layer structure you already know, just
> repeated 23 times.

### Why not output torques directly?

You *could* train a policy that outputs torques. Some research does. But
position targets with a PD loop underneath have huge practical advantages:

1. **A built-in stabilizer.** Even a random policy produces a robot that
   *holds a pose* rather than collapsing. Early training is much easier.
2. **Higher effective control rate.** The policy runs at 50 Hz, but the PD
   loop runs at the physics rate (200 Hz in sim, often 1 kHz+ on the motor
   driver). High-frequency stabilization happens "for free".
3. **It matches the hardware.** Real motor drivers for robots like this
   usually run their own PD loop onboard and accept position targets with
   gains. The sim models what the hardware does.
4. **It's more transferable.** Torque-level policies are very sensitive to
   motor modeling errors. Position-level policies are more forgiving.

The cost: the policy can't directly command a precise torque, and the PD
gains become part of the "physics" the policy learns. Change the gains on the
real robot, and your policy is out of distribution.

## 2.3 The actuator table

Here's the heart of `asimov_1.py`, summarized as a table:

| Group | Joints | Kp (N·m/rad) | Kd (N·m·s/rad) | Effort limit (N·m) | Armature (kg·m²) | Friction |
|-------|--------|---:|---:|---:|---:|---:|
| `hip_pitch` | `.*_hip_pitch_joint` | 150 | 5 | 45 | 0.0698 | 0.70 |
| `hip_roll` | `.*_hip_roll_joint` | 150 | 5 | 45 | 0.1400 | 0.20 |
| `hip_yaw` | `.*_hip_yaw_joint` | 150 | 5 | 28 | 0.0687 | 0.70 |
| `knee` | `.*_knee_joint` | 150 | 5 | 45 | 0.0330 | 0.70 |
| `ankle_pitch` | `.*_ankle_pitch_joint` | 110 | 5 | 40 | 0.0484 | 0.40 |
| `ankle_roll` | `.*_ankle_roll_joint` | 110 | 5 | 17 | 0.0484 | 0.40 |
| `waist` | `waist_yaw_joint` | 65 | 5 | 40 | 0.0698 | 0.70 |
| `shoulder_pitch` | `.*_shoulder_pitch_joint` | 57 | 5 | 30 | 0.1400 | 0.20 |
| `shoulder_roll` | `.*_shoulder_roll_joint` | 86 | 5 | 25 | 0.0330 | 0.70 |
| `shoulder_yaw` | `.*_shoulder_yaw_joint` | 96 | 5 | 20 | 0.0687 | 0.70 |
| `elbow_wrist` | `.*_elbow_joint`, `.*_wrist_yaw_joint` | 40 | 2 | 12 | 0.0242 | 0.40 |

Every group is a `DelayedPDActuatorCfg` with `min_delay=0`, `max_delay=5`.

Let's read this table like a hardware engineer.

**Legs are stiff, arms are soft.** Hips and knees use Kp = 150; elbows and
wrists use 40. Legs carry the whole body weight and must reject impacts. Arms
mostly swing for balance, and soft arms are safer and more natural.

**Ankle roll is weak.** Only 17 N·m. Many humanoids have a relatively weak
ankle roll, often because of the actuator packaging at the ankle. This
means the policy can't rely on "ankle strategy" to balance sideways; it must
step instead. That shapes the whole gait.

**Hip yaw is weaker than hip pitch.** 28 vs 45 N·m. Twisting the leg needs less
torque than lifting the body.

### Armature: the invisible inertia

**Armature** is the reflected inertia of the motor's rotor through the
gearbox. If a rotor with inertia `I_rotor` drives the joint through a gear
ratio `N`, the joint "feels" an extra inertia of `I_rotor · N²`. With typical
robot gear ratios, that squared term makes armature comparable to, or even
larger than, the inertia of a small link like a foot.

Leaving armature out is a classic sim-to-real bug: in sim, the ankle whips
around unrealistically fast; on hardware, it's sluggish. Putting it in makes
the simulated joint accelerate the way the real one does.

### Friction: the sticky joint

`friction` models joint friction in the gearbox (dry friction that opposes
motion). Without it, a simulated joint glides with zero resistance for small
commands. On real hardware, a small command might not move the joint at all.
The values here (0.2 to 0.7) were presumably identified from, or tuned
against, the real motors.

> 🔧 **Under the Hood: where do these numbers come from?**
> In a well-run robotics project, armature and friction come from **system
> identification**: you run known motions or torques on the real actuator,
> record the response, and fit the parameters. Gains often come from the
> hardware team's recommended settings. The repo doesn't include the
> sysid process itself, only the results. When you move to a new robot,
> this table is one of the first things you'll need to build.

### A back-of-the-envelope check: will actions saturate?

With `scale = 0.25`, a raw action of `1.0` shifts the target by 0.25 rad
(about 14°). If the joint were held exactly at its old target, the PD
torque would be:

| Joint | Kp × 0.25 | Effort limit | Saturates at \|a\| ≈ |
|-------|---:|---:|---:|
| hip pitch | 37.5 N·m | 45 | 1.2 |
| ankle roll | 27.5 N·m | 17 | 0.62 |
| elbow | 10 N·m | 12 | 1.2 |

So a raw action around 1 is already near the torque limit for most joints,
and past it for the ankle roll. That's why the action scale is 0.25: it puts
a unit-scale network output (what a Gaussian policy with std ≈ 1 naturally
produces early in training) in the "useful but not insane" range.

> ⚠️ **Pothole**
> If you change `ASIMOV_1_ACTION_SCALE`, you change the meaning of every
> action. An old checkpoint won't work anymore, and the reward weights on
> `action_rate_l2` suddenly mean something different. Treat the action scale
> as part of the model architecture, not a free hyperparameter.

## 2.4 The delay: why "DELAYED" matters

Now the special sauce:

```python
DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 5
...
"hip_pitch": DelayedPDActuatorCfg(
    ...
    min_delay=DELAY_MIN_LAG,
    max_delay=DELAY_MAX_LAG,
),
```

In Isaac Lab, a `DelayedPDActuator` is an ideal PD actuator that applies
the position target from **several physics steps ago** rather than the
current one. The lag is measured in physics steps and is sampled randomly
between `min_delay` and `max_delay` for each environment, and resampled when
that environment resets.

With a physics step of 5 ms (Chapter 3), a lag of 0 to 5 steps means a
**0 to 25 ms** delay between "the network decided" and "the motor acted on
it".

Why simulate this? Because on the real robot, the action has to travel:

```
policy inference on the onboard computer
  → serialize command
  → bus (CAN, EtherCAT, or similar)
  → motor driver
  → PD loop picks up the new target
```

That pipeline takes time, and the time varies. A policy trained with zero
delay learns to rely on instant reactions. On hardware it will oscillate,
because every correction arrives late, overshoots, and gets over-corrected.
Training with a *random* delay forces the policy to be robust to *any*
latency in the range.

> 🚗 **Driving Déjà Vu**
> You know this one intimately. Drive-by-wire has actuator latency. The
> steering command you publish isn't the steering angle you get, and a
> controller tuned on a zero-latency model oscillates in the real car. Many
> AD teams add a latency model to their simulators for exactly this reason.
> Same lesson, different body.

### The timing diagram

Here's how the clocks fit together (details of `decimation` come in the next
chapter):

```
time (ms):   0    5    10   15   20   25   30   35   40
physics:     |----|----|----|----|----|----|----|----|     200 Hz (dt = 5 ms)
policy:      A0                  A1                  A2    50 Hz (every 4 physics steps)
PD target:   (A0 applied after a random 0-5 step lag, i.e. 0-25 ms)
PD torque:   computed every physics step from the (delayed) target
```

A delay of up to 25 ms is more than one full policy step (20 ms). So the policy
must sometimes act knowing its previous command hasn't even arrived yet.
That's hard, and it's precisely why the observation includes the **last
action** (Chapter 4): the policy needs to remember what it asked for.

## 2.5 Putting it all together: one control step

Let's trace one policy step from start to finish:

1. **Observe.** Isaac Lab collects the 78-number observation.
2. **Infer.** The actor MLP outputs 23 raw actions `a`.
3. **Process.** The action manager computes
   `q_target = q_default + 0.25 · a`, in `ASIMOV_1_JOINT_NAMES` order.
4. **Repeat 4 times (decimation):**
   1. Each actuator group pushes the new target into its delay buffer and
      reads out the target from `lag` physics steps ago.
   2. It computes `τ = Kp(q_target_delayed − q) − Kd·q̇`.
   3. It clips `τ` to `±effort_limit`.
   4. PhysX simulates 5 ms with those torques, plus armature and friction.
5. **Evaluate.** The reward manager computes rewards; the termination manager
   checks for falls; the event manager might push the robot.
6. **Loop.**

That's the robot's "nervous system". Next, let's meet the world it lives in.

## 🏁 Pit Stop

1. Write the equation for the joint target given raw action `a`.
2. Why does the simulator's built-in drive have zero gains while the actuator
   config has non-zero gains?
3. What real-world effect does armature model? What goes wrong without it?
4. How many milliseconds of delay can the actuator model introduce, and why is
   that more than one policy step?
5. Which joint saturates first for a raw action of about 0.6, and why does that
   matter for balance?

<details>
<summary>Answers</summary>

1. `q_target = q_default + 0.25 · a`.
2. The explicit actuator model computes and applies the torques. The built-in
   drive is disabled so the two don't stack.
3. Reflected rotor inertia through the gearbox. Without it, simulated joints
   accelerate unrealistically fast, and the policy learns motions the real
   robot can't reproduce.
4. 0 to 5 physics steps × 5 ms = up to 25 ms. The policy step is 20 ms.
5. Ankle roll (Kp × 0.25 = 27.5 N·m vs a 17 N·m limit). The policy can't
   balance sideways with the ankle alone and must use stepping and hip
   strategies.

</details>

---

*[← Meet Asimov‑1](01-meet-asimov.md) · [Contents](README.md) · [Next: A Thousand Worlds per Second →](03-simulation-and-repo-tour.md)*
