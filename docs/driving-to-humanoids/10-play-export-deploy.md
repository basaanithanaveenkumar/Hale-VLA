# Chapter 10 — From Checkpoint to Concrete Floor: Play, Export, and the Deployment Contract

*[← Race Day](09-training-and-debugging.md) · [Contents](README.md) · [Next: Your First Five Experiments →](11-experiments.md)*

---

There's a moment every AD engineer remembers: the first time your model
drove a real car. Not a replay, not a sim, a real car with a safety driver
whose hand hovered near the wheel. Your heart rate was not normal.

The humanoid version of that moment is when a policy that has only ever
existed inside a GPU sends its first joint targets to real motors. This
chapter is about getting there safely. It covers what the repo gives you
(`play.py` and model export) and, just as importantly, the **contract** that
whoever writes the robot-side code must honor.

## 10.1 Watching the policy: `play.py`

```bash
./isaac_asimov.sh --play --task Asimov1-Velocity-AMP-Play-v0 --num_envs 32
```

What happens, in order (from `scripts/rsl_rl/play.py`):

**1. Find the checkpoint.** Three ways, in priority order:

```python
if args_cli.use_pretrained_checkpoint:
    resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    ...
elif args_cli.checkpoint:
    resume_path = retrieve_file_path(args_cli.checkpoint)
else:
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
```

- `--use_pretrained_checkpoint` asks Isaac Lab for a published checkpoint
  for the training task (the `-Play` suffix is stripped). For these custom
  tasks, expect it to print that none is available; it's a hook inherited
  from Isaac Lab's template.
- `--checkpoint <path>` (or its alias `--target <path>`) loads a specific
  file.
- Otherwise, it picks the latest checkpoint in the latest run under
  `logs/rsl_rl/<experiment_name>/`. Because the Play task uses the same
  agent config as training, `experiment_name` matches and it finds your
  training logs automatically.

**2. Build the Play environment** (32 robots, randomization off, as in
Chapter 6), wrap it for RSL‑RL, create an `OnPolicyRunner`, and load the
checkpoint.

**3. Export the policy.** Every time you play, the policy is exported next to
the checkpoint:

```python
export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
if version.parse(installed_version) >= version.parse("4.0.0"):
    runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
```

So after playing `logs/rsl_rl/asimov_velocity_amp/<run>/model_9999.pt`,
you'll find:

```
logs/rsl_rl/asimov_velocity_amp/<run>/exported/policy.pt     ← TorchScript
logs/rsl_rl/asimov_velocity_amp/<run>/exported/policy.onnx   ← ONNX
```

Optionally, `--onnx-output <path>` writes an extra ONNX copy to a file path
or a directory (then named by `--onnx-filename`, default `policy.onnx`).

The exported model is the **actor only**, running deterministically (the
Gaussian mean, no sampling). The critic and the discriminator stay behind;
they were training tools.

**4. Run the loop:**

```python
policy = runner.get_inference_policy(device=env.unwrapped.device)
obs = env.get_observations()
while simulation_app.is_running():
    start_time = time.time()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy.reset(dones)
    ...
    sleep_time = dt - (time.time() - start_time)
    if args_cli.real_time and sleep_time > 0:
        time.sleep(sleep_time)
```

`--real-time` throttles the loop to 50 Hz of wall-clock time so motion looks
natural on screen; without it, the sim runs as fast as it can. `--video`
records one clip of `--video_length` steps to `<run>/videos/play/` and exits.

`policy.reset(dones)` resets any recurrent state for finished episodes. The
MLP here has none, so it's a no-op, but keeping it means the script works
unchanged if you switch to an LSTM policy.

> 🚗 **Driving Déjà Vu**
> `play.py` is your "replay the model in the simulator and eyeball it" tool,
> and the export step is your "convert to ONNX for the target hardware" step,
> conveniently fused. Just as with your perception models, the ONNX file is
> the artifact that crosses the boundary between the research team and the
> on-vehicle (on-robot) stack.

## 10.2 The deployment contract

The repo stops at the ONNX file. The code that runs on the real Asimov‑1
(reading sensors, running inference, commanding motors) isn't in this
repository. Whoever writes or maintains it must reproduce the *simulated
interface* exactly. Here's the full contract, assembled from everything in
Chapters 1 through 8.

### Timing

| Item | Value |
|------|-------|
| Policy rate | **50 Hz** (every 20 ms) |
| Input dtype | float32 |
| Recurrent state | None (plain MLP) |

### Input: the 78-number observation, in this exact order

| Index | Size | Content | Formula | Scale |
|------:|-----:|---------|---------|------:|
| 0–2 | 3 | Base angular velocity (pelvis frame) | gyro | × 0.25 |
| 3–5 | 3 | Projected gravity (pelvis frame) | `R_baseᵀ · [0, 0, −1]` | × 1 |
| 6–8 | 3 | Command `[vx, vy, ωz]` (base frame) | from joystick/planner | × 1 |
| 9–17 | 9 | Joint pos − default, `SLOT_0_1` order | `q − q_default` | × 1 |
| 18–25 | 8 | Joint pos − default, `SLOT_2_3` order | | × 1 |
| 26–31 | 6 | Joint pos − default, `SLOT_4_5` order | | × 1 |
| 32–40 | 9 | Joint vel, `SLOT_0_1` order | `q̇` | × 0.1 |
| 41–48 | 8 | Joint vel, `SLOT_2_3` order | | × 0.1 |
| 49–54 | 6 | Joint vel, `SLOT_4_5` order | | × 0.1 |
| 55–77 | 23 | Previous raw action, **`ASIMOV_1_JOINT_NAMES` order** | last network output | × 1 |

Where the slot orders are (from `velocity_env_cfg.py`):

```
SLOT_0_1: left_hip_pitch, left_hip_roll, right_hip_pitch, right_hip_roll, waist_yaw,
          right_shoulder_pitch, right_shoulder_roll, left_shoulder_pitch, left_shoulder_roll
SLOT_2_3: left_hip_yaw, left_knee, right_hip_yaw, right_knee,
          right_shoulder_yaw, right_elbow, left_shoulder_yaw, left_elbow
SLOT_4_5: left_ankle_pitch, left_ankle_roll, right_ankle_pitch, right_ankle_roll,
          right_wrist_yaw, left_wrist_yaw
```

No normalizer: `obs_normalization=False` for the actor, so the ONNX model's
input is exactly this vector. No noise or delay at deployment; the real world
provides those for free.

### Output: 23 numbers → joint targets

```
q_target[i] = q_default[i] + 0.25 · a[i]     for i in ASIMOV_1_JOINT_NAMES order
```

with `q_default` from `ASIMOV_1_STANDING_INIT_STATE` (the *nominal* values,
not a randomized version).

### Motor side

- **PD gains:** the nominal Kp/Kd from the actuator table in Chapter 2. The
  policy was trained with gains randomized ±20% around these; outside that
  range, you're out of distribution.
- **Torque limits:** the effort limits from the same table.
- **Latency:** total sensing-to-actuation delay should fall inside what was
  simulated (actions delayed 0–25 ms; gyro 0–20 ms; gravity estimate
  0–40 ms). Measure it on the real system.

> ⚠️ **Pothole: the top five deployment bugs**
> Every one of these makes a robot fall on its first step, and every one has
> happened to real teams on real robots:
> 1. **Joint order.** Building the observation in action order instead of
>    slot order (or vice versa).
> 2. **Sign conventions.** A joint whose encoder direction on hardware is
>    opposite to the URDF axis. Mirrored left/right joints make this easy to
>    get wrong.
> 3. **Missing scales.** Forgetting ×0.25 on angular velocity or ×0.1 on joint
>    velocity. The policy sees 4× or 10× larger values than it was trained on.
> 4. **Wrong frame.** Using the IMU's frame instead of the pelvis frame for
>    angular velocity and gravity, or using a quaternion convention mismatch
>    (`wxyz` vs `xyzw`; Isaac Lab uses `wxyz`).
> 5. **Last action.** Feeding the *processed* joint target instead of the raw
>    network output, or feeding it in the wrong order.

### A reference loop (illustrative)

Here's what a minimal robot-side loop looks like. **This is not code from the
repo**; it's a sketch to make the contract concrete. Hardware I/O functions
are placeholders.

```python
import numpy as np
import onnxruntime as ort

ACTION_ORDER = [...]          # ASIMOV_1_JOINT_NAMES, 23 names
SLOT_0_1, SLOT_2_3, SLOT_4_5 = [...], [...], [...]
OBS_JOINT_ORDER = SLOT_0_1 + SLOT_2_3 + SLOT_4_5
Q_DEFAULT = {...}             # from ASIMOV_1_STANDING_INIT_STATE, by name
ACTION_SCALE = 0.25

session = ort.InferenceSession("policy.onnx")
input_name = session.get_inputs()[0].name
last_action = np.zeros(23, dtype=np.float32)

def projected_gravity(quat_wxyz):
    w, x, y, z = quat_wxyz
    # rotate world gravity [0, 0, -1] into the body frame (R^T @ g)
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.array([gx, gy, gz], dtype=np.float32)

while running():                                   # 50 Hz
    imu = read_imu()                               # pelvis-frame gyro + orientation
    q, qd = read_joints()                          # dicts by joint name
    cmd = read_command()                           # [vx, vy, wz] in base frame

    obs = np.concatenate([
        0.25 * imu.gyro,
        projected_gravity(imu.quat_wxyz),
        cmd,
        [q[n] - Q_DEFAULT[n] for n in OBS_JOINT_ORDER],
        [0.1 * qd[n] for n in OBS_JOINT_ORDER],
        last_action,
    ]).astype(np.float32)[None, :]                 # shape (1, 78)

    action = session.run(None, {input_name: obs})[0][0]   # shape (23,)
    last_action = action.astype(np.float32)

    targets = {n: Q_DEFAULT[n] + ACTION_SCALE * action[i] for i, n in enumerate(ACTION_ORDER)}
    send_joint_targets(targets)                    # motor drivers run PD with nominal gains
    sleep_until_next_tick()
```

Before this goes anywhere near motors, test it by **replaying simulator
logs**: record observations and actions in Isaac Lab, feed the recorded
observations into this code path, and check that it produces the same
actions to within float tolerance. That catches ordering, scale and export
bugs with zero risk.

## 10.3 Sim-to-sim: the dress rehearsal

The README acknowledges MuJoCo and mjlab. A common practice in legged
robotics, and a strongly recommended one, is **sim-to-sim transfer**:
run the exported ONNX policy in a *different* physics engine (such as
MuJoCo, loading the same robot model) with a deployment-style loop like the
one above.

Why it's so valuable:

- It exercises your deployment code (ordering, scaling, frames) end to end.
- Different contact models and solvers act like a *different reality*. A
  policy that exploited a PhysX quirk tends to fail in MuJoCo, which is
  exactly what you want to discover before hardware.
- It's cheap and safe to iterate on.

> 🚗 **Driving Déjà Vu**
> This is like validating a perception model on a second vendor's sensor data
> or a different city before a new launch. Different distribution, same task.
> If it holds up, you gain confidence; if not, you learned something for free.

## 10.4 First contact: a hardware safety checklist

This isn't in the repo, but no guide to deploying a locomotion policy would
be responsible without it:

1. **Gantry or harness.** The first runs should be with the robot suspended so
   a fall is caught. Start with feet just touching the ground.
2. **E-stop within reach**, tested before every session. Know what it does
   (cut power vs. go limp vs. hold position).
3. **Soft start.** Interpolate from the robot's current pose to the default
   pose over a few seconds *before* handing control to the policy.
4. **Zero command first.** Test standing (command `[0, 0, 0]`) for a long time
   before any walking command.
5. **Conservative limits.** Start with reduced torque limits and command
   ranges; widen gradually.
6. **Log everything.** Observations, actions, targets, measured torques,
   temperatures, timestamps. When something goes wrong, you'll want to
   replay it in sim.
7. **Watch the motors.** Temperatures and current. A policy that's smooth in
   sim can still overheat real motors if it holds high torques.
8. **One change at a time.** New checkpoint *or* new deployment code *or* new
   floor. Never all three.

> 🚗 **Driving Déjà Vu**
> This is your closed-course testing protocol with a safety driver,
> translated. Same principles: containment, a kill switch, gradual
> escalation, full logging, one variable at a time.

## 10.5 Beyond this repo: joining the community

The README mentions community livestreams where developer-contributed
policies are tested on the real Asimov‑1, organized through Menlo Research's
Discord. That's a rare opportunity for someone learning: a path to real
hardware without owning a robot. Check the README for the current details.

## 🏁 Pit Stop

1. Where does `play.py` write the exported policy files, and in which formats?
2. Is the ONNX model's input normalized internally? Why does that matter?
3. At deployment, what is the joint order of the joint-position block of the
   observation? And of the action?
4. Name three of the "top five" deployment bugs.
5. What's the value of replaying simulator logs through the deployment code?

<details>
<summary>Answers</summary>

1. `<checkpoint_dir>/exported/policy.pt` (TorchScript) and `policy.onnx`
   (ONNX), plus an optional extra ONNX via `--onnx-output`.
2. No; the actor has `obs_normalization=False`. The deployment code must
   produce exactly the scaled observation vector, with no normalizer
   statistics to apply.
3. Slot order (`SLOT_0_1` + `SLOT_2_3` + `SLOT_4_5`) for observations;
   `ASIMOV_1_JOINT_NAMES` order for actions (and for the last-action block).
4. Any three of: joint order, sign conventions, missing scales, wrong
   frame/quaternion convention, wrong last action.
5. It verifies the deployment code produces the same actions as the simulator
   pipeline, catching ordering/scaling/export bugs with zero hardware risk.

</details>

---

*[← Race Day](09-training-and-debugging.md) · [Contents](README.md) · [Next: Your First Five Experiments →](11-experiments.md)*
