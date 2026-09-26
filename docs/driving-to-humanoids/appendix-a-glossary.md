# Appendix A — The Rosetta Stone: AD ↔ Humanoid Robotics Glossary

*[← Epilogue](12-epilogue-roadmap.md) · [Contents](README.md) · [Appendix B: File-by-File Reference →](appendix-b-file-reference.md)*

---

## Part 1: Concept map

| In automated driving you say… | In this repo it's… | Notes |
|-------------------------------|--------------------|-------|
| Ego vehicle | The robot / articulation | `scene.robot` |
| Vehicle model / dynamics params | URDF + `ArticulationCfg` | Chapter 1 |
| `base_link` | `pelvis_link` (the floating base) | |
| TF tree | Kinematic tree of links and joints | |
| Drive-by-wire interface | Joint position targets → PD actuators | Chapter 2 |
| Actuator latency model | `DelayedPDActuatorCfg` (0–5 physics steps) | Chapter 2 |
| Low-level controller (PID) | PD loop inside each actuator | Chapter 2 |
| Mission / route goal | Velocity command `twist = [vx, vy, ωz]` | Chapter 4 |
| Model input tensor spec | Observation group (`policy`, 78-d) | Chapter 4 |
| Ground-truth / privileged labels | Critic observation group (93-d) | Chapter 4 |
| IMU roll/pitch from gravity | `projected_gravity` | Chapter 4 |
| Planner cost function | Reward terms (with the sign flipped) | Chapter 5 |
| Jerk minimization | `action_rate_l2` | Chapter 5 |
| Gain scheduling by speed | `variable_posture` σ tables | Chapter 5 |
| Data augmentation | Domain randomization (events) | Chapter 6 |
| Scenario generator knobs | `EventCfg` terms | Chapter 6 |
| Collision / off-road = failure | `fell_over` termination | Chapter 6 |
| Simulation timeout | `time_out` termination (bootstrapped) | Chapter 6 |
| Training dataset | Rollouts generated on the fly | Chapter 7 |
| Closed-loop training / DAgger | On-policy RL | Chapter 7 |
| Residual over a baseline model | Advantage over the value function | Chapter 7 |
| GAN for sim-to-real images | AMP discriminator | Chapter 8 |
| Config/version tracking | `logs/.../params/env.yaml`, `agent.yaml` | Chapter 9 |
| ONNX/TensorRT export | `play.py` → `exported/policy.onnx` | Chapter 10 |
| Closed-course testing with safety driver | Gantry + e-stop + soft start | Chapter 10 |
| Cross-city / cross-sensor validation | Sim-to-sim (e.g., MuJoCo) | Chapter 10 |
| ODD analysis | Robustness sweeps | Chapter 11 |

## Part 2: Terms, A to Z

**Action scale.** Multiplier from raw network output to joint-target offset.
Here 0.25 rad per unit.

**Advantage.** How much better an action turned out than the critic expected.
The "label" that weights PPO's policy update.

**AMP (Adversarial Motion Priors).** A technique that adds a learned style
reward from a discriminator trained to distinguish reference motion from
policy motion.

**Anchor body.** The body used as the reference frame for body-relative
features in the motion dataset; here `pelvis_link`.

**Armature.** Reflected rotor inertia of a motor through its gearbox, added
to the joint's inertia in simulation.

**Articulation.** A tree of rigid bodies connected by joints; Isaac Lab's
representation of a robot.

**Asymmetric actor-critic.** The critic gets more (privileged) information than
the actor; only the actor is deployed.

**Bootstrapping.** Using the value function's estimate in place of the
unknown future rewards, for example at a time-out.

**Command.** The task input to the policy (here, a desired body velocity).

**Contact sensor.** Isaac Lab sensor reporting contact forces, air time and
contact time for bodies.

**Critic / value function.** Network predicting expected discounted future
reward from a state.

**Curriculum.** Gradually increasing task difficulty (here, terrain difficulty
is available but `curriculum=False`).

**Damping (Kd).** The PD controller's velocity gain.

**Decimation.** The number of physics steps per policy step (here 4).

**Default joint position.** The standing pose; the reference point for
actions and joint observations.

**Discriminator.** In AMP, a network scoring whether a transition looks like
the reference motion.

**Domain randomization.** Randomizing simulation parameters during training so
the policy is robust to the real world's unknown parameters.

**DoF (degree of freedom).** An independent coordinate of motion; Asimov‑1 has
23 actuated DoF plus 6 floating-base DoF.

**Effort limit.** Maximum joint torque (N·m).

**Entropy bonus.** A reward for keeping the policy's action distribution wide,
to preserve exploration.

**Episode.** One run from reset to termination (up to 20 s here).

**Event term.** An Isaac Lab function that modifies the sim at startup, reset
or on an interval.

**Floating base.** A robot base that isn't fixed to the world; its 6 DoF are
unactuated.

**GAE (Generalized Advantage Estimation).** An exponentially weighted
multi-step advantage estimator controlled by λ.

**Gradient penalty.** A regularizer on the discriminator's input gradients
that keeps its output landscape smooth.

**Heading command.** A mode where turning rate is computed from a heading
error instead of sampled directly.

**Isaac Lab.** NVIDIA's robot-learning framework on top of Isaac Sim.

**Isaac Sim.** NVIDIA's robotics simulator (Omniverse, USD, PhysX).

**KL divergence (in PPO).** A measure of how much the policy changed in one
update; drives the adaptive learning rate.

**LSGAN.** Least-squares GAN; the discriminator regresses to +1/−1 instead of
using cross-entropy.

**Manager-based environment.** Isaac Lab's config-driven environment design,
where observations, rewards, events etc. are each handled by a manager.

**MDP (Markov Decision Process).** The formal model of an RL problem: states,
actions, transitions, rewards.

**Motion clip.** Recorded reference motion; here `policy_delay_walk_slow.npz`.

**Observation group.** A named set of observation terms concatenated into one
vector (`policy`, `critic`, `amp`).

**On-policy.** RL that only learns from data collected by the current policy.

**PD controller.** Torque = Kp·(target − position) − Kd·velocity.

**PhysX.** NVIDIA's physics engine, run on the GPU here.

**Play config.** An environment variant with randomization disabled, for
visualization.

**Policy / actor.** The network mapping observations to actions; the thing you
deploy.

**PPO (Proximal Policy Optimization).** A policy-gradient algorithm that clips
the probability ratio to limit update size.

**Prim path.** The path of an object in the USD scene graph.

**Projected gravity.** The gravity direction expressed in the robot's base
frame; encodes roll and pitch.

**Replay buffer.** Storage for past policy transitions; used here only for
the discriminator.

**Rollout.** A sequence of steps collected by running the policy.

**RSL‑RL.** ETH Zurich Robotic Systems Lab's RL library; provides PPO and the
runners.

**Self-collision.** Contact between two links of the same robot.

**Sim-to-real.** Transferring a policy trained in simulation to real hardware.

**Sim-to-sim.** Testing a policy in a different simulator as a proxy for
sim-to-real.

**Soft joint limits.** A fraction (here 90%) of the hard joint range, used for
penalties.

**Stiffness (Kp).** The PD controller's position gain.

**Support polygon.** The convex hull of ground contacts; the CoM projection
must stay near it for static balance.

**Sysid (system identification).** Measuring a real system's parameters
(friction, inertia, delays) to calibrate the simulator.

**Termination.** A condition that ends an episode.

**Transition (AMP).** A pair of consecutive AMP observations `(s_t, s_{t+1})`.

**Twist.** A linear + angular velocity; the name of the command here.

**URDF.** Unified Robot Description Format; XML describing links, joints,
inertias and geometry.

**USD.** Universal Scene Description; Isaac Sim's scene format.

---

*[← Epilogue](12-epilogue-roadmap.md) · [Contents](README.md) · [Appendix B: File-by-File Reference →](appendix-b-file-reference.md)*
