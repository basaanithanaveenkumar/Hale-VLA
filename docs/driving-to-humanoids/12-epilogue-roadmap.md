# Chapter 12 — Epilogue: The New Road

*[← Your First Five Experiments](11-experiments.md) · [Contents](README.md) · [Appendix A: The Rosetta Stone →](appendix-a-glossary.md)*

---

Think back to that night in the prologue: the lane-change dashboard on one
screen, the shoved-but-unbothered robot on the other. That video no longer
looks like magic to you.

You know the robot had 23 joints driven by PD loops with a random delay of up
to 25 ms. You know the policy was a three-layer MLP seeing 78 numbers at
50 Hz, including readings that were deliberately noisy and stale. You know it
was shoved every one to three seconds for a simulated half-year across 4,096
parallel copies of itself, with friction, gains, calibration and mass
distribution all randomized. You know eighteen reward terms and a
discriminator trained on 49 seconds of slow walking shaped *how* it moves. And
you know that the stumble-and-recover in that video is the `push_robot` event
and the disturbance-relaxed pose reward doing exactly what they were designed
to do.

That's the whole repository. Let's step back and see what it means for your
path.

## 12.1 What transferred, what's new

| Your AD skill | How it transfers |
|---------------|------------------|
| PyTorch, CUDA, batched tensor thinking | **Directly.** The whole sim is batched tensors. |
| Data pipelines, schema discipline | **Directly.** Observation ordering and scaling is the #1 deployment risk. |
| Model export (ONNX/TensorRT), on-target validation | **Directly.** Same pain, higher stakes. |
| Controls (PID, MPC, latency modeling) | **Directly.** PD actuators, gain scheduling, delay compensation. |
| Planning cost functions | **Strongly.** Reward design is cost design. |
| Imitation learning, DAgger, covariate shift | **Strongly.** On-policy RL is the closed-loop extreme; AMP is imitation of style. |
| GANs for sim-to-real imagery | **Strongly.** AMP is a GAN with an RL generator. |
| Scenario-based testing, ODD thinking | **Strongly, and rarer in robotics.** Robustness sweeps are a differentiator. |
| Perception (detection, segmentation, BEV) | **Later.** Comes back for perceptive locomotion and manipulation. |

| New for you | Where to learn it |
|-------------|-------------------|
| Rigid-body dynamics, kinematic trees, floating bases | *Modern Robotics* (Lynch & Park); Featherstone's *Rigid Body Dynamics Algorithms* |
| Legged-robot intuition (support polygon, CoM, angular momentum) | Tedrake's *Underactuated Robotics* (free online course notes) |
| RL theory (MDPs, policy gradients, value functions) | Sutton & Barto, *Reinforcement Learning: An Introduction* |
| Physics simulation quirks (contacts, solvers, stability) | Isaac Lab and MuJoCo docs; lots of hands-on time |
| Hardware realities (actuators, gearboxes, thermal limits, sysid) | Time with a real robot; talk to hardware engineers |

## 12.2 A twelve-week roadmap

Assuming evenings and weekends, alongside a day job.

**Weeks 1–2: Run it.** Install the repo. Run the quick test. Run a full
PPO and a full AMP training. Watch both in Play. Read this guide's Chapters 1–6
with the code open. *Milestone:* you can explain every line of
`velocity_env_cfg.py`.

**Weeks 3–4: Understand the learning.** Read the PPO and GAE papers. Read
`amp_ppo.py` end to end against RSL‑RL's `PPO`. Run the unit tests and write
one more (for example, that `MotionDataset` never returns a transition
crossing a clip boundary). *Milestone:* you can derive the PPO clipped
objective and the AMP reward on a whiteboard.

**Weeks 5–6: First experiments.** Experiments 1 and 2 from Chapter 11.
Two seeds each. Write a one-page report per experiment: hypothesis,
change, curves, video, conclusion. *Milestone:* you've changed the MDP and
predicted the outcome correctly at least once.

**Weeks 7–8: Robustness and evaluation.** Build the robustness-sweep script.
Try a sim-to-sim transfer to MuJoCo. *Milestone:* you have a quantitative
"operational envelope" for a policy.

**Weeks 9–10: Go deeper.** Experiment 3 (terrain + curriculum) or Experiment 4
(a new AMP clip). Read the AMP paper and the "AMP as reward substitute" paper.
*Milestone:* a policy with a capability the baseline doesn't have.

**Weeks 11–12: Share it.** Clean up one experiment, write it up as a blog
post, and share it with the community (the README points to Menlo's Discord
and its livestreams testing community policies on the real robot).
*Milestone:* public, reproducible work, which is exactly what hiring managers
in humanoid robotics look for.

## 12.3 The reading list

In roughly the order you'll want them:

**Foundations of the algorithms in this repo**
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Schulman et al., *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation* (2016).
- Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based
  Character Control* (SIGGRAPH 2021).
- Peng et al., *DeepMimic: Example-Guided Deep Reinforcement Learning of
  Physics-Based Character Skills* (SIGGRAPH 2018). AMP's predecessor.
- Mao et al., *Least Squares Generative Adversarial Networks* (2017). The
  discriminator loss.
- Mescheder et al., *Which Training Methods for GANs do actually Converge?*
  (2018). The gradient-penalty-on-real-data idea used here.

**Legged robots and sim-to-real**
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep
  Reinforcement Learning* (CoRL 2021). The origin of the GPU-parallel
  recipe that RSL‑RL and Isaac Lab's locomotion tasks descend from.
- Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots*
  (Science Robotics 2019). Actuator modeling for sim-to-real.
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks
  from Simulation to the Real World* (2017).
- Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*
  (2018). The critic-with-privileges idea.
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain*
  (Science Robotics 2020). Teacher–student training.
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (RSS 2021).
- Escontrela et al., *Adversarial Motion Priors Make Good Substitutes for
  Complex Reward Functions* (IROS 2022). AMP on real legged robots.
- Radosavovic et al., *Real-World Humanoid Locomotion with Reinforcement
  Learning* (Science Robotics 2024).

**A bridge from your AD past**
- Chen et al., *Learning by Cheating* (CoRL 2019). Privileged teachers in
  driving; compare it with this repo's asymmetric critic.

**Tools**
- The Isaac Lab documentation (especially the manager-based environment
  tutorials).
- The RSL‑RL source code (it's short and readable).
- The source code of the projects the README acknowledges: Isaac Lab,
  MuJoCo, `whole_body_tracking`, `beyondAMP` and `mjlab`.

## 12.4 Where you fit in the humanoid industry

The field needs people at every layer, and your AD background maps onto
several of them:

- **Locomotion / whole-body control RL engineer.** Exactly what this repo
  does. Your controls and cost-design experience is a head start.
- **Simulation and sim-to-real engineer.** Building worlds, sysid,
  randomization, evaluation. Your scenario-testing experience is valuable.
- **Robot learning infrastructure.** Massively parallel training, experiment
  tracking, evaluation pipelines. Your data-engine experience is directly
  relevant.
- **Perception for humanoids.** Depth, terrain mapping, object detection for
  manipulation. Your core AD expertise, in a new body.
- **Robot foundation models / VLAs.** Vision-language-action models that turn
  "go to the kitchen and pick up the cup" into commands. This is where it all
  connects.

## 12.5 The last connection

Look again at the command interface from Chapter 4: `twist = [vx, vy, ωz]`.
Three numbers. That's the entire API between "what the robot should do" and
"how the robot moves its 23 joints".

In AD, you learned that a clean interface between planning and control is what
lets each layer improve independently. The same is true here. A locomotion
policy like this one is a **low-level skill**. Above it sits whatever decides
*where* to go: a joystick today, a navigation stack tomorrow, and, increasingly,
a vision-language-action model that looks at the world, reads an instruction,
and emits commands.

You came from building the brains of cars. The humanoid world needs brains
at every level of that stack, and now you understand the level where the
robot meets the ground. That's the level everything else stands on.

Welcome to the new road. Mind the cobblestones.

---

*[← Your First Five Experiments](11-experiments.md) · [Contents](README.md) · [Appendix A: The Rosetta Stone →](appendix-a-glossary.md)*
