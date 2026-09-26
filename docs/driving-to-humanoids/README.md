# From Lane Lines to Leg Swings

### A driving engineer's field guide to humanoid locomotion, told through `menloresearch/isaac_asimov`

> *"The car never had to worry about falling over. That's the first thing you'll miss."*

This is a long-form, story-driven walk through the
[`menloresearch/isaac_asimov`](https://github.com/menloresearch/isaac_asimov)
repository: the open-source training code that teaches Menlo Research's
**Asimov‑1** humanoid robot to walk.

It is written for one specific reader: **you**, a deep learning engineer who
has spent years on automated driving (perception, prediction, planning, maybe
some controls) and who now wants to work on humanoid robots. You already know
PyTorch, backprop, CUDA OOM errors and the pain of a bad dataset. You may never
have trained a policy with reinforcement learning, tuned a PD gain or wondered
why a robot's ankle has an "armature".

Every chapter anchors a robotics idea to something you already know from
driving, then shows the real code in the repo, line by line.

---

## How to read this book

- **Read it in order the first time.** Each chapter builds on the last, like a
  road trip with stops.
- **Keep the repo open beside you.** Every code excerpt names its file, and
  the paths are relative to the `isaac_asimov` repository root.
- **Watch for the recurring boxes:**
  - 🚗 **Driving Déjà Vu**: the AD concept that maps to what you're reading.
  - 🔧 **Under the Hood**: a deeper dive you can skip on a first read.
  - ⚠️ **Pothole**: a mistake that costs real people real days.
  - 🏁 **Pit Stop**: a few questions to check you understood before moving on.

## The route

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 0 | [Prologue: The Last Lane Change](00-prologue.md) | Why this move is harder *and* easier than you think |
| 1 | [Meet Asimov‑1](01-meet-asimov.md) | The body: 23 joints, 26 links, one URDF |
| 2 | [Muscles Made of Math](02-actuators-and-pd.md) | PD control, actuators, delay, and what the network actually outputs |
| 3 | [A Thousand Worlds per Second](03-simulation-and-repo-tour.md) | Isaac Sim, Isaac Lab, installation, and the repo's layout |
| 4 | [The MDP Is a Config File](04-the-mdp-as-config.md) | Scene, commands, actions, observations, and the asymmetric critic |
| 5 | [Speaking Robot: Rewards](05-rewards.md) | All 18 reward terms, their math, and why each one exists |
| 6 | [Lying to Your Robot on Purpose](06-randomization-and-sim2real.md) | Domain randomization, resets, terminations, and the sim-to-real gap |
| 7 | [PPO for People Who Love Supervised Learning](07-ppo.md) | Policy gradients, GAE, clipping, and every PPO hyperparameter here |
| 8 | [Teaching Style with a GAN: AMP](08-amp.md) | Adversarial motion priors, the discriminator, and the replay buffer |
| 9 | [Race Day: Training and Debugging](09-training-and-debugging.md) | Running it, reading the curves, and fixing what breaks |
| 10 | [From Checkpoint to Concrete Floor](10-play-export-deploy.md) | Play, ONNX export, and what deployment must reproduce exactly |
| 11 | [Your First Five Experiments](11-experiments.md) | Hands-on changes, from easy to ambitious |
| 12 | [Epilogue: The New Road](12-epilogue-roadmap.md) | A learning roadmap, a reading list, and a career map |
| A | [Appendix A: The Rosetta Stone](appendix-a-glossary.md) | Glossary that maps AD terms to robotics terms |
| B | [Appendix B: File-by-File Reference](appendix-b-file-reference.md) | Every file in the repo and what it does |
| C | [Appendix C: The Numbers Card](appendix-c-numbers.md) | Every important constant on one page |

## A note on scope and honesty

The repository is small: about 3,300 lines of Python across two dozen files, plus two
git submodules (Isaac Lab and the robot's model). That smallness is a gift,
because you can understand *all* of it. Where this guide describes behavior
that lives in Isaac Lab or RSL‑RL rather than in `isaac_asimov` itself, it
says so. Where it infers intent the code does not state (for example, why
joints are grouped into "slots"), it marks the claim as an inference.

The guide was written against commit `bdf28f5` of `isaac_asimov` (the
open-source cleanup merge, September 2026), with Isaac Lab pinned at
`b0542fe` and RSL‑RL `5.0.1`.

Buckle up. Or rather, stand up. Robots don't get seats.
