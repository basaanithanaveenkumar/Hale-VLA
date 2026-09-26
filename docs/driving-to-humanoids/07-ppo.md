# Chapter 7 — PPO for People Who Love Supervised Learning

*[← Lying to Your Robot on Purpose](06-randomization-and-sim2real.md) · [Contents](README.md) · [Next: Teaching Style with a GAN: AMP →](08-amp.md)*

---

You've built a world, a body, senses, and a scoring system. Now you need a
way to turn scores into a better network. That's the algorithm, and in this
repo it's **PPO (Proximal Policy Optimization)**, as implemented by RSL‑RL and
extended by `isaac_asimov/algorithms/amp_ppo.py`.

If you've only ever trained with supervised losses, RL algorithms can feel
like black magic. They aren't. This chapter builds PPO from ideas you already
use every day, then maps every hyperparameter in
`agents/rsl_rl_ppo_cfg.py` to what it does.

## 7.1 The loop that replaces your dataloader

In supervised learning, your training loop looks like:

```
for batch in dataloader:            # data exists before training
    loss = criterion(model(x), y)
    loss.backward(); optimizer.step()
```

In on-policy RL, the data doesn't exist until the model creates it:

```
for iteration in range(max_iterations):
    # 1. COLLECT: run the current policy in 4096 envs for 24 steps
    rollout = []
    for step in range(24):
        actions = policy.sample(obs)                # stochastic!
        obs, rewards, dones = env.step(actions)
        rollout.append(...)
    # 2. SCORE: figure out which actions were better than expected
    advantages, returns = compute_gae(rollout, critic)
    # 3. LEARN: a few epochs of minibatch SGD on this fresh data
    for epoch in range(5):
        for minibatch in split(rollout, 4):
            loss = ppo_loss(minibatch)
            loss.backward(); optimizer.step()
    # 4. THROW AWAY the data. It's stale now: the policy has changed.
```

With this repo's settings:

| Quantity | Value | Where |
|----------|-------|-------|
| Environments | 4,096 | `num_envs` |
| Steps per env per iteration | 24 | `num_steps_per_env` |
| Transitions per iteration | 4,096 × 24 = **98,304** | |
| Simulated time per iteration | 24 × 0.02 s = 0.48 s per robot | |
| Learning epochs | 5 | `num_learning_epochs` |
| Minibatches per epoch | 4 | `num_mini_batches` |
| Minibatch size | 98,304 / 4 = **24,576** | |
| Gradient steps per iteration | 5 × 4 = **20** | |
| Max iterations | 10,000 | `max_iterations` |
| Total transitions (full run) | ≈ **983 million** | |
| Checkpoint every | 500 iterations | `save_interval` |

Note that 24 steps is only 0.48 seconds per robot per iteration, much less
than an episode. Episodes span many iterations. The rollout storage handles
this naturally, and the value function bootstraps across the cut.

> 🚗 **Driving Déjà Vu**
> On-policy RL is like **closed-loop training with DAgger**, taken to the
> extreme. In AD imitation learning, you learned that open-loop training on
> logged data causes compounding errors: the model never sees the
> consequences of its own mistakes. On-policy RL has no logged data at all.
> Every sample is a consequence of the current policy's own decisions. That's
> expensive (data expires every iteration), but it's why the policy
> learns to recover from its own errors.

## 7.2 The actor: a network that outputs a distribution

```python
actor = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128],
    activation="elu",
    obs_normalization=False,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
)
```

The actor is an MLP: 78 → 512 → 256 → 128 → 23, with ELU activations. Its
output is the **mean** of a Gaussian over actions. The **standard
deviation** is a separate learnable parameter per action dimension (not a
network output), initialized to 1.0. With `std_type="scalar"` the std is
stored directly (rather than as a log-std).

During training, actions are **sampled**: `a ~ N(μ(o), σ²)`. That randomness
is how the robot explores. During deployment, you use just the mean `μ(o)`.
As training progresses, σ shrinks; the policy becomes more confident.

Why ELU? It's smooth (no kink at zero like ReLU), which gives smoother
action outputs as a function of the observations. That's the convention across
RSL‑RL legged-robot configs.

Why `obs_normalization=False`? Many RSL‑RL setups normalize observations with a
running mean/std. Here, the observation terms are already hand-scaled
(`scale=0.25` on angular velocity, `0.1` on joint velocity) to a sensible range.
It also simplifies deployment: no normalizer statistics to export and apply.

## 7.3 The critic: learning to predict the future

```python
critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
```

Same architecture, but it takes the 93-d **critic** observation (Chapter 4)
and outputs one number: `V(s)`, the expected discounted future reward from this
state. The mapping from groups to networks is:

```python
obs_groups = {"actor": ["policy"], "critic": ["critic"]}
```

The critic is trained by plain **regression**, just like any value
prediction you've done: MSE between `V(s)` and the observed return.

## 7.4 Advantages: the "label" in RL

Here's the central idea. After a rollout, for each `(observation, action)` pair,
we want to know: **was that action better or worse than what the policy
usually does in that situation?** That's the **advantage**:

```
A(s, a) = [what actually happened after taking a] − [what we expected, V(s)]
```

Positive advantage: that action was a pleasant surprise, so do it more.
Negative: do it less.

The simplest estimate uses one step:

```
δ_t = r_t + γ · V(s_{t+1}) − V(s_t)          ← the "TD error"
```

PPO uses **Generalized Advantage Estimation (GAE)**, an exponentially weighted
sum of these one-step errors:

```
A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...
```

Two knobs:

- **`gamma = 0.99`**: the discount. Rewards `k` steps in the future count
  `0.99^k`. The effective horizon is about `1 / (1 − γ)` = 100 steps = **2
  seconds** at 50 Hz. The robot cares about the next couple of strides, not
  the next minute.
- **`lam = 0.95`**: the bias–variance knob. λ = 0 trusts the critic
  completely (low variance, biased if the critic is wrong); λ = 1 trusts only
  real rewards (unbiased, high variance). 0.95 leans toward real rewards.

The "returns" target for the critic is `A_t + V(s_t)`.

With `normalize_advantage_per_mini_batch=False`, RSL‑RL normalizes advantages
(zero mean, unit std) once over the whole rollout rather than per minibatch.

> 🚗 **Driving Déjà Vu**
> The advantage is the RL version of a **residual**. In prediction models you
> often predict a residual over a baseline (say, a constant-velocity
> model). The critic is the baseline: "what normally happens from here". The
> advantage is the residual: "how much better did *this* action do?" Using a
> baseline doesn't change what's optimal, but it massively reduces variance,
> exactly as residual prediction does.

## 7.5 The PPO loss: weighted imitation with a leash

Now the loss. Here's the core, straight from
`isaac_asimov/algorithms/amp_ppo.py` (which reproduces RSL‑RL's PPO update
and adds AMP on top):

```python
ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
surrogate = -torch.squeeze(batch.advantages) * ratio
surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

Let's build intuition from supervised learning.

**Step 1: behavior cloning.** If you had a perfect expert, you'd maximize
`log π(a_expert | o)`. That's cross-entropy/MSE-style imitation.

**Step 2: advantage-weighted imitation.** We don't have an expert, but we have
our own actions and their advantages. So: imitate your own actions,
*weighted by how good they turned out*. Good actions (A > 0) get their
probability pushed up; bad ones (A < 0) get pushed down. That's the
**policy gradient**: `∇ E[A · log π(a|o)]`.

**Step 3: the ratio.** We reuse each rollout for 20 gradient steps. After the
first step, the policy has changed, and the data came from the *old* policy.
The ratio `π_new(a|o) / π_old(a|o)` corrects for that (importance sampling).
The objective becomes `E[ratio · A]`.

**Step 4: the leash (the "proximal" in PPO).** If you optimize `ratio · A`
freely, the policy can change drastically on 98k samples and destroy itself.
So PPO **clips** the ratio to `[1 − 0.2, 1 + 0.2]` and takes the pessimistic
(max-loss) version. Once an action's probability has moved 20% in the helpful
direction, the gradient stops pushing it further. That's a cheap, effective
trust region.

> 🔧 **Under the Hood: reading the `max`**
> The code negates everything (it minimizes a loss rather than maximizing an
> objective), so the pessimistic choice is `torch.max` of two negatives.
> For A > 0: the loss is `−A · min(ratio, 1.2)`. Once ratio > 1.2 there's no
> gradient. For A < 0: the loss is `−A · max(ratio, 0.8)` = `|A| · max(ratio, 0.8)`.
> Once ratio < 0.8 there's no gradient. In both cases: *don't move more than
> 20% per update*.

### The value loss

```python
if self.use_clipped_value_loss:
    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
    value_losses = (values - batch.returns).pow(2)
    value_losses_clipped = (value_clipped - batch.returns).pow(2)
    value_loss = torch.max(value_losses, value_losses_clipped).mean()
```

MSE against the returns, with the same kind of clipping so the critic doesn't
jump too far per update either (`use_clipped_value_loss=True`).

### The entropy bonus

```python
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
```

`entropy_coef = 0.005` gives a small reward for keeping the action
distribution wide. It fights premature convergence: without it, σ can
collapse early, exploration stops, and the policy gets stuck in a mediocre
gait.

`value_loss_coef = 1.0`: the actor and critic losses are summed and trained
with one optimizer.

## 7.6 The adaptive learning rate

```python
schedule="adaptive", learning_rate=1.0e-3, desired_kl=0.01,
```

Instead of a fixed schedule, the learning rate reacts to how much the policy
changes per update, measured by the KL divergence between old and new action
distributions:

```python
if kl_mean > self.desired_kl * 2.0:            # changing too fast (KL > 0.02)
    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:   # too slow (KL < 0.005)
    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
for param_group in self.optimizer.param_groups:
    param_group["lr"] = self.learning_rate
```

It's a bang-bang controller on the KL, keeping it near 0.01. In multi-GPU
runs, the KL is averaged across GPUs (`all_reduce`), rank 0 decides, and the
new learning rate is broadcast so every GPU stays in sync.

> 🚗 **Driving Déjà Vu**
> A feedback controller for the optimizer! Setpoint: KL = 0.01. Measurement:
> KL of the latest minibatch. Actuator: the learning rate. You've tuned
> controllers like this for speed tracking. Here the "plant" is the neural
> network.

Watch out for one subtlety in the AMP variant: the loop sets the learning
rate on **all** optimizer parameter groups, and (as we'll see in Chapter 8)
the discriminator's parameters live in the same optimizer. So the
discriminator's learning rate is also driven by the *policy's* KL.

## 7.7 Gradient clipping

```python
max_grad_norm=1.0
...
def _clip_actor_critic_gradients(self):
    return nn.utils.clip_grad_norm_(
        chain(self.actor.parameters(), self.critic.parameters()),
        self.max_grad_norm,
    )
```

The combined actor + critic gradient norm is clipped to 1.0, as one vector.
The test `test_amp_ppo_clips_one_combined_actor_critic_gradient_norm` checks
exactly this: with all-ones gradients over 9 parameters (3·2 + 3·1), the
original norm is 3.0 and the clipped norm is ≤ 1.0. The discriminator is
deliberately *not* included in this clip.

## 7.8 The full hyperparameter card

```python
@configclass
class Asimov1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "asimov1_velocity"
    run_name = "ppo"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"))
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        optimizer="adam",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,             # no curiosity (Random Network Distillation)
        symmetry_cfg=None,        # no left/right symmetry augmentation
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

| Param | Value | If you increase it… |
|-------|-------|---------------------|
| `num_steps_per_env` | 24 | Longer rollouts per update; better advantage estimates, fewer updates per sample |
| `clip_param` | 0.2 | Bigger steps per update; faster but riskier |
| `entropy_coef` | 0.005 | More exploration; slower to converge, may stay jittery |
| `num_learning_epochs` | 5 | More reuse of each rollout; more sample-efficient until it overfits the batch |
| `num_mini_batches` | 4 | Smaller minibatches, more gradient steps |
| `learning_rate` | 1e‑3 | Only the *initial* value; the adaptive schedule takes over |
| `gamma` | 0.99 | Longer horizon; can help with long-term balance but raises variance |
| `lam` | 0.95 | Less bias, more variance in advantages |
| `desired_kl` | 0.01 | Larger policy changes per iteration |
| `max_grad_norm` | 1.0 | Allows larger gradient steps |

`rnd_cfg=None` and `symmetry_cfg=None` are features RSL‑RL supports but this
repo doesn't use. The `AMPPPO` class carries the code paths for both (it copies
RSL‑RL's update loop), but they're inactive. Symmetry augmentation, in
particular, is a common next step for humanoid gaits (Chapter 11).

## 7.9 Why PPO, and why it works so well here

PPO is not the most sample-efficient RL algorithm. Off-policy methods like SAC
reuse old data and need far fewer environment steps. But in massively
parallel simulation, **samples are nearly free** and **wall-clock time is what
matters**. PPO's simplicity, its stability, and the fact that it
parallelizes trivially across thousands of environments make it the
workhorse of legged-robot learning. With 4,096 environments, one iteration
collects about 98k transitions in the time an off-policy method would spend
carefully reusing a few thousand.

## 🏁 Pit Stop

1. How many transitions does one PPO iteration collect, and how many gradient
   steps does it take on them?
2. What does the advantage measure, and what plays the role of the "baseline"?
3. What's the effective planning horizon implied by `gamma = 0.99` at 50 Hz?
4. What stops PPO from changing the policy too much in one update? Name two
   mechanisms.
5. Why can the critic use privileged observations but the actor can't?
6. In the AMP variant, what else does the adaptive learning rate affect?

<details>
<summary>Answers</summary>

1. 98,304 transitions; 20 gradient steps (5 epochs × 4 minibatches).
2. How much better an action turned out than expected. The critic's value
   estimate `V(s)` is the baseline.
3. About 100 steps, which is 2 seconds.
4. Ratio clipping (`clip_param = 0.2`) and the adaptive learning rate that
   targets KL = 0.01. (Gradient-norm clipping also helps.)
5. The critic only exists during training; the actor is deployed on the real
   robot, where privileged information isn't available.
6. The discriminator's learning rate, because its parameters live in the same
   optimizer.

</details>

---

*[← Lying to Your Robot on Purpose](06-randomization-and-sim2real.md) · [Contents](README.md) · [Next: Teaching Style with a GAN: AMP →](08-amp.md)*
