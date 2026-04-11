# ADR 004 — RL optimizer: contextual bandit with tabular Q-learning

**Status**: Accepted  
**Decided**: 2026-04-10 (implemented across PRs 1–4 / `memory_guard/bandit_state.py`,
             `memory_guard/reward.py`, `memory_guard/bandit.py`,
             `memory_guard/guard.py`)  
**Applies to**: `memory_guard.bandit.BanditPolicy`,
               `memory_guard.bandit_state.{StateKey, ConfigAction, DeviceFingerprint,
               ModelFingerprint}`, `memory_guard.reward.{RewardSignal, compute_reward}`,
               `memory_guard.guard.MemoryGuard.{preflight, preflight_inference,
               record_result}`

---

## Context

v0.3 uses a formula-based estimator plus a binary-search downgrade loop to
find a memory-safe training configuration.  The formula is deliberately
conservative: it cannot model framework-specific behaviors (operator fusion,
activation recomputation schedules, CUDA graph amortization) and so often
under-utilizes available memory.

The calibration layer introduced alongside the formula records `(estimated,
actual)` pairs and applies a median correction factor per backend.  This
improves accuracy over time but is still a scalar multiplier — it cannot
learn *which* `(batch_size, lora_rank)` combination best saturates the
device without OOM.

v0.4 adds a learned optimizer that picks the training configuration from
experience rather than from the formula alone.  Several design axes had to be
decided:

1. Full sequential RL vs. contextual bandit  
2. Neural policy vs. tabular Q-learning  
3. Shared policy vs. per-device isolation  
4. Decaying epsilon to zero vs. permanent exploration floor  

---

## Decision 1 — Contextual bandit, not full sequential RL

**Chosen**: contextual bandit (single-step).

A training run is a single episode: we choose a `(batch_size, lora_rank,
seq_length)` tuple *before* the run starts, observe the outcome (success /
OOM, peak memory) *after* the run ends, and update the policy.  There is no
sequence of decisions within a run that a sequential RL agent could improve.

Full sequential RL (e.g., PPO, SAC) models a *trajectory* of decisions
`s₀ → a₀ → s₁ → a₁ → … → sₙ`.  Applying it here would mean treating each
gradient step within a training run as a separate state-action pair, which
requires a reward signal at every step, an online policy that runs inside the
training loop, and a replay buffer that accumulates within-run experience.
None of this is justified: the gradient-step reward signal is undefined (we
only know the outcome after the full run), and adding RL overhead inside the
training loop would slow training on hardware that is already memory-bound.

The contextual bandit formulation is exact for our setting:
- Context (state): device fingerprint × model fingerprint
- Action: one `ConfigAction` chosen from a discrete set
- Reward: one scalar, observed once per run, after the run ends

---

## Decision 2 — Tabular Q-learning, not a neural policy

**Chosen**: tabular `Dict[StateKey, Dict[ConfigAction, float]]`.

**Zero-dependency constraint**: `memory_guard` must be installable with no
ML framework present.  Adding `torch`, `jax`, or even `numpy` as a runtime
dependency would break the use case where users install the guard on a fresh
machine *before* installing PyTorch or MLX.  A Python `dict` has no external
dependencies.

**Action space is small and discrete**: the candidate set is the Cartesian
product of six batch sizes × six LoRA ranks × a fixed sequence length = 36
training actions, and eleven `max_num_seqs` values for inference.  Neural
function approximation is unnecessary at this scale; tabular Q-values converge
faster and are easier to inspect and debug.

**Interpretability**: a human can read the JSON policy file and understand
which config the policy has learned to prefer.  This is important for
debugging unexpected behavior (e.g., policy favoring batch_size=1 after a
series of OOMs) and for regression analysis.

**Q-update rule** (incremental exponential moving average toward the reward):

    Q[s][a] += alpha * (reward − Q[s][a])

This is equivalent to a 1-step temporal-difference update with no bootstrapping
(the episode ends immediately after the action), which is correct for the
bandit setting.

---

## Decision 3 — Per-device isolation (no cross-device generalization)

**Chosen**: `StateKey = (DeviceFingerprint, ModelFingerprint)` — separate
Q-table row per device tier × backend × platform × model class × bits.

Memory behavior is hardware-specific.  A batch size that saturates an A100-80GB
GPU without OOM may crash an RTX 3090 24GB.  Allowing policy entries from one
device to influence another would corrupt the Q-table and could cause OOM on
devices with less memory.

**Bucketing** is used instead of exact values so that Q-table entries generalize
across sessions on the same physical machine.  Free memory fluctuates between
runs (background processes, CUDA context overhead); requiring exact equality
would cause a cold start on every launch.  Five memory tiers and five parameter
classes give enough resolution to distinguish qualitatively different hardware
configurations while being coarse enough to accumulate experience within each
bucket.

**Future work**: transfer learning across device tiers (e.g., warm-start the
48–80 GB policy from the 24–48 GB policy, scaled by the memory ratio) is a
future concern.  It is not implemented in v0.4 because the action space
differs enough across tiers that naive transfer would introduce bias.

---

## Decision 4 — Epsilon never reaches zero (permanent exploration floor)

**Chosen**: `epsilon = max(epsilon_floor, epsilon * epsilon_decay)` with
`epsilon_floor = 0.05`.

The training landscape changes continuously: new model architectures arrive
with different memory footprints, frameworks release updates that change
memory allocation patterns, and users switch between jobs of varying size.
A policy that stops exploring will eventually become stale — it will keep
recommending a configuration that was optimal six months ago but is no longer
appropriate.

The 5 % floor means that roughly 1 in 20 runs explores via the existing
binary-search downgrade path.  This is cheap (the binary search is already
running in v0.3 and takes O(log(max_num_seqs)) estimator calls) and provides
a continuous stream of fresh signal.

Setting `epsilon_floor = 0.0` would cause the policy to stop exploring after
enough updates, making it impossible to recover from a run of bad luck (e.g.,
a sequence of OOMs that drove the policy toward overly-conservative configs).

---

## Alternatives rejected

### Alternative: Bayesian optimization (e.g., Gaussian Process)

Would require `scipy` or `scikit-learn` at runtime — violates the
zero-dependency constraint.  Also better suited to continuous action spaces;
the discrete grid here is handled more efficiently by tabular Q-learning.

### Alternative: multi-armed bandit without context (ignoring device/model)

Ignoring the state entirely means the same action (e.g., batch_size=8) is
recommended regardless of whether the current device is an A100-80GB or an
M2 8GB.  This would cause frequent OOMs on smaller devices.

### Alternative: online gradient descent / linear regression

Would require the policy to store gradients and maintain a weight vector per
feature.  More complex to implement correctly, harder to serialize, and
provides no interpretability advantage over tabular Q-values for this problem.

---

## Consequences

**Positive**:
- No new runtime dependencies; `memory_guard` remains installable with `pip
  install ml-memguard` on any machine.
- Cold-start behavior is identical to v0.3: `select_action` returns `None`
  on a fresh install, and the binary-search path runs as before.
- The policy file is human-readable JSON and can be inspected, edited, or
  deleted without code changes.
- Per-device isolation prevents cross-hardware contamination.
- The 5 % exploration floor ensures the policy never locks in permanently.

**Negative / tradeoffs**:
- The policy does not generalize across device tiers.  A user who upgrades
  from an RTX 3090 to an A100 starts from cold on the new device.
- The Q-table grows unboundedly if a user runs on many different device/model
  combinations.  In practice the table is small (tens of KB) because the
  action space is bounded and most users have one or two devices.
- Tabular Q-learning requires many runs to converge compared to a neural
  policy with function approximation.  The calibration store mitigates this
  by ensuring the formula-based fallback remains accurate while the policy
  accumulates experience.
