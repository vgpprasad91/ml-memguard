# RL Optimizer — Reference Guide (v0.4)

`memory_guard` v0.4 adds a contextual bandit that learns which training
configuration works best on your specific device and model over time.  On a
fresh install it behaves exactly like v0.3 — the binary-search downgrade path
runs unchanged until the policy has gathered enough experience to make
confident recommendations.

---

## Quick overview

```
┌───────────────────────────────────────────┐
│           MemoryGuard.preflight()         │
│                                           │
│  1. Build StateKey (device + model info)  │
│  2. Ask BanditPolicy for best action      │
│     ├─ Cold start / explore → None        │
│     │   └─ Run binary-search (v0.3 path)  │
│     └─ Exploit → ConfigAction             │
│         ├─ Validate with estimator        │
│         │   ├─ Fits → return SafeConfig   │
│         │   └─ Doesn't fit → binary search│
│  3. Record _last_action + _last_state_key │
└───────────────────────────────────────────┘

        ↓  after your training run ends  ↓

┌───────────────────────────────────────────┐
│        MemoryGuard.record_result()        │
│                                           │
│  1. Measure actual peak memory            │
│  2. compute_reward(estimated, actual,     │
│       budget, oom_occurred)               │
│  3. policy.update(state_key, action,      │
│       reward.combined)                    │
│  4. policy.save()  ← atomic JSON write   │
└───────────────────────────────────────────┘
```

---

## State, action, and reward — plain language

### State (what the policy observes)

The policy groups your setup into a `StateKey` made of two fingerprints:

**DeviceFingerprint** — describes your hardware:
| Field | How it's computed | Example values |
|---|---|---|
| `memory_tier` | available RAM bucketed into 5 tiers | `sub_8gb`, `8_24gb`, `24_48gb`, `48_80gb`, `80plus_gb` |
| `backend` | auto-detected framework | `cuda`, `apple_silicon`, `cpu`, `rocm` |
| `os_platform` | `sys.platform` | `darwin`, `linux`, `win32` |

**ModelFingerprint** — describes your model:
| Field | How it's computed | Example values |
|---|---|---|
| `param_class` | parameter count bucketed into 5 classes | `sub_1b`, `1_7b`, `7_13b`, `13_35b`, `35plus_b` |
| `bits` | quantisation width, snapped to `{4, 8, 16, 32}` | `4`, `8`, `16` |

Bucketing means the policy generalises across sessions.  If you have 39.2 GB
free today and 38.7 GB tomorrow (because a background process is running),
both land in the same `24_48gb` tier and reuse the same Q-table row.

### Action (what the policy recommends)

For training, the policy chooses from a grid of `(batch_size, lora_rank)`
pairs at your requested `seq_length`:
- Batch sizes: 1, 2, 4, 8, 16, 32
- LoRA ranks: 0, 4, 8, 16, 32, 64

For inference, the policy chooses a `max_num_seqs` value from:
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 (capped at your requested max).

Every candidate action is validated by the formula-based estimator before
being returned.  The estimator is the safety net — if the policy recommends
something that the estimator says won't fit, the binary-search path runs as a
fallback.

### Reward (what the policy learns from)

After each run you call `guard.record_result(actual_peak_mb=..., oom_occurred=...)`.
The reward is:

```
combined = 0.6 × outcome + 0.4 × efficiency_bonus
```

| Component | Value |
|---|---|
| `outcome` | +1.0 on success, -1.0 on OOM |
| `efficiency_bonus` | `clamp(1 − |estimated − actual| / budget, 0, 1)` |
| `combined` | range [-0.6, 1.0] |

A run that succeeded and whose memory estimate was accurate gets a reward
close to 1.0.  A run that OOMed gets -0.6 regardless of estimate accuracy.
A clean run with a wildly inaccurate estimate (large `|estimated − actual|`)
gets `0.6 × 1.0 + 0.4 × 0.0 = 0.6`.

The Q-update rule:
```
Q[state][action] += alpha × (reward − Q[state][action])
```
This is an exponential moving average toward the reward.  With `alpha=0.1`
(the default), the Q-value decays toward a new reward over ~20 updates.

---

## Cold-start guarantee

On a fresh install — or on a new device the policy has never seen — the
Q-table has no entry for the current state.  `policy.select_action()` returns
`None`, and `preflight()` / `preflight_inference()` run the v0.3 binary-search
path exactly as before.  **No configuration change is required; behavior on a
fresh install is identical to v0.3.**

After a handful of runs the policy starts accumulating Q-values for the
current (device, model) pair.  With `epsilon=0.9` (the default), 90 % of
early runs still explore (fall back to binary search) while the policy learns.
As `epsilon` decays toward the 5 % floor, the policy gradually takes over.

---

## Disabling the bandit

To reproduce exact v0.3 behaviour:

```python
guard = MemoryGuard.auto(enable_bandit=False)
# or
guard = MemoryGuard(enable_bandit=False)
```

When `enable_bandit=False`, no policy is loaded, `_policy` is `None`, and
`record_result()` still records calibration data but skips the Q-table update.

---

## Inspecting the learned Q-table

The policy is stored at `~/.memory-guard/rl_policy.json` (default).  It is
a plain JSON file you can read, edit, or delete:

```bash
cat ~/.memory-guard/rl_policy.json | python -m json.tool | head -40
```

Example structure:
```json
{
  "version": "0.4.0",
  "epsilon": 0.312,
  "epsilon_decay": 0.995,
  "epsilon_floor": 0.05,
  "alpha": 0.1,
  "num_updates": 47,
  "q_table": {
    "24_48gb|apple_silicon|darwin|7_13b|4": {
      "4|16|2048|0": 0.847,
      "2|16|2048|0": 0.631,
      "8|16|2048|0": -0.183
    }
  }
}
```

The key format is:
- State key: `memory_tier|backend|os_platform|param_class|bits`
- Action key: `batch_size|lora_rank|seq_length|max_num_seqs`

In the example above, on an Apple Silicon Mac with 24–48 GB memory running a
7–13B model at 4-bit:
- `batch_size=4, lora_rank=16` has Q ≈ 0.85 → good (clean runs, accurate estimate)
- `batch_size=2, lora_rank=16` has Q ≈ 0.63 → acceptable but wastes memory
- `batch_size=8, lora_rank=16` has Q ≈ -0.18 → bad (caused OOM)

### Reading Q-values in Python

```python
from memory_guard.bandit import BanditPolicy
from memory_guard.bandit_state import StateKey, ConfigAction

policy = BanditPolicy.load()   # loads ~/.memory-guard/rl_policy.json

sk = StateKey.from_values(
    available_mb=38_000,
    backend="apple_silicon",
    model_params=8e9,
    model_bits=4,
)
action = ConfigAction(batch_size=4, lora_rank=16, seq_length=2048)

print(f"Q-value: {policy.q_value(sk, action):.3f}")
print(f"Updates so far: {policy.num_updates}")
print(f"States learned: {policy.num_states}")
print(f"Current epsilon: {policy.epsilon:.3f}")
```

---

## Resetting the policy for a new device

If you move to a new machine, the existing policy entries for your old device
remain but don't interfere — they're keyed to a different `DeviceFingerprint`.
The policy starts cold on the new device.

To wipe all learned data and start fresh:

```bash
rm ~/.memory-guard/rl_policy.json
```

To reset only the entries for a specific state (e.g., because you know the
learned Q-values are wrong after a hardware upgrade):

```python
from memory_guard.bandit import BanditPolicy
from memory_guard.bandit_state import StateKey

policy = BanditPolicy.load()

# Remove a specific state's Q-row
bad_sk = StateKey.from_values(
    available_mb=38_000,
    backend="apple_silicon",
    model_params=8e9,
    model_bits=4,
)
policy._q.pop(bad_sk, None)
policy.save()
```

---

## Full training workflow

```python
from memory_guard import MemoryGuard

guard = MemoryGuard.auto()   # loads policy from disk

# Pre-flight: policy recommends if it has learned, otherwise binary search
safe = guard.preflight(
    model_params=8e9, model_bits=4,
    hidden_dim=4096, num_heads=32, num_layers=32,
    batch_size=4, seq_length=2048,
    lora_rank=16, lora_layers=16,
)
print(safe)  # shows chosen batch_size / lora_rank

# Run your training
oom = False
try:
    with guard.monitor(safe.batch_size) as mon:
        for step in range(num_steps):
            train_step(batch_size=mon.current_batch_size)
except MemoryError:
    oom = True

# Record result — updates the policy and the calibration store
guard.record_result(
    actual_peak_mb=get_peak_memory(),   # or None for auto-detect
    oom_occurred=oom,
    model_name="llama-3-8b",
)
```

---

## Parameters reference

### `BanditPolicy.__init__`

| Parameter | Default | Description |
|---|---|---|
| `epsilon` | 0.9 | Initial exploration rate. 0 = always exploit, 1 = always explore |
| `epsilon_decay` | 0.995 | Multiplicative decay applied after each update |
| `epsilon_floor` | 0.05 | Minimum epsilon. Exploration never drops below this |
| `alpha` | 0.1 | Q-learning rate. Higher = faster adaptation, noisier |

### `MemoryGuard.__init__` / `.auto()`

| Parameter | Default | Description |
|---|---|---|
| `enable_bandit` | `True` | Load and use the RL policy |
| `enable_calibration` | `True` | Apply learned correction factors |
| `safety_ratio` | 0.80 | Use at most this fraction of available memory |

### `MemoryGuard.record_result()`

| Parameter | Default | Description |
|---|---|---|
| `actual_peak_mb` | `None` | Measured peak memory. Auto-detected from MLX/CUDA if None |
| `oom_occurred` | `False` | Set to True if the run ended with OOM |
| `policy_update` | `True` | Whether to update the bandit Q-table |
| `model_name` | `""` | Human-readable identifier for calibration logs |

---

## Policy file location

| Platform | Default path |
|---|---|
| macOS / Linux | `~/.memory-guard/rl_policy.json` |
| Windows | `C:\Users\<user>\.memory-guard\rl_policy.json` |

Override the path at load/save time:

```python
from pathlib import Path
from memory_guard.bandit import BanditPolicy

custom_path = Path("/shared/cluster/ml-memguard/rl_policy.json")
policy = BanditPolicy.load(custom_path)
# ... train ...
policy.save(custom_path)
```

---

## Design decisions

See [ADR 004 — RL optimizer: contextual bandit with tabular Q-learning](decisions/004-rl-contextual-bandit.md)
for the rationale behind:

- Why contextual bandit and not full sequential RL
- Why tabular Q-learning and not a neural policy
- Why per-device isolation (no cross-device generalization)
- Why epsilon never reaches zero
