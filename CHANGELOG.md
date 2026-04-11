# Changelog

## [0.5.0] - 2026-04-11

### Added — Auto-Heal Recovery API

- **`BanditPolicy.recommend(state_key) → Optional[ConfigAction]`** — deterministic
  argmax over the Q-table for *state_key*.  Unlike `select_action`, epsilon is
  ignored entirely and no candidate list is required.  Returns `None` on cold start
  (state never seen).  Safe to call from auto-heal watchdogs and recovery paths where
  random exploration would cause a second crash.

- **`BanditPolicy.recommend_conservative(state_key, margin=0.15) → Optional[ConfigAction]`**
  — calls `recommend()` then applies a 15 % safety margin: `batch_size` and
  `max_num_seqs` are reduced by *margin* (floored at 1 and 0 respectively);
  `lora_rank` and `seq_length` are left unchanged (reducing either changes model
  semantics, not just memory headroom).  Returns `None` on cold start.  This is the
  correct method to call from an OOM recovery loop.

- **`BanditPolicy.confidence → float`** — `min(num_updates / 20, 1.0)`.  A
  cold-start gate: 0.0 at zero updates, reaches 1.0 after 20 real training or
  inference runs.  Values below 0.5 signal that recommendations exist but carry
  little weight; callers should prefer binary-search fallback.
  `MIN_UPDATES_FOR_CONFIDENCE = 20` is exposed at the module level.

- **`VLLMWatchdog`** — subprocess supervisor that auto-heals vLLM inference servers
  on OOM crashes, compressing MTTR from 60–80 min (engineer-assisted) to under 2 min
  (fully automated).
  - OOM detection requires **both** a suspicious exit code (`{1, -9, 137}`) **and** an
    OOM pattern in stderr (8 patterns: `"cuda out of memory"`, `"oom"`, `"killed"`,
    etc.) — pure exit-code matches (e.g. config errors exiting 1) are never retried.
  - On confirmed OOM: calls `bandit.recommend_conservative(state_key)` to get a
    guaranteed-safer config, patches `--max-num-seqs` and `--gpu-memory-utilization`
    in the CLI command, backs off for `backoff_seconds`, and relaunches.
  - Cold-start fallback (no Q-table data): applies a fixed *conservative_margin*
    reduction to the current flag values directly.
  - `alert_callback(message, attempt, max_retries)` is fired on every recovery
    attempt and on final failure — wire this to PagerDuty, Slack, or any webhook.
  - `VLLMWatchdog.stop()` sends SIGTERM and prevents restart after the process exits.
  - `_patch_flag(cmd, flag, new_value)` handles both `--flag value` and
    `--flag=value` forms; never mutates the original list.

- **`guard_vllm_watchdog(model, *, host, port, max_num_seqs, gpu_memory_utilization,
  tensor_parallel_size, model_params, model_bits, bandit, max_retries,
  backoff_seconds, conservative_margin, alert_callback, extra_args) → VLLMWatchdog`**
  — convenience constructor.  Builds the vLLM CLI command, derives the `StateKey`
  from the current platform, loads or reuses a `BanditPolicy`, and returns a
  ready-to-run watchdog.  `extra_args` are appended verbatim.

- **`KVCacheMonitor.critical_threshold`** (default `0.95`) — KV cache usage fraction
  above which a planned graceful restart is triggered.

- **`KVCacheMonitor.restart_callback`** — zero-argument callable invoked when
  utilization stays at or above `critical_threshold` for `critical_ticks` consecutive
  poll ticks.  The caller wires this to the process supervisor (e.g.
  `VLLMWatchdog.stop` + relaunch).  `None` disables the feature.

- **`KVCacheMonitor.critical_ticks`** (default `3`) — number of consecutive ticks
  above `critical_threshold` required before `restart_callback` fires.  A single
  transient spike never triggers a restart; the consecutive counter resets to 0 on
  any tick below the threshold and on every `monitor.start()` call.

### Tests

- **`tests/test_bandit_recommend.py`** — 22 tests covering `recommend()` argmax,
  cold-start `None`, determinism across 50 calls, state-key isolation;
  `recommend_conservative()` 15 % margin math, floor clamping at 1 / 0,
  lora_rank / seq_length unchanged; `confidence` linearity, threshold clamp,
  no-exceed-1.0.
- **`tests/test_watchdog.py`** — 62 tests covering `_is_oom_exit()` with all
  8 OOM patterns × 3 exit codes (24 parametrized cases); false-positive guards
  (exit-code-only, pattern-only with exit 0, exit code 2, empty stderr);
  case-insensitive matching; `_patch_flag()` space and equals forms, append-when-
  absent, no mutation; `_apply_action_to_cmd()` patching and zero/None skip guards;
  `guard_vllm_watchdog()` command construction and attribute propagation.
- **`tests/test_kvcache_critical.py`** — 20 tests covering attribute defaults;
  fires after N ticks; does not fire on single spike, N-1 ticks, alternating
  above/below; counter resets on `start()`; crashing callback doesn't kill thread.
- **`tests/conftest.py`** — v0.5.0 baseline recorded: **591 passed, 1 skipped
  (intentional smoke-test gate), 0 failed** across all 17 test files.

### Fixed

- Calculator "How?" panel copy for the KV cache monitor now accurately describes
  the implemented behaviour — "polls KV cache usage every 5 s; when usage exceeds
  95 % for 3 consecutive ticks, triggers a planned graceful restart before a crash"
  — replacing a vague claim about a feature that was not yet built.

---

## [0.4.0] - 2026-04-10

### Added — RL Contextual Bandit Optimizer

- **`memory_guard.bandit_state`** — pure data layer for the RL policy.
  - `DeviceFingerprint`: bucketed available memory (5 tiers: sub-8 GB to 80+ GB) ×
    backend × platform.  Two sessions on the same machine with similar free memory
    hash to the same fingerprint so the Q-table generalises across sessions.
  - `ModelFingerprint`: bucketed parameter count (5 classes: sub-1B to 35+B) ×
    normalised quantisation bits (snapped to 4/8/16/32).
  - `ConfigAction`: frozen, hashable discrete action — `(batch_size, lora_rank,
    seq_length, max_num_seqs)`.  `max_num_seqs=0` signals training-only configs.
  - `StateKey`: `NamedTuple(DeviceFingerprint, ModelFingerprint)` — the Q-table
    row key.
  - Bucket boundary constants exposed at module level: `MEMORY_TIER_BOUNDARIES_MB`,
    `PARAM_CLASS_BOUNDARIES`, `MEMORY_TIER_LABELS`, `PARAM_CLASS_LABELS`.
  - `StateKey.from_values(available_mb, backend, model_params, model_bits)` and
    equivalent `from_values` classmethods on both fingerprint types.

- **`memory_guard.reward`** — reward signal for the bandit.
  - `RewardSignal(outcome, efficiency_bonus, combined)`: frozen dataclass with an
    `is_oom` property.
  - `compute_reward(estimated_mb, actual_peak_mb, budget_mb, oom_occurred, ...)`:
    `outcome` = ±1; `efficiency_bonus` = `clamp(1 − |est − act| / budget, 0, 1)`;
    `combined` = `0.6 × outcome + 0.4 × efficiency_bonus` (range [-0.6, 1.0]).
  - `RewardSignal` and `compute_reward` re-exported from `memory_guard` top level.

- **`memory_guard.bandit.BanditPolicy`** — epsilon-greedy tabular Q-learner.
  - `select_action(state_key, candidates, epsilon=None) → Optional[ConfigAction]`:
    returns `None` on cold start (state unseen) or on the exploration branch,
    prompting the caller to fall back to binary search.  Never raises.
  - `update(state_key, action, reward, alpha=None)`: applies
    `Q[s][a] += alpha × (reward − Q[s][a])` then decays epsilon:
    `epsilon = max(epsilon_floor, epsilon × epsilon_decay)`.
  - `q_value(state_key, action) → float`: returns the current Q-value (0.0 for
    unseen state/action pairs).
  - `save(path=None)`: atomic JSON write (tempfile + `os.replace`, chmod 0o600).
    Defaults to `~/.memory-guard/rl_policy.json`.  Fails silently with a warning
    so a save error never crashes a training run.
  - `BanditPolicy.load(path=None) → BanditPolicy`: deserialises from disk;
    returns a fresh cold-start policy on absent, unreadable, or corrupt files.
    Never raises.
  - Serialisation: `StateKey` → `"tier|backend|os|param_class|bits"`;
    `ConfigAction` → `"batch_size|lora_rank|seq_length|max_num_seqs"` —
    human-readable JSON object keys.
  - Policy format version `"0.4.0"` stored in the file for forward compatibility.

- **`MemoryGuard` integration** — bandit wired into the existing API.
  - `MemoryGuard.__init__` and `.auto()` gain `enable_bandit: bool = True`.
    On construction the policy is loaded from disk (cold-start safe; no file
    required).  Set `enable_bandit=False` to reproduce exact v0.3 behaviour.
  - `preflight()`: builds `StateKey`, generates `(batch_size, lora_rank)`
    candidate grid, calls `policy.select_action()` first.  If the policy
    returns an action it is validated by the formula-based estimator (safety net
    preserved); if it fits the budget the `SafeConfig` is returned immediately,
    skipping the binary-search downgrade.  On cold start or exploration the v0.3
    binary-search path runs unchanged.  `_last_action` and `_last_state_key`
    are always recorded for subsequent `record_result()` calls.
  - `preflight_inference()`: same pattern for `max_num_seqs` — the bandit picks
    from `_BANDIT_NUM_SEQS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)`.
  - `record_result()` gains `oom_occurred: bool = False` and
    `policy_update: bool = True` parameters.  It now returns the `RewardSignal`
    from `record_training_result` and calls `policy.update(state_key, action,
    reward.combined)` then `policy.save()` when conditions are met.

### Changed

- **`record_training_result()`** (in `memory_guard.calibration`): now returns
  `RewardSignal` instead of `None`.  Existing callers that discard the return
  value are unaffected.  Two new keyword parameters: `budget_mb: float = 0.0`
  and `oom_occurred: bool = False`.

### Added — Documentation

- **`docs/decisions/004-rl-contextual-bandit.md`** (ADR 004): records why
  contextual bandit was chosen over full sequential RL, why tabular Q-learning
  over a neural policy, why per-device isolation over cross-device
  generalisation, and why epsilon has a permanent floor.
- **`docs/rl_optimizer.md`**: user-facing reference guide — state/action/reward
  in plain terms, cold-start guarantee, inspecting and resetting the Q-table,
  full training workflow, and parameters reference.

### Tests

- 63 tests for `bandit_state` (state/action encoding, bucketing, fingerprints)
- 26 tests for `reward` (RewardSignal, compute_reward, record_training_result)
- 37 tests for `bandit` (Q-formula, epsilon decay, persistence round-trips)
- 37 integration tests for MemoryGuard+bandit (`test_bandit_integration.py`)
- Total: **488 tests passing** (0 regressions from v0.3)

---

## [0.3.0] - 2026-04-10

### Added — Inference Memory Monitoring

- **`estimate_serving_memory()`** — KV cache ceiling estimator for inference
  serving workloads.  Computes the worst-case memory required when all
  `max_num_seqs` sequences are at full `max_seq_len` length:
  `2 × num_layers × num_kv_heads × head_dim × max_seq_len × max_num_seqs × dtype_bytes`.
  GQA-aware (`num_kv_heads` rather than `num_heads`).  Returns an
  `InferenceServingEstimate` dataclass with per-component breakdown and a
  `.fits_in(budget_mb)` helper.

- **`MemoryGuard.preflight_inference()`** — binary search over `max_num_seqs`
  to find the largest value that fits within the memory budget.  Returns an
  `InferenceSafeConfig` with `max_num_seqs`, `max_seq_len`, `gpu_memory_utilization`,
  and a `monitor` field (see below) ready to pass to vLLM / SGLang CLI flags.

- **`InferenceSafeConfig.monitor`** — `KVCacheMonitor` attached by the adapter
  functions below.  Unstarted on return; use `safe.monitor.start()` or
  `with safe.monitor.session(): ...` when ready to serve.

- **`KVCacheMonitor`** — background-thread KV cache utilization monitor for
  inference serving.  Polls a caller-supplied `poll_fn: () → (used, total)`,
  fires `on_warning(u)` at ≥ 80 % and `on_shed_load(u)` at ≥ 92 %
  utilization.  Shed-load takes priority over warning.  Both callbacks are
  subject to a per-level cooldown (default 30 s).  The monitor never reads or
  writes any attribute of the serving engine (ADR 003 — signals only, no
  engine mutation).  Use via `monitor.session()` context manager or explicit
  `start()` / `stop()`.

- **vLLM adapter** (`pip install ml-memguard[vllm]`)
  - `guard_vllm(llm, guard=None, **preflight_overrides) -> InferenceSafeConfig`
    accepts `vllm.LLM`, `vllm.AsyncLLMEngine`, or a bare `vllm.LLMEngine`.
    Reads `model_config.hf_config` for architecture params.
  - Back-calculates `max_num_seqs` from `cache_config.num_gpu_blocks`:
    `blocks_per_seq = ceil(max_seq_len / block_size)`;
    `actual_max_seqs = num_gpu_blocks // blocks_per_seq`.
    Keeps the preflight estimate and live utilization on the same scale.
  - Refines `gpu_memory_utilization` from the actual measured KV MB vs
    available memory.
  - Wires `KVCacheMonitor` poll_fn to
    `scheduler.block_manager.get_num_free_gpu_blocks()` and
    `.get_num_total_gpu_blocks()`.
  - Quantization detection: AWQ / GPTQ / AWQ-Marlin / GPTQ-Marlin / FP8 /
    SqueezeLLM → 4 bits; BitsAndBytes / SmoothQuant → 8 bits; else 16 bits.
  - Returns `InferenceSafeConfig` with `safe.monitor` set (unstarted).

- **SGLang adapter** (`pip install ml-memguard[sglang]`)
  - `guard_sglang(engine, guard=None, **preflight_overrides) -> InferenceSafeConfig`
    accepts `sglang.Runtime` (unwrapped via `.engine`) or any bare engine
    object with `server_args`.
  - Reads `server_args.{context_length, dtype, quantization}` and walks
    `tp_worker.model_runner.model.config` for HF architecture fields.
  - Back-calculates `max_num_seqs` from the actual token pool:
    `actual_max_seqs = total_token_slots // max_seq_len`.
  - Polls `engine.token_to_kv_pool.get_available_size()` (SGLang ≥ 0.3.0,
    preferred) or `engine.mem_pool.available` (older SGLang), with a
    `scheduler.get_stats()` fallback and a null fallback with a one-time
    warning.
  - **3-reading rolling-max smoothing**: SGLang's RadixAttention prefix-cache
    can evict large KV blocks suddenly, causing utilization to drop below the
    shed-load threshold.  The `poll_fn` tracks the last 3 raw utilization
    values and reports their maximum, suppressing transient drops without
    delaying recovery detection once pressure genuinely recedes.
  - Returns `InferenceSafeConfig` with `safe.monitor` set (unstarted).

- **ADR 003** (`docs/decisions/003-inference-signals-only.md`) — documents the
  signals-only design: why vLLM `max_num_seqs` cannot be hot-reconfigured,
  why mutation is invasive, and how the callback design composes with nginx,
  Envoy, PagerDuty, and Kubernetes autoscalers.

- **`docs/inference.md`** — new reference guide covering `estimate_serving_memory`,
  `preflight_inference`, `InferenceSafeConfig`, the `KVCacheMonitor` hook table
  (poll_fn contract, thresholds, callbacks, cooldown), and full vLLM / SGLang
  workflow examples.

- **New extras in `pyproject.toml`**:
  `vllm = ["vllm>=0.4"]`, `sglang = ["sglang>=0.3"]`.
  Both are included in `all`.

### Tests

- 122 new unit tests across `tests/test_inference_estimator.py`,
  `tests/test_kv_cache_monitor.py`, `tests/test_vllm_adapter.py`, and
  `tests/test_sglang_adapter.py`.  No vLLM or SGLang installation required —
  all framework objects are simulated with `MagicMock`.  Total suite: 325 tests.

---

## [0.2.0] - 2026-04-10

### Added — Framework Adapters

- **`memory_guard.adapters` subpackage** (pure-Python, zero new hard dependencies)
  - `introspect_model(model)` — reads `hidden_size`, `num_attention_heads`,
    `num_hidden_layers`, `num_key_value_heads`, `model_bits`, and `num_parameters`
    from any HuggingFace-style model without importing torch or transformers
  - `optional_import(name, extra)` — lazy import helper that raises a clear
    `pip install ml-memguard[<extra>]` hint on missing dependencies

- **HuggingFace Transformers adapter** (`pip install ml-memguard[hf]`)
  - `guard_trainer(trainer, guard=None, **preflight_overrides)` — one-call setup:
    introspects `trainer.model`, runs `preflight()`, writes the safe config to
    `trainer.args`, and appends `MemoryGuardCallback`
  - `MemoryGuardCallback(TrainerCallback)` — memory monitoring for the HF
    training loop:
    - `on_train_begin` starts `guard.monitor(per_device_train_batch_size)`
    - `on_step_begin` records a pending batch-size downgrade when the monitor
      signals pressure; sets `control.should_log` to flush a warning entry
    - `on_epoch_begin` applies any pending downgrade atomically at the epoch
      boundary, scaling `gradient_accumulation_steps` by `old_bs // new_bs`
      so the effective batch size is preserved
    - `on_log` surfaces sustained pressure warnings through the Trainer logger
    - `on_train_end` stops the monitor and calls `guard.record_result()` for
      auto-calibration

- **Unsloth adapter** (`pip install ml-memguard[unsloth]`)
  - `guard_unsloth_model(model, guard=None, **preflight_overrides)` — run
    preflight *before* `FastLanguageModel.get_peft_model`; returns `SafeConfig`
    with `lora_rank`, `lora_layers`, `seq_length` ready to thread in
  - `guard_sft_trainer(trainer, guard=None, **preflight_overrides)` — identical
    to `guard_trainer` but named for TRL `SFTTrainer` workflows
  - BnB double-quantization detection (`bnb_4bit_use_double_quant=True`): applies
    a documented 5 % correction to `num_parameters`; `model_bits` stays 4;
    auto-calibration refines the estimate after 3+ runs

- **Lazy package exports** — all adapter symbols registered via `__getattr__` in
  `memory_guard/__init__.py` so `import memory_guard` is safe on a torch-free
  machine
- **New extras** in `pyproject.toml`:
  `hf = ["transformers>=4.30", "torch>=2.0", "accelerate>=1.1.0"]`,
  `unsloth = ["unsloth", "transformers>=4.30", "torch>=2.0", "accelerate>=1.1.0"]`

### Changed

- README `With HuggingFace Transformers` example replaced with the one-call
  `guard_trainer` version
- README new **Framework Adapters** section: introspection field table,
  `preflight_overrides` guide, QLoRA double-quant note
- README **API Reference** extended with all four adapter functions

### Tests

- 43 new unit tests across `test_adapters_base.py`, `test_adapters_huggingface.py`,
  `test_adapters_unsloth.py`, and `test_hf_mid_training_downgrade.py`
- `tests/test_adapters_smoke.py` — integration smoke test: `guard_trainer` +
  `MemoryGuardCallback` survive 2 real training steps on distilgpt2 (skipped
  unless transformers is installed; run with `pip install ml-memguard[hf]`)

---

## [0.1.0] - 2026-04-01

### Core
- Proactive memory estimation for training and inference
- Auto-downgrade in quality-preserving order (grad checkpoint, batch size, seq length, rank, layers)
- `MemoryGuard` unified API: `preflight()` + `monitor()` + `record_result()`
- `ModelSpec` presets for Llama, Mistral, Qwen, Phi, Mixtral, DeepSeek-MoE, LLaVA

### Estimation
- Per-projection activation buffers (Q/K/V/O) per HyC-LoRA and LoRA-FA research
- FlashAttention-aware: O(n) attention memory vs O(n^2) standard
- GQA-aware KV cache (uses `num_kv_heads`, not `num_heads`)
- MoE routing buffers and active expert FFN activations
- Multi-modal encoder memory (vision/audio)
- Full fine-tuning, LoRA, QLoRA (double quantization), DoRA support
- MLX lazy evaluation discount (20% activation reduction)
- Swap headroom credit (50% of available swap added to budget)

### Platform Support
- macOS: Mach `host_statistics` + `sysctlbyname` via ctypes (zero subprocess calls)
- macOS: ARM64 ABI-compliant `argtypes` on all ctypes calls (CPython #42880)
- macOS: `total - (active + wired) * 0.85` available memory formula
- macOS: MLX `mx.metal.get_active_memory()` ground-truth in runtime monitor
- Linux: PSI (`/proc/pressure/memory`), cgroups v1/v2
- Linux: `memory.high` preferred over `memory.max` (CockroachDB/DuckDB finding)
- Linux: Full cgroup hierarchy walk for nested containers (K8s, systemd slices)
- Linux: `memory.high` 90% discount for overshoot under concurrent allocations
- Linux: Docker, Podman, Kubernetes, systemd-nspawn, LXC detection
- Windows: `GlobalMemoryStatusEx` via ctypes
- CUDA: `torch.cuda.mem_get_info`, OOM catch-and-retry, binary search batch finder
- ROCm/HIP: Detection via `torch.version.hip`

### Runtime Monitoring
- Background thread with configurable thresholds (warning 70%, critical 85%, emergency 92%)
- MLX Metal memory leak detection (monotonic growth pattern from mlx-examples#1262)
- Cooldown between downgrades (30s default)
- Pressure history (last 60 readings)
- Thread-safe: `_mach_lock` around all ctypes Mach kernel calls

### Calibration
- Persistent calibration store (`~/.memory-guard/calibration.json`)
- Median correction factor (robust to outliers, requires 3+ data points)
- Auto-reads peak from `mx.metal.get_peak_memory()` or `torch.cuda.max_memory_allocated()`

### Quality
- `py.typed` marker (PEP 561)
- 100+ tests across estimation, platform, downgrade, calibration, thread safety
- Zero external dependencies (ctypes-only for macOS/Windows, /proc for Linux)
- Optional dependencies: `torch` (CUDA), `mlx` (Apple Silicon), `psutil` (fallback)
