# memory-guard

**memguard tells you that you're reserving 4×A10G but your true peak fits in 2×A10G 94% of the time, then books the right-sizing ticket automatically. OOM prevention is how it earns the trust to do that.**

[![PyPI version](https://img.shields.io/pypi/v/ml-memguard.svg)](https://pypi.org/project/ml-memguard/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ml-memguard.svg)](https://pypi.org/project/ml-memguard/)
[![Works with vLLM](https://img.shields.io/badge/vLLM-%E2%89%A50.4-green.svg)](https://github.com/vllm-project/vllm)
[![Works with SGLang](https://img.shields.io/badge/SGLang-%E2%89%A50.3-green.svg)](https://github.com/sgl-project/sglang)
[![Works with Unsloth](https://img.shields.io/badge/Unsloth-supported-green.svg)](https://github.com/unslothai/unsloth)


```bash
pip install ml-memguard                  # core (zero dependencies)
pip install ml-memguard[hf]             # + HuggingFace Transformers adapter
pip install ml-memguard[unsloth]        # + Unsloth adapter
pip install ml-memguard[apple]          # + MLX Metal ground-truth monitoring
pip install ml-memguard[cuda]           # + CUDA OOM recovery
pip install ml-memguard[vllm]           # + vLLM inference serving adapter
pip install ml-memguard[sglang]         # + SGLang inference serving adapter
```

## OOM Prevention Quickstart

No API key. No Worker. No Cloudflare account. Works on first install.

```bash
pip install ml-memguard[vllm]
```

```python
from memory_guard import guard_vllm
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.9)
safe = guard_vllm(llm)                                         # finds safe max_num_seqs
safe.monitor.on_shed_load = lambda u: lb.reduce_weight(host, 0)

with safe.monitor.session():
    server.serve_forever()
```

---

## Cost Optimization

Once your Worker is deployed and `MEMGUARD_API_URL` is set, memguard watches your inference fleet's true memory footprint over a rolling window, computes the 94th-percentile peak per (source × model) pair, and tells you exactly which GPU tier you can safely downgrade to — and how much that saves per month.

```
$ memguard-efficiency --fleet

SOURCE       MODEL            CURRENT      RECOMMENDS     P94 MB   WASTE   SAVINGS/MO   CONF
──────────────────────────────────────────────────────────────────────────────────────────────
prod-llm     mistral-7b       4×A10G       2×A10G         18,432    61%      $727/mo    HIGH
batch-llm    llama-3-8b       2×A100-40    1×A100-40      34,102    57%      $495/mo    HIGH
dev-serve    phi-3-mini       1×A10G       1×T4            5,218    44%      $212/mo    MED

3 sources · fleet savings: $1,434/mo
```

### Quickstart

> Steps 2–3 require a deployed memguard-cloud Worker — complete [memguard-cloud/DEPLOYMENT.md](memguard-cloud/DEPLOYMENT.md) first, then return here.

```bash
pip install ml-memguard
export MEMGUARD_API_KEY=<your-key>  MEMGUARD_API_URL=https://<your-worker>.workers.dev
memguard-efficiency --fleet
```

### CLI reference

| Flag | Description |
|------|-------------|
| `--fleet` | Show all sources sorted by waste fraction (highest first) |
| `--source-id SOURCE` | Filter output to a single source ID |
| `--model MODEL` | Filter output to a specific model name |
| `--lookback-days N` | Rolling window length in days (default: 30) |
| `--json` | Output machine-readable JSON instead of a table |

### Weekly Digest

Register a Slack or Teams webhook and memguard-cloud fires it every Monday at 09:00 UTC whenever your fleet's total potential savings exceed $100 — a standing nudge to keep right-sizing tickets moving.

```bash
curl -X PUT "$MEMGUARD_API_URL/v1/settings/webhook" \
  -H "Authorization: Bearer $MEMGUARD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"}'
```

---

## The Problem

### Inference serving (vLLM / SGLang / Ollama)

`No available memory for the cache blocks. Try increasing gpu_memory_utilization` is vLLM's most-filed error. It appears when `--gpu-memory-utilization`, `--max-num-seqs`, and `--max-num-batched-tokens` are misconfigured — which they almost always are on first deploy. There is no formula; the official advice is to tune until it stops crashing. When it does crash mid-serving, it takes live user traffic down with it.

- vLLM has 30+ distinct numbered OOM issues. SGLang has a dedicated OOM tracking issue. Ollama makes the host unresponsive.
- KV cache grows linearly with context length × batch size × layers. A 128k-context Llama 3 70B needs ~40 GB of KV cache on top of ~140 GB for weights.
- There is no built-in tool that tells you the right `max_num_seqs` before you launch.

### Fine-tuning (Unsloth / HuggingFace / mlx_lm)

- **Apple Silicon**: No OOM exception exists. When you exceed memory, macOS silently swaps to disk, your Mac freezes for minutes, and eventually the OS kills your process.
- **CUDA**: `torch.cuda.OutOfMemoryError` crashes your training run. You restart, guess a smaller batch size, and pray.
- **Containers**: cgroups silently kill your process with no warning when you hit the memory limit.

Existing solutions (PyTorch Lightning BatchSizeFinder, HuggingFace `accelerate`) are CUDA-only and reactive — they catch OOM exceptions that don't exist on Apple Silicon and do nothing for inference servers.

## Quick-Start

**→ [vLLM: stop KV cache crashes in 3 minutes](docs/quickstart/vllm.md)**

```bash
pip install ml-memguard[vllm]
```

```python
from memory_guard import guard_vllm

safe = guard_vllm(engine)                                         # pre-flight: finds safe max_num_seqs
safe.monitor.on_shed_load = lambda u: lb.reduce_weight(host, 0)  # fire when KV cache hits 92%

with safe.monitor.session():
    server.serve_forever()
```

That is the entire integration. No config files. No separate process. The monitor runs in a background
thread and fires your callback — the server keeps running, your load balancer stops sending traffic here.

See the [full quick-start guide](docs/quickstart/vllm.md) for the before/after terminal output, CLI
usage, Kubernetes sidecar setup, and common troubleshooting.

### Collect KV Cache Telemetry via OTel Collector

Route your vLLM KV cache signals to memguard-cloud through the OpenTelemetry Collector with a single
pipeline addition — no code changes to your inference server. Set `VLLM_OTEL_KV_METRICS_ENABLED=true`
and point your collector's `otlphttp` exporter at `https://api.memguard.io/v1/ingest/otlp` with your
memguard API key in the `Authorization` header. See the
[OTel Collector integration guide](docs/integrations/otel-collector.md) for the complete
copy-paste config snippet and verification steps.

---

## The Solution

`memory-guard` is **proactive, not reactive**. For inference: it calculates the safe `max_num_seqs` before you launch the server and monitors KV cache utilization at runtime. For fine-tuning: it estimates peak memory before training starts and auto-adjusts batch size, LoRA rank, and sequence length to fit. Both paths use an RL optimizer that learns your specific device over time.

**Inference serving (vLLM)**

```python
from memory_guard import guard_vllm
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.9)

safe = guard_vllm(llm)
# InferenceSafeConfig:
#   max_num_seqs:     32         <- largest concurrent batch that fits
#   max_seq_len:      4096
#   estimated memory: 18,240 MB
#   budget:           19,456 MB

# Wire the KV cache monitor — fires on_shed_load at 92% utilization
safe.monitor.on_shed_load = lambda u: load_balancer.reduce_weight("primary", 0)
safe.monitor.on_warning   = lambda u: logger.warning("KV cache at %.0f%%", u * 100)

with safe.monitor.session():
    server.serve_forever()   # monitor runs in background thread
```

**Fine-tuning (Unsloth / HuggingFace / mlx_lm)**

```python
from memory_guard import MemoryGuard

guard = MemoryGuard.auto()

# Pre-flight: estimate memory and auto-downgrade config
safe = guard.preflight(
    model_params=9_000_000_000,  # 9B parameter model
    model_bits=4,                # 4-bit quantized
    hidden_dim=4096,
    num_heads=32,
    num_layers=32,
    batch_size=4,
    seq_length=2048,
    lora_rank=32,
    lora_layers=16,
)

print(safe)
# SafeConfig (FITS):
#   batch_size:       2          <- auto-reduced from 4
#   grad_checkpoint:  True       <- auto-enabled
#   grad_accumulation:4          <- compensates for smaller batch
#   estimated memory: 3835 MB
#   budget:           4643 MB

# Runtime monitoring: polls memory pressure every 5s
with guard.monitor(safe.batch_size) as mon:
    for step in range(1000):
        # Batch size may decrease mid-training if pressure rises
        train_step(batch_size=mon.current_batch_size)
```

## Features

### Inference serving

| Feature | vLLM | SGLang | Ollama / custom |
|---------|:---:|:---:|:---:|
| Safe `max_num_seqs` pre-flight | Yes | Yes | Yes (via `preflight_inference`) |
| KV cache utilization monitoring | Yes | Yes | Yes (`KVCacheMonitor`) |
| Load-shed signal at 92% KV utilization | Yes | Yes | Yes |
| Warning signal at 80% KV utilization | Yes | Yes | Yes |
| RL optimizer (learns per device/model) | Yes | Yes | Yes |
| Architecture auto-introspection | Yes (hf_config) | Yes (server_args) | Manual |

### Fine-tuning

| Feature | Apple Silicon | CUDA | Linux CPU | Windows |
|---------|:---:|:---:|:---:|:---:|
| Proactive memory estimation | Yes | Yes | Yes | Yes |
| Auto-downgrade config | Yes | Yes | Yes | Yes |
| RL optimizer (learns per device) | Yes | Yes | Yes | Yes |
| Runtime pressure monitoring | Yes (Mach kernel + MLX Metal) | Yes (torch.cuda) | Yes (PSI, cgroups) | Yes (kernel32) |
| MLX Metal ground-truth | Yes (mx.metal.get_active_memory) | N/A | N/A | N/A |
| OOM catch & retry | N/A (no OOM on Metal) | Yes | N/A | N/A |
| Container-aware (cgroups v1/v2) | N/A | Yes | Yes | N/A |
| Auto-calibration | Yes | Yes | Yes | Yes |
| FlashAttention-aware | Yes | Yes | Yes | Yes |
| GQA / MoE / Multi-modal | Yes | Yes | Yes | Yes |

## How It Works

### Inference serving path

`preflight_inference()` computes the memory footprint of model weights + KV cache at a given `max_num_seqs` and `max_seq_len`, then binary-searches for the largest concurrent batch that fits within your GPU budget (default: 80% of available VRAM). The RL optimizer learns which `max_num_seqs` value worked well on your device and model over time, replacing binary search with a confident recommendation after a handful of runs.

`KVCacheMonitor` runs a background thread polling the live KV cache token counts from vLLM or SGLang. At 80% utilization it fires `on_warning`; at 92% it fires `on_shed_load`. Neither callback does anything by default — they are signals. Your load balancer or health endpoint decides what to do (reduce upstream weight, return 503, etc.). The engine is never mutated while serving.

### Fine-tuning path

### 1. Proactive Estimation

Calculates peak memory from model architecture, accounting for:
- Per-projection LoRA input buffers (Q, K, V, O)
- FlashAttention O(n) vs standard O(n^2) attention scores
- GQA-aware KV cache (uses `num_kv_heads`, not `num_heads`)
- MoE routing buffers and active expert activations
- Optimizer states (Adam 3x, SGD 2x, Adafactor 1.5x)
- MLX lazy evaluation discount (20% reduction on Apple Silicon)
- Framework overhead (25% proportional + 400MB fixed runtime cost)

With gradient checkpointing, activation memory drops to `sqrt(layers)`.

### 2. Auto-Downgrade (quality-preserving order)

When estimate exceeds budget (available × 80%):

1. Enable gradient checkpointing (free quality, ~40% activation savings)
2. Halve batch size (compensate with gradient accumulation)
3. Halve sequence length
4. Halve LoRA rank
5. Halve LoRA layers

### 3. Runtime Monitoring

Background thread polls memory pressure every 5 seconds:

- **Apple Silicon**: `mx.metal.get_active_memory()` (ground-truth from Metal allocator), with `kern.memorystatus_level` as fallback. Detects the monotonic memory growth pattern from [mlx-examples#1262](https://github.com/ml-explore/mlx-examples/issues/1262).
- **CUDA**: `torch.cuda.memory_allocated()` vs total VRAM
- **Linux**: `/proc/pressure/memory` (PSI), cgroup-aware (`memory.high` preferred over `memory.max`)
- **Windows**: `GlobalMemoryStatusEx`

When pressure exceeds 85%, batch size is halved mid-training.

### 4. Auto-Calibration

After each training run, the actual peak memory (from `mx.metal.get_peak_memory()` or `torch.cuda.max_memory_allocated()`) is recorded alongside the formula estimate. After 3+ runs, a median correction factor is applied to future estimates, narrowing the gap between predicted and actual memory usage over time.

### 5. RL Optimizer (v0.4)

A contextual bandit that learns which `(batch_size, lora_rank)` combination works best on your specific device and model. On cold start it falls back to the binary-search path from step 2. After a handful of runs it starts recommending configs it has learned are safe and efficient — and still falls back to binary search on the 5 % exploration floor so novel model architectures always get probed.

```python
guard = MemoryGuard.auto()   # loads ~/.memory-guard/rl_policy.json on disk

safe = guard.preflight(...)  # bandit recommends once it has learned; binary search until then

# ... training loop ...

guard.record_result(
    actual_peak_mb=get_peak_memory(),
    oom_occurred=False,          # set True if training crashed with OOM
)
# → updates the Q-table and saves the policy file atomically
```

The policy is a plain JSON file — human-readable, editable, and deletable.  See [`docs/rl_optimizer.md`](docs/rl_optimizer.md) for the full reference and [`docs/decisions/004-rl-contextual-bandit.md`](docs/decisions/004-rl-contextual-bandit.md) for the design rationale.

## Framework Integration

### With vLLM

```python
from memory_guard import guard_vllm
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.9)
safe = guard_vllm(llm)

# safe.max_num_seqs  → pass to --max-num-seqs
# safe.monitor       → KVCacheMonitor, ready to start

safe.monitor.on_shed_load = lambda u: load_balancer.reduce_weight("primary", 0)
safe.monitor.on_warning   = lambda u: logger.warning("KV cache %.0f%%", u * 100)

with safe.monitor.session():
    server.serve_forever()
```

`guard_vllm` reads architecture directly from `model_config.hf_config` — no manual `hidden_dim` or `num_layers` required.

### With SGLang

```python
from memory_guard import guard_sglang
from sglang import Runtime

runtime = Runtime(model_path="meta-llama/Llama-3.1-8B-Instruct")
safe = guard_sglang(runtime)

safe.monitor.on_shed_load = lambda u: nginx.upstream_weight("primary", 0)

with safe.monitor.session():
    runtime.wait()
```

Polls `token_to_kv_pool` (preferred) or `scheduler.get_stats()` as fallback. Rolling-max smoothing suppresses false-recovery signals from RadixAttention prefix-cache evictions.

### With mlx_lm (Apple Silicon)

```python
import mlx.optimizers as optim
from memory_guard import MemoryGuard
from mlx_lm import load
from mlx_lm.tuner.trainer import train, TrainingArgs
from mlx_lm.tuner.utils import linear_to_lora_layers

guard = MemoryGuard.auto()
model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")

safe = guard.preflight(
    model_params=9e9, model_bits=4,
    hidden_dim=4096, num_heads=32, num_layers=32,
    batch_size=4, seq_length=2048,
    lora_rank=32, lora_layers=16,
)

model.freeze()
linear_to_lora_layers(
    model, safe.lora_layers,
    {"rank": safe.lora_rank, "scale": 20.0, "dropout": 0.0},
)
optimizer = optim.Adam(learning_rate=1e-4)

# The monitor runs in the background and logs if memory pressure rises.
# Note: mlx_lm's train() uses a fixed batch size. For dynamic adjustment,
# use a custom training loop that reads mon.current_batch_size each step.
with guard.monitor(safe.batch_size) as mon:
    train(
        model=model, optimizer=optimizer, train_dataset=train_set,
        args=TrainingArgs(
            batch_size=safe.batch_size,
            iters=1000,
            max_seq_length=safe.seq_length,
            grad_checkpoint=safe.grad_checkpoint,
            adapter_file="adapters.safetensors",
        ),
    )
```

### With HuggingFace Transformers (CUDA / CPU)

One call reads the model's architecture, runs preflight, patches `trainer.args`,
and attaches `MemoryGuardCallback` for mid-training batch-size downgrade.

```python
from memory_guard import guard_trainer
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./output", max_steps=1000),
    train_dataset=train_set,
)

guard_trainer(trainer)   # reads model, runs preflight, patches args + callback
trainer.train()
```

Pass `preflight_overrides` to lock in specific values the adapter can't infer
(e.g. `guard_trainer(trainer, batch_size=8, seq_length=4096, lora_rank=32)`).

### With Unsloth

Three lines — no manual architecture spelunking required.  `guard_unsloth_model`
introspects the loaded model, runs preflight, and returns a `SafeConfig` you
thread directly into `get_peft_model`.

```python
from memory_guard import guard_unsloth_model, guard_sft_trainer
from unsloth import FastLanguageModel
from trl import SFTTrainer

# 1. Load model (before LoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Preflight — reads architecture automatically, auto-downgrades if needed
safe = guard_unsloth_model(model)   # ← the one line that replaces all the math

# 3. Attach LoRA using the safe values
model = FastLanguageModel.get_peft_model(
    model,
    r=safe.lora_rank,
    lora_alpha=safe.lora_rank * 2,
    max_seq_length=safe.seq_length,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 4. Train with mid-training downgrade protection
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)
guard_sft_trainer(trainer)   # patches trainer.args + adds MemoryGuardCallback
trainer.train()
```

> **BnB double-quantization**: Unsloth loads with `bnb_4bit_use_double_quant=True`
> by default.  memory-guard detects this and applies a 5 % correction to the
> weight-memory estimate.  Auto-calibration refines the correction after 3+
> training runs.

## Framework Adapters

*New in v0.2.0* — adapters read the model's architecture automatically so you
don't have to look up `hidden_size`, `num_heads`, or `num_layers`.
Inference serving adapters added in v0.3.0; RL optimizer integrated in v0.4.0.
Full reference: [`docs/adapters.md`](docs/adapters.md).

### How model introspection works

`introspect_model(model)` reads directly from `model.config` and
`model.parameters()` without importing torch or transformers at the call site:

| Field read | Source |
|---|---|
| `hidden_size` | `model.config.hidden_size` |
| `num_attention_heads` | `model.config.num_attention_heads` |
| `num_hidden_layers` | `model.config.num_hidden_layers` |
| `num_key_value_heads` | `model.config.num_key_value_heads` (falls back to `num_attention_heads` for MHA) |
| `model_bits` | `quantization_config.load_in_4bit / load_in_8bit`, else `model.dtype` (fp16/bf16 → 16, fp32 → 32) |
| `num_parameters` | `sum(p.numel() for p in model.parameters())` |

### When to pass `preflight_overrides`

Introspected values cover most cases.  Override when:

| Scenario | Pass |
|---|---|
| Specific training batch size | `batch_size=8` |
| Non-default sequence length | `seq_length=4096` |
| Fixed LoRA config | `lora_rank=32, lora_layers=24` |
| Model loaded at different precision | `model_bits=16` |

```python
# Override batch_size and lora_rank; everything else is introspected
guard_trainer(trainer, batch_size=8, lora_rank=32)
safe = guard_unsloth_model(model, seq_length=4096, lora_rank=16)
```

### QLoRA with BnB double-quantization

When Unsloth (or any HF model) loads with `bnb_4bit_use_double_quant=True`,
`guard_unsloth_model` automatically applies a 5 % correction to the weight-memory
estimate.  `model_bits` stays 4; only `num_parameters` is scaled down to proxy
the reduced quantization-constant footprint.  After 3+ runs, auto-calibration
refines the correction further.

## API Reference

### Inference Serving (v0.3+)

#### `guard.preflight_inference(...) -> InferenceSafeConfig`
Find the largest `max_num_seqs` that fits in your GPU budget. Binary-searches from your requested max down to 1; uses the RL optimizer if it has learned this device/model combination.

```python
safe = guard.preflight_inference(
    model_params=8e9, model_bits=4,
    hidden_dim=4096, num_kv_heads=8, num_layers=32,
    max_seq_len=4096, max_num_seqs=256,
)
# safe.max_num_seqs  → int, largest safe concurrent batch
# safe.monitor       → KVCacheMonitor (not yet started)
```

#### `guard_vllm(llm, ...) -> InferenceSafeConfig`
One call. Reads architecture from `model_config.hf_config`, runs preflight, returns `InferenceSafeConfig`. No manual config spelunking.

#### `guard_sglang(runtime, ...) -> InferenceSafeConfig`
Same as `guard_vllm` for SGLang. Reads `server_args.context_length` and `server_args.max_running_requests`.

#### `KVCacheMonitor`
Background-thread monitor. Fires `on_warning` at 80% KV utilization, `on_shed_load` at 92%. Start with `safe.monitor.session()` context manager or `safe.monitor.start()` / `safe.monitor.stop()` directly.

### Fine-Tuning

#### `MemoryGuard.auto(safety_ratio=0.80)`
Create with auto-detected platform. `safety_ratio` controls headroom (0.80 = use 80% of available).

#### `guard.preflight(**config) -> SafeConfig`
Estimate memory and auto-downgrade. Returns safe config.

#### `guard.monitor(batch_size) -> RuntimeMonitor`
Context manager for runtime monitoring. Use `mon.current_batch_size` in training loop.

#### `guard.estimate(**config) -> MemoryEstimate`
Pure estimation without auto-downgrade.

#### `estimate_training_memory(**config) -> MemoryEstimate`
Standalone estimation function.

#### `auto_downgrade(budget_mb, **config) -> DowngradeResult`
Standalone downgrade function.

#### `CUDAOOMRecovery(initial_batch_size)`
CUDA-specific OOM catch-and-retry wrapper.

### RL Optimizer (v0.4)

#### `guard.record_result(actual_peak_mb=None, oom_occurred=False, policy_update=True, model_name="")`
Call after each training run to update the calibration store and the RL
policy.  `actual_peak_mb` is auto-detected from MLX/CUDA if not supplied.
Set `oom_occurred=True` if the run ended with OOM — the policy learns a
negative reward and avoids that config in future.

#### `BanditPolicy.load(path=None) -> BanditPolicy`
Load the policy from disk (defaults to `~/.memory-guard/rl_policy.json`).
Returns a fresh cold-start policy silently if the file is absent or corrupt.

#### `BanditPolicy.q_value(state_key, action) -> float`
Read the current Q-value for a `(StateKey, ConfigAction)` pair (0.0 for
unseen entries).

#### `StateKey.from_values(available_mb, backend, model_params, model_bits)`
Convenience constructor for use with `BanditPolicy.q_value()` and
`BanditPolicy.update()` directly.

Full reference: [`docs/rl_optimizer.md`](docs/rl_optimizer.md).

### Fine-Tuning Adapters (v0.2, `pip install ml-memguard[hf]`)

#### `guard_trainer(trainer, guard=None, **preflight_overrides) -> SafeConfig`
Attach memory-guard to a HuggingFace `Trainer` in one call.  Introspects the
model, runs preflight, writes safe values to `trainer.args`, and appends
`MemoryGuardCallback`.

#### `MemoryGuardCallback(guard)`
`TrainerCallback` subclass.  `on_train_begin` starts the monitor;
`on_step_begin` records a pending batch-size downgrade when the monitor signals
pressure; `on_epoch_begin` applies it (scales `gradient_accumulation_steps` to
preserve effective batch); `on_train_end` stops the monitor and records
calibration data.

#### `guard_unsloth_model(model, guard=None, **preflight_overrides) -> SafeConfig`
Run preflight on an Unsloth model before `FastLanguageModel.get_peft_model` is
called.  Thread `safe.lora_rank`, `safe.lora_layers`, `safe.seq_length` into
`get_peft_model`.  Detects BnB double-quantization and applies a 5 % correction.

#### `guard_sft_trainer(trainer, guard=None, **preflight_overrides) -> SafeConfig`
Identical to `guard_trainer` but named for TRL `SFTTrainer` workflows.

**Design constraint** (ADR 003): `guard_vllm` and `guard_sglang` emit signals only — they never mutate a running engine. Load-shedding requires a load balancer or health endpoint in front of the engine. See [`docs/adapters.md`](docs/adapters.md) for the full reference.

## Supported Hardware

**Tested** = verified on real hardware. **Reported** = community-reported. **Planned** = implementation exists, not yet verified on real hardware. **—** = not applicable.

[→ Report your config](https://github.com/memguard-project/ml-memguard/issues/new?template=hardware_report.yml) | [→ Full hardware discussion](https://github.com/memguard-project/ml-memguard/discussions)

| GPU / Device         | VRAM   | vLLM    | SGLang  | Unsloth | HF Trainer | mlx_lm  |
|----------------------|--------|---------|---------|---------|------------|---------|
| A100 40 GB           | 40 GB  | Planned | Planned | —       | Planned    | —       |
| A100 80 GB           | 80 GB  | Planned | Planned | —       | Planned    | —       |
| H100 80 GB           | 80 GB  | Planned | Planned | —       | Planned    | —       |
| RTX 4090             | 24 GB  | Planned | Planned | Planned | Planned    | —       |
| RTX 3090             | 24 GB  | Planned | Planned | Planned | Planned    | —       |
| M1/M2 MacBook Air    | 8–16 GB | —      | Planned | —       | Planned    | Planned |
| M3/M4 MacBook Pro    | 18–48 GB | —     | Planned | —       | Planned    | **Tested** |
| M4 Max (36 GB)       | 36 GB  | —       | Planned | —       | Planned    | **Tested** |
| CPU (Linux / macOS)  | —      | —       | —       | —       | **Tested** | —       |
| AMD ROCm (RX 7900)   | 24 GB  | Planned | —       | —       | —          | —       |

The M4 Max 36 GB + mlx_lm row is the only configuration verified end-to-end on real hardware.
All other "Planned" entries have working code paths and pass unit tests but have not been
verified against a running vLLM/SGLang server on those specific GPUs.

**Help us fill in the gaps**: open a [Hardware Config Report](https://github.com/memguard-project/ml-memguard/issues/new?template=hardware_report.yml)
with your GPU, framework, and whether memguard prevented an OOM.
Each report turns a "Planned" into a "Reported" or "Tested" cell.

---

## Estimation Accuracy

Measured accuracy on real training runs. **We need your help expanding this table** — see [Contributing](#contributing) below.

| Model | Device | Batch | Seq | Rank | Estimated | Actual | Error |
|-------|--------|------:|----:|-----:|----------:|-------:|------:|
| Qwen3.5-9B-4bit | M4 Max 36GB | 1 | 512 | 8 | 6,193 MB | 7,048 MB | 12.1% under |
| Qwen3.5-9B-4bit | M4 Max 36GB | 1 | 128 | 16 | 9,522 MB | 8,879 MB | 7.2% over |

**What's tested**:
- LoRA fine-tuning with mlx_lm on Apple Silicon (M4 Max)
- HuggingFace `Trainer` + `MemoryGuardCallback` end-to-end (distilgpt2, CPU, fp32) — integration smoke test passes in CI
- BnB 4-bit + double-quantization detection and 5 % correction (unit-tested against mock models)

**What's NOT tested yet**:
- CUDA GPUs (RTX 3060/4090, A100, H100)
- AMD ROCm (RX 7900, MI300X)
- Smaller devices (M1/M2 MacBook Air 8-16GB)
- Models below 7B or above 13B
- MoE architectures (Mixtral, DeepSeek-MoE)
- Multi-modal models (LLaVA, Qwen-VL)
- DoRA, full fine-tuning
- PyTorch Lightning, Axolotl, LitGPT

The estimation formula is based on published research (FlashAttention, HyC-LoRA, LoRA-FA) and verified on one configuration. Auto-calibration improves accuracy after 3+ runs on any given setup.

## Known Limitations

- **Single validation point**: Estimation accuracy is verified on one model/device combination. Your results may differ significantly — please report them.
- **Inference monitoring**: `KVCacheMonitor` (v0.3.0) emits signals only — it never mutates a running vLLM or SGLang engine. Load-shedding requires a load balancer or health-endpoint pattern in front of the engine.
- **Calibration cold start**: Auto-calibration needs 3+ training runs on a given device before corrections kick in.
- **Custom kernels**: Frameworks with heavily fused kernels (Unsloth) use less memory than the formula predicts. Calibration corrects this over time.
- **MLX Metal thread safety**: `mx.metal.get_active_memory()` is called from a background thread. MLX's Metal backend has [known thread safety limitations](https://github.com/ml-explore/mlx/issues/2133). Memory counter reads work in practice but aren't guaranteed thread-safe by the MLX API.
- **Windows**: CUDA path uses well-tested `torch.cuda` APIs. The CPU-only fallback (`GlobalMemoryStatusEx`) hasn't been validated across Windows versions.

## Contributing

### Help Us Benchmark

The single most valuable contribution right now is **running the benchmark on your hardware** and sharing the results. This directly improves estimation accuracy for everyone.

```bash
# Install
pip install ml-memguard mlx-lm

# Run with default small model (fast, ~2 minutes)
python bench/bench_accuracy.py

# Run with a specific model
python bench/bench_accuracy.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Generate a pre-formatted GitHub issue with your results
python bench/bench_accuracy.py --model mlx-community/Qwen3.5-9B-MLX-4bit --submit
```

Then open a [GitHub issue](https://github.com/memguard-project/ml-memguard/issues/new) with the output. We'll add your results to the accuracy table above.

**Devices we especially need data from**:
- M1/M2 MacBook Air (8GB, 16GB)
- M3/M4 MacBook Pro (18GB, 36GB)
- RTX 3060/3090, RTX 4070/4090
- A100, H100
- AMD Radeon RX 7900 / MI300X
- Docker/Kubernetes containers with memory limits

### Other Contributions

- **Framework adapters**: PyTorch Lightning, Axolotl, LitGPT wrappers (HF Transformers and Unsloth ship in v0.2.0; vLLM and SGLang in v0.3.0; RL optimizer in v0.4.0)
- **Accuracy data**: Real training runs on CUDA or non-Apple hardware — see the table above
- **Bug reports**: If the estimate was off by >30%, that's a bug — please report it with your config

---

## License

Apache 2.0
