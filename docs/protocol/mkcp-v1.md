# MemGuard KV Cache Protocol (MKCP) — v1.0.0

MKCP is a lightweight, transport-agnostic telemetry protocol for KV cache
memory pressure signals in LLM inference servers (vLLM, SGLang, Ollama, and
compatible engines).  It fills the gap left by existing observability standards:
OpenTelemetry's Gen AI semantic conventions cover request latency, token counts,
and throughput — none of them standardise the **sub-second memory dynamics** that
precede an OOM crash.

This document is the authoritative specification for MKCP v1.  It covers:

1. [Motivation and scope](#motivation-and-scope)
2. [Metric definitions](#metric-definitions)
3. [Event type definitions](#event-type-definitions)
4. [Wire format — JSON over HTTP](#wire-format--json-over-http)
5. [Wire format — Prometheus exposition](#wire-format--prometheus-exposition)
6. [Versioning policy](#versioning-policy)
7. [Reference implementation](#reference-implementation)
8. [Security considerations](#security-considerations)

---

## Motivation and scope

vLLM and SGLang both expose a Prometheus `/metrics` endpoint.  The existing
gauges are **snapshot metrics** — they report the current utilisation fraction
but cannot distinguish a stable 80 % utilisation (safe) from a spike to 80 %
over 8 seconds (imminent OOM).  More importantly, they do not expose:

- the **rate of change** of KV cache usage (velocity)
- **block fragmentation** (OOM at 65 % utilisation due to no contiguous window)
- **scheduler preemption events** (the internal signal that the engine is
  already responding to memory pressure)
- **allocation near-misses** (succeeded with < N MB headroom — requires a
  custom allocator to measure)

MKCP defines canonical names for all of these signals so that any inference
engine can emit them and any observability backend can consume them without
per-engine adapter code.

MKCP does **not** define:

- request-level traces (use OpenTelemetry)
- model accuracy metrics (out of scope)
- GPU hardware utilisation (use DCGM / node_exporter)

---

## Metric definitions

All MKCP metrics use dot-separated namespaces.  Prometheus implementations
replace dots with underscores per the Prometheus naming convention
(`kvcache.utilization` → `kvcache_utilization`).

### `kvcache.utilization`

| Property | Value |
|---|---|
| Type | Gauge |
| Unit | fraction (0.0 – 1.0) |
| Description | Fraction of KV cache blocks currently allocated |
| vLLM source | `vllm_kv_cache_usage_perc / 100` |
| SGLang source | `sglang_token_usage / sglang_max_total_num_tokens` |

The foundational utilisation metric.  Used as a feature baseline; on its own
insufficient to predict OOM — see `kvcache.allocation_velocity_mbps`.

---

### `kvcache.allocation_velocity_mbps`

| Property | Value |
|---|---|
| Type | Gauge |
| Unit | MB/s |
| Description | Rate of KV cache growth over the last poll interval |
| Derivation | `Δ(allocated_blocks × block_size_mb) / Δt` |

The earliest-leading OOM predictor.  A spike from 50 % to 80 % utilisation in
8 seconds is critically different from a slow drift to 80 % over 30 minutes;
both look identical in `kvcache.utilization`.

Zero means the KV cache is stable or shrinking (sequences completing).

---

### `kvcache.eviction_rate`

| Property | Value |
|---|---|
| Type | Gauge |
| Unit | preemptions/second |
| Description | Rate at which the scheduler is preempting in-flight sequences to reclaim blocks |
| vLLM source | delta of internal `Scheduler.num_preemption` counter / poll interval |
| SGLang source | delta of RadixAttention eviction counter / poll interval |

Non-zero eviction rate at moderate utilisation is the clearest sign that the
engine is already under memory pressure.  It is an internal vLLM/SGLang signal
that is never surfaced in either framework's Prometheus endpoint today.

---

### `kvcache.fragmentation_ratio`

| Property | Value |
|---|---|
| Type | Gauge |
| Unit | fraction (0.0 – 1.0) |
| Description | Fraction of allocated KV cache blocks that are not part of a contiguous free region |
| Derivation | `(total_blocks − max_contiguous_free_blocks) / total_blocks` |

Engines can OOM at surprisingly low utilisation (< 70 %) when block
fragmentation prevents the allocator from finding a contiguous window large
enough to hold a new sequence.  This metric is currently invisible to all
existing observability tooling.

---

### `memory.pressure_level`

| Property | Value |
|---|---|
| Type | Gauge |
| Unit | MB above high-watermark |
| Description | OS-level memory pressure reported by eBPF cgroup probes |
| Source | `cgroup memory.high` threshold breach events (Linux only) |

Zero means no OS-level pressure signal.  Non-zero values indicate that the
Linux kernel has detected memory usage above the cgroup high-watermark and is
applying throttling.  Requires `CAP_BPF` or eBPF probes to measure.

---

### `memory.oom_risk_score`

| Property | Value |
|---|---|
| Type | Gauge |
| Unit | probability (0.0 – 1.0) |
| Description | GBT model prediction: probability of OOM in the next 30 – 120 seconds |
| Source | `POST /v1/predict` response from memguard-cloud |

This field is populated by the memguard sidecar after calling the fleet
intelligence API.  Inference engines that embed memguard natively can emit it
directly; standalone Prometheus scrapers will see it as 0.0 until the sidecar
populates it.

The threshold at which `shed_load` events are emitted (default: 0.70) is
configurable via `MemGuardPolicy.spec.shedThreshold` or the
`MEMGUARD_SHED_THRESHOLD` environment variable.

---

### `memory.near_miss_count`

| Property | Value |
|---|---|
| Type | Counter |
| Unit | allocations |
| Description | Cumulative GPU memory allocations that succeeded with less than the headroom threshold remaining |
| Source | memguard-allocator CUDA/Metal allocator hook |
| Default headroom | 512 MB |

Requires `memguard-allocator` to be installed and active.  The counter resets
to zero on allocator restart.  Without the custom allocator, this field is
always 0 — it is not estimable from framework-level metrics alone.

---

### `scheduler.preemption_rate`

| Property | Value |
|---|---|
| Type | Gauge |
| Unit | preemptions/second |
| Description | Scheduler-level request preemptions, distinct from block-level evictions |
| vLLM source | delta of `vllm_num_preemptions_total` / poll interval |
| SGLang source | delta of internal preemption counter / poll interval |

Conceptually related to `kvcache.eviction_rate` but at a coarser granularity:
one scheduler preemption may cause multiple block evictions.  Both signals are
included because they have different leading times — block evictions appear
first, followed by scheduler preemptions as pressure intensifies.

---

## Event type definitions

MKCP defines three named event types that capture **discrete state transitions**
(as opposed to the continuous gauge signals above).  Events are emitted
alongside the metric payload when a threshold is crossed.

### `warn`

Emitted when `memory.oom_risk_score` crosses the warning threshold (default: 0.50)
or when `kvcache.utilization` exceeds 0.85.  Indicates elevated risk; no
traffic changes are made.  Consumers should alert on this event.

### `shed_load`

Emitted when `memory.oom_risk_score` crosses the shed threshold (default: 0.70).
The memguard sidecar returns HTTP 503 from `/readyz` for this interval, causing
Kubernetes to stop routing new requests to this pod.  In-flight requests
continue to completion.

### `critical_restart`

Emitted immediately before the `VLLMWatchdog` initiates a graceful restart
sequence (SIGTERM → drain → SIGKILL).  Consumers should expect the pod to
become unavailable within 60 seconds.

---

## Wire format — JSON over HTTP

Emitters `POST` a JSON payload to the configured `MKCP_ENDPOINT` (or
`VLLM_MKCP_ENDPOINT` / `SGLANG_MKCP_ENDPOINT` environment variables).

### Request

```
POST /v1/ingest/mkcp
Content-Type: application/json
Authorization: Bearer <api_key>
```

```json
{
  "mkcp_version": "1.0.0",
  "ts": "2026-04-12T10:00:00.000Z",
  "source_id": "pod/vllm-prod-abc123",
  "runtime": "kubernetes",
  "model_name": "meta-llama/Llama-3-70B-Instruct",
  "backend": "cuda",
  "metrics": {
    "kvcache.utilization":             0.74,
    "kvcache.allocation_velocity_mbps": 18.3,
    "kvcache.eviction_rate":            0.0,
    "kvcache.fragmentation_ratio":      0.11,
    "memory.pressure_level":            0.0,
    "memory.oom_risk_score":            0.42,
    "memory.near_miss_count":           3,
    "scheduler.preemption_rate":        0.0
  },
  "event": null
}
```

When a threshold is crossed the `event` field carries the event type:

```json
{
  "event": {
    "type": "shed_load",
    "triggered_by": "memory.oom_risk_score",
    "value": 0.71,
    "threshold": 0.70
  }
}
```

### Response

```
202 Accepted
Content-Type: application/json

{"ok": true, "ingested_at": "2026-04-12T10:00:00.123Z"}
```

### Batching

Emitters may batch up to 100 records per request by wrapping the payload in an
array under the `batch` key:

```json
{"batch": [ <record>, <record>, ... ]}
```

The server processes each record atomically; partial success is not possible
within a batch — the entire batch succeeds or fails.

### Retry policy

Emitters must implement exponential back-off with jitter on 5xx responses.
Initial delay: 100 ms.  Maximum delay: 30 s.  Maximum attempts: 5.  Network
errors and 429 responses are treated as transient; 400 responses indicate a
malformed payload and must not be retried.

---

## Wire format — Prometheus exposition

For engines that already expose a Prometheus `/metrics` endpoint, MKCP metrics
can be emitted as standard Prometheus gauges/counters.  The dot-separator is
replaced with underscore per Prometheus convention.

```
# HELP kvcache_utilization Fraction of KV cache blocks currently allocated (0.0-1.0)
# TYPE kvcache_utilization gauge
kvcache_utilization{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 0.74

# HELP kvcache_allocation_velocity_mbps Rate of KV cache growth in MB/s
# TYPE kvcache_allocation_velocity_mbps gauge
kvcache_allocation_velocity_mbps{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 18.3

# HELP kvcache_eviction_rate Scheduler block evictions per second
# TYPE kvcache_eviction_rate gauge
kvcache_eviction_rate{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 0.0

# HELP kvcache_fragmentation_ratio Fraction of KV cache that is non-contiguous (0.0-1.0)
# TYPE kvcache_fragmentation_ratio gauge
kvcache_fragmentation_ratio{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 0.11

# HELP memory_pressure_level OS cgroup memory pressure in MB above high-watermark
# TYPE memory_pressure_level gauge
memory_pressure_level{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 0.0

# HELP memory_oom_risk_score GBT model OOM probability for the next 30-120 seconds (0.0-1.0)
# TYPE memory_oom_risk_score gauge
memory_oom_risk_score{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 0.42

# HELP memory_near_miss_count_total Cumulative GPU allocations that succeeded with < headroom threshold
# TYPE memory_near_miss_count_total counter
memory_near_miss_count_total{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 3

# HELP scheduler_preemption_rate Scheduler request preemptions per second
# TYPE scheduler_preemption_rate gauge
scheduler_preemption_rate{model="meta-llama/Llama-3-70B-Instruct",backend="cuda"} 0.0
```

Prometheus-mode emitters do not emit event types — events are expressed as
threshold crossings that consumers must detect by comparing gauge values to
configured thresholds.

---

## Versioning policy

MKCP uses **semantic versioning** (`MAJOR.MINOR.PATCH`).

| Change type | Version bump | Backwards compatible? |
|---|---|---|
| New metric or event type added | MINOR | Yes — consumers ignore unknown fields |
| Existing metric renamed or removed | MAJOR | No |
| Required field added to payload | MAJOR | No |
| Optional field added to payload | MINOR | Yes |
| Documentation / description change only | PATCH | Yes |

The `mkcp_version` field in the JSON payload carries the spec version used by
the emitter.  Ingest endpoints must accept any payload with the same MAJOR
version and ignore fields they do not recognise.

MKCP v1 is stable.  Breaking changes (MAJOR version bump) require a 6-month
deprecation window with dual-version support from the memguard-cloud ingest
endpoint.

---

## Reference implementation

The memguard sidecar (`memory_guard/sidecar.py`) and
`KVCacheMonitor` (`memory_guard/inference_monitor.py`) constitute the
reference implementation of MKCP emission.

`InferenceTelemetry` (`memory_guard/telemetry.py`) maps directly to the MKCP
metric payload:

| `InferenceTelemetry` field | MKCP metric |
|---|---|
| `kv_velocity_mbps` | `kvcache.allocation_velocity_mbps` |
| `fragmentation_ratio` | `kvcache.fragmentation_ratio` |
| `eviction_rate` | `kvcache.eviction_rate` |
| `near_miss_count` | `memory.near_miss_count` |
| `memory_pressure_level` | `memory.pressure_level` |
| `page_fault_rate` | (eBPF supplemental; not in base MKCP v1) |
| `preemption_count` / poll_interval | `scheduler.preemption_rate` |

The `memory.oom_risk_score` field is populated by the memguard-cloud
`POST /v1/predict` response — it is not derived from `InferenceTelemetry`
directly.

JSON Schema: `docs/protocol/mkcp-v1.schema.json`
Protobuf schema: `docs/protocol/mkcp-v1.proto`
Changelog: `docs/protocol/CHANGELOG.md`

---

## Security considerations

**Authentication**: All HTTP POST emissions must carry a Bearer token in the
`Authorization` header.  The ingest endpoint rejects unauthenticated requests
with `401 Unauthorized`.

**Data sensitivity**: MKCP payloads contain model names, backend identifiers,
and memory pressure signals.  They do not contain user prompts, completions, or
PII.  Payloads are safe to transmit over TLS to third-party aggregators.

**Replay attacks**: Each payload carries a `ts` (RFC 3339 timestamp) field.
The ingest endpoint rejects payloads where `|now − ts| > 5 minutes` to prevent
replay of stale events.

**Emitter trust**: The `source_id` field is informational and not used for
access control.  API key authentication governs ingest permissions.
