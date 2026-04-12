# memguard-allocator Integration

Add GPU allocation-level telemetry to `KVCacheMonitor` with four lines of code.

## Overview

`ml-memguard` monitors KV cache utilization through vLLM's Prometheus `/metrics`
endpoint.  `memguard-allocator` goes one level deeper: it intercepts every
`cudaMalloc` / Metal buffer allocation before it reaches the GPU, recording
size, headroom, and velocity into an in-process ring buffer.

Together they form a three-tier observability stack:

```
┌─────────────────────────────────────────────────────────────────┐
│  ml-memguard  (OSS / Apache-2.0 after 2030-04-11)              │
│  • KVCacheMonitor — KV block utilization → load-shed callbacks  │
│  • VLLMWatchdog  — process health + auto-restart                │
│  • MemGuardPolicy CRD — Kubernetes hot-reload                   │
├─────────────────────────────────────────────────────────────────┤
│  memguard-allocator  (BSL-1.1)                                  │
│  • CUDA: replaces PyTorch's caching allocator (mgcuda_malloc)   │
│  • Metal: polls mlx.core.get_active_memory() every 50 ms       │
│  • ring buffer: near_miss_count, allocation_velocity_mbps,      │
│    max_headroom_mb → extended_poll_fn → InferenceTelemetry      │
├─────────────────────────────────────────────────────────────────┤
│  Cloud telemetry backend  (optional)                            │
│  • InferenceTelemetry upload (MEMGUARD_API_KEY / MEMGUARD_ORG)  │
└─────────────────────────────────────────────────────────────────┘
```

The allocator layer is optional.  `KVCacheMonitor` works without it; adding it
gives you allocation-rate visibility that KV block counts alone cannot provide.

---

## Install

**NVIDIA CUDA (Linux / WSL)**

```bash
pip install ml-memguard
pip install memguard-allocator[cuda]
memguard-allocator-build          # compiles libmemguard_allocator.so
```

**Apple Silicon (macOS)**

```bash
pip install ml-memguard
pip install memguard-allocator[metal]
# No build step required — polling-only mode uses pure Python
# Optional: memguard-allocator-build for C-level ring buffer
```

---

## Four-line setup

```python
import memguard_allocator
from memguard_allocator.poll_fn import make_extended_poll_fn, detect_gpu_total_mb
from memory_guard.inference_monitor import KVCacheMonitor

memguard_allocator.install()                                           # 1
gpu_mb      = detect_gpu_total_mb()                                    # 2 auto-detect
extended_fn = make_extended_poll_fn(total_memory_mb=gpu_mb)           # 3
monitor     = KVCacheMonitor(poll_fn=my_kv_poll, extended_poll_fn=extended_fn)

with monitor.session():
    server.serve_forever()
```

`detect_gpu_total_mb()` reads `torch.cuda.get_device_properties(0).total_memory`
on CUDA and `sysctl hw.memsize` on macOS — no manual lookup needed.

---

## How it works

### CUDA path

`memguard_allocator.install()` registers `libmemguard_allocator.so` as
PyTorch's current CUDA allocator via
`torch.cuda.memory.change_current_allocator()`.  Every `cudaMalloc` in the
process flows through `mgcuda_malloc`, which:

1. Records `(timestamp, requested_bytes, free_headroom_bytes)` into a
   per-device ring buffer (capacity 100 events, lock-free).
2. Returns control to PyTorch immediately — no allocation latency added.

### Metal path (Apple Silicon)

`install()` starts a background daemon thread that samples
`mlx.core.get_active_memory()` every 50 ms (configurable via `poll_interval_s`).
When usage increases, the delta and headroom are appended to a Python
`RingBuffer`.  Optional ObjC swizzle (`install(swizzle=True)`) gives
zero-latency kernel-level tracking but requires an entitlement on newer macOS.

### extended_poll_fn merge

Every `KVCacheMonitor` poll cycle (default 5 s), `extended_poll_fn` is called.
It drains the ring buffer and returns:

| Field | Type | Description |
|-------|------|-------------|
| `near_miss_count` | `int` | Allocations that left < `threshold_mb` free |
| `allocation_velocity_mbps` | `float` | Average allocation rate (MB/s) |
| `fragmentation_ratio` | `float` | `1 - max_headroom_mb / total_memory_mb` |

These fields are merged into the `InferenceTelemetry` snapshot uploaded to
the cloud backend on every telemetry interval.

---

## Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMGUARD_API_KEY` | — | Cloud telemetry API key (required for upload) |
| `MEMGUARD_ORG` | — | Organisation slug (required for upload) |
| `MEMGUARD_THRESHOLD_MB` | `512` | Near-miss threshold in MB |
| `MEMGUARD_POLL_INTERVAL_S` | `0.05` | Metal polling interval (seconds) |
| `ENABLE_ALLOCATOR` | `false` | Docker Compose flag — see below |

---

## Docker Compose: optional allocator service

The `examples/vllm-quickstart/docker-compose.yml` ships an optional
`memguard-allocator-build` service you can enable with:

```bash
ENABLE_ALLOCATOR=true docker compose --profile allocator up
```

This service compiles `libmemguard_allocator.so` into a named volume and
mounts it into the sidecar container so `install()` finds the library on
startup.  Without the flag the sidecar runs in polling-only mode (Metal) or
raises a `RuntimeError` on CUDA asking you to run `memguard-allocator-build`.

---

## Full example

See [`examples/vllm_allocator_integration.py`](../../examples/vllm_allocator_integration.py)
for a complete, runnable integration including:

- Auto GPU memory detection via `detect_gpu_total_mb()`
- vLLM `/metrics` Prometheus scraper as `poll_fn`
- Load-shed callbacks wired to `on_warning` / `on_shed_load`
- Standalone demo mode (`--demo` flag) that runs without a live vLLM server

```bash
# Standalone demo (no vLLM required)
python examples/vllm_allocator_integration.py --demo

# Live integration
VLLM_URL=http://localhost:8000 python examples/vllm_allocator_integration.py
```
