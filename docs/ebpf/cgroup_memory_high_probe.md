# cgroup memory.high Probe — Timing Advantage over Polling

`CgroupMemoryHighProbe` attaches to the `cgroup:cgroup_memory_high` kernel
tracepoint.  It delivers a `MemguardBPFEvent` **200–500 ms before the kernel
OOM killer selects a victim** — a lead time that no polling approach can
replicate.

---

## Why polling always arrives late

When a cgroup exceeds its `memory.high` soft limit the sequence of events is:

```
T+0 ms   kernel fires cgroup:cgroup_memory_high tracepoint
         → CgroupMemoryHighProbe dispatches on_high / on_oom_imminent
           immediately, in the same CPU instruction stream

T+50 ms  /proc/meminfo reflects the elevated RSS (page cache reclaim may
         partially mask the spike depending on allocation pattern)

T+100 ms vLLM Prometheus /metrics endpoint is scraped by the next poll cycle
         (default interval: 15–30 s — only relevant as a long-running trend)

T+200–   kernel throttles processes in the cgroup (sleep injection); the
T+500 ms allocating request blocks on reclaim

T+500–   kernel selects an OOM victim; SIGKILL sent to the target process
T+1500 ms (actual value depends on slab/file cache availability)
```

The tracepoint fires at T+0.  A 5-second polling loop first sees the
pressure at its next tick — anywhere from 5 ms to 5,000 ms later.  A
15-second Prometheus scrape interval means up to 15 s lag.  By the time the
Prometheus signal is actionable, the process may already be dead.

The `MemguardBPFEvent` delivered to `on_oom_imminent` arrives in the same
200–500 ms window that the kernel uses to throttle the cgroup before killing
anything — the only window in which a graceful restart is still possible.

---

## What `pressure_bytes` measures

```
pressure_bytes = memory.usage_in_bytes − memory.high
```

A value of zero means the cgroup just touched its watermark.  A value of
256 MiB means it has blown past the soft limit by 256 MiB; at this point
active reclaim is likely consuming CPU and the OOM killer may fire within
seconds.

`oom_imminent_threshold_mb` (default: 512 MB) controls when
`on_oom_imminent` fires.  Tune this per workload:

| Model size  | Recommended threshold |
|-------------|----------------------|
| 7B (4-bit)  | 256 MB               |
| 13B (4-bit) | 512 MB               |
| 70B (4-bit) | 1024 MB              |

Setting it too low produces false positives under bursty allocation.
Setting it too high leaves insufficient time for a graceful restart.

---

## Usage

```python
from memory_guard.ebpf.probes import CgroupMemoryHighProbe

def on_high(event):
    print(f"memory.high crossed: {event.cgroup_id}  "
          f"+{event.pressure_bytes // 1024**2} MiB over limit")

def on_oom_imminent(event):
    # PR 56 wires this to VLLMWatchdog.trigger_graceful_restart()
    print(f"OOM imminent — {event.pressure_bytes // 1024**2} MiB over limit")

probe = CgroupMemoryHighProbe(
    on_high=on_high,
    on_oom_imminent=on_oom_imminent,
    oom_imminent_threshold_mb=512.0,
    cgroup_filter="/kubepods/",          # ignore system slices
)
probe.load()

import time
try:
    while True:
        probe.poll(timeout_ms=10)
finally:
    probe.detach()
```

Or use it inside a `MemguardBPFSession` (recommended for production):

```python
from memory_guard.ebpf import MemguardBPFSession

with MemguardBPFSession(on_high=on_high, on_oom=on_oom_imminent) as session:
    if session.available:
        print("eBPF probes active")
    server.serve_forever()
```

---

## Platform requirements

| Requirement                          | Notes                                 |
|--------------------------------------|---------------------------------------|
| Linux kernel ≥ 5.8                   | `cgroup:cgroup_memory_high` tracepoint|
| cgroup v2 (unified hierarchy)        | Required for `memory.high` semantics  |
| `CAP_BPF` + `CAP_PERFMON`            | Or `CAP_SYS_ADMIN` (root)            |
| `bcc` Python package                 | `pip install 'ml-memguard[ebpf]'`    |

`BPFProbeLoader` checks all four requirements and logs a single
`logger.warning` if any check fails — the process continues without eBPF.

---

## Performance overhead

The BPF program runs in kernel space and completes in under 1 μs per event.
In typical LLM inference workloads the `cgroup:cgroup_memory_high` tracepoint
fires at most a few times per second during peak allocation bursts — the
CPU overhead is immeasurable compared to GPU compute time.
