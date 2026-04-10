# ADR 003 — Inference monitor: signals only, no engine mutation

**Status**: Accepted  
**Decided**: 2026-04-10 (implemented in PR 2 / `memory_guard/inference_monitor.py`)  
**Applies to**: `memory_guard.inference_monitor.KVCacheMonitor`,
               `memory_guard.adapters.vllm.guard_vllm` (PR 4),
               `memory_guard.adapters.sglang.guard_sglang` (PR 5)

---

## Context

When `KVCacheMonitor` detects that KV cache utilization has crossed the
shed-load threshold (92 %), it needs to do *something* to protect the serving
engine.  Two options were considered:

**Option A — Mutate the engine** (rejected)  
Call into the running engine to reduce the maximum number of concurrent
sequences (`max_num_seqs` in vLLM, the equivalent parameter in SGLang).

**Option B — Emit a signal, let the caller decide** (chosen)  
Fire the caller-supplied `on_shed_load` callback with the current utilization
value and do nothing else.  The operator's load balancer, autoscaler, or
custom controller handles the signal.

---

## Why Option A is not viable

### vLLM does not support hot-reconfiguration of `max_num_seqs`

`max_num_seqs` is read once at engine initialisation in
`vllm.engine.async_llm_engine.AsyncLLMEngine.__init__` and baked into the
scheduler.  There is no public API to lower it at runtime.  Patching the
private attribute (`engine.scheduler.max_num_seqs`) would:

- Break on any vLLM release that changes internal scheduler layout.
- Leave the active request queue in an inconsistent state (pending requests
  already accepted above the new limit would still be scheduled).
- Require `memory-guard` to carry version-specific monkey-patches for each
  vLLM release.

### SGLang has the same constraint

SGLang's memory pool size and concurrency limit are resolved at engine startup
in `sglang.srt.server.TokenizerManager` and committed to CUDA memory via
`torch.cuda.caching_allocator_alloc`.  There is no documented mechanism to
reduce concurrency at runtime without restarting the engine process.

### Mutation is invasive by design

A library that silently mutates a running production inference engine creates
an audit problem: operators cannot reason about why throughput changed by
reading the code alone.  The mutation happens asynchronously in a daemon
thread, with no transaction boundary and no rollback path.

---

## Why Option B is the right design

### The operator always knows better

A load balancer in front of vLLM/SGLang has information the memory monitor
does not: request priority, SLA deadlines, client retry budgets, shadow traffic
fractions.  The right actor to shed load is the one that can weigh those
factors — not a memory library.

### Signals compose; mutations do not

`on_shed_load` is a plain Python callable.  Operators wire it to whatever makes
sense for their stack:

```python
# Route new requests to a secondary replica
monitor = KVCacheMonitor(
    poll_fn=poll,
    on_shed_load=lambda u: nginx_upstream.set_weight("primary", 0),
)

# Decrement a token-bucket rate limiter
monitor = KVCacheMonitor(
    poll_fn=poll,
    on_shed_load=lambda u: rate_limiter.reduce(by=0.5),
)

# Log and page on-call
monitor = KVCacheMonitor(
    poll_fn=poll,
    on_shed_load=lambda u: pagerduty.trigger(f"KV cache {u:.0%}"),
)
```

None of these require `memory-guard` to know about nginx, rate limiters, or
PagerDuty.

### Non-invasive integrations survive framework upgrades

Because `KVCacheMonitor` only calls `poll_fn()` and the two callbacks, it
couples to vLLM/SGLang at exactly one point (the `poll_fn` the user writes).
When vLLM renames an internal API, only the user's `poll_fn` needs updating —
not the library.

### Compatible with Kubernetes and autoscalers

In cloud deployments, the correct response to memory pressure is to spin up an
additional replica and update the load balancer — neither of which can be done
from inside the engine process.  Signals are the only design that works here.

---

## Decision

**`KVCacheMonitor` fires `on_warning` and `on_shed_load` callbacks.
It never reads or writes any attribute of the serving engine.**

Specifically:

- `KVCacheMonitor` accepts `poll_fn: Callable[[], tuple[int, int]]` — the
  **only** coupling point to the serving framework.
- `on_warning(utilization: float)` is called when `used / total ≥ 0.80`.
- `on_shed_load(utilization: float)` is called when `used / total ≥ 0.92`.
  At 92 %+ only `on_shed_load` fires; `on_warning` is suppressed.
- Both callbacks are subject to a per-level cooldown (default 30 s) so they
  fire at most once per cooldown window during sustained pressure.
- The `guard_vllm` and `guard_sglang` adapters (PRs 4 and 5) follow the same
  contract: they construct and return a `KVCacheMonitor`; they never call into
  the engine except through the caller-supplied `poll_fn`.

---

## Consequences

### Single-process deployments without a load balancer

An operator running a single vLLM process with no load balancer in front of it
must handle shed-load signals themselves.  The simplest approach is to expose a
health endpoint that returns HTTP 503 when the monitor has signalled:

```python
shed_load_active = threading.Event()

monitor = KVCacheMonitor(
    poll_fn=poll,
    on_shed_load=lambda _: shed_load_active.set(),
    on_warning=lambda _: shed_load_active.clear(),  # recover on lower pressure
)

@app.get("/health")
def health():
    if shed_load_active.is_set():
        raise HTTPException(status_code=503, detail="KV cache pressure")
    return {"status": "ok"}
```

An upstream proxy (nginx, Envoy, AWS ALB) then routes new requests away from
the 503-returning replica automatically.

### No automatic recovery

`KVCacheMonitor` fires `on_shed_load` when pressure is high.  It does not fire
a corresponding "pressure cleared" callback when utilization drops below the
threshold.  The operator must poll `monitor.current_utilization` or
`monitor.utilization_history` to detect recovery, or use `on_warning` as a
partial signal (warning fires at 80 %; if only warning fires, pressure has
dropped below 92 %).  A future `on_recover` callback can be added in v0.4
if operators consistently need it.

### Threshold values are engineering defaults, not measurements

`KV_CACHE_WARNING_THRESHOLD = 0.80` and `KV_CACHE_SHED_LOAD_THRESHOLD = 0.92`
are documented constants in `memory_guard/constants.py`.  They are not derived
from empirical data.  Operators should override them by subclassing
`KVCacheMonitor` or by setting the class attributes before instantiation if
their workload profiles differ.

---

## Dependency graph note

```
PR 1 (estimate_serving_memory + preflight_inference)
  └─ PR 2 (KVCacheMonitor)
       └─ PR 3 (ADR 003 — this document)
            ├─ PR 4 (vLLM adapter)
            └─ PR 5 (SGLang adapter)
                 └─ PR 6 (docs + v0.3.0)
```

PRs 4 and 5 can be developed in parallel after PR 2 merges; PR 3 documents
the design intent that both must honour.
