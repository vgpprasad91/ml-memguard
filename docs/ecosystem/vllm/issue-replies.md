# Draft Replies for vLLM GitHub Issues

Post these after the vLLM PR is open. Replace `{PR_URL}` with the actual PR link.

---

## Issue #27934 — V1 Engine memory allocation failures and crashes with 7B-13B models on RTX 3060 12GB

> **URL:** https://github.com/vllm-project/vllm/issues/27934

```markdown
This is a symptom of two compounding problems: zombie GPU processes keeping memory
allocated after a crashed vLLM container, and the V1 engine's pre-load memory check
counting that stale allocation against the budget before the model even loads.

I ran into the same pattern and built a small tool to handle both sides:
**[ml-memguard](https://github.com/vgpprasad91/ml-memguard)** — it runs a pre-flight
check that binary-searches for the largest `max_num_seqs` that actually fits your live
GPU budget (accounting for what's already allocated), and starts a background watchdog
that auto-restarts vLLM cleanly if the process exits.

Two lines to add to your launch script:

```python
from memory_guard import guard_vllm

safe = guard_vllm(engine)          # finds max_num_seqs that fits
safe.monitor.on_shed_load = lambda u: print(f"KV cache at {u:.0%} — shedding load")

with safe.monitor.session():
    server.serve_forever()
```

The pre-flight report prints the exact memory breakdown so you can see whether the RTX
3060's 12 GB is enough at your chosen `max_seq_len`:

```
gpu_total:     12,288 MB
weights_mb:     7,629 MB  (7B BF16)
kv_budget:      2,131 MB
max_num_seqs:       4    ← binary-searched
status:         FITS
```

For the zombie-process problem specifically, the `VLLMWatchdog` restarts with a clean
CUDA context each time so stale memory from the previous crash doesn't carry over.

3-minute setup guide: https://github.com/vgpprasad91/ml-memguard/blob/main/docs/quickstart/vllm.md

I also opened a PR to add this to the vLLM integrations docs: {PR_URL}
```

---

## Issue #28230 — GPU VRAM continuously increases during Qwen3-VL usage over days until OOM

> **URL:** https://github.com/vllm-project/vllm/issues/28230

```markdown
The slow VRAM growth pattern you're describing — where reported utilization stays
moderate but free memory dwindles until new requests OOM — is exactly the failure mode
that `kv_cache_usage_perc` misses. That metric is a snapshot of block utilization, not
a velocity signal. By the time it shows danger, you're already out of headroom.

I built **[ml-memguard](https://github.com/vgpprasad91/ml-memguard)** specifically to
catch this: it tracks `kv_velocity_mbps` (rate of memory growth, not just the level) and
`fragmentation_ratio` (contiguous free blocks, not just total free blocks) as separate
signals. When velocity is high and fragmentation is rising, the `on_shed_load` callback
fires *before* the OOM — your load balancer stops routing new requests, existing requests
finish cleanly, and the server keeps running without a restart.

For a long-running Qwen3-VL deployment it would look like this:

```python
safe = guard_vllm(engine)

def on_shed_load(util):
    # Stop new vision encoder requests routing here
    load_balancer.reduce_weight(replica_id, weight=0)

def on_warning(util):
    # Early signal at 80% — useful for dashboards
    metrics.increment("vllm.kvcache.warning")

safe.monitor.on_shed_load = on_shed_load
safe.monitor.on_warning   = on_warning

with safe.monitor.session():
    server.serve_forever()
```

The monitor runs in a background thread and never touches the vLLM engine — it only
emits signals. Your replica keeps taking traffic until you decide to shed.

This won't fix the underlying vision encoder leak (that needs a vLLM fix), but it
prevents the leak from killing your serving availability in the meantime.

3-minute setup: https://github.com/vgpprasad91/ml-memguard/blob/main/docs/quickstart/vllm.md

I also submitted a PR to add memguard to the vLLM integrations docs: {PR_URL}
```
