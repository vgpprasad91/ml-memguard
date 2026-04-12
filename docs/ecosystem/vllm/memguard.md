# memguard

[ml-memguard](https://github.com/vgpprasad91/ml-memguard) is a proactive KV cache monitor
and OOM prevention layer for vLLM. It calculates the largest safe `max_num_seqs` before the
server starts, monitors KV cache utilization in a background thread, and fires load-shed
signals before the engine runs out of memory — without touching vLLM internals or
requiring a GPU or framework restart.

## What it addresses

vLLM's most common OOM failure modes share a root cause: the KV cache is allocated at a
fixed size at startup, but actual memory consumption at runtime depends on the sequence
length distribution of live requests — a variable the operator cannot control. Three
failure modes arise from this:

1. **KV cache exhaustion under high concurrency.** When many long sequences are in flight
   simultaneously, total KV cache demand exceeds the block pool. vLLM raises
   `No available memory for the cache blocks` and the server process exits. This is the
   most-filed vLLM error on GitHub.

2. **Long-sequence burst traffic.** A sudden arrival of requests with 32k–128k context
   lengths causes KV cache demand to spike from safe utilization to exhaustion within a
   few seconds — faster than any polling-based detection can act.

3. **Fragmentation-triggered OOM at moderate utilization.** Block fragmentation means the
   engine can run out of contiguous free blocks at 65–70% reported utilization, making
   the reported `kv_cache_usage_perc` metric an unreliable OOM predictor.

memguard tracks three derived signals that expose all three failure modes:
`kv_velocity_mbps` (rate of change, not just level), `fragmentation_ratio` (contiguous
free block fraction), and `eviction_rate` (scheduler preemption pressure). When any
combination of these signals crosses a threshold, `on_shed_load` fires and your load
balancer stops routing new requests — the server keeps running, existing requests
complete, and no process restart is needed.

## Installation

```bash
pip install ml-memguard[vllm]
```

## Usage

```python
from memory_guard import guard_vllm
from vllm import AsyncLLMEngine, AsyncEngineArgs

args   = AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct", gpu_memory_utilization=0.90)
engine = AsyncLLMEngine.from_engine_args(args)

# Pre-flight: binary-search for the largest safe max_num_seqs
safe = guard_vllm(engine)

# Wire the load-shed signal
safe.monitor.on_shed_load = lambda u: load_balancer.reduce_weight(replica_id, weight=0)
safe.monitor.on_warning   = lambda u: metrics.gauge("vllm.kvcache.util", u)

with safe.monitor.session():
    server.serve_forever()
```

`guard_vllm` reads architecture directly from `engine.model_config.hf_config` — no manual
`hidden_dim` or `num_layers` required.

## Kubernetes sidecar (no code changes to vLLM)

For deployments where modifying the launch script is not possible, a sidecar container
polls vLLM's existing Prometheus `/metrics` endpoint and exposes `/readyz`:

```yaml
containers:
  - name: memguard-sidecar
    image: python:3.11-slim
    command:
      - sh
      - -c
      - |
        pip install 'ml-memguard[cloud]' -q &&
        python -m memory_guard.sidecar \
          --vllm-url http://localhost:8000 \
          --port 8001
    readinessProbe:
      httpGet:
        path: /readyz
        port: 8001
      periodSeconds: 10
      failureThreshold: 1
```

When KV cache exceeds the OOM threshold, `/readyz` returns `503`. Kubernetes removes the
pod from the Service endpoint set — no new traffic, zero vLLM code changes.

## Further reading

- [Quick-start guide (3 minutes)](https://github.com/vgpprasad91/ml-memguard/blob/main/docs/quickstart/vllm.md)
- [PyPI](https://pypi.org/project/ml-memguard/)
- [GitHub](https://github.com/vgpprasad91/ml-memguard)
