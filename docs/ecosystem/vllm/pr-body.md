# PR Body: Add memguard to vLLM integrations docs

> **Target repo:** https://github.com/vllm-project/vllm
> **Target file:** `docs/deployment/integrations/memguard.md` (new file)
> **Target branch:** `main`
>
> Copy `memguard.md` from this directory into `docs/deployment/integrations/` in a fork of
> the vLLM repo, then open a PR with the title and body below.

---

## PR Title

```
docs: add memguard integration — proactive KV cache monitoring and OOM prevention
```

---

## PR Body

### What this adds

Adds a new integration page for [ml-memguard](https://github.com/vgpprasad91/ml-memguard),
an open-source library that provides proactive KV cache monitoring and OOM prevention for
vLLM, under `docs/deployment/integrations/memguard.md`.

### Why

`No available memory for the cache blocks` is vLLM's most-filed error class. Three
distinct failure modes cause it — KV cache exhaustion at high concurrency, burst
long-sequence traffic, and fragmentation-triggered OOM at moderate utilization — and they
share no common static-threshold fix. memguard addresses all three by tracking velocity,
fragmentation ratio, and eviction rate alongside raw utilization, then firing a
`on_shed_load` callback *before* the engine crashes so the load balancer can act without
a process restart.

The integration requires two lines of Python added to an existing vLLM launch script, or
zero lines if deployed as a Kubernetes sidecar that polls vLLM's existing `/metrics`
endpoint.

### Checklist

- [x] New file only — no existing vLLM source files modified
- [x] Follows the same structure as existing integration pages (h1 title, linked
      description, installation, code snippet, further reading)
- [x] Tested against vLLM ≥ 0.4 (the `ml-memguard[vllm]` extra pins this requirement)
- [x] No new dependencies added to vLLM itself

### Related issues

Directly addresses the failure mode described in:
- #27934 — V1 Engine memory allocation failures on RTX 3060 12GB
- #28230 — GPU VRAM continuously increases during Qwen3-VL usage until OOM

---

## Checklist before opening the PR

1. Fork https://github.com/vllm-project/vllm
2. Create branch: `git checkout -b docs/memguard-integration`
3. Copy `memguard.md` → `docs/deployment/integrations/memguard.md`
4. Verify the docs build locally: `cd docs && pip install -r requirements.txt && make html`
5. Open PR against `vllm-project/vllm:main` with the title and body above
6. After PR is open, post the replies in `issue-replies.md` to issues #27934 and #28230,
   linking to the PR
