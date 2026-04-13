# MKCP Changelog

All changes to the MemGuard KV Cache Protocol specification are documented here.
MKCP follows [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH.

Breaking changes (MAJOR) require a 6-month deprecation window with dual-version
support from the memguard-cloud ingest endpoint.

---

## [1.0.0] — 2026-04-12

Initial stable release of the MemGuard KV Cache Protocol.

### Defined

**8 canonical metric names:**

| Metric | Type | Unit |
|---|---|---|
| `kvcache.utilization` | Gauge | fraction 0.0–1.0 |
| `kvcache.allocation_velocity_mbps` | Gauge | MB/s |
| `kvcache.eviction_rate` | Gauge | evictions/s |
| `kvcache.fragmentation_ratio` | Gauge | fraction 0.0–1.0 |
| `memory.pressure_level` | Gauge | MB above high-watermark |
| `memory.oom_risk_score` | Gauge | probability 0.0–1.0 |
| `memory.near_miss_count` | Counter | allocations |
| `scheduler.preemption_rate` | Gauge | preemptions/s |

**3 event types:**

| Event | Trigger | Traffic effect |
|---|---|---|
| `warn` | `memory.oom_risk_score` ≥ 0.50 or `kvcache.utilization` ≥ 0.85 | None — alert only |
| `shed_load` | `memory.oom_risk_score` ≥ 0.70 (configurable) | Sidecar `/readyz` → 503 |
| `critical_restart` | `VLLMWatchdog` initiates graceful restart | Pod goes NotReady |

**Wire formats:**

- JSON over HTTP POST (`Content-Type: application/json`)
- Protobuf over HTTP POST (`Content-Type: application/x-protobuf`)
- Prometheus exposition text (scrape endpoint; no event support)
- gRPC (`MKCPIngestService`) — optional

**Versioning policy** established:
- MINOR bump for additive changes (new metrics, new optional fields)
- MAJOR bump for renames, removals, or new required fields
- 6-month dual-version support window for MAJOR changes

**Reference implementation:**
- `memory_guard/sidecar.py` — HTTP server, `/readyz` gate, MKCP emitter
- `memory_guard/inference_monitor.py` — `KVCacheMonitor`, signal collection
- `memory_guard/telemetry.py` — `InferenceTelemetry` ↔ MKCP metric mapping

**Schemas:**
- `docs/protocol/mkcp-v1.schema.json` — JSON Schema draft-07
- `docs/protocol/mkcp-v1.proto` — Protobuf 3

### Intentionally deferred to v1.1

- `kvcache.prefix_cache_hit_rate` — SGLang RadixAttention prefix cache;
  will be added as an optional MINOR field once vLLM's prefix cache hit metric
  is stabilised upstream
- `memory.page_fault_rate` — eBPF supplemental signal; present in
  `InferenceTelemetry` but deferred from base MKCP to keep v1 emittable without
  eBPF probes
- Ollama-specific metric mappings — pending Ollama upstream instrumentation

---

*See `docs/protocol/mkcp-v1.md` for the full specification.*
