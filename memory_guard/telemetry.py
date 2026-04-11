"""Telemetry data structures for inference-signal upload to memguard-cloud.

``InferenceTelemetry`` captures the seven signals that are invisible to
vLLM/SGLang's own Prometheus endpoints but are critical for predicting OOM
events before they happen.  The ``KVCacheMonitor`` populates these on every
monitoring cycle and hands them to ``cloud.upload_inference_telemetry``.

Signals
-------
kv_velocity_mbps:
    Rate of KV-cache growth in MB/s (computed from successive block readings
    multiplied by ``kv_block_size_mb``; stored as blocks/s when block size is
    unknown).  A velocity spike — not sustained high utilization — is the
    earliest OOM predictor.

fragmentation_ratio:
    Fraction of KV cache that is allocated but not part of a contiguous free
    region: ``(total_blocks - max_contiguous_free) / total_blocks``.  High
    fragmentation causes OOM at surprisingly low overall utilization because
    the scheduler cannot find a large-enough contiguous window for a new
    sequence.

eviction_rate:
    Preemption events per second (vLLM/SGLang scheduler evicting in-flight
    sequences to reclaim blocks).  Non-zero eviction_rate at moderate
    utilization is the clearest leading indicator of impending exhaustion.

avg_seq_len:
    Mean token-length of in-flight sequences at the time of polling.  Five
    requests at 128k tokens each can exhaust the same KV cache that handles
    500 requests at 512 tokens — utilization gauges alone cannot distinguish
    these cases.

near_miss_count:
    Number of allocation attempts that succeeded with less than a configurable
    headroom (default: 512 MB).  Invisible without a custom allocator but
    estimable from block-level accounting.

preemption_count:
    Cumulative scheduler preemptions since the last telemetry upload.

weights_mb / kvcache_mb / activations_mb / cuda_ctx_mb:
    Cross-layer GPU memory breakdown.  Required for accurate OOM prediction
    because KV-cache pressure interacts with the fixed weight footprint and
    CUDA context overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InferenceTelemetry:
    """Full payload sent to ``POST /v1/telemetry`` for an inference monitoring cycle.

    All numeric fields default to ``0.0`` / ``0`` so partial uploads are safe
    — the cloud stores whatever the client supplies and treats zero as
    "not measured" rather than "zero pressure".

    Attributes
    ----------
    kv_velocity_mbps:
        KV cache growth rate in MB/s (or blocks/s when block size is unknown).
    fragmentation_ratio:
        Fraction 0.0–1.0 of KV cache that is non-contiguous.
    eviction_rate:
        Scheduler preemptions per second.
    avg_seq_len:
        Mean in-flight sequence length at poll time.
    near_miss_count:
        Allocations that succeeded with < headroom_mb to spare.
    preemption_count:
        Cumulative preemptions since last upload.
    weights_mb:
        GPU memory consumed by model weights.
    kvcache_mb:
        GPU memory consumed by the KV cache.
    activations_mb:
        GPU memory consumed by activation buffers.
    cuda_ctx_mb:
        GPU memory consumed by the CUDA context / driver overhead.
    model_name:
        Serving model identifier (e.g. ``"meta-llama/Llama-3-8B-Instruct"``).
    backend:
        Inference backend string (``"cuda"``, ``"metal"``, ``"cpu"`` …).
    os_platform:
        Platform string (``"linux"``, ``"darwin"`` …).
    """

    kv_velocity_mbps:    float = 0.0
    fragmentation_ratio: float = 0.0
    eviction_rate:       float = 0.0
    avg_seq_len:         float = 0.0
    near_miss_count:     int   = 0
    preemption_count:    int   = 0
    weights_mb:          float = 0.0
    kvcache_mb:          float = 0.0
    activations_mb:      float = 0.0
    cuda_ctx_mb:         float = 0.0
    model_name:          str   = ""
    backend:             str   = ""
    os_platform:         str   = ""

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for the telemetry POST body."""
        return {
            "kv_velocity_mbps":    self.kv_velocity_mbps,
            "fragmentation_ratio": self.fragmentation_ratio,
            "eviction_rate":       self.eviction_rate,
            "avg_seq_len":         self.avg_seq_len,
            "near_miss_count":     self.near_miss_count,
            "preemption_count":    self.preemption_count,
            "weights_mb":          self.weights_mb,
            "kvcache_mb":          self.kvcache_mb,
            "activations_mb":      self.activations_mb,
            "cuda_ctx_mb":         self.cuda_ctx_mb,
            "model_name":          self.model_name,
            "backend":             self.backend,
            "os_platform":         self.os_platform,
            # Required by the runs schema; inference telemetry rows are not OOM events
            "oom_occurred":        0,
        }
