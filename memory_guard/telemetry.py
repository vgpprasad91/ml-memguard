"""Telemetry data structures for inference-signal collection.

``InferenceTelemetry`` captures the seven signals that are invisible to
vLLM/SGLang's own Prometheus endpoints but are critical for predicting OOM
events before they happen.  The ``KVCacheMonitor`` populates these on every
monitoring cycle and passes them to the active backend plugin (if any).

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

cuda_graph_mb:
    CUDA graph reservation block in MB.  vLLM pre-captures CUDA graphs at
    startup; the resulting allocation (typically 2–4 GB on A10G/A100) is
    invisible to KV-utilization gauges.  Snapshotted once at engine startup
    as ``torch.cuda.memory_reserved() − (weights_mb + kvcache_mb)``.
    Zero means "not measured" (non-CUDA backend or PyTorch unavailable).

prefill_peak_activation_mb:
    Peak transient activation allocation observed during the current
    telemetry interval (MB).  Computed as
    ``torch.cuda.memory_allocated() − (weights_mb + kvcache_mb + cuda_graph_mb)``
    whenever KV velocity is positive (prefill is in progress).  On Linux
    without PyTorch access, falls back to the eBPF ``mmap_growth_mb``
    excess over expected KV growth.  Zero means "not measured".
    This is a running max reset after each upload, not an instantaneous
    value, so it reflects the worst-case activation spike in the interval.

max_seq_len_in_flight:
    Approximate peak sequence-length demand in the current batch,
    computed as ``num_running_seqs × avg_prompt_len`` from the vLLM
    ``/metrics`` Prometheus endpoint.  Zero when the endpoint is not
    configured or unreachable.  Long sequences consume disproportionately
    large activation buffers — 1 request at 128 k tokens can spike
    activations by several GB.

total_peak_mb:
    High-water mark of ``torch.cuda.memory_allocated() / 1024**2``
    observed across all poll ticks within the current upload interval.
    Running maximum — reset to 0 after each telemetry upload.  Zero
    means "not measured" (PyTorch unavailable or CUDA absent).
    Numerator for per-interval efficiency scoring:
    ``total_peak_mb / reserved_vram_mb``.

reserved_vram_mb:
    Total physical GPU VRAM in MB, from
    ``torch.cuda.get_device_properties(0).total_memory / 1024**2``.
    Snapshotted once at engine startup.  Static across the lifetime of
    the monitor.  Denominator for efficiency scoring.  Zero means
    "not measured" (PyTorch unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InferenceTelemetry:
    """Inference monitoring signals collected during one poll cycle.

    All numeric fields default to ``0.0`` / ``0`` so partial records are safe
    — the backend plugin treats zero as "not measured" rather than "zero pressure".

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
    cuda_graph_mb:
        CUDA graph reservation block in MB (snapshotted once at startup).
        Zero when not measured.
    prefill_peak_activation_mb:
        Peak transient activation allocation observed this interval (MB).
        Running max, reset after each telemetry upload. Zero = not measured.
    max_seq_len_in_flight:
        Proxy for peak activation demand: num_running_seqs × avg_prompt_len.
        Zero when vLLM /metrics endpoint is not configured.
    total_peak_mb:
        High-water mark of torch.cuda.memory_allocated() within this interval (MB).
        Running max, reset after each upload. Zero = not measured.
    reserved_vram_mb:
        Total physical GPU VRAM in MB from torch.cuda.get_device_properties(0).
        Static across the monitor lifetime. Zero = not measured.
    model_name:
        Serving model identifier (e.g. ``"meta-llama/Llama-3-8B-Instruct"``).
    backend:
        Inference backend string (``"cuda"``, ``"metal"``, ``"cpu"`` …).
    os_platform:
        Platform string (``"linux"``, ``"darwin"`` …).
    device_count:
        Number of visible CUDA devices at monitor startup.  1 = single-GPU
        (default and fallback when PyTorch is unavailable).  N > 1 indicates
        a tensor-parallel pod where ``reserved_vram_mb`` is the *sum* across
        all N devices (e.g., 4 × 24,576 = 98,304 MB for a 4×A10G node).
        Used by the efficiency engine to look up the correct multi-GPU catalog
        SKU rather than matching against single-GPU entries.
    """

    kv_velocity_mbps:    float = 0.0
    fragmentation_ratio: float = 0.0
    eviction_rate:       float = 0.0
    avg_seq_len:         float = 0.0
    near_miss_count:     int   = 0
    preemption_count:    int   = 0
    weights_mb:           float = 0.0
    kvcache_mb:           float = 0.0
    activations_mb:       float = 0.0
    cuda_ctx_mb:          float = 0.0
    cuda_graph_mb:                float = 0.0
    prefill_peak_activation_mb:   float = 0.0
    max_seq_len_in_flight:        int   = 0
    total_peak_mb:                float = 0.0
    reserved_vram_mb:             float = 0.0
    model_name:                   str   = ""
    backend:              str   = ""
    os_platform:          str   = ""
    # eBPF signals (PR 56) — populated when an ebpf_session is active;
    # zero means "not measured" (no eBPF), not "zero pressure".
    memory_pressure_level: float = 0.0
    page_fault_rate:       float = 0.0
    # PR 72: number of visible CUDA devices (tensor-parallel pool size).
    # 1 = single-GPU (default); N > 1 = multi-GPU NVLink/SXM pod.
    # Combined reserved_vram_mb = per-device VRAM × device_count.
    device_count:          int   = 1

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for the telemetry POST body."""
        return {
            "kv_velocity_mbps":     self.kv_velocity_mbps,
            "fragmentation_ratio":  self.fragmentation_ratio,
            "eviction_rate":        self.eviction_rate,
            "avg_seq_len":          self.avg_seq_len,
            "near_miss_count":      self.near_miss_count,
            "preemption_count":     self.preemption_count,
            "weights_mb":           self.weights_mb,
            "kvcache_mb":           self.kvcache_mb,
            "activations_mb":       self.activations_mb,
            "cuda_ctx_mb":          self.cuda_ctx_mb,
            "cuda_graph_mb":               self.cuda_graph_mb,
            "prefill_peak_activation_mb":  self.prefill_peak_activation_mb,
            "max_seq_len_in_flight":       self.max_seq_len_in_flight,
            "total_peak_mb":               self.total_peak_mb,
            "reserved_vram_mb":            self.reserved_vram_mb,
            "model_name":                  self.model_name,
            "backend":              self.backend,
            "os_platform":          self.os_platform,
            "memory_pressure_level": self.memory_pressure_level,
            "page_fault_rate":      self.page_fault_rate,
            # PR 72: multi-GPU topology — 1 for single-GPU pods
            "device_count":         self.device_count,
            # Required by the runs schema; inference telemetry rows are not OOM events
            "oom_occurred":         0,
        }
