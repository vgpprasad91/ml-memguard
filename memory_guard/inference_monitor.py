"""KV cache utilization monitor for inference serving workloads.

Polls a caller-supplied function every ``poll_interval`` seconds and fires
callbacks when KV cache utilization crosses the warning (80 %) and
shed-load (92 %) thresholds.  The monitor is framework-agnostic — the
caller provides ``poll_fn``; this module never imports vLLM or SGLang.

Usage::

    from memory_guard.inference_monitor import KVCacheMonitor

    def poll():
        # vLLM example:
        bm = engine.scheduler.block_manager
        used = bm.get_num_used_gpu_blocks()    # or total - free
        total = bm.get_num_total_gpu_blocks()
        return used, total

    monitor = KVCacheMonitor(
        poll_fn=poll,
        on_warning=lambda u: logger.warning("KV cache %.0f%%", u * 100),
        on_shed_load=lambda u: load_balancer.reduce_concurrency(),
    )

    with monitor.session() as mon:
        server.serve_forever()
        # mon.current_utilization and mon.utilization_history are available
        # throughout; the monitor never touches the engine.
"""

from __future__ import annotations

import collections
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Optional

from .constants import (
    KV_CACHE_SHED_LOAD_THRESHOLD,
    KV_CACHE_WARNING_THRESHOLD,
    MONITOR_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)

# Default history window — matches RuntimeMonitor
_HISTORY_SIZE = 60

# Default: upload inference telemetry every 30 s (6 × 5 s poll ticks)
_DEFAULT_TELEMETRY_INTERVAL = 30.0


class KVCacheMonitor:
    """Background-thread KV cache utilization monitor.

    ``poll_fn`` is the sole coupling point to the serving framework.  It
    must be a zero-argument callable that returns ``(used_blocks, total_blocks)``
    as plain integers and must be thread-safe (it is called from a daemon
    thread, not the main thread).

    Thresholds:
        - ``THRESHOLD_WARNING``   (80 %): fires ``on_warning``
        - ``THRESHOLD_SHED_LOAD`` (92 %): fires ``on_shed_load``

    Each callback is subject to a per-level cooldown so it fires at most
    once per ``cooldown_seconds``, preventing log-spam during sustained
    pressure.  The shed-load threshold takes priority: when utilization
    exceeds 92 %, only ``on_shed_load`` fires (not ``on_warning``).

    The monitor never writes to the engine — callbacks decide what to do
    with the signal (log, alert load balancer, etc.).
    """

    THRESHOLD_WARNING: float = KV_CACHE_WARNING_THRESHOLD
    THRESHOLD_SHED_LOAD: float = KV_CACHE_SHED_LOAD_THRESHOLD

    def __init__(
        self,
        poll_fn: Callable[[], tuple[int, int]],
        poll_interval: float = MONITOR_POLL_INTERVAL,
        on_warning: Optional[Callable[[float], None]] = None,
        on_shed_load: Optional[Callable[[float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cooldown_seconds: float = 30.0,
        history_size: int = _HISTORY_SIZE,
        critical_threshold: float = 0.95,
        restart_callback: Optional[Callable[[], None]] = None,
        critical_ticks: int = 3,
        # --- Inference telemetry parameters (PR 23) ---
        kv_block_size_mb: float = 0.0,
        extended_poll_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        telemetry_upload_interval: float = _DEFAULT_TELEMETRY_INTERVAL,
        telemetry_model_name: str = "",
        telemetry_backend: str = "",
        telemetry_os_platform: str = "",
        # --- eBPF cgroup probe (PR 35) ---
        use_ebpf: bool = False,
        # --- eBPF session (PR 56) — page-fault + mmap probes ---
        ebpf_session: Optional[Any] = None,
        # --- Prefill activation spike detection (PR 65) ---
        prefill_spike_threshold_mb: float = 512.0,
        vllm_metrics_url: str = "",
        # --- Source baseline (PR 67) ---
        source_id: str = "",
        total_vram_mb: float = 0.0,
    ) -> None:
        """
        Args:
            poll_fn:            Zero-argument callable returning
                                ``(used_blocks, total_blocks)``.  Called from
                                the background thread — must be thread-safe.
            poll_interval:      Seconds between polls (default 5 s).
            on_warning:         ``Callable[[float], None]`` fired when
                                utilization ≥ THRESHOLD_WARNING (80 %).
                                Receives the utilization value (0.0–1.0).
                                Called at most once per ``cooldown_seconds``.
            on_shed_load:       ``Callable[[float], None]`` fired when
                                utilization ≥ THRESHOLD_SHED_LOAD (92 %).
                                Receives the utilization value (0.0–1.0).
                                Called at most once per ``cooldown_seconds``.
                                Takes priority over ``on_warning`` at 92 %+.
            on_log:             Callback for human-readable log messages
                                (default: ``logger.warning``).
            cooldown_seconds:   Minimum seconds between consecutive firings
                                of the same callback level.
            history_size:       Maximum number of utilization readings to
                                retain in ``utilization_history`` (default 60).
            critical_threshold: KV cache usage fraction above which a planned
                                graceful restart is triggered (default 0.95).
                                When utilization stays at or above this level
                                for ``critical_ticks`` consecutive poll ticks,
                                ``restart_callback`` is invoked and the
                                consecutive counter resets.
            restart_callback:   Zero-argument callable invoked when utilization
                                exceeds ``critical_threshold`` for
                                ``critical_ticks`` consecutive ticks.  The
                                caller wires this to the process supervisor
                                (e.g. ``VLLMWatchdog.stop`` + relaunch).
                                ``None`` disables the planned-restart feature.
            critical_ticks:     Number of consecutive poll ticks above
                                ``critical_threshold`` required before
                                ``restart_callback`` fires (default 3).
                                Prevents a single transient spike from
                                triggering an unnecessary restart.
            kv_block_size_mb:   Size of a single KV cache block in MB.
                                When non-zero, ``kv_velocity_mbps`` is
                                reported in true MB/s; otherwise it is
                                stored as blocks/s.  Obtain from
                                ``engine.cache_config.block_size`` in
                                vLLM (default 0 = unknown).
            extended_poll_fn:   Optional zero-argument callable that
                                returns a ``dict`` with any subset of the
                                keys: ``fragmentation_ratio``,
                                ``eviction_rate``, ``avg_seq_len``,
                                ``near_miss_count``, ``preemption_count``,
                                ``weights_mb``, ``kvcache_mb``,
                                ``activations_mb``, ``cuda_ctx_mb``.
                                Called on each monitoring tick; missing
                                keys default to ``0.0``.  Must be
                                thread-safe.
            telemetry_upload_interval:
                                Seconds between cloud telemetry uploads
                                (default 30 s).  Independent of
                                ``poll_interval``.
            telemetry_model_name:
                                Model identifier stored in the telemetry
                                row (e.g. ``"meta-llama/Llama-3-8B"``).
            telemetry_backend:  Backend string (``"cuda"``, ``"metal"`` …).
            telemetry_os_platform:
                                OS platform string (``"linux"`` …).
            ebpf_session:       Optional :class:`~memory_guard.ebpf.MemguardBPFSession`
                                (or duck-typed object with ``available``,
                                ``page_fault_rate``, ``mmap_growth_mbps``, and
                                ``memory_pressure_bytes`` properties).  When
                                provided and ``session.available`` is ``True``,
                                BPF-derived ``page_fault_rate`` and
                                ``memory_pressure_level`` are merged into each
                                :class:`~memory_guard.telemetry.InferenceTelemetry`
                                snapshot.  Zero behavior change when ``None``.
        """
        self.poll_fn = poll_fn
        self.poll_interval = poll_interval
        self.on_warning = on_warning
        self.on_shed_load = on_shed_load
        self.on_log = on_log or (lambda msg: logger.warning(msg))
        self.cooldown_seconds = cooldown_seconds
        self.critical_threshold: float = critical_threshold
        self.restart_callback: Optional[Callable[[], None]] = restart_callback
        self.critical_ticks: int = critical_ticks

        # Inference telemetry state
        self._kv_block_size_mb: float = max(0.0, kv_block_size_mb)
        self._extended_poll_fn: Optional[Callable[[], Dict[str, Any]]] = extended_poll_fn
        self._telemetry_upload_interval: float = max(1.0, telemetry_upload_interval)
        self._telemetry_model_name: str = telemetry_model_name
        self._telemetry_backend: str = telemetry_backend
        self._telemetry_os_platform: str = telemetry_os_platform
        # Velocity tracking
        self._prev_used_blocks: Optional[int] = None
        self._prev_poll_time: float = 0.0
        self._last_telemetry_upload: float = 0.0
        # Predictive OOM state (PR 24)
        # Separate cooldown so predictive restart doesn't loop tightly.
        self._last_predictive_restart: float = 0.0
        _PREDICTIVE_RESTART_COOLDOWN = 120.0   # module-level constant for tests
        self._predictive_restart_cooldown: float = _PREDICTIVE_RESTART_COOLDOWN

        # CUDA graph memory baseline (PR 64) — snapshotted once at start()
        self._cuda_graph_baseline_mb: float = 0.0

        # Prefill activation spike detection (PR 65)
        self._prefill_spike_threshold_mb: float = max(0.0, prefill_spike_threshold_mb)
        self._vllm_metrics_url: str             = vllm_metrics_url
        # Running max over the current telemetry interval; reset after each upload
        self._prefill_peak_activation_mb: float = 0.0
        # Latest value from vLLM /metrics scrape
        self._max_seq_len_in_flight: int        = 0

        # Source baseline (PR 67) — posted once at start()
        self._source_id: str            = source_id
        self._total_vram_mb: float      = max(0.0, total_vram_mb)
        # PR 68: physical GPU VRAM — sum across all visible devices (PR 72).
        # snapshotted once at start(); falls back to _total_vram_mb when torch absent.
        self._reserved_vram_mb: float   = 0.0
        # PR 72: number of visible CUDA devices at start(); 1 on single-GPU / non-CUDA.
        self._device_count: int         = 1
        # PR 68: running max of torch.cuda.memory_allocated() across poll ticks;
        # reset to 0 after each telemetry upload.
        self._total_peak_mb: float      = 0.0
        # Last true_available_headroom_mb returned by /v1/predict.
        # inf = "not yet received"; used by sidecar /readyz headroom gate.
        self._last_true_available_headroom_mb: float = float("inf")

        # eBPF state (PR 35)
        self._use_ebpf: bool = use_ebpf
        self._ebpf_wake: threading.Event = threading.Event()
        self._ebpf_manager: Optional[object] = None  # EBPFProbeManager
        # eBPF session (PR 56) — page-fault + mmap probes
        self._ebpf_session: Optional[Any] = ebpf_session

        self._history: collections.deque[float] = collections.deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_warning_time: float = 0.0
        self._last_shed_load_time: float = 0.0
        self._critical_consecutive: int = 0
        # PR 31: last cloud-predicted OOM probability (0.0 = no prediction yet)
        self._last_oom_probability: float = 0.0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_utilization(self) -> float:
        """Latest KV cache utilization reading (0.0–1.0).

        Returns 0.0 if no readings have been taken yet.
        """
        with self._lock:
            return self._history[-1] if self._history else 0.0

    @property
    def utilization_history(self) -> list[float]:
        """All retained utilization readings, oldest first (up to history_size)."""
        with self._lock:
            return list(self._history)

    @property
    def is_running(self) -> bool:
        """True while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_oom_probability(self) -> float:
        """Most recent OOM probability returned by the cloud predict API.

        Returns 0.0 until the first successful ``POST /v1/predict`` call.
        Read by the sidecar ``/readyz`` endpoint to drive Kubernetes
        readiness-gate load-shedding (PR 31).
        """
        return self._last_oom_probability

    @property
    def last_true_available_headroom_mb(self) -> float:
        """Most recent true available headroom (MB) returned by ``/v1/predict``.

        Returns ``float('inf')`` until the first successful predict call that
        includes source baseline data.  Read by the sidecar ``/readyz``
        endpoint to trip 503 when headroom falls below the configured
        ``headroom_threshold_mb`` — catching weight/overhead OOMs that the
        probabilistic model may not flag in time (PR 67).
        """
        return self._last_true_available_headroom_mb

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start polling in a daemon background thread.

        If the monitor is already running, the existing thread is stopped
        cleanly before the new one starts (prevents orphaned threads).
        """
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        with self._lock:
            self._history.clear()
            self._last_warning_time = 0.0
            self._last_shed_load_time = 0.0
            self._critical_consecutive = 0
        self._prev_used_blocks = None
        self._prev_poll_time = 0.0
        self._last_telemetry_upload = 0.0
        self._last_predictive_restart = 0.0
        self._last_oom_probability = 0.0
        self._prefill_peak_activation_mb = 0.0
        self._max_seq_len_in_flight = 0
        self._last_true_available_headroom_mb = float("inf")
        self._total_peak_mb = 0.0
        self._stop.clear()
        self._ebpf_wake.clear()

        # Snapshot CUDA graph reservation once at startup (best-effort)
        self._snapshot_cuda_graph_baseline()

        # PR 68: snapshot total physical VRAM once at startup (best-effort)
        self._snapshot_reserved_vram()

        # PR 67: POST the startup memory footprint to the cloud (best-effort)
        self._post_source_baseline()

        # Load eBPF probes when requested (Linux + bcc only; silently skips otherwise)
        if self._use_ebpf:
            self._start_ebpf()

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="kv-cache-monitor"
        )
        self._thread.start()
        try:
            self.on_log(
                f"[memory-guard] KVCacheMonitor started: poll={self.poll_interval}s, "
                f"warning={self.THRESHOLD_WARNING:.0%}, "
                f"shed_load={self.THRESHOLD_SHED_LOAD:.0%}"
            )
        except Exception:
            logger.debug("on_log raised during KVCacheMonitor.start", exc_info=True)

    def stop(self) -> None:
        """Signal the background thread to stop and join it."""
        self._stop.set()
        self._ebpf_wake.set()  # unblock _ebpf_wake.wait() immediately
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_interval + 2)
        self._thread = None

        # Stop eBPF manager if running
        if self._ebpf_manager is not None:
            try:
                self._ebpf_manager.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            self._ebpf_manager = None

    def session(self) -> "_KVCacheSession":
        """Return a context manager that starts and stops this monitor."""
        return _KVCacheSession(self)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            # --- poll ---------------------------------------------------
            try:
                used, total = self.poll_fn()
                utilization = used / total if total > 0 else 0.0
            except Exception as exc:
                logger.debug("KVCacheMonitor poll_fn raised: %s", exc)
                self._stop.wait(self.poll_interval)
                continue

            now = time.time()

            # --- velocity (delta used-blocks / elapsed seconds → MB/s) --
            kv_velocity = self._compute_velocity(used, now)

            # --- prefill activation spike (PR 65) -----------------------
            self._update_prefill_signals(kv_velocity)

            # --- record -------------------------------------------------
            with self._lock:
                self._history.append(utilization)
                warn_ready = (now - self._last_warning_time) >= self.cooldown_seconds
                shed_ready = (now - self._last_shed_load_time) >= self.cooldown_seconds

            # --- predictive OOM check (PR 24) — runs before reactive checks
            # so that when the predictive path fires on_shed_load it updates
            # _last_shed_load_time and the reactive path won't double-fire
            # within the cooldown window.
            self._run_predict_oom(kv_velocity, utilization, shed_ready)

            # --- critical threshold (consecutive tick counting) ----------
            if utilization >= self.critical_threshold:
                self._critical_consecutive += 1
                if (self._critical_consecutive >= self.critical_ticks
                        and self.restart_callback is not None):
                    self._emit_log(
                        f"[memory-guard] KV cache critical: {utilization:.1%} \u2265 "
                        f"{self.critical_threshold:.0%} for "
                        f"{self._critical_consecutive} consecutive ticks "
                        f"\u2014 triggering planned graceful restart"
                    )
                    self._fire_restart()
                    self._critical_consecutive = 0
            else:
                self._critical_consecutive = 0

            # --- dispatch callbacks (outside lock) ----------------------
            if utilization >= self.THRESHOLD_SHED_LOAD and shed_ready:
                with self._lock:
                    self._last_shed_load_time = time.time()
                self._emit_log(
                    f"[memory-guard] KV cache shed-load: "
                    f"{utilization:.1%} \u2265 {self.THRESHOLD_SHED_LOAD:.0%}"
                )
                self._fire(self.on_shed_load, utilization, "on_shed_load")

            elif utilization >= self.THRESHOLD_WARNING and warn_ready:
                with self._lock:
                    self._last_warning_time = time.time()
                self._emit_log(
                    f"[memory-guard] KV cache warning: "
                    f"{utilization:.1%} \u2265 {self.THRESHOLD_WARNING:.0%}"
                )
                self._fire(self.on_warning, utilization, "on_warning")

            # --- inference telemetry upload (every N seconds) -----------
            if (now - self._last_telemetry_upload) >= self._telemetry_upload_interval:
                self._upload_inference_telemetry(kv_velocity)
                self._last_telemetry_upload = now

            # Wait up to poll_interval; eBPF events short-circuit the wait
            self._ebpf_wake.wait(self.poll_interval)
            self._ebpf_wake.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_ebpf(self) -> None:
        """Load and start the EBPFProbeManager.  Silently skips on failure."""
        try:
            from .ebpf import EBPFProbeManager

            def _on_high(event: object) -> None:
                logger.debug("[memory-guard] eBPF memory.high: %s", event)

            def _on_oom(event: object) -> None:
                logger.warning("[memory-guard] eBPF OOM killer invoked: %s", event)

            mgr = EBPFProbeManager(
                on_high=_on_high,
                on_oom=_on_oom,
                ebpf_wake=self._ebpf_wake,
            )
            mgr.load()
            mgr.start()
            self._ebpf_manager = mgr
            logger.debug("[memory-guard] EBPFProbeManager loaded and started")
        except (ImportError, PermissionError, OSError) as exc:
            logger.debug(
                "[memory-guard] eBPF probes unavailable (%s: %s) — "
                "falling back to poll-based detection",
                type(exc).__name__,
                exc,
            )

    def _snapshot_cuda_graph_baseline(self) -> None:
        """Snapshot the CUDA graph reservation block at engine startup.

        Estimates the fixed overhead reserved by vLLM's CUDA graph capture
        (typically 2–4 GB on A10G/A100) as::

            reserved_mb − weights_mb − kvcache_mb

        The result is stored in ``_cuda_graph_baseline_mb`` and included as a
        static feature in every telemetry upload and OOM predict call.

        Best-effort: silently skips if PyTorch is unavailable, CUDA is absent,
        or the extended_poll_fn raises.  Zero means "not measured".
        """
        try:
            import torch  # type: ignore[import]
            if not torch.cuda.is_available():
                return
            reserved_mb = torch.cuda.memory_reserved() / (1024.0 * 1024.0)

            extra: Dict[str, Any] = {}
            if self._extended_poll_fn is not None:
                try:
                    extra = self._extended_poll_fn() or {}
                except Exception:
                    pass

            weights_mb  = float(extra.get("weights_mb",  0.0))
            kvcache_mb  = float(extra.get("kvcache_mb",  0.0))
            baseline_mb = reserved_mb - weights_mb - kvcache_mb
            self._cuda_graph_baseline_mb = max(0.0, baseline_mb)
            logger.debug(
                "[memory-guard] CUDA graph baseline: %.1f MB "
                "(reserved=%.1f MB, weights=%.1f MB, kvcache=%.1f MB)",
                self._cuda_graph_baseline_mb, reserved_mb, weights_mb, kvcache_mb,
            )
        except Exception as exc:
            logger.debug("[memory-guard] CUDA graph snapshot skipped: %s", exc)

    def _snapshot_reserved_vram(self) -> None:
        """Snapshot total physical GPU VRAM across *all* visible CUDA devices.

        Stored in ``_reserved_vram_mb`` and emitted in every telemetry upload
        as the denominator for efficiency scoring.  The number of devices is
        stored in ``_device_count`` and forwarded in the source baseline POST
        so the Worker can match the combined pool against multi-GPU catalog
        entries (e.g., 4×A10G = 98,304 MB).

        PR 72 fix: sum ``total_memory`` across ``torch.cuda.device_count()``
        devices instead of reading device 0 only.  On a 4×A10G node the
        previous implementation recorded 24,576 MB (one card) rather than
        98,304 MB (the full tensor-parallel pool), causing the efficiency
        engine to match the single-A10G catalog tier and miss the correct
        4×A10G → 2×A10G recommendation.

        Falls back to the ``total_vram_mb`` constructor param when PyTorch or
        CUDA is unavailable.  If neither is available, stays at 0.0 /
        device_count stays at 1.
        """
        try:
            import torch  # type: ignore[import]
            n = max(1, torch.cuda.device_count())
            total_bytes = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(n)
            )
            self._reserved_vram_mb = total_bytes / (1024.0 * 1024.0)
            self._device_count     = n
            logger.debug(
                "[memory-guard] reserved_vram_mb: %.0f MB (%d device(s): %s)",
                self._reserved_vram_mb,
                n,
                ", ".join(
                    getattr(torch.cuda.get_device_properties(i), "name", "?")
                    for i in range(n)
                ),
            )
        except Exception:
            # Fall back to user-provided value (may be 0.0 when not set)
            self._reserved_vram_mb = self._total_vram_mb
            self._device_count     = 1

    def _update_prefill_signals(self, kv_velocity: float) -> None:
        """Measure prefill activation spike on each poll tick.

        Updates ``_prefill_peak_activation_mb`` (running max over the current
        telemetry upload interval, reset to 0 after each upload) and
        ``_max_seq_len_in_flight`` (latest value from vLLM /metrics).

        Primary path:
            Sum of ``torch.cuda.memory_allocated(i)`` across all
            ``_device_count`` devices, minus
            ``(weights_mb + kvcache_mb + cuda_graph_mb)``.
            The sum is required for multi-GPU tensor-parallel pods where
            allocation is distributed across devices; a single-device fast
            path (``_device_count == 1``) skips the loop.
            Only the prefill spike subtraction is gated on ``kv_velocity > 0``
            (KV cache growing ⟹ prefill in progress).  The pool-wide
            ``allocated_mb`` is recorded unconditionally as the interval peak
            so it matches the ``reserved_vram_mb`` denominator used for
            efficiency scoring.

            Assumption for ``known_mb``: ``extended_poll_fn`` is expected to
            return *pool-wide* totals for multi-GPU deployments (e.g. vLLM
            reports total KV-cache blocks across the tensor-parallel group,
            not per-shard).  If a custom ``extended_poll_fn`` returns
            per-device values instead, multiply each component by
            ``self._device_count`` before the subtraction.

        eBPF fallback (Linux, no PyTorch):
            If ``mmap_growth_mb`` in this poll interval exceeds the expected
            KV growth by more than ``prefill_spike_threshold_mb``, the excess
            is recorded as the activation spike.

        Both paths are best-effort and silently skip on any exception.
        """
        try:
            extra: Dict[str, Any] = {}
            if self._extended_poll_fn is not None:
                try:
                    extra = self._extended_poll_fn() or {}
                except Exception:
                    pass

            weights_mb    = float(extra.get("weights_mb",    0.0))
            kvcache_mb    = float(extra.get("kvcache_mb",    0.0))
            cuda_graph_mb = float(extra.get("cuda_graph_mb", self._cuda_graph_baseline_mb))

            prefill_mb   = 0.0
            torch_ok     = False

            # --- primary: pool-wide torch.cuda.memory_allocated() -----------
            try:
                import torch  # type: ignore[import]
                if torch.cuda.is_available():
                    # PR 74: sum across all visible devices so allocated_mb is
                    # pool-wide and matches the reserved_vram_mb denominator
                    # (PR 72).  Single-GPU fast path avoids the loop overhead.
                    if self._device_count > 1:
                        allocated_mb = sum(
                            torch.cuda.memory_allocated(i)
                            for i in range(self._device_count)
                        ) / (1024.0 * 1024.0)
                    else:
                        allocated_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)

                    # PR 68: unconditionally update the interval peak (all phases)
                    if allocated_mb > self._total_peak_mb:
                        self._total_peak_mb = allocated_mb
                    if kv_velocity > 0:
                        # known_mb assumes extended_poll_fn returns pool-wide
                        # totals — see docstring for per-device caveat.
                        known_mb   = weights_mb + kvcache_mb + cuda_graph_mb
                        prefill_mb = max(0.0, allocated_mb - known_mb)
                    torch_ok = True
            except Exception:
                pass

            # --- fallback: eBPF mmap_growth excess -------------------------
            if not torch_ok and (
                self._ebpf_session is not None
                and getattr(self._ebpf_session, "available", False)
            ):
                mmap_mbps      = float(getattr(self._ebpf_session, "mmap_growth_mbps", 0.0))
                mmap_growth_mb = mmap_mbps * self.poll_interval
                expected_kv_mb = max(0.0, kv_velocity * self.poll_interval)
                excess_mb      = mmap_growth_mb - expected_kv_mb
                if excess_mb > self._prefill_spike_threshold_mb:
                    prefill_mb = excess_mb

            # Keep running max (reset to 0 after each telemetry upload)
            if prefill_mb > self._prefill_peak_activation_mb:
                self._prefill_peak_activation_mb = prefill_mb
                logger.debug(
                    "[memory-guard] prefill spike: %.1f MB (velocity=%.2f MB/s %s)",
                    prefill_mb, kv_velocity, "torch" if torch_ok else "ebpf",
                )

            # --- max_seq_len_in_flight from vLLM /metrics ------------------
            if self._vllm_metrics_url:
                seq_len = self._fetch_max_seq_len_in_flight()
                if seq_len > 0:
                    self._max_seq_len_in_flight = seq_len

        except Exception as exc:
            logger.debug("[memory-guard] _update_prefill_signals raised: %s", exc)

    def _fetch_max_seq_len_in_flight(self) -> int:
        """Scrape vLLM /metrics for ``num_running_seqs × avg_prompt_len``.

        Parses the Prometheus text exposition format emitted by the vLLM
        OpenAI-compatible server at ``/metrics``.  Returns 0 on any failure
        (connection refused, timeout, parse error) — callers must treat 0 as
        "not measured".

        The product of running sequences × average prompt length is a proxy
        for peak activation memory demand: a single 128k-token request can
        spike activations by several GB while prefill computes attention.
        """
        try:
            import urllib.request
            with urllib.request.urlopen(self._vllm_metrics_url, timeout=2) as resp:
                text = resp.read().decode("utf-8", errors="replace")

            num_running: float = 0.0
            avg_prompt:  float = 0.0
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Prometheus text: metric_name{labels} value [timestamp]
                # We match by prefix (labels may vary)
                parts = line.split()
                if len(parts) < 2:
                    continue
                metric_part = parts[0]
                value_str   = parts[1]   # value is always index 1
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                if metric_part.startswith("vllm:num_running_seqs"):
                    num_running = value
                elif metric_part.startswith("vllm:avg_prompt_len"):
                    avg_prompt = value

            return int(num_running * avg_prompt)
        except Exception as exc:
            logger.debug("[memory-guard] _fetch_max_seq_len_in_flight: %s", exc)
            return 0

    def _run_predict_oom(
        self,
        kv_velocity: float,
        utilization: float,
        shed_ready: bool,
    ) -> None:
        """Call POST /v1/predict and fire on_shed_load or restart_callback.

        Falls back silently to existing rule-based thresholds when the cloud
        API is unavailable (predict_oom returns None).

        Probability → action mapping (mirrors the Worker):
          p > 0.92 → planned restart (if restart_callback set, with 120 s cooldown)
          p > 0.70 → on_shed_load  (respects the normal per-level cooldown)
          p ≤ 0.70 → no action (let reactive thresholds decide)
        """
        try:
            from .backends import predict_oom as _predict_oom

            extra: Dict[str, Any] = {}
            if self._extended_poll_fn is not None:
                try:
                    extra = self._extended_poll_fn() or {}
                except Exception as exc:
                    logger.debug("extended_poll_fn raised in predict: %s", exc)

            signals: Dict[str, Any] = {
                "kv_velocity_mbps":    kv_velocity,
                "fragmentation_ratio": float(extra.get("fragmentation_ratio", 0.0)),
                "eviction_rate":       float(extra.get("eviction_rate", 0.0)),
                "avg_seq_len":         float(extra.get("avg_seq_len", 0.0)),
                "near_miss_count":     int(extra.get("near_miss_count", 0)),
                "preemption_count":    int(extra.get("preemption_count", 0)),
                "weights_mb":                 float(extra.get("weights_mb", 0.0)),
                "kvcache_mb":                 float(extra.get("kvcache_mb", 0.0)),
                "cuda_graph_mb":              float(extra.get("cuda_graph_mb", self._cuda_graph_baseline_mb)),
                "prefill_peak_activation_mb": self._prefill_peak_activation_mb,
                "max_seq_len_in_flight":      self._max_seq_len_in_flight,
                # PR 67: source_id routes the Worker to the correct baseline row
                "source_id": self._source_id,
            }

            # Merge BPF-derived signals into the predict payload (PR 56)
            if (
                self._ebpf_session is not None
                and getattr(self._ebpf_session, "available", False)
            ):
                signals["page_fault_rate"] = float(
                    getattr(self._ebpf_session, "page_fault_rate", 0.0)
                )
                signals["memory_pressure_level"] = (
                    float(getattr(self._ebpf_session, "memory_pressure_bytes", 0.0))
                    / (1024 * 1024)
                )

            result = _predict_oom(
                signals,
                model_name=self._telemetry_model_name,
                backend_str=self._telemetry_backend,
            )

            if result is None:
                # Cloud unavailable — fall through to reactive thresholds (no-op here)
                return

            p            = float(result.get("oom_probability", 0.0))
            horizon      = result.get("horizon_seconds", "?")
            model_source = result.get("model_source", "unknown")

            # PR 31: expose for sidecar /readyz probe
            self._last_oom_probability = p

            # PR 67: cache headroom for sidecar /readyz headroom gate
            headroom = result.get("true_available_headroom_mb")
            if headroom is not None:
                self._last_true_available_headroom_mb = float(headroom)

            logger.debug(
                "[memory-guard] predict_oom: p=%.3f source=%s horizon=%s",
                p, model_source, horizon,
            )

            if p > 0.92 and self.restart_callback is not None:
                now = time.time()
                if (now - self._last_predictive_restart) >= self._predictive_restart_cooldown:
                    self._emit_log(
                        f"[memory-guard] Predictive OOM p={p:.2f} ≥ 0.92 "
                        f"[{model_source}] → planned restart (horizon ≈ {horizon}s)"
                    )
                    self._last_predictive_restart = now
                    self._fire_restart()

            elif p > 0.70 and shed_ready:
                with self._lock:
                    self._last_shed_load_time = time.time()
                self._emit_log(
                    f"[memory-guard] Predictive OOM p={p:.2f} > 0.70 "
                    f"[{model_source}] → shed_load (horizon ≈ {horizon}s)"
                )
                self._fire(self.on_shed_load, utilization, "on_shed_load (predictive)")

        except Exception as exc:
            logger.debug("[memory-guard] _run_predict_oom raised: %s", exc)

    def _post_source_baseline(self) -> None:
        """POST the startup memory footprint to ``POST /v1/ingest/baseline``.

        Called once in :meth:`start` after :meth:`_snapshot_cuda_graph_baseline`
        so that ``_cuda_graph_baseline_mb`` is already populated.

        Silently skips when no ``source_id`` is configured, no backend is
        installed, or any step fails.
        """
        if not self._source_id:
            return
        try:
            from .backends import upload_source_baseline as _upload_baseline

            extra: Dict[str, Any] = {}
            if self._extended_poll_fn is not None:
                try:
                    extra = self._extended_poll_fn() or {}
                except Exception as exc:
                    logger.debug(
                        "[memory-guard] extended_poll_fn raised in _post_source_baseline: %s",
                        exc,
                    )

            baseline = {
                "source_id":        self._source_id,
                "reserved_vram_mb": self._reserved_vram_mb,
                "weights_mb":       float(extra.get("weights_mb",  0.0)),
                "cuda_ctx_mb":      float(extra.get("cuda_ctx_mb", 0.0)),
                "cuda_graph_mb":    self._cuda_graph_baseline_mb,
                # PR 72: multi-GPU topology — lets the Worker store device count
                # alongside the combined VRAM and look up the correct catalog SKU
                "device_count":     self._device_count,
            }
            _upload_baseline(baseline)
            logger.debug(
                "[memory-guard] Source baseline posted: source_id=%r "
                "reserved_vram=%.0fMB device_count=%d "
                "weights=%.0fMB cuda_ctx=%.0fMB cuda_graph=%.0fMB",
                self._source_id,
                self._reserved_vram_mb,
                self._device_count,
                baseline["weights_mb"],
                baseline["cuda_ctx_mb"],
                baseline["cuda_graph_mb"],
            )
        except Exception as exc:
            logger.debug("[memory-guard] _post_source_baseline raised: %s", exc)

    def _compute_velocity(self, used_blocks: int, now: float) -> float:
        """Return KV cache growth rate.

        Returns MB/s when ``kv_block_size_mb`` is set; blocks/s otherwise.
        Resets tracking state on the first call after ``start()`` (returns 0).
        """
        velocity = 0.0
        if self._prev_used_blocks is not None and now > self._prev_poll_time:
            elapsed = now - self._prev_poll_time
            delta_blocks = used_blocks - self._prev_used_blocks
            rate_blocks_per_sec = delta_blocks / elapsed
            if self._kv_block_size_mb > 0.0:
                velocity = rate_blocks_per_sec * self._kv_block_size_mb
            else:
                velocity = rate_blocks_per_sec
        self._prev_used_blocks = used_blocks
        self._prev_poll_time = now
        return velocity

    # ------------------------------------------------------------------
    # Local SQLite telemetry persistence (PR 79)
    # ------------------------------------------------------------------

    _LOCAL_DB_SCHEMA = """
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY,
            source_id       TEXT,
            model_name      TEXT,
            reserved_vram_mb REAL,
            total_peak_mb   REAL,
            device_count    INTEGER,
            recorded_at     INTEGER DEFAULT (strftime('%s','now'))
        )
    """

    def _write_local_telemetry(self, telemetry: Any) -> None:
        """Persist one telemetry record to ~/.memory-guard/telemetry.db.

        Uses stdlib sqlite3 — no new dependencies.  Errors are swallowed so
        a disk-full or permissions problem never propagates to the monitor loop.
        """
        try:
            db_dir = os.path.expanduser("~/.memory-guard")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "telemetry.db")
            with sqlite3.connect(db_path, timeout=5) as conn:
                conn.execute(self._LOCAL_DB_SCHEMA)
                conn.execute(
                    "INSERT INTO runs "
                    "(source_id, model_name, reserved_vram_mb, total_peak_mb, device_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        self._source_id,
                        getattr(telemetry, "model_name", ""),
                        getattr(telemetry, "reserved_vram_mb", 0.0),
                        getattr(telemetry, "total_peak_mb", 0.0),
                        getattr(telemetry, "device_count", 1),
                    ),
                )
        except Exception as exc:
            logger.debug("KVCacheMonitor local telemetry write failed: %s", exc)

    def _upload_inference_telemetry(self, kv_velocity: float) -> None:
        """Collect extended signals and post to cloud.upload_inference_telemetry.

        Silently skips when no API key is configured or any step fails.
        """
        try:
            from .backends import upload_inference_signals as _upload_signals
            from .telemetry import InferenceTelemetry

            extra: Dict[str, Any] = {}
            if self._extended_poll_fn is not None:
                try:
                    extra = self._extended_poll_fn() or {}
                except Exception as exc:
                    logger.debug("KVCacheMonitor extended_poll_fn raised: %s", exc)

            # Pull BPF-derived signals when an ebpf_session is active (PR 56)
            bpf_page_fault_rate: float = 0.0
            bpf_pressure_mb:     float = 0.0
            if (
                self._ebpf_session is not None
                and getattr(self._ebpf_session, "available", False)
            ):
                bpf_page_fault_rate = float(
                    getattr(self._ebpf_session, "page_fault_rate", 0.0)
                )
                bpf_pressure_mb = (
                    float(getattr(self._ebpf_session, "memory_pressure_bytes", 0.0))
                    / (1024 * 1024)
                )

            # Snapshot running maxima and reset so the next interval starts fresh
            prefill_mb  = self._prefill_peak_activation_mb
            self._prefill_peak_activation_mb = 0.0
            total_peak  = self._total_peak_mb
            self._total_peak_mb = 0.0

            signals = InferenceTelemetry(
                kv_velocity_mbps           = kv_velocity,
                fragmentation_ratio        = float(extra.get("fragmentation_ratio", 0.0)),
                eviction_rate              = float(extra.get("eviction_rate", 0.0)),
                avg_seq_len                = float(extra.get("avg_seq_len", 0.0)),
                near_miss_count            = int(extra.get("near_miss_count", 0)),
                preemption_count           = int(extra.get("preemption_count", 0)),
                weights_mb                 = float(extra.get("weights_mb", 0.0)),
                kvcache_mb                 = float(extra.get("kvcache_mb", 0.0)),
                activations_mb             = float(extra.get("activations_mb", 0.0)),
                cuda_ctx_mb                = float(extra.get("cuda_ctx_mb", 0.0)),
                cuda_graph_mb              = float(extra.get("cuda_graph_mb", self._cuda_graph_baseline_mb)),
                prefill_peak_activation_mb = prefill_mb,
                max_seq_len_in_flight      = self._max_seq_len_in_flight,
                # PR 68: per-interval peak allocation + physical VRAM capacity
                total_peak_mb              = total_peak,
                reserved_vram_mb           = self._reserved_vram_mb,
                model_name                 = self._telemetry_model_name,
                backend                    = self._telemetry_backend,
                os_platform                = self._telemetry_os_platform,
                memory_pressure_level      = bpf_pressure_mb,
                page_fault_rate            = bpf_page_fault_rate,
                # PR 72: multi-GPU device count; 1 = single-GPU (default)
                device_count               = self._device_count,
            )
            # PR 79: always write locally first; cloud upload is optional
            self._write_local_telemetry(signals)
            _upload_signals(signals)
        except Exception as exc:
            logger.debug("KVCacheMonitor telemetry upload raised: %s", exc)

    def _emit_log(self, msg: str) -> None:
        try:
            self.on_log(msg)
        except Exception:
            logger.debug("KVCacheMonitor on_log raised", exc_info=True)

    def _fire(
        self,
        cb: Optional[Callable[[float], None]],
        utilization: float,
        name: str,
    ) -> None:
        if cb is None:
            return
        try:
            cb(utilization)
        except Exception:
            logger.debug("KVCacheMonitor %s raised", name, exc_info=True)

    def _fire_restart(self) -> None:
        if self.restart_callback is None:
            return
        try:
            self.restart_callback()
        except Exception:
            logger.debug("KVCacheMonitor restart_callback raised", exc_info=True)


class _KVCacheSession:
    """Context manager for KVCacheMonitor lifecycle."""

    def __init__(self, monitor: KVCacheMonitor) -> None:
        self._monitor = monitor

    def __enter__(self) -> KVCacheMonitor:
        self._monitor.start()
        return self._monitor

    def __exit__(self, *args: object) -> None:
        try:
            self._monitor.stop()
        except Exception:
            logger.debug("KVCacheMonitor stop raised in __exit__", exc_info=True)
