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

        self._history: collections.deque[float] = collections.deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_warning_time: float = 0.0
        self._last_shed_load_time: float = 0.0
        self._critical_consecutive: int = 0

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
        self._stop.clear()
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
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_interval + 2)
        self._thread = None

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

            # --- record -------------------------------------------------
            with self._lock:
                self._history.append(utilization)
                warn_ready = (now - self._last_warning_time) >= self.cooldown_seconds
                shed_ready = (now - self._last_shed_load_time) >= self.cooldown_seconds

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

            self._stop.wait(self.poll_interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _upload_inference_telemetry(self, kv_velocity: float) -> None:
        """Collect extended signals and post to cloud.upload_inference_telemetry.

        Silently skips when no API key is configured or any step fails.
        """
        try:
            from . import cloud as _cloud
            if not _cloud.api_key():
                return
            from .telemetry import InferenceTelemetry

            extra: Dict[str, Any] = {}
            if self._extended_poll_fn is not None:
                try:
                    extra = self._extended_poll_fn() or {}
                except Exception as exc:
                    logger.debug("KVCacheMonitor extended_poll_fn raised: %s", exc)

            signals = InferenceTelemetry(
                kv_velocity_mbps    = kv_velocity,
                fragmentation_ratio = float(extra.get("fragmentation_ratio", 0.0)),
                eviction_rate       = float(extra.get("eviction_rate", 0.0)),
                avg_seq_len         = float(extra.get("avg_seq_len", 0.0)),
                near_miss_count     = int(extra.get("near_miss_count", 0)),
                preemption_count    = int(extra.get("preemption_count", 0)),
                weights_mb          = float(extra.get("weights_mb", 0.0)),
                kvcache_mb          = float(extra.get("kvcache_mb", 0.0)),
                activations_mb      = float(extra.get("activations_mb", 0.0)),
                cuda_ctx_mb         = float(extra.get("cuda_ctx_mb", 0.0)),
                model_name          = self._telemetry_model_name,
                backend             = self._telemetry_backend,
                os_platform         = self._telemetry_os_platform,
            )
            _cloud.upload_inference_telemetry(signals)
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
