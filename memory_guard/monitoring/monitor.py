"""Runtime memory monitor — polls memory pressure during training.

Runs in a background thread and triggers batch size downgrades
when memory pressure exceeds configurable thresholds.

Apple Silicon: Uses MLX mx.metal.get_active_memory() as ground truth,
  falls back to Mach kernel host_statistics via ctypes.
CUDA: Polls torch.cuda.memory_allocated.
Linux: Reads /proc/pressure/memory (PSI) or /proc/meminfo.
Windows: Reads GlobalMemoryStatusEx.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from ..constants import (
    PRESSURE_THRESHOLD_WARNING,
    PRESSURE_THRESHOLD_CRITICAL,
    PRESSURE_THRESHOLD_EMERGENCY,
)
from .platforms import (
    Backend,
    detect_platform,
    get_memory_pressure,
    get_mlx_active_memory_mb,
    get_mlx_peak_memory_mb,
)

logger = logging.getLogger(__name__)


class RuntimeMonitor:
    """Background thread that monitors memory and triggers downgrades.

    On Apple Silicon, uses MLX's mx.metal.get_active_memory() for
    ground-truth GPU allocator readings when available, falling back
    to macOS Mach kernel vm_statistics. This catches the memory leak
    pattern documented in https://github.com/ml-explore/mlx-examples/issues/1262 where active
    memory rises continuously until crash.

    Usage:
        monitor = RuntimeMonitor(on_pressure=lambda bs: print(f"New batch: {bs}"))
        monitor.start(batch_size=4)

        for step in training_loop:
            actual_bs = monitor.current_batch_size
            train_step(batch_size=actual_bs)

        monitor.stop()

    Or as context manager:
        with RuntimeMonitor(on_pressure=callback).session(batch_size=4) as mon:
            for step in training_loop:
                train_step(batch_size=mon.current_batch_size)
    """

    THRESHOLD_WARNING = PRESSURE_THRESHOLD_WARNING
    THRESHOLD_CRITICAL = PRESSURE_THRESHOLD_CRITICAL
    THRESHOLD_EMERGENCY = PRESSURE_THRESHOLD_EMERGENCY

    def __init__(
        self,
        poll_interval: float = 5.0,  # See constants.MONITOR_POLL_INTERVAL
        on_pressure: Optional[Callable[[int], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        backend: Optional[Backend] = None,
        max_downgrades: int = 3,
        cooldown_seconds: float = 30.0,
        memory_limit_mb: Optional[float] = None,
    ):
        """
        Args:
            poll_interval: Seconds between memory pressure checks.
            on_pressure: Callback(new_batch_size) when batch size changes.
            on_log: Callback(message) for logging (default: logger.warning).
            backend: Force a specific backend, or None for auto-detect.
            max_downgrades: Maximum number of automatic downgrades per session.
            cooldown_seconds: Minimum time between consecutive downgrades.
            memory_limit_mb: Explicit memory limit. If set, MLX active memory
                           is compared against this instead of using pressure.
        """
        self.poll_interval = poll_interval
        self.on_pressure = on_pressure
        self.on_log = on_log or (lambda msg: logger.warning(msg))
        self.max_downgrades = max_downgrades
        self.cooldown_seconds = cooldown_seconds
        self.memory_limit_mb = memory_limit_mb

        self._backend = backend or detect_platform().backend
        self._stop = threading.Event()
        self._thread = None
        self._batch_size = 1
        self._lock = threading.Lock()
        self._downgrades_used = 0
        self._last_downgrade_time = 0.0
        self._last_warning_time = 0.0
        self._pressure_history: list[float] = []
        self._mlx_memory_history: list[float] = []  # MLX active memory readings
        self._has_mlx_metal = False

        # MLX Metal availability
        if self._backend in (Backend.APPLE_SILICON, Backend.APPLE_INTEL):
            test = get_mlx_active_memory_mb()
            self._has_mlx_metal = test is not None

    @property
    def current_batch_size(self) -> int:
        """Thread-safe access to current batch size."""
        with self._lock:
            return self._batch_size

    @property
    def downgrades_remaining(self) -> int:
        with self._lock:
            return self.max_downgrades - self._downgrades_used

    @property
    def pressure_history(self) -> list[float]:
        """Recent pressure readings (last 60 samples)."""
        with self._lock:
            return list(self._pressure_history)

    @property
    def mlx_memory_history(self) -> list[float]:
        """Recent MLX Metal active memory readings in MB (last 60)."""
        with self._lock:
            return list(self._mlx_memory_history)

    @property
    def peak_mlx_memory_mb(self) -> Optional[float]:
        """Peak MLX Metal memory observed during this session."""
        with self._lock:
            if self._mlx_memory_history:
                return max(self._mlx_memory_history)
        return get_mlx_peak_memory_mb()

    def start(self, batch_size: int):
        """Start monitoring in background thread.

        If already running, stops the existing thread first to prevent
        orphaned daemon threads that leak resources.
        """
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        with self._lock:
            self._batch_size = batch_size
            self._downgrades_used = 0
            self._mlx_memory_history.clear()
            self._pressure_history.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="memory-guard")
        self._thread.start()
        mlx_note = " + MLX Metal" if self._has_mlx_metal else ""
        try:
            self.on_log(
                f"[memory-guard] Monitor started: backend={self._backend.value}{mlx_note}, "
                f"poll={self.poll_interval}s, batch_size={batch_size}"
            )
        except Exception:
            logger.debug("on_log callback raised during start", exc_info=True)

    def stop(self):
        """Stop monitoring and join thread."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_interval + 2)
        self._thread = None

    def session(self, batch_size: int):
        """Context manager for monitor lifecycle."""
        return _MonitorSession(self, batch_size)

    def _get_effective_pressure(self) -> float:
        """Get memory pressure, preferring MLX Metal APIs when available.

        On Apple Silicon with MLX:
          - Uses mx.metal.get_active_memory() as ground truth
          - Compares against memory_limit_mb or total system memory
          - This catches Metal allocator leaks that vm_stat misses

        Falls back to OS-level pressure for non-MLX backends.
        """
        # MLX Metal ground truth (Apple Silicon)
        if self._has_mlx_metal:
            active_mb = get_mlx_active_memory_mb()
            if active_mb is not None:
                with self._lock:
                    self._mlx_memory_history.append(active_mb)
                    if len(self._mlx_memory_history) > 60:
                        self._mlx_memory_history.pop(0)

                # Compute pressure from MLX active memory
                if self.memory_limit_mb and self.memory_limit_mb > 0:
                    limit = self.memory_limit_mb
                else:
                    # Use total system memory as reference
                    try:
                        from .platforms import _sysctl_int64
                        limit = _sysctl_int64("hw.memsize") / (1024 * 1024)
                    except Exception:
                        limit = 0

                if limit > 0:
                    mlx_pressure = active_mb / limit

                    # Also check OS-level pressure for non-MLX memory usage
                    os_pressure = get_memory_pressure(self._backend)

                    # Use the HIGHER of the two — either could indicate danger
                    return max(mlx_pressure, os_pressure)

        # Fallback: OS-level pressure only
        return get_memory_pressure(self._backend)

    def _loop(self):
        while not self._stop.is_set():
            try:
                pressure = self._get_effective_pressure()

                with self._lock:
                    self._pressure_history.append(pressure)
                    if len(self._pressure_history) > 60:
                        self._pressure_history.pop(0)

                    now = time.time()
                    in_cooldown = (now - self._last_downgrade_time) < self.cooldown_seconds

                # Check for MLX memory leak pattern: monotonically increasing
                if self._has_mlx_metal:
                    with self._lock:
                        hist = list(self._mlx_memory_history)
                    if len(hist) >= 6:
                        recent = hist[-6:]
                        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                            growth_mb = recent[-1] - recent[0]
                            from ..constants import MLX_LEAK_GROWTH_THRESHOLD_MB
                            if growth_mb > MLX_LEAK_GROWTH_THRESHOLD_MB:
                                if not in_cooldown:
                                    self._trigger_downgrade(pressure, "MLX_LEAK")
                                    continue

                if pressure >= self.THRESHOLD_EMERGENCY and not in_cooldown:
                    self._trigger_downgrade(pressure, "EMERGENCY")
                elif pressure >= self.THRESHOLD_CRITICAL and not in_cooldown:
                    self._trigger_downgrade(pressure, "CRITICAL")
                elif pressure >= self.THRESHOLD_WARNING:
                    with self._lock:
                        recent_pressures = list(self._pressure_history[-3:])
                    if len(recent_pressures) >= 3:
                        if all(p >= self.THRESHOLD_WARNING for p in recent_pressures):
                            now_warn = time.time()
                            if (now_warn - self._last_warning_time) >= self.cooldown_seconds:
                                self._last_warning_time = now_warn
                                try:
                                    self.on_log(
                                        f"[memory-guard] Sustained pressure: "
                                        f"{pressure:.0%} (3 consecutive readings)"
                                    )
                                except Exception:
                                    pass

            except Exception as e:
                logger.debug(f"Monitor poll error: {e}")

            self._stop.wait(self.poll_interval)

    def _trigger_downgrade(self, pressure: float, level: str):
        # All state reads/writes under lock. ALL callbacks OUTSIDE lock
        # to prevent deadlock if callback reads mon.current_batch_size.
        msg = None
        new_bs = None

        with self._lock:
            if self._downgrades_used >= self.max_downgrades:
                msg = (
                    f"[memory-guard] {level} pressure ({pressure:.0%}) but "
                    f"max downgrades ({self.max_downgrades}) exhausted"
                )
                # Don't return inside lock — break out and log outside
            elif self._batch_size <= 1:
                msg = (
                    f"[memory-guard] {level} pressure ({pressure:.0%}), "
                    f"batch_size=1, cannot downgrade further"
                )
            else:
                old_bs = self._batch_size
                self._batch_size = max(1, old_bs // 2)
                self._downgrades_used += 1
                self._last_downgrade_time = time.time()

                new_bs = self._batch_size
                remaining = self.max_downgrades - self._downgrades_used
                mlx_info = ""
                if self._has_mlx_metal and self._mlx_memory_history:
                    mlx_info = f", MLX active={self._mlx_memory_history[-1]:.0f}MB"

                msg = (
                    f"[memory-guard] {level} ({pressure:.0%}{mlx_info}): "
                    f"batch_size {old_bs} -> {new_bs} "
                    f"({remaining} downgrades left)"
                )

        # Callbacks outside lock — safe even if callback reads monitor state
        if msg:
            try:
                self.on_log(msg)
            except Exception:
                logger.debug("on_log callback raised", exc_info=True)

        if new_bs is not None and self.on_pressure:
            try:
                self.on_pressure(new_bs)
            except Exception:
                logger.debug("on_pressure callback raised", exc_info=True)


class _MonitorSession:
    """Context manager wrapper for RuntimeMonitor."""

    def __init__(self, monitor: RuntimeMonitor, batch_size: int):
        self._monitor = monitor
        self._batch_size = batch_size

    def __enter__(self) -> RuntimeMonitor:
        self._monitor.start(self._batch_size)
        return self._monitor

    def __exit__(self, *args):
        try:
            self._monitor.stop()
        except Exception:
            logger.debug("Monitor stop failed", exc_info=True)
