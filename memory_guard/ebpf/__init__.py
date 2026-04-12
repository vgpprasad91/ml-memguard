"""eBPF probe manager for memory-guard.

Provides :class:`EBPFProbeManager`, which loads :class:`CgroupMemoryProbe`
and :class:`PreemptionProbe`, runs a background polling thread, and dispatches
``on_high``, ``on_oom``, and ``on_preemption`` callbacks.

Usage::

    from memory_guard.ebpf import EBPFProbeManager

    def on_high(event):
        logger.warning("cgroup memory.high crossed: %s", event)

    def on_oom(event):
        logger.critical("OOM killer invoked: %s", event)

    mgr = EBPFProbeManager(on_high=on_high, on_oom=on_oom)
    mgr.load()   # may raise ImportError, PermissionError, OSError
    mgr.start()
    # … background thread dispatches events …
    mgr.stop()

Graceful degradation
--------------------
When ``bcc`` is not installed or the kernel version is too old,
:meth:`EBPFProbeManager.load` raises the appropriate exception.
Callers are expected to wrap the call in a ``try/except`` block and
fall back to poll-based detection when eBPF is unavailable.

Public re-exports
-----------------
All public types from the sub-modules are importable from this package::

    from memory_guard.ebpf import (
        EBPFProbeManager,
        CgroupMemoryProbe, MemPressureEvent, LEVEL_HIGH, LEVEL_OOM,
        PreemptionProbe,   PreemptionEvent,
    )
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from .cgroup_memory import (
    CgroupMemoryProbe,
    LEVEL_HIGH,
    LEVEL_OOM,
    MemPressureEvent,
)
from .preemption import PreemptionEvent, PreemptionProbe

logger = logging.getLogger(__name__)

__all__ = [
    "EBPFProbeManager",
    "CgroupMemoryProbe",
    "MemPressureEvent",
    "LEVEL_HIGH",
    "LEVEL_OOM",
    "PreemptionProbe",
    "PreemptionEvent",
]


class EBPFProbeManager:
    """Lifecycle manager for cgroup memory and preemption eBPF probes.

    Loads :class:`CgroupMemoryProbe` (and optionally
    :class:`PreemptionProbe` when ``worker_pid`` is given), runs a
    background thread that drains the perf ring buffers, and dispatches
    ``on_high``, ``on_oom``, and ``on_preemption`` callbacks.

    Parameters
    ----------
    on_high:
        ``Callable[[MemPressureEvent], None]`` — fired when a cgroup
        crosses its ``memory.high`` soft limit (``LEVEL_HIGH``).
        Called from the background polling thread — must be thread-safe.
    on_oom:
        ``Callable[[MemPressureEvent], None]`` — fired when the OOM killer
        selects a cgroup victim (``LEVEL_OOM``).
    on_preemption:
        ``Callable[[PreemptionEvent], None]`` — fired when the monitored
        worker process (``worker_pid``) exits.  Ignored when
        ``worker_pid`` is ``None`` (no :class:`PreemptionProbe` is
        loaded in that case).
    worker_pid:
        PID of the vLLM worker process to watch for preemption.
        When ``None`` (default), the preemption probe is not loaded.
    poll_timeout_ms:
        Milliseconds passed to each ``perf_buffer_poll`` call (default 10).
    ebpf_wake:
        Optional :class:`threading.Event` that the polling thread will
        ``set()`` whenever a ``LEVEL_HIGH`` event fires — allows the
        :class:`~memory_guard.inference_monitor.KVCacheMonitor` to
        wake immediately instead of waiting for the next poll tick.

    Raises
    ------
    ImportError
        If ``bcc`` is not installed.
    PermissionError
        If the process lacks the required Linux capabilities.
    OSError
        If the kernel does not support the required tracepoints or kprobes.
    """

    def __init__(
        self,
        on_high: Optional[Callable[[MemPressureEvent], None]] = None,
        on_oom: Optional[Callable[[MemPressureEvent], None]] = None,
        on_preemption: Optional[Callable[[PreemptionEvent], None]] = None,
        worker_pid: Optional[int] = None,
        poll_timeout_ms: int = 10,
        ebpf_wake: Optional[threading.Event] = None,
    ) -> None:
        self._on_high = on_high
        self._on_oom = on_oom
        self._on_preemption = on_preemption
        self._worker_pid = worker_pid
        self._poll_timeout_ms = poll_timeout_ms
        self._ebpf_wake = ebpf_wake

        self._cgroup_probe: Optional[CgroupMemoryProbe] = None
        self._preemption_probe: Optional[PreemptionProbe] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Compile and attach all BPF probes.

        Must be called before :meth:`start`.  Safe to call multiple times
        (subsequent calls are no-ops when already loaded).
        """
        if self._cgroup_probe is None:
            probe = CgroupMemoryProbe(on_event=self._dispatch_mem_event)
            probe.load()
            self._cgroup_probe = probe

        if self._worker_pid is not None and self._preemption_probe is None:
            pp = PreemptionProbe(
                target_pid=self._worker_pid,
                on_event=self._dispatch_preemption_event,
            )
            pp.load()
            self._preemption_probe = pp

    def start(self) -> None:
        """Start the background perf-buffer polling thread.

        :meth:`load` must be called first.  If the probes have not been
        loaded, this method is a no-op (logs a warning).
        """
        if self._cgroup_probe is None:
            logger.warning(
                "[EBPFProbeManager] start() called without load() — no-op"
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return  # already running
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="ebpf-probe-manager",
        )
        self._thread.start()
        logger.debug("[EBPFProbeManager] polling thread started")

    def stop(self) -> None:
        """Signal the polling thread to stop and detach all probes."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._cgroup_probe is not None:
            try:
                self._cgroup_probe.detach()
            except Exception:
                pass
            self._cgroup_probe = None

        if self._preemption_probe is not None:
            try:
                self._preemption_probe.detach()
            except Exception:
                pass
            self._preemption_probe = None

    @property
    def is_loaded(self) -> bool:
        """True once :meth:`load` has completed successfully."""
        return self._cgroup_probe is not None and self._cgroup_probe.is_loaded

    @property
    def is_running(self) -> bool:
        """True while the background polling thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def __repr__(self) -> str:
        state = "running" if self.is_running else ("loaded" if self.is_loaded else "unloaded")
        pid_part = f", worker_pid={self._worker_pid}" if self._worker_pid is not None else ""
        return f"EBPFProbeManager(state={state!r}{pid_part})"

    # ------------------------------------------------------------------
    # Background polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._cgroup_probe is not None:
                    self._cgroup_probe.poll(timeout_ms=self._poll_timeout_ms)
                if self._preemption_probe is not None:
                    self._preemption_probe.poll(timeout_ms=self._poll_timeout_ms)
            except Exception as exc:
                logger.debug("[EBPFProbeManager] poll error: %s", exc)

    # ------------------------------------------------------------------
    # Event dispatchers
    # ------------------------------------------------------------------

    def _dispatch_mem_event(self, event: MemPressureEvent) -> None:
        if event.level == LEVEL_HIGH:
            if self._on_high is not None:
                try:
                    self._on_high(event)
                except Exception:
                    logger.debug("[EBPFProbeManager] on_high raised", exc_info=True)
            # Wake the KVCacheMonitor poll loop immediately
            if self._ebpf_wake is not None:
                self._ebpf_wake.set()

        elif event.level == LEVEL_OOM:
            if self._on_oom is not None:
                try:
                    self._on_oom(event)
                except Exception:
                    logger.debug("[EBPFProbeManager] on_oom raised", exc_info=True)

    def _dispatch_preemption_event(self, event: PreemptionEvent) -> None:
        if self._on_preemption is not None:
            try:
                self._on_preemption(event)
            except Exception:
                logger.debug("[EBPFProbeManager] on_preemption raised", exc_info=True)
