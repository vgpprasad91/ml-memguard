"""MemguardBPFSession — graceful context manager wrapping EBPFProbeManager.

:class:`MemguardBPFSession` converts :class:`EBPFProbeManager`'s exception-
raising :meth:`~EBPFProbeManager.load` into a graceful no-op when BPF is
unavailable.  Callers never need to wrap the session in a ``try/except``:

::

    with MemguardBPFSession(on_high=my_callback) as session:
        if session.available:
            logger.info("eBPF probes are active (backend=%s)", session.manager)
        else:
            logger.info("Running without eBPF — poll-based fallback only")
        server.serve_forever()

When ``eBPF`` is unavailable (non-Linux, missing capabilities, no BPF
library, or kernel too old), the context manager logs a single
``logger.warning`` and returns a session where :attr:`available` is ``False``
and :attr:`manager` is ``None``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional

from . import EBPFProbeManager
from ._loader import BPFProbeLoader
from .cgroup_memory import MemPressureEvent
from .preemption import PreemptionEvent

logger = logging.getLogger(__name__)


class MemguardBPFSession:
    """Graceful context manager wrapping :class:`EBPFProbeManager`.

    Converts BPF load failures into logged warnings rather than raised
    exceptions.  The calling code continues whether or not eBPF is active.

    Parameters
    ----------
    on_high:
        Callback ``Callable[[MemPressureEvent], None]`` — fired when a
        cgroup crosses its ``memory.high`` soft limit.  Called from the
        background polling thread; must be thread-safe.
    on_oom:
        Callback ``Callable[[MemPressureEvent], None]`` — fired when the
        OOM killer selects a cgroup victim.
    on_preemption:
        Callback ``Callable[[PreemptionEvent], None]`` — fired when the
        watched worker process exits.  Only meaningful when ``worker_pid``
        is also set.
    worker_pid:
        PID of the vLLM / SGLang worker process to watch.  When ``None``
        (default) the preemption probe is not loaded.
    poll_timeout_ms:
        Milliseconds per ``perf_buffer_poll`` call (default 10).
    ebpf_wake:
        Optional :class:`threading.Event` that the polling thread sets on
        every ``LEVEL_HIGH`` event — lets :class:`KVCacheMonitor` wake
        immediately instead of waiting for its next poll tick.
    loader:
        Override the :class:`BPFProbeLoader` used for capability detection.
        Primarily for testing.

    Attributes
    ----------
    available:
        ``True`` only while BPF probes are actually running inside the
        context block.
    manager:
        The live :class:`EBPFProbeManager` instance, or ``None`` when
        unavailable.
    page_fault_rate:
        User-space page faults per second from the :class:`PageFaultProbe`
        rolling window.  Returns ``0.0`` when the probe is not loaded.
    mmap_growth_mbps:
        Anonymous memory commitment rate in MB/s from the
        :class:`MmapGrowthProbe` rolling window.  Returns ``0.0`` when the
        probe is not loaded.
    memory_pressure_bytes:
        Bytes the cgroup was over its ``memory.high`` watermark at the time
        of the last OOM event.  Updated by the wrapped ``on_oom`` handler;
        ``0.0`` until the first event fires.
    """

    def __init__(
        self,
        on_high:       Optional[Callable[[MemPressureEvent], None]] = None,
        on_oom:        Optional[Callable[[MemPressureEvent], None]] = None,
        on_preemption: Optional[Callable[[PreemptionEvent], None]] = None,
        worker_pid:    Optional[int] = None,
        poll_timeout_ms: int = 10,
        ebpf_wake:     Optional[threading.Event] = None,
        loader:        Optional[BPFProbeLoader] = None,
    ) -> None:
        self._on_high         = on_high
        self._on_oom          = on_oom
        self._on_preemption   = on_preemption
        self._worker_pid      = worker_pid
        self._poll_timeout_ms = poll_timeout_ms
        self._ebpf_wake       = ebpf_wake
        self._loader          = loader or BPFProbeLoader()

        self._manager: Optional[EBPFProbeManager] = None
        self._active:  bool                       = False

        # PR 56 — extended probe metrics and OOM-imminent callbacks
        self._oom_imminent_cbs: List[Callable[[], None]] = []
        self._last_pressure_bytes: float = 0.0
        self._page_fault_probe: Optional[Any] = None   # PageFaultProbe
        self._mmap_probe:       Optional[Any] = None   # MmapGrowthProbe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` if BPF probes are actually running inside the ``with`` block."""
        return self._active

    @property
    def manager(self) -> Optional[EBPFProbeManager]:
        """The live :class:`EBPFProbeManager`, or ``None`` when unavailable."""
        return self._manager

    @property
    def page_fault_rate(self) -> float:
        """User-space page faults per second (rolling window).

        Returns ``0.0`` when :class:`PageFaultProbe` is not loaded.
        """
        if self._page_fault_probe is not None:
            return self._page_fault_probe.fault_rate_per_s
        return 0.0

    @property
    def mmap_growth_mbps(self) -> float:
        """Anonymous memory growth rate in MB/s (rolling window).

        Returns ``0.0`` when :class:`MmapGrowthProbe` is not loaded.
        """
        if self._mmap_probe is not None:
            return self._mmap_probe.growth_rate_mbps
        return 0.0

    @property
    def memory_pressure_bytes(self) -> float:
        """Bytes the cgroup was over its ``memory.high`` watermark at the last OOM event.

        Updated by the internal ``on_oom`` wrapper; ``0.0`` until an OOM fires.
        """
        return self._last_pressure_bytes

    def add_oom_imminent_callback(self, fn: Callable[[], None]) -> None:
        """Register *fn* to be called when an OOM-kill event fires.

        Callbacks registered before or after entering the context block are
        both honoured — the list is checked at event-dispatch time.

        *fn* is called from the BPF polling thread; it must be thread-safe.
        Exceptions raised by *fn* are silently swallowed.

        Parameters
        ----------
        fn:
            Zero-argument callable invoked when the OOM killer selects a
            cgroup victim.  Intended for graceful-restart triggers.
        """
        self._oom_imminent_cbs.append(fn)

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemguardBPFSession":
        if not self._loader.available:
            logger.warning(
                "[MemguardBPFSession] eBPF unavailable — running without kernel "
                "probes. Reason: %s",
                self._loader.unavailable_reason,
            )
            return self

        # Wrap on_oom so registered OOM-imminent callbacks are also fired.
        _session = self

        def _oom_wrapper(event: Any) -> None:
            _session._last_pressure_bytes = float(
                getattr(event, "pressure_bytes", 0)
            )
            for cb in _session._oom_imminent_cbs:
                try:
                    cb()
                except Exception:
                    pass
            if _session._on_oom is not None:
                try:
                    _session._on_oom(event)
                except Exception:
                    pass

        mgr = EBPFProbeManager(
            on_high         = self._on_high,
            on_oom          = _oom_wrapper,
            on_preemption   = self._on_preemption,
            worker_pid      = self._worker_pid,
            poll_timeout_ms = self._poll_timeout_ms,
            ebpf_wake       = self._ebpf_wake,
        )
        try:
            mgr.load()
            mgr.start()
            self._manager = mgr
            self._active  = True
            logger.debug(
                "[MemguardBPFSession] probes active (backend=%s)",
                self._loader.backend,
            )
        except Exception as exc:
            logger.warning(
                "[MemguardBPFSession] probe load failed — running without eBPF. "
                "Error: %s",
                exc,
            )
            try:
                mgr.stop()
            except Exception:
                pass
            return self

        # Load optional page-fault and mmap probes (silently skip failures).
        try:
            from .probes.page_fault import PageFaultProbe  # noqa: PLC0415
            pf = PageFaultProbe()
            pf.load()
            self._page_fault_probe = pf
        except Exception as exc:
            logger.debug("[MemguardBPFSession] PageFaultProbe unavailable: %s", exc)

        try:
            from .probes.mmap_growth import MmapGrowthProbe  # noqa: PLC0415
            mp = MmapGrowthProbe()
            mp.load()
            self._mmap_probe = mp
        except Exception as exc:
            logger.debug("[MemguardBPFSession] MmapGrowthProbe unavailable: %s", exc)

        return self

    def __exit__(self, *_: object) -> None:
        for probe in (self._page_fault_probe, self._mmap_probe):
            if probe is not None:
                try:
                    probe.detach()
                except Exception:
                    pass
        self._page_fault_probe = None
        self._mmap_probe       = None

        if self._manager is not None:
            try:
                self._manager.stop()
            except Exception as exc:
                logger.debug("[MemguardBPFSession] stop error: %s", exc)
            self._manager = None
        self._active = False

    def __repr__(self) -> str:
        state   = "active"  if self._active else "inactive"
        backend = self._loader.backend if self._active else "none"
        return f"MemguardBPFSession(state={state!r}, backend={backend!r})"
