"""Python wrapper for the cgroup memory.high BPF probe.

:class:`CgroupMemoryHighProbe` loads ``cgroup_memory.bpf.c``, opens the
``cgroup_mem_high_events`` perf ring buffer, and dispatches typed
:class:`~memory_guard.ebpf._event.MemguardBPFEvent` callbacks:

* ``on_high`` — every crossing of the ``memory.high`` soft limit
* ``on_oom_imminent`` — crossings where ``pressure_bytes`` exceeds a
  configurable threshold, indicating that OOM kill is imminent

Timing advantage
----------------
The ``cgroup:cgroup_memory_high`` tracepoint fires 200–500 ms before
the kernel OOM killer selects a victim.  No ``/proc/meminfo`` poll or
vLLM Prometheus scrape can match this latency.  See
``docs/ebpf/cgroup_memory_high_probe.md`` for the full explanation.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Callable, List, Optional

from .._event import EVENT_MEMORY_HIGH, MemguardBPFEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the BPF C source file (same directory as this module)
# ---------------------------------------------------------------------------

_PROBE_DIR   = os.path.dirname(os.path.abspath(__file__))
_BPF_C_FILE  = os.path.join(_PROBE_DIR, "cgroup_memory.bpf.c")

# Default OOM-imminent threshold: 512 MB over the memory.high watermark.
_DEFAULT_OOM_IMMINENT_THRESHOLD_MB: float = 512.0


# ---------------------------------------------------------------------------
# ctypes mirror of the BPF wire format
# ---------------------------------------------------------------------------

class _CgroupMemHighEvent(ctypes.Structure):
    """Mirrors ``struct cgroup_mem_high_event`` in cgroup_memory.bpf.c.

    Field order, sizes, and alignment **must** match the C struct exactly.
    """
    _fields_: List = [
        ("timestamp_ns",   ctypes.c_uint64),
        ("pressure_bytes", ctypes.c_uint64),
        ("pid",            ctypes.c_uint32),
        ("_pad",           ctypes.c_uint32),
        ("cgroup_id",      ctypes.c_char * 128),
    ]


# ---------------------------------------------------------------------------
# CgroupMemoryHighProbe
# ---------------------------------------------------------------------------

class CgroupMemoryHighProbe:
    """BPF probe for cgroup ``memory.high`` pressure events.

    Attaches to the ``cgroup:cgroup_memory_high`` tracepoint (Linux ≥ 5.8,
    cgroup v2).  Fires 200–500 ms before the OOM killer acts — earlier than
    any poll-based approach.

    Parameters
    ----------
    on_high:
        Called for every ``memory.high`` crossing regardless of magnitude.
        Receives a :class:`~memory_guard.ebpf._event.MemguardBPFEvent` with
        ``event_type == EVENT_MEMORY_HIGH`` and ``pressure_bytes`` set to the
        bytes the cgroup is over its watermark.
        Called from the perf-buffer polling thread — must be thread-safe.
    on_oom_imminent:
        Called when ``pressure_bytes > oom_imminent_threshold_mb * 1024²``.
        This is the signal that PR 56 wires to the graceful-restart path.
    oom_imminent_threshold_mb:
        MB threshold above the ``memory.high`` watermark at which
        ``on_oom_imminent`` fires (default: 512 MB).
    cgroup_filter:
        Optional cgroup path prefix.  When set, events whose ``cgroup_id``
        does **not** start with this string are silently dropped.  Use
        ``"/kubepods/"`` to watch only Kubernetes workload cgroups and
        ignore system slices.

    Raises
    ------
    OSError
        If called on a non-Linux platform.
    PermissionError
        If the process lacks ``CAP_BPF`` + ``CAP_PERFMON`` (or root).
    ImportError
        If ``bcc`` is not installed.

    Examples
    --------
    ::

        probe = CgroupMemoryHighProbe(
            on_high=lambda e: logger.warning("memory.high: %s", e),
            on_oom_imminent=lambda e: trigger_graceful_restart(e.cgroup_id),
            oom_imminent_threshold_mb=256.0,
            cgroup_filter="/kubepods/",
        )
        probe.load()
        try:
            while True:
                probe.poll(timeout_ms=10)
        finally:
            probe.detach()
    """

    def __init__(
        self,
        on_high:                   Optional[Callable[[MemguardBPFEvent], None]] = None,
        on_oom_imminent:           Optional[Callable[[MemguardBPFEvent], None]] = None,
        oom_imminent_threshold_mb: float = _DEFAULT_OOM_IMMINENT_THRESHOLD_MB,
        cgroup_filter:             Optional[str] = None,
    ) -> None:
        self._on_high                   = on_high
        self._on_oom_imminent           = on_oom_imminent
        self._oom_imminent_threshold_b  = int(oom_imminent_threshold_mb * 1024 * 1024)
        self._cgroup_filter             = cgroup_filter
        self._bpf: Optional[object]     = None   # bcc.BPF instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read, compile, and attach the BPF probe.

        Raises ``OSError`` on non-Linux, ``PermissionError`` without
        capabilities, ``ImportError`` when bcc is absent, and ``OSError``
        if the kernel rejects the program (e.g. kernel < 5.8, cgroup v1).
        """
        if sys.platform != "linux":
            raise OSError(
                f"CgroupMemoryHighProbe is Linux-only "
                f"(current platform: {sys.platform!r})"
            )

        try:
            from bcc import BPF  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "bcc is required for CgroupMemoryHighProbe. "
                "Install with: pip install 'ml-memguard[ebpf]'"
            ) from exc

        with open(_BPF_C_FILE) as fh:
            bpf_text = fh.read()

        bpf = BPF(text=bpf_text)

        # Register the perf buffer callback
        probe_self = self

        def _raw_callback(cpu: int, data: ctypes.c_void_p, size: int) -> None:
            try:
                raw = ctypes.cast(
                    data, ctypes.POINTER(_CgroupMemHighEvent)
                ).contents
                probe_self._dispatch(raw)
            except Exception:
                pass  # never crash the perf buffer thread

        bpf.open_perf_buffer(  # type: ignore[attr-defined]
            "cgroup_mem_high_events",
            callback=_raw_callback,
        )
        self._bpf = bpf
        logger.debug(
            "[CgroupMemoryHighProbe] loaded (oom_threshold=%d MB, filter=%r)",
            self._oom_imminent_threshold_b // (1024 * 1024),
            self._cgroup_filter,
        )

    def poll(self, timeout_ms: int = 10) -> None:
        """Drain the perf ring buffer and dispatch pending callbacks.

        Call this in a tight loop from a dedicated thread:

            while not stop.is_set():
                probe.poll(timeout_ms=10)
        """
        if self._bpf is not None:
            self._bpf.perf_buffer_poll(timeout=timeout_ms)  # type: ignore[attr-defined]

    def detach(self) -> None:
        """Detach all probes and release BPF resources."""
        if self._bpf is not None:
            try:
                self._bpf.cleanup()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._bpf = None
        logger.debug("[CgroupMemoryHighProbe] detached")

    @property
    def is_loaded(self) -> bool:
        """``True`` after :meth:`load` completes successfully."""
        return self._bpf is not None

    def __repr__(self) -> str:
        state     = "loaded" if self.is_loaded else "unloaded"
        threshold = self._oom_imminent_threshold_b // (1024 * 1024)
        return (
            f"CgroupMemoryHighProbe(state={state!r}, "
            f"oom_threshold_mb={threshold}, "
            f"cgroup_filter={self._cgroup_filter!r})"
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, raw: "_CgroupMemHighEvent") -> None:
        """Parse the raw BPF struct, apply filter, dispatch callbacks."""
        cgroup_id = raw.cgroup_id.decode("utf-8", errors="replace").rstrip("\x00")

        # Apply cgroup path prefix filter
        if self._cgroup_filter is not None:
            if not cgroup_id.startswith(self._cgroup_filter):
                return

        event = MemguardBPFEvent(
            ts_ns          = int(raw.timestamp_ns),
            event_type     = EVENT_MEMORY_HIGH,
            pressure_bytes = int(raw.pressure_bytes),
            pid            = int(raw.pid),
            cgroup_id      = cgroup_id,
        )

        # on_high fires for every crossing
        if self._on_high is not None:
            try:
                self._on_high(event)
            except Exception:
                logger.debug("[CgroupMemoryHighProbe] on_high raised", exc_info=True)

        # on_oom_imminent fires only when pressure exceeds threshold
        if (self._on_oom_imminent is not None
                and event.pressure_bytes > self._oom_imminent_threshold_b):
            try:
                self._on_oom_imminent(event)
            except Exception:
                logger.debug(
                    "[CgroupMemoryHighProbe] on_oom_imminent raised", exc_info=True
                )
