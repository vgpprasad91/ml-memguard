"""Python wrapper for the mmap/brk growth BPF probe.

:class:`MmapGrowthProbe` loads ``mmap.bpf.c``, opens the ``mmap_events``
perf ring buffer, and dispatches typed
:class:`~memory_guard.ebpf._event.MemguardBPFEvent` callbacks.

Use case — pre-RSS expansion detection:
    When vLLM / SGLang commits anonymous memory via ``mmap()`` or ``brk()``,
    the pages are allocated but not yet faulted in.  The allocation is
    invisible to ``/proc/meminfo`` (RSS) and to vLLM's Prometheus metrics
    until the pages are touched.  This probe captures the commitment
    immediately, reporting ``mmap_growth_mbps`` before any poll-based
    approach sees the pressure.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Callable, List, Optional, Set

from .._event import EVENT_MMAP_GROWTH, MemguardBPFEvent
from ._rolling_window import _RollingWindow

logger = logging.getLogger(__name__)

_PROBE_DIR  = os.path.dirname(os.path.abspath(__file__))
_BPF_C_FILE = os.path.join(_PROBE_DIR, "mmap.bpf.c")

#: BPF event_subtype value for ``sys_enter_mmap`` events.
_SUBTYPE_MMAP: int = 0
#: BPF event_subtype value for ``sys_enter_brk`` events.
_SUBTYPE_BRK:  int = 1


# ---------------------------------------------------------------------------
# ctypes mirror of the BPF wire format
# ---------------------------------------------------------------------------

class _MmapEvent(ctypes.Structure):
    """Mirrors ``struct mmap_event`` in mmap.bpf.c.

    Field order and sizes **must** match the C struct exactly.
    """
    _fields_: List = [
        ("timestamp_ns",  ctypes.c_uint64),
        ("alloc_bytes",   ctypes.c_uint64),
        ("pid",           ctypes.c_uint32),
        ("event_subtype", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# MmapGrowthProbe
# ---------------------------------------------------------------------------

class MmapGrowthProbe:
    """BPF probe for anonymous memory map growth rate.

    Attaches to ``syscalls:sys_enter_mmap`` (anonymous, private mappings)
    and ``syscalls:sys_enter_brk`` (heap expansions) — Linux ≥ 4.9.

    Parameters
    ----------
    on_growth:
        Called for each allocation event that passes the PID filter.
        Receives a :class:`~memory_guard.ebpf._event.MemguardBPFEvent` with
        ``event_type == EVENT_MMAP_GROWTH``, ``pressure_bytes`` = bytes
        committed, and ``extra["subtype"]`` = ``"mmap"`` or ``"brk"``.
        Called from the perf-buffer polling thread — must be thread-safe.
    pid_allowlist:
        Optional set of PIDs to watch.  ``None`` / empty set = watch all
        PIDs.  Pass ``{vllm_pid}`` to monitor only a specific process.
    rate_window_s:
        Width of the rolling window used by :attr:`growth_rate_mbps`
        (default: 5.0 s).

    Raises
    ------
    OSError
        On non-Linux platforms.
    PermissionError
        Without ``CAP_BPF`` + ``CAP_PERFMON`` or root.
    ImportError
        When ``bcc`` is not installed.

    Examples
    --------
    ::

        probe = MmapGrowthProbe(
            on_growth=lambda e: logger.info(
                "mmap growth: +%d MB via %s",
                e.pressure_bytes // 1024**2,
                e.extra["subtype"],
            ),
            pid_allowlist={vllm_worker_pid},
        )
        probe.load()
        try:
            while True:
                probe.poll(timeout_ms=10)
                if probe.growth_rate_mbps > 500:
                    logger.warning("rapid allocation: %.0f MB/s", probe.growth_rate_mbps)
        finally:
            probe.detach()
    """

    def __init__(
        self,
        on_growth:     Optional[Callable[[MemguardBPFEvent], None]] = None,
        pid_allowlist: Optional[Set[int]] = None,
        rate_window_s: float = 5.0,
    ) -> None:
        self._on_growth   = on_growth
        self._pid_set:    Set[int]        = set(pid_allowlist) if pid_allowlist else set()
        self._window      = _RollingWindow(window_s=rate_window_s)
        self._bpf: Optional[object]      = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def growth_rate_mbps(self) -> float:
        """Anonymous memory allocation rate in MB/s over the rolling window."""
        return self._window.rate() / (1024 * 1024)

    @property
    def is_loaded(self) -> bool:
        """``True`` after :meth:`load` completes successfully."""
        return self._bpf is not None

    # ------------------------------------------------------------------
    # PID allowlist management
    # ------------------------------------------------------------------

    def add_pid(self, pid: int) -> None:
        """Add *pid* to the allowlist.

        Also inserts the PID into the BPF ``pid_allowlist_mmap`` map when
        the probe is already loaded.
        """
        self._pid_set.add(pid)
        if self._bpf is not None:
            try:
                tbl  = self._bpf["pid_allowlist_mmap"]      # type: ignore[index]
                tbl[ctypes.c_uint32(pid)] = ctypes.c_uint8(1)
                cnt  = self._bpf["pid_filter_count_mmap"]   # type: ignore[index]
                cnt[ctypes.c_uint32(0)] = ctypes.c_uint32(len(self._pid_set))
            except Exception as exc:
                logger.debug("[MmapGrowthProbe] add_pid BPF update error: %s", exc)

    def remove_pid(self, pid: int) -> None:
        """Remove *pid* from the allowlist."""
        self._pid_set.discard(pid)
        if self._bpf is not None:
            try:
                tbl = self._bpf["pid_allowlist_mmap"]       # type: ignore[index]
                del tbl[ctypes.c_uint32(pid)]
                cnt = self._bpf["pid_filter_count_mmap"]    # type: ignore[index]
                cnt[ctypes.c_uint32(0)] = ctypes.c_uint32(len(self._pid_set))
            except Exception as exc:
                logger.debug("[MmapGrowthProbe] remove_pid BPF update error: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read, compile, and attach the BPF probe."""
        if sys.platform != "linux":
            raise OSError(
                f"MmapGrowthProbe is Linux-only "
                f"(current platform: {sys.platform!r})"
            )
        try:
            from bcc import BPF  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "bcc is required for MmapGrowthProbe. "
                "Install with: pip install 'ml-memguard[ebpf]'"
            ) from exc

        with open(_BPF_C_FILE) as fh:
            bpf_text = fh.read()

        bpf   = BPF(text=bpf_text)
        _self = self

        def _raw_callback(cpu: int, data: ctypes.c_void_p, size: int) -> None:
            try:
                raw = ctypes.cast(data, ctypes.POINTER(_MmapEvent)).contents
                _self._dispatch(raw)
            except Exception:
                pass

        bpf.open_perf_buffer("mmap_events", callback=_raw_callback)  # type: ignore[attr-defined]
        self._bpf = bpf

        for pid in self._pid_set:
            self.add_pid(pid)

        logger.debug(
            "[MmapGrowthProbe] loaded (pid_allowlist=%s)",
            self._pid_set or "all",
        )

    def poll(self, timeout_ms: int = 10) -> None:
        """Drain the perf ring buffer and dispatch pending callbacks."""
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
        logger.debug("[MmapGrowthProbe] detached")

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "unloaded"
        pids  = sorted(self._pid_set) if self._pid_set else "all"
        return f"MmapGrowthProbe(state={state!r}, pids={pids!r})"

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, raw: "_MmapEvent") -> None:
        """Parse the raw BPF struct, apply PID filter, dispatch callback."""
        pid = int(raw.pid)

        # Python-side PID filter (mirrors BPF-side filter; also works in tests)
        if self._pid_set and pid not in self._pid_set:
            return

        subtype = "brk" if int(raw.event_subtype) == _SUBTYPE_BRK else "mmap"

        event = MemguardBPFEvent(
            ts_ns          = int(raw.timestamp_ns),
            event_type     = EVENT_MMAP_GROWTH,
            pressure_bytes = int(raw.alloc_bytes),
            pid            = pid,
            cgroup_id      = "",
            extra          = {"subtype": subtype},
        )

        # Update rolling window (stores bytes; growth_rate_mbps divides by 1 MiB)
        self._window.add(float(raw.alloc_bytes))

        if self._on_growth is not None:
            try:
                self._on_growth(event)
            except Exception:
                logger.debug("[MmapGrowthProbe] on_growth raised", exc_info=True)
