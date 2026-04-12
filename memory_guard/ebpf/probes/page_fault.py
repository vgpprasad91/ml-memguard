"""Python wrapper for the page_fault BPF probe.

:class:`PageFaultProbe` loads ``page_fault.bpf.c``, opens the
``page_fault_events`` perf ring buffer, and dispatches typed
:class:`~memory_guard.ebpf._event.MemguardBPFEvent` callbacks.

Use case — silent-kill detection:
    When a GPU-serving process (vLLM, SGLang) begins swapping, page faults
    spike 100-1000x above baseline 300–800 ms before the OOM killer fires
    SIGKILL.  The Linux OOM killer bypasses Python's signal machinery, so
    the current VLLMWatchdog misses the kill entirely.  This probe fires at
    the exact kernel exit timestamp.

See ``docs/ebpf/cgroup_memory_high_probe.md`` for the timing context.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Callable, List, Optional, Set

from .._event import EVENT_PAGE_FAULT, MemguardBPFEvent
from ._rolling_window import _RollingWindow

logger = logging.getLogger(__name__)

_PROBE_DIR  = os.path.dirname(os.path.abspath(__file__))
_BPF_C_FILE = os.path.join(_PROBE_DIR, "page_fault.bpf.c")


# ---------------------------------------------------------------------------
# ctypes mirror of the BPF wire format
# ---------------------------------------------------------------------------

class _PageFaultEvent(ctypes.Structure):
    """Mirrors ``struct page_fault_event`` in page_fault.bpf.c.

    Field order and sizes **must** match the C struct exactly.
    """
    _fields_: List = [
        ("timestamp_ns",  ctypes.c_uint64),
        ("fault_address", ctypes.c_uint64),
        ("error_code",    ctypes.c_uint32),
        ("pid",           ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# PageFaultProbe
# ---------------------------------------------------------------------------

class PageFaultProbe:
    """BPF probe for user-space page fault rate monitoring.

    Attaches to ``exceptions:page_fault_user`` (Linux ≥ 4.9).  Fires on
    every user-space page fault — both minor (soft) and major (hard) faults.

    Parameters
    ----------
    on_fault:
        Called for each page fault event that passes the PID filter.
        Receives a :class:`~memory_guard.ebpf._event.MemguardBPFEvent` with
        ``event_type == EVENT_PAGE_FAULT`` and ``extra`` containing
        ``fault_address`` and ``error_code``.
        Called from the perf-buffer polling thread — must be thread-safe.
    pid_allowlist:
        Optional set of PIDs to watch.  ``None`` / empty set = watch all
        PIDs.  Pass ``{vllm_pid}`` to monitor only a specific process.
    rate_window_s:
        Width of the rolling window used by :attr:`fault_rate_per_s`
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

        probe = PageFaultProbe(
            on_fault=lambda e: logger.warning("fault: %s", e),
            pid_allowlist={vllm_worker_pid},
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
        on_fault:      Optional[Callable[[MemguardBPFEvent], None]] = None,
        pid_allowlist: Optional[Set[int]] = None,
        rate_window_s: float = 5.0,
    ) -> None:
        self._on_fault    = on_fault
        self._pid_set:    Set[int]        = set(pid_allowlist) if pid_allowlist else set()
        self._window      = _RollingWindow(window_s=rate_window_s)
        self._bpf: Optional[object]      = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fault_rate_per_s(self) -> float:
        """Page faults per second over the rolling window."""
        return self._window.rate()

    @property
    def is_loaded(self) -> bool:
        """``True`` after :meth:`load` completes successfully."""
        return self._bpf is not None

    # ------------------------------------------------------------------
    # PID allowlist management
    # ------------------------------------------------------------------

    def add_pid(self, pid: int) -> None:
        """Add *pid* to the allowlist.

        Also inserts the PID into the BPF ``pid_allowlist`` map when
        the probe is already loaded, enabling kernel-side filtering.
        """
        self._pid_set.add(pid)
        if self._bpf is not None:
            try:
                tbl = self._bpf["pid_allowlist"]        # type: ignore[index]
                key = ctypes.c_uint32(pid)
                val = ctypes.c_uint8(1)
                tbl[key] = val
                # Increment filter count
                cnt_tbl = self._bpf["pid_filter_count"]  # type: ignore[index]
                zero    = ctypes.c_uint32(0)
                current = ctypes.c_uint32(len(self._pid_set))
                cnt_tbl[zero] = current
            except Exception as exc:
                logger.debug("[PageFaultProbe] add_pid BPF update error: %s", exc)

    def remove_pid(self, pid: int) -> None:
        """Remove *pid* from the allowlist."""
        self._pid_set.discard(pid)
        if self._bpf is not None:
            try:
                tbl = self._bpf["pid_allowlist"]         # type: ignore[index]
                key = ctypes.c_uint32(pid)
                del tbl[key]
                cnt_tbl = self._bpf["pid_filter_count"]  # type: ignore[index]
                zero    = ctypes.c_uint32(0)
                cnt_tbl[zero] = ctypes.c_uint32(len(self._pid_set))
            except Exception as exc:
                logger.debug("[PageFaultProbe] remove_pid BPF update error: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read, compile, and attach the BPF probe."""
        if sys.platform != "linux":
            raise OSError(
                f"PageFaultProbe is Linux-only "
                f"(current platform: {sys.platform!r})"
            )
        try:
            from bcc import BPF  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "bcc is required for PageFaultProbe. "
                "Install with: pip install 'ml-memguard[ebpf]'"
            ) from exc

        with open(_BPF_C_FILE) as fh:
            bpf_text = fh.read()

        bpf   = BPF(text=bpf_text)
        _self = self

        def _raw_callback(cpu: int, data: ctypes.c_void_p, size: int) -> None:
            try:
                raw = ctypes.cast(data, ctypes.POINTER(_PageFaultEvent)).contents
                _self._dispatch(raw)
            except Exception:
                pass

        bpf.open_perf_buffer("page_fault_events", callback=_raw_callback)  # type: ignore[attr-defined]
        self._bpf = bpf

        # Seed the BPF allowlist if PIDs were registered before load()
        for pid in self._pid_set:
            self.add_pid(pid)

        logger.debug(
            "[PageFaultProbe] loaded (pid_allowlist=%s)",
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
        logger.debug("[PageFaultProbe] detached")

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "unloaded"
        pids  = sorted(self._pid_set) if self._pid_set else "all"
        return f"PageFaultProbe(state={state!r}, pids={pids!r})"

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, raw: "_PageFaultEvent") -> None:
        """Parse the raw BPF struct, apply PID filter, dispatch callback."""
        pid = int(raw.pid)

        # Python-side PID filter (mirrors BPF-side filter; also works in tests)
        if self._pid_set and pid not in self._pid_set:
            return

        event = MemguardBPFEvent(
            ts_ns          = int(raw.timestamp_ns),
            event_type     = EVENT_PAGE_FAULT,
            pressure_bytes = 0,
            pid            = pid,
            cgroup_id      = "",
            extra          = {
                "fault_address": int(raw.fault_address),
                "error_code":    int(raw.error_code),
            },
        )

        # Always update the rolling window
        self._window.add(1.0)

        if self._on_fault is not None:
            try:
                self._on_fault(event)
            except Exception:
                logger.debug("[PageFaultProbe] on_fault raised", exc_info=True)
