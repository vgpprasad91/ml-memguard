"""BPF programs for cgroup memory pressure monitoring.

Attaches two kernel instrumentation points:

1. ``TRACEPOINT_PROBE(cgroup, memory_high)``
   Fires when a cgroup exceeds its ``memory.high`` soft limit — the
   kernel throttles the cgroup's processes before the OOM killer fires.
   Available on Linux ≥ 5.8 (cgroup v2).

2. ``kprobe`` on ``mem_cgroup_handle_oom``
   Fires when the kernel OOM killer selects a cgroup victim.  This fires
   later than ``memory_high`` and means the process is about to be killed.

Both probes write into a shared ``BPF_PERF_OUTPUT`` ring buffer named
``mem_events``.  The Python side reads events from that buffer and wraps
each one in a :class:`MemPressureEvent` namedtuple::

    MemPressureEvent(level=0, cgroup_path=b'/kubepods/pod-abc', timestamp_ns=...)

Level constants:
    :data:`LEVEL_HIGH` = 0 — memory.high threshold crossed (soft limit)
    :data:`LEVEL_OOM`  = 1 — OOM killer invoked (hard limit exhausted)
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections import namedtuple
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: Emitted for every cgroup memory pressure event.
MemPressureEvent = namedtuple(
    "MemPressureEvent",
    ["level", "cgroup_path", "timestamp_ns"],
)

LEVEL_HIGH: int = 0  #: memory.high threshold crossed (soft limit, throttling)
LEVEL_OOM:  int = 1  #: OOM killer invoked (hard limit exhausted)

# ---------------------------------------------------------------------------
# BPF C program
# ---------------------------------------------------------------------------

#: Embedded BPF C program loaded by :class:`CgroupMemoryProbe`.
BPF_PROG: str = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

#define CGROUP_PATH_LEN 128

/* Event struct — must match _MemEvent below exactly (same field order/types). */
struct mem_event_t {
    u64  timestamp_ns;
    u32  level;                        /* 0=LEVEL_HIGH, 1=LEVEL_OOM */
    u32  pid;
    char cgroup_path[CGROUP_PATH_LEN];
};

BPF_PERF_OUTPUT(mem_events);

/* ---- cgroup:memory_high tracepoint (Linux >= 5.8, cgroup v2) ---- */
/* Fires when a cgroup's memory usage exceeds its memory.high limit.  */
/* args->path   — the cgroup path in the cgroup fs                    */
TRACEPOINT_PROBE(cgroup, memory_high) {
    struct mem_event_t ev = {};
    ev.timestamp_ns = bpf_ktime_get_ns();
    ev.level        = 0;   /* LEVEL_HIGH */
    ev.pid          = (u32)(bpf_get_current_pid_tgid() >> 32);
    bpf_probe_read_kernel_str(ev.cgroup_path, sizeof(ev.cgroup_path), args->path);
    mem_events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}

/* ---- kprobe: mem_cgroup_handle_oom ---- */
/* Fires when the OOM killer selects this cgroup as a victim.         */
int kprobe__mem_cgroup_handle_oom(struct pt_regs *ctx) {
    struct mem_event_t ev = {};
    ev.timestamp_ns = bpf_ktime_get_ns();
    ev.level        = 1;   /* LEVEL_OOM */
    ev.pid          = (u32)(bpf_get_current_pid_tgid() >> 32);
    /* cgroup path unavailable from kprobe args — leave zeroed */
    mem_events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}
"""

# ---------------------------------------------------------------------------
# ctypes mirror of the BPF struct
# ---------------------------------------------------------------------------

class _MemEvent(ctypes.Structure):
    """Mirrors ``struct mem_event_t`` in the BPF program above.

    Must match field order and sizes exactly so the perf_buffer callback
    can cast the raw bytes from the ring buffer.
    """
    _fields_: List = [
        ("timestamp_ns", ctypes.c_uint64),
        ("level",        ctypes.c_uint32),
        ("pid",          ctypes.c_uint32),
        ("cgroup_path",  ctypes.c_char * 128),
    ]


# ---------------------------------------------------------------------------
# CgroupMemoryProbe
# ---------------------------------------------------------------------------

class CgroupMemoryProbe:
    """Load and manage the cgroup memory BPF probes.

    Requires:
      - Linux kernel ≥ 5.8 (cgroup v2, ``cgroup:memory_high`` tracepoint)
      - ``bcc`` Python package installed  (``pip install bcc``)
      - Running as root or with ``CAP_BPF`` + ``CAP_PERFMON``

    Parameters
    ----------
    on_event:
        Callback ``Callable[[MemPressureEvent], None]`` invoked for every
        memory pressure event received from the perf buffer.
        Called from the perf buffer polling thread — must be thread-safe.

    Raises
    ------
    ImportError
        If ``bcc`` is not installed.
    PermissionError
        If the process lacks the required Linux capabilities.
    OSError
        If the kernel does not support the required tracepoints or kprobes.
    """

    def __init__(self, on_event: Callable[[MemPressureEvent], None]) -> None:
        self._on_event = on_event
        self._bpf: Optional[object] = None   # bcc.BPF instance (typed as object)

    def load(self) -> None:
        """Compile and load the BPF program; attach all probes.

        Raises ``ImportError`` if bcc is not installed.
        Raises ``PermissionError`` if the process is not root / lacks CAP_BPF.
        Raises ``OSError`` if the kernel rejects the program or probes.
        """
        if sys.platform != "linux":
            raise OSError(
                "CgroupMemoryProbe is Linux-only — "
                f"current platform is {sys.platform!r}"
            )
        if os.getuid() != 0 and not _has_cap_bpf():
            raise PermissionError(
                "CgroupMemoryProbe requires root or CAP_BPF + CAP_PERFMON. "
                "Re-run with sudo or grant the capabilities to the interpreter."
            )
        try:
            from bcc import BPF  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "bcc (BPF Compiler Collection) is not installed. "
                "Install it with: pip install bcc  "
                "(also requires libbcc-dev on the host)"
            ) from exc

        self._bpf = BPF(text=BPF_PROG)

        # Register the perf buffer callback
        def _raw_callback(cpu: int, data: ctypes.c_void_p, size: int) -> None:
            ev = ctypes.cast(data, ctypes.POINTER(_MemEvent)).contents
            event = MemPressureEvent(
                level        = int(ev.level),
                cgroup_path  = ev.cgroup_path.decode("utf-8", errors="replace"),
                timestamp_ns = int(ev.timestamp_ns),
            )
            try:
                self._on_event(event)
            except Exception:
                pass  # never crash the perf_buffer thread

        self._bpf.open_perf_buffer("mem_events", callback=_raw_callback)  # type: ignore[attr-defined]

    def poll(self, timeout_ms: int = 10) -> None:
        """Drain the perf buffer and dispatch callbacks.

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

    @property
    def is_loaded(self) -> bool:
        """True once :meth:`load` has completed successfully."""
        return self._bpf is not None

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "unloaded"
        return f"CgroupMemoryProbe(state={state!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cap_bpf() -> bool:
    """Return True if the process has CAP_BPF and CAP_PERFMON capabilities."""
    try:
        import struct
        # CAP_BPF = 39, CAP_PERFMON = 38 (Linux 5.8+)
        # Read effective capabilities from /proc/self/status
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    cap_eff = int(line.split(":")[1].strip(), 16)
                    cap_bpf     = (cap_eff >> 39) & 1
                    cap_perfmon = (cap_eff >> 38) & 1
                    return bool(cap_bpf and cap_perfmon)
    except Exception:
        pass
    return False
