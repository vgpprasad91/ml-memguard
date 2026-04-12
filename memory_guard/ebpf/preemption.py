"""BPF program for vLLM worker preemption detection.

Attaches to the ``sched:sched_process_exit`` tracepoint, filtered on
the target PID.  Emits a :class:`PreemptionEvent` when the monitored
process exits — catching the *silent-kill* case where the Linux OOM
killer or cgroup memory controller terminates the vLLM worker without
delivering SIGKILL through the normal Python signal machinery.

Problem this solves
-------------------
The :class:`~memory_guard.watchdog.VLLMWatchdog` monitors the worker
process via ``waitpid()`` and poll ticks.  When the kernel OOM-kills
the process directly (e.g. via cgroup ``memory.limit_in_bytes``), the
exit can go undetected for 2+ seconds until the next poll tick.

This probe fires at the **exact kernel exit timestamp** — the Python
watchdog callback is invoked within microseconds of the worker's death
rather than waiting for the next poll cycle.

Usage::

    from memory_guard.ebpf.preemption import PreemptionProbe, PreemptionEvent

    def on_exit(event: PreemptionEvent) -> None:
        print(f"Worker PID {event.pid} exited (code {event.exit_code})")
        watchdog.restart_worker()

    probe = PreemptionProbe(target_pid=worker_pid, on_event=on_exit)
    probe.load()
    # probe.poll() called from a background thread

``PreemptionEvent`` fields:
    pid          — the exited process PID (always == target_pid)
    exit_code    — raw kernel exit code (signal number in low bits)
    timestamp_ns — monotonic kernel timestamp (ns since boot)
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

#: Emitted when the monitored vLLM worker process exits.
PreemptionEvent = namedtuple(
    "PreemptionEvent",
    ["pid", "exit_code", "timestamp_ns"],
)

# ---------------------------------------------------------------------------
# BPF C program template
# ---------------------------------------------------------------------------
# ``{target_pid}`` is substituted with the actual PID at load time.
# Double-braces (``{{``, ``}}``) are Python format-string escapes.

_BPF_TEMPLATE: str = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

/* ---- Filter to a single target PID ---- */
#define TARGET_PID {target_pid}u

struct preemption_event_t {{
    u64  timestamp_ns;
    u32  pid;
    int  exit_code;
}};

BPF_PERF_OUTPUT(preemption_events);

/* sched:sched_process_exit fires on every process/thread exit.
   We filter to TARGET_PID so only the vLLM worker triggers an event.
   The tracepoint args are:
     char comm[TASK_COMM_LEN]  — task name
     pid_t pid                 — task PID
     int prio                  — task priority
   The exit code is read from the task_struct directly.            */
TRACEPOINT_PROBE(sched, sched_process_exit) {{
    u32 pid = (u32)args->pid;
    if (pid != TARGET_PID) return 0;

    struct preemption_event_t ev = {{}};
    ev.timestamp_ns = bpf_ktime_get_ns();
    ev.pid          = pid;

    /* Read exit_code from the task struct.
       bit 0–7: signal that killed the process (0 = exited normally)
       bit 8–15: exit status passed to _exit()                     */
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    bpf_probe_read_kernel(&ev.exit_code, sizeof(ev.exit_code), &task->exit_code);

    preemption_events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}}
"""


# ---------------------------------------------------------------------------
# ctypes mirror of the BPF struct
# ---------------------------------------------------------------------------

class _PreemptionEvent(ctypes.Structure):
    """Mirrors ``struct preemption_event_t`` in the BPF program."""
    _fields_: List = [
        ("timestamp_ns", ctypes.c_uint64),
        ("pid",          ctypes.c_uint32),
        ("exit_code",    ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# PreemptionProbe
# ---------------------------------------------------------------------------

class PreemptionProbe:
    """Load and manage the preemption-detection BPF probe.

    Requires:
      - Linux kernel ≥ 5.4 (``sched:sched_process_exit`` tracepoint)
      - ``bcc`` Python package installed
      - Running as root or with ``CAP_BPF`` + ``CAP_PERFMON``

    Parameters
    ----------
    target_pid:
        The PID of the vLLM worker process to monitor.  Only exit events
        from this PID trigger the ``on_event`` callback.
    on_event:
        ``Callable[[PreemptionEvent], None]`` invoked when the monitored
        process exits.  Called from the perf buffer polling thread.

    Raises
    ------
    ImportError
        If ``bcc`` is not installed.
    PermissionError
        If the process lacks the required Linux capabilities.
    OSError
        If the kernel does not support the required tracepoint.
    """

    def __init__(
        self,
        target_pid: int,
        on_event: Callable[[PreemptionEvent], None],
    ) -> None:
        self._target_pid = target_pid
        self._on_event   = on_event
        self._bpf: Optional[object] = None

    def load(self) -> None:
        """Compile, load, and attach the BPF program for the target PID."""
        if sys.platform != "linux":
            raise OSError(
                f"PreemptionProbe is Linux-only — platform is {sys.platform!r}"
            )
        if os.getuid() != 0 and not _has_cap_bpf():
            raise PermissionError(
                "PreemptionProbe requires root or CAP_BPF + CAP_PERFMON."
            )
        try:
            from bcc import BPF  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("bcc is not installed — pip install bcc") from exc

        bpf_src = _BPF_TEMPLATE.format(target_pid=self._target_pid)
        self._bpf = BPF(text=bpf_src)

        def _raw_callback(cpu: int, data: ctypes.c_void_p, size: int) -> None:
            ev = ctypes.cast(data, ctypes.POINTER(_PreemptionEvent)).contents
            event = PreemptionEvent(
                pid          = int(ev.pid),
                exit_code    = int(ev.exit_code),
                timestamp_ns = int(ev.timestamp_ns),
            )
            try:
                self._on_event(event)
            except Exception:
                pass

        self._bpf.open_perf_buffer("preemption_events", callback=_raw_callback)  # type: ignore[attr-defined]

    def poll(self, timeout_ms: int = 10) -> None:
        """Drain the perf buffer and dispatch callbacks."""
        if self._bpf is not None:
            self._bpf.perf_buffer_poll(timeout=timeout_ms)  # type: ignore[attr-defined]

    def detach(self) -> None:
        """Detach the probe and release BPF resources."""
        if self._bpf is not None:
            try:
                self._bpf.cleanup()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._bpf = None

    @property
    def is_loaded(self) -> bool:
        return self._bpf is not None

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "unloaded"
        return (
            f"PreemptionProbe(target_pid={self._target_pid}, state={state!r})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cap_bpf() -> bool:
    """Return True if the process has CAP_BPF and CAP_PERFMON."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    cap_eff = int(line.split(":")[1].strip(), 16)
                    return bool((cap_eff >> 39) & 1 and (cap_eff >> 38) & 1)
    except Exception:
        pass
    return False
