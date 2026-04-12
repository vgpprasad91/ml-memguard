"""eBPF probe programs for memguard.

Each sub-module pairs a BPF C source file (.bpf.c) with a Python wrapper
that loads it via bcc, parses the perf-output ring buffer, and dispatches
typed :class:`~memory_guard.ebpf._event.MemguardBPFEvent` callbacks.

Available probes
----------------
:class:`CgroupMemoryHighProbe`
    Attaches to ``cgroup:cgroup_memory_high`` (Linux ≥ 5.8 + cgroup v2).
    Fires 200–500 ms before the OOM killer acts; exports ``pressure_bytes``
    (bytes the cgroup is over its ``memory.high`` watermark).

:class:`PageFaultProbe`
    Attaches to ``exceptions:page_fault_user`` (Linux ≥ 4.9).
    Detects when a GPU-serving process starts faulting into swap — the
    "silent kill" scenario the VLLMWatchdog currently misses entirely.
    Exposes ``fault_rate_per_s`` rolling-window metric.

:class:`MmapGrowthProbe`
    Attaches to ``syscalls:sys_enter_mmap`` and ``syscalls:sys_enter_brk``
    (Linux ≥ 4.9).  Tracks anonymous memory commitment per PID before it
    appears in RSS or any vLLM metric.  Exposes ``growth_rate_mbps``.
"""

from .cgroup_memory_high import CgroupMemoryHighProbe
from .mmap_growth import MmapGrowthProbe
from .page_fault import PageFaultProbe

__all__ = [
    "CgroupMemoryHighProbe",
    "PageFaultProbe",
    "MmapGrowthProbe",
]
