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
"""

from .cgroup_memory_high import CgroupMemoryHighProbe

__all__ = ["CgroupMemoryHighProbe"]
