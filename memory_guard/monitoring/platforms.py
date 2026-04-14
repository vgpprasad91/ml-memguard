"""Platform detection and memory introspection.

Cross-platform memory introspection:
  - macOS Apple Silicon: Mach host_statistics via ctypes (no subprocess)
  - macOS Intel: Same API, 4KB page size (not 16KB)
  - Linux: /proc/meminfo, PSI (/proc/pressure/memory), cgroups v1/v2
  - Linux containers: nested cgroups, systemd slices, Docker/Podman/K8s
  - Windows: GlobalMemoryStatusEx + CUDA for GPU memory
  - CUDA: torch.cuda (NVIDIA GPUs)
  - ROCm/HIP: torch with HIP backend (AMD GPUs)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..constants import FALLBACK_AVAILABLE_RATIO

# Global lock for ctypes Mach kernel calls. ctypes releases the GIL
# during foreign function calls, which means our monitor's background
# thread could race with the training thread on Mach APIs.
# Ref: https://github.com/python/cpython/issues/127945
_mach_lock = threading.Lock()

logger = logging.getLogger(__name__)


# Backend Enum

class Backend(Enum):
    APPLE_SILICON = "apple_silicon"
    APPLE_INTEL = "apple_intel"
    CUDA = "cuda"
    ROCM = "rocm"
    LINUX_CPU = "linux_cpu"
    WINDOWS_CPU = "windows_cpu"
    UNKNOWN = "unknown"


# Platform Info

@dataclass
class PlatformInfo:
    """Detected platform and memory capabilities."""
    backend: Backend
    system: str
    arch: str
    total_memory_mb: float
    gpu_memory_mb: float
    unified_memory: bool
    chip_name: str
    num_gpus: int = 1
    page_size: int = 4096
    in_container: bool = False
    container_memory_limit_mb: Optional[float] = None
    swap_available_mb: float = 0.0     # Available swap / compressor headroom


_platform_cache = None

def detect_platform() -> PlatformInfo:
    """Auto-detect hardware platform and memory configuration. Cached."""
    global _platform_cache
    if _platform_cache is not None:
        return _platform_cache
    system = platform.system()
    arch = platform.machine()

    if system == "Darwin":
        _platform_cache = _detect_macos(arch)
    elif system == "Linux":
        _platform_cache = _detect_linux(arch)
    elif system == "Windows":
        _platform_cache = _detect_windows(arch)
    else:
        _platform_cache = PlatformInfo(
            backend=Backend.UNKNOWN, system=system, arch=arch,
            total_memory_mb=_get_total_ram_fallback(), gpu_memory_mb=0,
            unified_memory=False, chip_name="unknown",
        )
    return _platform_cache


# macOS Detection

def _detect_macos(arch: str) -> PlatformInfo:
    is_arm = arch == "arm64"
    page_size = _mach_page_size()
    total_mb = _sysctl_int64("hw.memsize") / (1024 * 1024)

    chip = "unknown"
    try:
        chip = _sysctl_string("machdep.cpu.brand_string").lower().replace(" ", "_")
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    # macOS compressor acts as swap — compressed pages are effectively extra headroom
    # Detect swap via sysctl (vm.swapusage returns a struct, use string parse)
    # This is a one-time call during platform detection, not in the hot path.
    swap_mb = 0.0
    try:
        swap_str = _sysctl_string("vm.swapusage")
        # Format: "total = 6144.00M  used = 2048.00M  free = 4096.00M  ..."
        # Parse as key=value pairs to avoid index() ambiguity when used==free
        parts = swap_str.replace("=", " ").split()
        for i, token in enumerate(parts):
            if token == "free" and i + 1 < len(parts):
                val = parts[i + 1].rstrip("M.,")
                swap_mb = float(val)
                break
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    return PlatformInfo(
        backend=Backend.APPLE_SILICON if is_arm else Backend.APPLE_INTEL,
        system="Darwin", arch=arch,
        total_memory_mb=total_mb, gpu_memory_mb=0,
        unified_memory=is_arm, chip_name=chip,
        page_size=page_size,
        swap_available_mb=swap_mb,
    )


_page_size_cache = None
_mach_host_cache = None


def _clear_mach_caches():
    """Reset cached Mach handles after fork().

    Mach ports are per-task and become invalid in child processes.
    Without this, forked workers (multiprocessing, DataLoader) get
    silent KERN_INVALID_ARGUMENT from host_statistics() and fall
    back to FALLBACK_PRESSURE with no error.
    """
    global _libc_cache, _page_size_cache, _mach_host_cache, _platform_cache
    _libc_cache = None
    _page_size_cache = None
    _mach_host_cache = None
    _platform_cache = None


# Register fork handler so forked children get fresh Mach ports
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_clear_mach_caches)

def _mach_page_size() -> int:
    """Get page size via Mach kernel (16KB on ARM64, 4KB on Intel). Cached, thread-safe."""
    global _page_size_cache
    if _page_size_cache is not None:
        return _page_size_cache
    with _mach_lock:
        if _page_size_cache is not None:  # Double-check after acquiring lock
            return _page_size_cache
        try:
            libc = _get_libc()
            page_size = ctypes.c_ulong.in_dll(libc, "vm_page_size")
            _page_size_cache = page_size.value
        except Exception:
            _page_size_cache = 16384 if platform.machine() == "arm64" else 4096
        return _page_size_cache


_libc_cache = None

def _get_libc():
    """Get cached libc handle with proper argtypes set (ARM64 ABI compliance)."""
    global _libc_cache
    if _libc_cache is not None:
        return _libc_cache
    libc = ctypes.CDLL(ctypes.util.find_library("c"))

    # ARM64 Apple Silicon requires explicit argtypes for correct calling
    # convention. Without these, arguments can be silently corrupted
    # due to variadic vs non-variadic ABI differences.
    # Ref: https://bugs.python.org/issue42880
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,              # name
        ctypes.c_void_p,              # oldp
        ctypes.POINTER(ctypes.c_size_t),  # oldlenp
        ctypes.c_void_p,              # newp
        ctypes.c_size_t,              # newlen
    ]
    libc.sysctlbyname.restype = ctypes.c_int

    libc.mach_host_self.argtypes = []
    libc.mach_host_self.restype = ctypes.c_uint32  # mach_port_t

    libc.host_statistics.argtypes = [
        ctypes.c_uint32,              # host (mach_port_t)
        ctypes.c_int,                 # flavor
        ctypes.c_void_p,              # host_info_out
        ctypes.POINTER(ctypes.c_uint32),  # host_info_outCnt
    ]
    libc.host_statistics.restype = ctypes.c_int  # kern_return_t

    _libc_cache = libc
    return libc


def _sysctl_int64(name: str) -> int:
    """Read a sysctl value as int64 via ctypes (no subprocess).

    Uses explicit argtypes for ARM64 ABI compliance.
    Thread-safe: acquires _mach_lock before calling into kernel.
    """
    libc = _get_libc()
    buf = ctypes.c_int64(0)
    buf_size = ctypes.c_size_t(8)

    with _mach_lock:
        ret = libc.sysctlbyname(
            name.encode("utf-8"),
            ctypes.byref(buf), ctypes.byref(buf_size),
            None, ctypes.c_size_t(0),
        )
    if ret != 0:
        raise OSError(f"sysctlbyname({name}) failed: {ret}")
    return buf.value


def _sysctl_string(name: str) -> str:
    """Read a sysctl value as string via ctypes. Thread-safe."""
    libc = _get_libc()
    buf_size = ctypes.c_size_t(256)
    buf = ctypes.create_string_buffer(256)

    with _mach_lock:
        ret = libc.sysctlbyname(
            name.encode("utf-8"),
            buf, ctypes.byref(buf_size),
            None, ctypes.c_size_t(0),
        )
    if ret != 0:
        raise OSError(f"sysctlbyname({name}) failed")
    return buf.value.decode("utf-8")


# Mach VM Statistics (no subprocess)

# Mach host_statistics constants
# Use HOST_VM_INFO (not 64) — the 32-bit version uses natural_t (uint32) counts
# which are correct. The 64-bit version's struct layout varies across macOS versions.
HOST_VM_INFO = 2
HOST_VM_INFO_COUNT = 15  # sizeof(vm_statistics_data_t) / sizeof(integer_t)

class _VMStatistics(ctypes.Structure):
    """Mach vm_statistics_data_t — 32-bit counters (stable across macOS versions)."""
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint32),
        ("reactivations", ctypes.c_uint32),
        ("pageins", ctypes.c_uint32),
        ("pageouts", ctypes.c_uint32),
        ("faults", ctypes.c_uint32),
        ("cow_faults", ctypes.c_uint32),
        ("lookups", ctypes.c_uint32),
        ("hits", ctypes.c_uint32),
        ("purgeable_count", ctypes.c_uint32),
        ("purges", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
    ]


def _mach_vm_stats() -> Optional[_VMStatistics]:
    """Get VM statistics via Mach host_statistics (zero-cost, no fork).

    Uses _get_libc() with explicit argtypes for ARM64 ABI compliance.
    Thread-safe: acquires _mach_lock before calling into kernel.
    """
    if platform.system() != "Darwin":
        return None
    try:
        libc = _get_libc()

        with _mach_lock:
            # Cache mach_host_self() to avoid leaking Mach port send rights.
            # Each call allocates a port right that's never deallocated.
            # At 5s polling, that's 720 leaks/hour → hits 65536 limit in ~91 hours.
            global _mach_host_cache
            if _mach_host_cache is None:
                _mach_host_cache = libc.mach_host_self()
            host = _mach_host_cache
            stats = _VMStatistics()
            count = ctypes.c_uint32(HOST_VM_INFO_COUNT)
            ret = libc.host_statistics(
                host, HOST_VM_INFO,
                ctypes.byref(stats), ctypes.byref(count),
            )

        if ret != 0:
            return None
        return stats
    except Exception:
        return None


def _mach_memory_pressure() -> float:
    """Get macOS memory pressure via Mach kernel (no subprocess).

    Returns 0.0 (no pressure) to 1.0 (critical).
    """
    # Method 1: sysctl kern.memorystatus_level (0-100, 100 = no pressure)
    try:
        level = _sysctl_int64("kern.memorystatus_level")
        return max(0.0, min(1.0, 1.0 - (level / 100.0)))
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    # Method 2: Mach host_statistics — (active+wired)/total
    stats = _mach_vm_stats()
    if stats:
        page_size = _mach_page_size()
        active = stats.active_count * page_size
        wired = stats.wire_count * page_size
        total = _sysctl_int64("hw.memsize")
        if total > 0:
            used_ratio = (active + wired) / total
            return max(0.0, min(1.0, used_ratio))

    from ..constants import FALLBACK_PRESSURE; return FALLBACK_PRESSURE


def _mach_available_mb() -> float:
    """Get available memory on macOS via multiple methods, cross-validated.

    macOS memory accounting is notoriously imprecise because:
    - vm_stat fields don't add up (malloc'd written memory is invisible)
    - inactive pages include dirty pages needing I/O
    - purgeable_count can lag behind actual reclaimable state

    Strategy: use total - (active + wired) as the upper bound of what
    COULD be freed, then apply a conservative discount. This matches
    what Activity Monitor shows as "Memory Used" (active + wired)
    vs "Memory Available" (everything else the OS can reclaim).
    """
    stats = _mach_vm_stats()
    if stats:
        page_size = _mach_page_size()
        try:
            total = _sysctl_int64("hw.memsize")
        except Exception:
            total = 0

        if total > 0:
            # Method: total - (active + wired) = reclaimable upper bound
            # This is what Activity Monitor effectively reports.
            active = stats.active_count * page_size
            wired = stats.wire_count * page_size
            used = active + wired
            reclaimable = total - used

            if reclaimable <= 0:
                # Transient state: compressor accounting can make
                # active+wired > total during pressure spikes.
                # Fall through to fallback instead of returning 0.
                logger.debug("Negative reclaimable memory (%d bytes), using fallback", reclaimable)
            else:
                from ..constants import MACOS_RECLAIMABLE_DISCOUNT
                available = reclaimable * MACOS_RECLAIMABLE_DISCOUNT
                return available / (1024 * 1024)

    # Fallback
    try:
        total = _sysctl_int64("hw.memsize")
        return (total * FALLBACK_AVAILABLE_RATIO) / (1024 * 1024)
    except Exception:
        from ..constants import FALLBACK_MEMORY_MB; return FALLBACK_MEMORY_MB


# Linux Detection

def _detect_linux(arch: str) -> PlatformInfo:
    total_mb = _linux_total_ram_mb()
    in_container, container_limit = _detect_container()

    # Check CUDA first
    cuda_info = _detect_cuda()
    if cuda_info:
        return PlatformInfo(
            backend=Backend.CUDA, system="Linux", arch=arch,
            total_memory_mb=total_mb,
            gpu_memory_mb=cuda_info["vram_mb"],
            unified_memory=False, chip_name=cuda_info["name"],
            num_gpus=cuda_info["count"],
            in_container=in_container,
            container_memory_limit_mb=container_limit,
        )

    # Check ROCm
    rocm_info = _detect_rocm()
    if rocm_info:
        return PlatformInfo(
            backend=Backend.ROCM, system="Linux", arch=arch,
            total_memory_mb=total_mb,
            gpu_memory_mb=rocm_info["vram_mb"],
            unified_memory=False, chip_name=rocm_info["name"],
            num_gpus=rocm_info["count"],
            in_container=in_container,
            container_memory_limit_mb=container_limit,
        )

    return PlatformInfo(
        backend=Backend.LINUX_CPU, system="Linux", arch=arch,
        total_memory_mb=total_mb, gpu_memory_mb=0,
        unified_memory=False, chip_name=f"cpu_{arch}",
        in_container=in_container,
        container_memory_limit_mb=container_limit,
    )


def _linux_total_ram_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return _get_total_ram_fallback()


def _detect_container() -> tuple[bool, Optional[float]]:
    """Detect if running in a container and its memory limit.

    Handles: Docker, Podman, Kubernetes, systemd-nspawn, LXC.
    Supports cgroups v1 and v2, including nested cgroups and systemd slices.
    """
    in_container = False
    limit_mb = None

    # Detect container environment
    container_signals = [
        Path("/.dockerenv").exists(),
        Path("/run/.containerenv").exists(),
        os.environ.get("KUBERNETES_SERVICE_HOST") is not None,
        os.environ.get("container") is not None,
    ]
    # Check /proc/1/cgroup for container indicators
    try:
        with open("/proc/1/cgroup") as f:
            cgroup_content = f.read()
            if any(sig in cgroup_content for sig in ["docker", "kubepods", "lxc", "containerd"]):
                container_signals.append(True)
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    in_container = any(container_signals)

    # Get cgroup memory limit
    limit_mb = _cgroup_memory_limit_mb()

    return in_container, limit_mb


def _cgroup_memory_limit_mb() -> Optional[float]:
    """Get effective cgroup memory limit in MB.

    Handles v1, v2, nested cgroups, systemd slices, Docker, K8s.

    Key insight (from production findings in CockroachDB and DuckDB):
    Refs:
      https://github.com/cockroachdb/cockroach/issues/114774
      https://github.com/duckdb/duckdb/issues/15080
    - memory.high is the throttling limit (preferred — processes get
      throttled and reclaimed here, which effectively acts as OOM)
    - memory.max is the hard kill limit (processes get OOM-killed)
    - We use min(memory.high, memory.max) as the effective limit

    For nested cgroups: walks the ENTIRE hierarchy from our cgroup
    to the root, collecting ALL limits, and returns the tightest one.
    This handles K8s pods where a parent cgroup constrains children.
    """

    # cgroups v2
    try:
        cgroup_path = None
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3 and parts[1] == "":  # cgroups v2
                    cgroup_path = parts[2]
                    break

        if cgroup_path:
            base = Path("/sys/fs/cgroup")
            current = base / cgroup_path.lstrip("/")
            tightest_limit = float("inf")
            tightest_is_high = False  # Track if limit came from memory.high

            # Walk the ENTIRE hierarchy to find tightest constraint
            while current != base.parent and current != current.parent:
                for mem_file in ["memory.high", "memory.max"]:
                    p = current / mem_file
                    if p.exists():
                        try:
                            val = p.read_text().strip()
                            if val not in ("max", "infinity"):
                                limit = int(val)
                                if limit < 1024 * 1024 * 1024 * 1024:
                                    if limit < tightest_limit:
                                        tightest_limit = limit
                                        tightest_is_high = (mem_file == "memory.high")
                        except (ValueError, OSError) as _exc:
                            logger.debug("Platform detection fallback: %s", _exc)
                current = current.parent

            if tightest_limit < float("inf"):
                limit_mb = tightest_limit / (1024 * 1024)
                # memory.high can be overshot by 10%+ under concurrent
                # allocations. Apply 90% discount to account for this.
                # Ref: https://www.kernel.org/doc/Documentation/cgroup-v2.txt
                # "memory.high [...] can be exceeded"
                if tightest_is_high:
                    from ..constants import CGROUP_HIGH_DISCOUNT
                    limit_mb *= CGROUP_HIGH_DISCOUNT
                return limit_mb
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    # cgroups v2 at standard root paths
    for mem_file in ["memory.high", "memory.max"]:
        try:
            p = Path("/sys/fs/cgroup") / mem_file
            if p.exists():
                val = p.read_text().strip()
                if val not in ("max", "infinity"):
                    limit = int(val)
                    if limit < 1024 * 1024 * 1024 * 1024:
                        limit_mb = limit / (1024 * 1024)
                        if mem_file == "memory.high":
                            from ..constants import CGROUP_HIGH_DISCOUNT
                            limit_mb *= CGROUP_HIGH_DISCOUNT
                        return limit_mb
        except Exception as _exc:
            logger.debug("Platform detection fallback: %s", _exc)

    # cgroups v1
    # Also walk hierarchy for v1 nested cgroups
    try:
        cgroup_path = None
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3 and "memory" in parts[1]:
                    cgroup_path = parts[2]
                    break

        if cgroup_path:
            base = Path("/sys/fs/cgroup/memory")
            current = base / cgroup_path.lstrip("/")
            tightest = float("inf")

            while current != base.parent and current != current.parent:
                limit_file = current / "memory.limit_in_bytes"
                if limit_file.exists():
                    try:
                        limit = int(limit_file.read_text().strip())
                        if limit < 2**62:  # Not "unlimited"
                            tightest = min(tightest, limit)
                    except (ValueError, OSError) as _exc:
                        logger.debug("Platform detection fallback: %s", _exc)
                current = current.parent

            if tightest < float("inf"):
                return tightest / (1024 * 1024)
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    # Direct v1 standard v1 paths
    for path in [
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/docker/memory.limit_in_bytes",
    ]:
        try:
            limit = int(Path(path).read_text().strip())
            if limit < 2**62:
                return limit / (1024 * 1024)
        except Exception as _exc:
            logger.debug("Platform detection fallback: %s", _exc)

    return None


def _linux_available_mb() -> float:
    """Get available memory on Linux, container-aware."""
    # Check cgroup limit first (container budget)
    limit_mb = _cgroup_memory_limit_mb()
    if limit_mb is not None:
        usage_mb = _cgroup_usage_mb()
        if usage_mb is not None:
            return limit_mb - usage_mb

    # Fallback: /proc/meminfo MemAvailable
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    return _linux_total_ram_mb() * FALLBACK_AVAILABLE_RATIO


def _cgroup_usage_mb() -> Optional[float]:
    """Get current cgroup memory usage in MB."""
    for path in [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]:
        try:
            with open(path) as f:
                return int(f.read().strip()) / (1024 * 1024)
        except Exception as _exc:
            logger.debug("Platform detection fallback: %s", _exc)
    return None


def _linux_pressure() -> float:
    """Linux memory pressure via PSI, cgroup, or /proc/meminfo."""
    # PSI (Pressure Stall Information) — best signal on modern kernels
    try:
        with open("/proc/pressure/memory") as f:
            for line in f:
                if line.startswith("some"):
                    for part in line.split():
                        if part.startswith("avg10="):
                            from ..constants import PSI_CRITICAL_THRESHOLD
                            psi = float(part.split("=")[1])
                            return min(1.0, psi / PSI_CRITICAL_THRESHOLD)
    except Exception as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    # Container-aware: check cgroup (uses memory.high as effective limit)
    limit_mb = _cgroup_memory_limit_mb()
    if limit_mb is not None:
        usage_mb = _cgroup_usage_mb()
        if usage_mb is not None and limit_mb > 0:
            return min(1.0, usage_mb / limit_mb)

    # Fallback: MemAvailable ratio
    available = _linux_available_mb()
    total = _linux_total_ram_mb()
    if total > 0:
        return max(0.0, min(1.0, 1.0 - (available / total)))

    from ..constants import FALLBACK_PRESSURE; return FALLBACK_PRESSURE


# Windows Detection

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _detect_windows(arch: str) -> PlatformInfo:
    total_mb = _windows_total_ram_mb()

    # Check CUDA (NVIDIA GPU on Windows)
    cuda_info = _detect_cuda()
    if cuda_info:
        return PlatformInfo(
            backend=Backend.CUDA, system="Windows", arch=arch,
            total_memory_mb=total_mb,
            gpu_memory_mb=cuda_info["vram_mb"],
            unified_memory=False, chip_name=cuda_info["name"],
            num_gpus=cuda_info["count"],
        )

    return PlatformInfo(
        backend=Backend.WINDOWS_CPU, system="Windows", arch=arch,
        total_memory_mb=total_mb, gpu_memory_mb=0,
        unified_memory=False, chip_name=f"cpu_{arch}",
    )


def _windows_memory_status() -> Optional[_MEMORYSTATUSEX]:
    """Get Windows memory status via kernel32."""
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat
    except Exception:
        return None


def _windows_total_ram_mb() -> float:
    stat = _windows_memory_status()
    if stat:
        return stat.ullTotalPhys / (1024 * 1024)
    return _get_total_ram_fallback()


def _windows_available_mb() -> float:
    stat = _windows_memory_status()
    if stat:
        return stat.ullAvailPhys / (1024 * 1024)
    return _windows_total_ram_mb() * FALLBACK_AVAILABLE_RATIO


def _windows_pressure() -> float:
    stat = _windows_memory_status()
    if stat:
        return stat.dwMemoryLoad / 100.0
    from ..constants import FALLBACK_PRESSURE; return FALLBACK_PRESSURE


# GPU Detection

def _detect_cuda() -> Optional[dict]:
    """Detect NVIDIA CUDA GPU via torch."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": props.name,
                "vram_mb": props.total_mem / (1024 * 1024),
                "count": torch.cuda.device_count(),
            }
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return None


def _detect_rocm() -> Optional[dict]:
    """Detect AMD ROCm/HIP GPU."""
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return {
                    "name": props.name,
                    "vram_mb": props.total_mem / (1024 * 1024),
                    "count": torch.cuda.device_count(),
                }
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return None


# CUDA Memory

def _cuda_available_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return 0.0


def _cuda_pressure() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_mem
            return allocated / total if total > 0 else 0.0
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return 0.0


# MLX Metal Memory (ground-truth on Apple Silicon)

def get_mlx_active_memory_mb() -> Optional[float]:
    """Get MLX Metal active GPU memory in MB.

    This is the GROUND TRUTH for Apple Silicon memory usage during
    training — it reads directly from the Metal allocator, unlike
    vm_stat which only sees the OS-level page tables.

    Returns None if MLX is not available.
    """
    try:
        import mlx.core as mx
        # New API (MLX >= 0.31): mx.get_active_memory()
        if hasattr(mx, 'get_active_memory'):
            return mx.get_active_memory() / (1024 * 1024)
        # Deprecated API (MLX < 0.31): mx.metal.get_active_memory()
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_active_memory'):
            return mx.metal.get_active_memory() / (1024 * 1024)
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return None


def get_mlx_peak_memory_mb() -> Optional[float]:
    """Get MLX Metal peak GPU memory in MB.

    Returns the high-water mark of Metal memory allocation since
    the last reset. Use for post-training calibration.

    Returns None if MLX is not available.
    """
    try:
        import mlx.core as mx
        if hasattr(mx, 'get_peak_memory'):
            return mx.get_peak_memory() / (1024 * 1024)
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_peak_memory'):
            return mx.metal.get_peak_memory() / (1024 * 1024)
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return None


def reset_mlx_peak_memory() -> bool:
    """Reset MLX Metal peak memory counter. Returns True if successful."""
    try:
        import mlx.core as mx
        if hasattr(mx, 'reset_peak_memory'):
            mx.reset_peak_memory()
            return True
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'reset_peak_memory'):
            mx.metal.reset_peak_memory()
            return True
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)
    return False


# Fallback

def _get_total_ram_fallback() -> float:
    """Last-resort RAM detection via os.sysconf or psutil."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024 * 1024)
    except (ValueError, OSError) as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    try:
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024)
    except ImportError as _exc:
        logger.debug("Platform detection fallback: %s", _exc)

    from ..constants import FALLBACK_MEMORY_MB
    logger.debug("All RAM detection failed, using fallback: %dMB", FALLBACK_MEMORY_MB)
    return FALLBACK_MEMORY_MB


# Public API

def get_available_memory_mb(backend: Optional[Backend] = None) -> float:
    """Get currently available memory in MB.

    For CUDA: free GPU VRAM.
    For Apple Silicon: free + inactive pages (Mach kernel, no subprocess).
    For Linux: MemAvailable (container-aware via cgroups).
    For Windows: available physical memory.
    """
    if backend is None:
        backend = detect_platform().backend

    dispatch = {
        Backend.CUDA: _cuda_available_mb,
        Backend.ROCM: _cuda_available_mb,
        Backend.APPLE_SILICON: _mach_available_mb,
        Backend.APPLE_INTEL: _mach_available_mb,
        Backend.LINUX_CPU: _linux_available_mb,
        Backend.WINDOWS_CPU: _windows_available_mb,
    }

    fn = dispatch.get(backend)
    if fn:
        result = fn()
        if result > 0:
            return result

    fallback = _get_total_ram_fallback() * FALLBACK_AVAILABLE_RATIO
    logger.debug(
        "All memory detection methods failed for backend=%s, "
        "using fallback: %.0fMB", backend, fallback
    )
    return fallback


def get_memory_pressure(backend: Optional[Backend] = None) -> float:
    """Get current memory pressure (0.0 = idle, 1.0 = critical).

    Apple Silicon: Mach host_statistics via ctypes (no subprocess).
    CUDA: allocated / total VRAM.
    Linux: PSI (avg10) or cgroup usage/limit or MemAvailable ratio.
    Windows: dwMemoryLoad from GlobalMemoryStatusEx.
    """
    if backend is None:
        backend = detect_platform().backend

    dispatch = {
        Backend.CUDA: _cuda_pressure,
        Backend.ROCM: _cuda_pressure,
        Backend.APPLE_SILICON: _mach_memory_pressure,
        Backend.APPLE_INTEL: _mach_memory_pressure,
        Backend.LINUX_CPU: _linux_pressure,
        Backend.WINDOWS_CPU: _windows_pressure,
    }

    fn = dispatch.get(backend)
    if fn:
        return fn()

    from ..constants import FALLBACK_PRESSURE; return FALLBACK_PRESSURE
