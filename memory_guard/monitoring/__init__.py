"""Runtime monitoring — watch live processes and emit signals."""
from .monitor import RuntimeMonitor
from .inference_monitor import KVCacheMonitor
from .platforms import (
    Backend,
    PlatformInfo,
    detect_platform,
    get_available_memory_mb,
    get_memory_pressure,
    get_mlx_active_memory_mb,
    get_mlx_peak_memory_mb,
    reset_mlx_peak_memory,
)
from .cuda_recovery import CUDAOOMRecovery

__all__ = [
    "RuntimeMonitor",
    "KVCacheMonitor",
    "Backend",
    "PlatformInfo",
    "detect_platform",
    "get_available_memory_mb",
    "get_memory_pressure",
    "get_mlx_active_memory_mb",
    "get_mlx_peak_memory_mb",
    "reset_mlx_peak_memory",
    "CUDAOOMRecovery",
]
