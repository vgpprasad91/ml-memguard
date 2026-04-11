"""memory-guard — Cross-platform memory guard for ML training.

Prevents OOM crashes on Apple Silicon, CUDA GPUs, and CPU-only systems
by proactive memory estimation before training and runtime pressure
monitoring during training.

Works with any training framework: mlx_lm, Unsloth, HuggingFace, PyTorch.

Quick start:
    from memory_guard import MemoryGuard

    guard = MemoryGuard.auto()
    safe = guard.preflight(
        model_params=9e9, model_bits=4,
        hidden_dim=4096, num_heads=32, num_layers=32,
        batch_size=4, seq_length=2048, lora_rank=32, lora_layers=16,
    )
    print(safe)  # Auto-downgraded if needed

    with guard.monitor(batch_size=safe.batch_size) as mon:
        for step in range(1000):
            train_step(batch_size=mon.current_batch_size)
"""

__version__ = "0.3.0"

# ---------------------------------------------------------------------------
# Lazy imports — HF/Unsloth adapters are only resolved on first attribute
# access so that ``import memory_guard`` works on a torch-free machine.
# ---------------------------------------------------------------------------

_LAZY_ADAPTER_ATTRS = {
    "MemoryGuardCallback": "memory_guard.adapters.huggingface",
    "guard_trainer": "memory_guard.adapters.huggingface",
    "guard_unsloth_model": "memory_guard.adapters.unsloth",
    "guard_sft_trainer": "memory_guard.adapters.unsloth",
    "guard_vllm": "memory_guard.adapters.vllm",
    "guard_sglang": "memory_guard.adapters.sglang",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_ADAPTER_ATTRS:
        import importlib
        mod = importlib.import_module(_LAZY_ADAPTER_ATTRS[name])
        obj = getattr(mod, name)
        # Cache on the module so subsequent accesses skip __getattr__
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'memory_guard' has no attribute {name!r}")


from .estimator import (
    InferenceServingEstimate,
    MemoryEstimate,
    ModelSpec,
    TrainSpec,
    FinetuneMethod,
    ModelArch,
    estimate_training_memory,
    estimate_inference_memory,
    estimate_serving_memory,
)
from .platforms import (
    detect_platform,
    get_available_memory_mb,
    get_memory_pressure,
    get_mlx_active_memory_mb,
    get_mlx_peak_memory_mb,
    reset_mlx_peak_memory,
    PlatformInfo,
    Backend,
)
from .calibration import CalibrationStore, record_training_result
from .reward import RewardSignal, compute_reward
from .guard import MemoryGuard, SafeConfig, InferenceSafeConfig
from .inference_monitor import KVCacheMonitor
from .monitor import RuntimeMonitor
from .cuda_recovery import CUDAOOMRecovery
from .downgrade import auto_downgrade, DowngradeResult

__all__ = [
    "MemoryGuard",
    "SafeConfig",
    "InferenceSafeConfig",
    "InferenceServingEstimate",
    "MemoryEstimate",
    "ModelSpec",
    "TrainSpec",
    "FinetuneMethod",
    "ModelArch",
    "estimate_training_memory",
    "estimate_inference_memory",
    "estimate_serving_memory",
    "detect_platform",
    "get_available_memory_mb",
    "get_memory_pressure",
    "get_mlx_active_memory_mb",
    "get_mlx_peak_memory_mb",
    "reset_mlx_peak_memory",
    "PlatformInfo",
    "Backend",
    "RuntimeMonitor",
    "KVCacheMonitor",
    "CUDAOOMRecovery",
    "CalibrationStore",
    "record_training_result",
    "RewardSignal",
    "compute_reward",
    "auto_downgrade",
    "DowngradeResult",
    # HuggingFace adapter (lazy — resolved only on access)
    "MemoryGuardCallback",
    "guard_trainer",
    # Unsloth adapter (lazy — resolved only on access)
    "guard_unsloth_model",
    "guard_sft_trainer",
    # vLLM adapter (lazy — resolved only on access)
    "guard_vllm",
    # SGLang adapter (lazy — resolved only on access)
    "guard_sglang",
]
