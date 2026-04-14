"""vLLM adapter for memory-guard.

Provides:
    guard_vllm(llm, guard=None, **preflight_overrides) -> InferenceSafeConfig

    One-call setup: introspects the running engine, calls
    ``guard.preflight_inference()``, wires a ``KVCacheMonitor`` to the block
    manager, and returns an ``InferenceSafeConfig`` with ``safe.monitor`` set.

Design contract (ADR 003)
--------------------------
``guard_vllm`` never mutates the running engine.  The ``KVCacheMonitor``
at ``safe.monitor`` only reads ``block_manager.get_num_free_gpu_blocks()``
and ``get_num_total_gpu_blocks()`` from a daemon thread.  All load-shedding
decisions are delegated to the caller via ``on_warning`` / ``on_shed_load``
callbacks on the monitor.

KV cache back-calculation
--------------------------
When ``engine.cache_config.num_gpu_blocks`` is available, the adapter
derives the actual maximum concurrent sequences that vLLM can serve at
``max_seq_len`` tokens each:

    blocks_per_seq  = ceil(max_seq_len / block_size)   # block_size default 16
    actual_max_seqs = num_gpu_blocks // blocks_per_seq

This ensures the ``max_num_seqs`` passed to ``preflight_inference()`` is
calibrated to vLLM's real pre-allocation rather than an external guess.
The ``gpu_memory_utilization`` field is then refined using the actual KV MB:

    actual_kv_mb = num_gpu_blocks × block_size × 2 × num_layers
                   × num_kv_heads × head_dim × dtype_bytes  (in bytes / MB)

Usage::

    from vllm import LLM
    from memory_guard import guard_vllm

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", ...)

    safe = guard_vllm(llm)
    print(safe)                         # InferenceSafeConfig
    print(safe.max_num_seqs)            # safe --max-num-seqs
    print(safe.gpu_memory_utilization)  # safe --gpu-memory-utilization

    with safe.monitor.session():
        server.serve_forever()
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from ..constants import MONITOR_POLL_INTERVAL
from ..guard import InferenceSafeConfig, MemoryGuard
from ..monitoring.inference_monitor import KVCacheMonitor
from .base import optional_import

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def guard_vllm(
    llm: object,
    guard: Optional[MemoryGuard] = None,
    **preflight_overrides,
) -> InferenceSafeConfig:
    """Preflight check and KV cache monitor for a vLLM engine.

    Introspects ``llm.llm_engine.model_config`` for architecture fields,
    calls ``guard.preflight_inference()``, then constructs a
    ``KVCacheMonitor`` whose ``poll_fn`` reads
    ``llm.llm_engine.scheduler.block_manager.get_num_free_gpu_blocks()``
    and total blocks.  The monitor is unstarted — use ``safe.monitor.start()``
    or ``with safe.monitor.session(): ...`` when ready to serve.

    ``cache_config.num_gpu_blocks`` is read to back-calculate the actual KV
    cache capacity vLLM pre-allocated, so the preflight estimate and the live
    utilization signal are on the same scale.

    Args:
        llm:                  A ``vllm.LLM``, ``vllm.AsyncLLMEngine``, or
                              bare ``vllm.LLMEngine`` instance.
        guard:                Optional ``MemoryGuard`` (auto-created if None).
        **preflight_overrides: Any keyword argument accepted by
                              ``MemoryGuard.preflight_inference()``.  Values
                              passed here override the introspected defaults.
                              Common overrides: ``max_num_seqs``,
                              ``max_seq_len``, ``model_bits``.

    Returns:
        ``InferenceSafeConfig`` with ``safe.monitor`` set to a wired-but-
        unstarted ``KVCacheMonitor``.

    Raises:
        ImportError: if vLLM is not installed
            (``pip install ml-memguard[vllm]``).
    """
    optional_import("vllm", "vllm")

    if guard is None:
        guard = MemoryGuard.auto(enable_calibration=False)

    engine = _get_llm_engine(llm)
    info = _extract_model_info(engine)

    # --- back-calculate from vLLM's actual block allocation -----------------
    cache_config = getattr(engine, "cache_config", None)
    num_gpu_blocks: int = int(getattr(cache_config, "num_gpu_blocks", 0) or 0)
    block_size: int = int(getattr(cache_config, "block_size", 16) or 16)
    block_size = max(1, block_size)

    if num_gpu_blocks > 0 and "max_num_seqs" not in preflight_overrides:
        blocks_per_seq = max(1, math.ceil(info["max_seq_len"] / block_size))
        actual_max_seqs = max(1, num_gpu_blocks // blocks_per_seq)
        preflight_overrides = {"max_num_seqs": actual_max_seqs, **preflight_overrides}
        logger.debug(
            "[memory-guard] vLLM cache_config: %d blocks (block_size=%d) "
            "→ %d max_num_seqs at seq_len=%d",
            num_gpu_blocks, block_size, actual_max_seqs, info["max_seq_len"],
        )
    elif "max_num_seqs" not in preflight_overrides:
        sc = getattr(engine, "scheduler_config", None)
        preflight_overrides = {
            "max_num_seqs": getattr(sc, "max_num_seqs", 256) if sc else 256,
            **preflight_overrides,
        }

    # --- run preflight with introspected + caller-supplied values -----------
    safe = guard.preflight_inference(
        model_params=info["model_params"],
        model_bits=info["model_bits"],
        num_kv_heads=info["num_kv_heads"],
        head_dim=info["head_dim"],
        num_layers=info["num_layers"],
        max_seq_len=info["max_seq_len"],
        dtype_bytes=info["dtype_bytes"],
        hidden_dim=info["hidden_dim"],
        **preflight_overrides,
    )

    # --- refine gpu_memory_utilization from actual KV allocation ------------
    if num_gpu_blocks > 0:
        actual_kv_mb = (
            num_gpu_blocks
            * block_size
            * 2
            * info["num_layers"]
            * info["num_kv_heads"]
            * info["head_dim"]
            * info["dtype_bytes"]
        ) / (1024 * 1024)
        available = guard.available_mb
        if available > 0:
            safe.gpu_memory_utilization = round(
                min(0.95, actual_kv_mb / available), 4
            )
        logger.debug(
            "[memory-guard] vLLM actual KV cache: %.0f MB "
            "(%d blocks × block_size %d)",
            actual_kv_mb, num_gpu_blocks, block_size,
        )

    # --- wire monitor to block manager --------------------------------------
    safe.monitor = KVCacheMonitor(poll_fn=_make_poll_fn(engine))

    return safe


# ---------------------------------------------------------------------------
# Internal helpers — engine introspection
# ---------------------------------------------------------------------------


def _get_llm_engine(llm: object) -> object:
    """Return the underlying ``LLMEngine`` from any vLLM wrapper type.

    - ``vllm.LLM``            exposes the engine as ``llm.llm_engine``
    - ``vllm.AsyncLLMEngine`` exposes it as ``llm.engine``
    - ``vllm.LLMEngine``      is returned directly
    """
    if hasattr(llm, "llm_engine"):
        return llm.llm_engine
    if hasattr(llm, "engine"):
        return llm.engine
    return llm


def _extract_model_info(engine: object) -> dict:
    """Read model architecture metadata from a vLLM LLMEngine.

    Returns a dict with keys matching ``preflight_inference()`` parameters.
    Falls back to conservative defaults for any missing field.
    """
    mc = getattr(engine, "model_config", None)
    hf = getattr(mc, "hf_config", None) if mc else None

    # --- attention geometry -------------------------------------------------
    num_heads: int = getattr(hf, "num_attention_heads", 32) if hf else 32
    num_kv_heads: int = (
        getattr(hf, "num_key_value_heads", num_heads) if hf else num_heads
    )
    num_layers: int = getattr(hf, "num_hidden_layers", 32) if hf else 32
    hidden_size: int = getattr(hf, "hidden_size", 4096) if hf else 4096
    head_dim: int = hidden_size // num_heads if num_heads > 0 else 128

    # --- sequence length ----------------------------------------------------
    max_seq_len: int = getattr(mc, "max_model_len", 8192) if mc else 8192

    # --- dtype → bytes ------------------------------------------------------
    dtype = getattr(mc, "dtype", None) if mc else None
    dtype_str = str(dtype) if dtype is not None else ""
    if "float32" in dtype_str:
        dtype_bytes = 4
    elif "int8" in dtype_str:
        dtype_bytes = 1
    else:
        dtype_bytes = 2  # fp16 / bf16

    # --- quantization → model_bits ------------------------------------------
    quantization: Optional[str] = getattr(mc, "quantization", None) if mc else None
    _4BIT = {"awq", "awq_marlin", "gptq", "gptq_marlin", "squeezellm", "fp8"}
    _8BIT = {"bitsandbytes", "smooth_quant"}
    if quantization in _4BIT:
        model_bits = 4
    elif quantization in _8BIT:
        model_bits = 8
    else:
        model_bits = 16

    # --- parameter count ----------------------------------------------------
    num_params: int = getattr(hf, "num_parameters", 0) if hf else 0
    if num_params == 0:
        num_params = int(12 * (hidden_size ** 2) * num_layers)

    return {
        "model_params": num_params,
        "model_bits": model_bits,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "max_seq_len": max_seq_len,
        "dtype_bytes": dtype_bytes,
        "hidden_dim": hidden_size,
    }


# ---------------------------------------------------------------------------
# Internal helpers — poll_fn construction
# ---------------------------------------------------------------------------


def _make_poll_fn(engine: object):
    """Return a zero-argument callable that reads ``(used, total)`` GPU blocks.

    Accesses ``engine.scheduler.block_manager``.  In vLLM ≥ 0.4.0 the
    scheduler may be a list; the first element is used in that case.
    Falls back to a null poll (returns ``(0, 1)``) if the block manager is
    not accessible, with a one-time warning.
    """
    scheduler = getattr(engine, "scheduler", None)
    if isinstance(scheduler, list):
        scheduler = scheduler[0] if scheduler else None

    block_manager = getattr(scheduler, "block_manager", None) if scheduler else None

    if block_manager is None:
        logger.warning(
            "[memory-guard] vLLM block manager not found on engine.scheduler; "
            "KVCacheMonitor will return 0 utilization.  "
            "Verify you are using a supported vLLM version."
        )

        def _null_poll() -> tuple[int, int]:
            return 0, 1

        return _null_poll

    def _poll() -> tuple[int, int]:
        free: int = block_manager.get_num_free_gpu_blocks()
        total: int = block_manager.get_num_total_gpu_blocks()
        used: int = total - free
        return used, total

    return _poll
