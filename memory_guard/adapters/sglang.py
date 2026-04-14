"""SGLang adapter for memory-guard.

Provides:
    guard_sglang(engine, guard=None, **preflight_overrides) -> InferenceSafeConfig

    One-call setup: introspects the running engine, calls
    ``guard.preflight_inference()``, wires a smoothed ``KVCacheMonitor`` to the
    token pool, and returns an ``InferenceSafeConfig`` with ``safe.monitor`` set.

Design contract (ADR 003)
--------------------------
``guard_sglang`` never mutates the running engine.  The ``KVCacheMonitor``
at ``safe.monitor`` only reads token-pool counters from a daemon thread.
All load-shedding decisions are delegated to the caller via ``on_warning`` /
``on_shed_load`` callbacks on the monitor.

Rolling-max smoothing
---------------------
SGLang's RadixAttention prefix-cache evicts KV token slots when a cached
prefix is freed, causing utilization to drop suddenly.  Without smoothing,
such drops reset the cooldown and delay the next shed-load signal.

To avoid false positives the ``poll_fn`` applies a **3-reading rolling
maximum**: it tracks the last three raw utilization values and reports the
maximum of those readings.  This suppresses transient drops (prefix eviction)
while still allowing the signal to clear once pressure genuinely recedes for
three consecutive polls.

Token pool paths
-----------------
Tried in order:

1. ``engine.token_to_kv_pool`` (SGLang ≥ 0.3.0)
   - ``.size``                 → total token slots
   - ``.get_available_size()`` → free token slots

2. ``engine.mem_pool`` (older SGLang)
   - ``.size``         → total token slots
   - ``.available``    → free token slots (attribute, not method)

3. Null fallback — returns ``(0, 1)`` with a one-time warning.

Usage::

    import sglang as sgl
    from memory_guard import guard_sglang

    runtime = sgl.Runtime(model_path="meta-llama/Meta-Llama-3-8B", ...)

    safe = guard_sglang(runtime)
    print(safe.max_num_seqs)            # pass to --max-running-requests
    print(safe.gpu_memory_utilization)  # pass to --mem-fraction-static

    with safe.monitor.session():
        runtime.wait()
"""

from __future__ import annotations

import collections
import logging
from typing import Callable, Optional

from ..guard import InferenceSafeConfig, MemoryGuard
from ..monitoring.inference_monitor import KVCacheMonitor
from .base import optional_import

logger = logging.getLogger(__name__)

_ROLLING_MAX_WINDOW = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def guard_sglang(
    engine: object,
    guard: Optional[MemoryGuard] = None,
    **preflight_overrides,
) -> InferenceSafeConfig:
    """Preflight check and KV cache monitor for a SGLang engine.

    Introspects ``engine.server_args`` for model configuration, calls
    ``guard.preflight_inference()``, and wires a ``KVCacheMonitor`` with
    a 3-reading rolling-max ``poll_fn`` to guard against false positives
    from RadixAttention prefix-cache evictions.

    If ``engine.token_to_kv_pool.size`` (or ``engine.mem_pool.size``) is
    accessible, the adapter back-calculates ``max_num_seqs`` from the actual
    token capacity, just as the vLLM adapter uses ``cache_config.num_gpu_blocks``.

    Args:
        engine:               A ``sglang.Runtime`` (unwrapped via ``.engine``),
                              ``TokenizerManager``, or any SGLang engine object
                              with ``server_args`` and a token pool.
        guard:                Optional ``MemoryGuard`` (auto-created if None).
        **preflight_overrides: Any keyword argument accepted by
                              ``MemoryGuard.preflight_inference()``.  Values
                              passed here override the introspected defaults.

    Returns:
        ``InferenceSafeConfig`` with ``safe.monitor`` set to a wired-but-
        unstarted ``KVCacheMonitor``.

    Raises:
        ImportError: if SGLang is not installed
            (``pip install ml-memguard[sglang]``).
    """
    optional_import("sglang", "sglang")

    if guard is None:
        guard = MemoryGuard.auto(enable_calibration=False)

    inner = _get_inner_engine(engine)
    info = _extract_model_info(inner)

    # --- back-calculate from SGLang's actual token pool ---------------------
    pool = _find_token_pool(inner)
    total_token_slots: int = _pool_total(pool)

    if total_token_slots > 0 and "max_num_seqs" not in preflight_overrides:
        actual_max_seqs = max(1, total_token_slots // info["max_seq_len"])
        preflight_overrides = {"max_num_seqs": actual_max_seqs, **preflight_overrides}
        logger.debug(
            "[memory-guard] SGLang token pool: %d total slots / %d max_seq_len"
            " → %d max_running_requests",
            total_token_slots, info["max_seq_len"], actual_max_seqs,
        )
    elif "max_num_seqs" not in preflight_overrides:
        server_args = _get_server_args(inner)
        preflight_overrides = {
            "max_num_seqs": int(getattr(server_args, "max_running_requests", None) or 256),
            **preflight_overrides,
        }

    # --- run preflight ------------------------------------------------------
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

    # --- refine gpu_memory_utilization from actual pool ---------------------
    if total_token_slots > 0:
        actual_kv_mb = (
            total_token_slots
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

    # --- build smoothed monitor ---------------------------------------------
    raw_poll = _make_raw_poll_fn(inner, pool)
    safe.monitor = KVCacheMonitor(
        poll_fn=_make_smoothed_poll_fn(raw_poll, window=_ROLLING_MAX_WINDOW)
    )

    return safe


# ---------------------------------------------------------------------------
# Internal helpers — engine unwrapping
# ---------------------------------------------------------------------------


def _get_inner_engine(engine: object) -> object:
    """Unwrap a SGLang Runtime to reach the inner engine.

    ``sglang.Runtime`` exposes the inner engine via ``.engine``.
    Objects that already have ``server_args`` at the top level are the inner
    engine and are returned directly.
    """
    if hasattr(engine, "engine") and not hasattr(engine, "server_args"):
        return engine.engine
    return engine


def _get_server_args(engine: object) -> object:
    """Return ``engine.server_args`` or a sentinel that returns None for any attr."""
    return getattr(engine, "server_args", None) or _NoArgs()


class _NoArgs:
    def __getattr__(self, name: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Internal helpers — model architecture extraction
# ---------------------------------------------------------------------------


def _extract_model_info(engine: object) -> dict:
    """Read model architecture metadata from a SGLang engine.

    Reads ``server_args`` for dtype, quantization, and sequence-length
    limits.  Falls back to conservative defaults for any missing field.
    """
    server_args = _get_server_args(engine)

    # --- sequence length ----------------------------------------------------
    max_seq_len: int = int(
        getattr(server_args, "context_length", None)
        or getattr(server_args, "max_total_tokens", None)
        or 8192
    )

    # --- dtype → bytes ------------------------------------------------------
    dtype_str = str(getattr(server_args, "dtype", "") or "")
    if "float32" in dtype_str or "fp32" in dtype_str:
        dtype_bytes = 4
    elif "int8" in dtype_str:
        dtype_bytes = 1
    else:
        dtype_bytes = 2  # fp16 / bf16

    # --- quantization → model_bits ------------------------------------------
    quantization: Optional[str] = getattr(server_args, "quantization", None)
    _4BIT = {"awq", "gptq", "fp8", "marlin", "squeezellm"}
    _8BIT = {"bitsandbytes", "smooth_quant"}
    if quantization in _4BIT:
        model_bits = 4
    elif quantization in _8BIT:
        model_bits = 8
    else:
        model_bits = 16

    # --- HF config (optional; accessible via tp_worker in some versions) ----
    hf = _try_hf_config(engine)
    num_heads: int = _safe_int(getattr(hf, "num_attention_heads", None), 32)
    num_kv_heads: int = _safe_int(getattr(hf, "num_key_value_heads", None), num_heads)
    num_layers: int = _safe_int(getattr(hf, "num_hidden_layers", None), 32)
    hidden_size: int = _safe_int(getattr(hf, "hidden_size", None), 4096)
    head_dim: int = hidden_size // num_heads if num_heads > 0 else 128

    # --- parameter count ----------------------------------------------------
    num_params: int = _safe_int(getattr(hf, "num_parameters", None), 0)
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


def _try_hf_config(engine: object) -> Optional[object]:
    """Attempt to reach model.config through tp_worker.model_runner."""
    for path in (
        ("tp_worker", "model_runner", "model", "config"),
        ("scheduler", "tp_worker", "model_runner", "model", "config"),
    ):
        obj = engine
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            return obj  # full path resolved
    return None


# ---------------------------------------------------------------------------
# Internal helpers — token pool access
# ---------------------------------------------------------------------------


def _find_token_pool(engine: object) -> Optional[object]:
    """Return the first accessible token pool object, or None."""
    # SGLang >= 0.3.0
    pool = getattr(engine, "token_to_kv_pool", None)
    if pool is not None:
        return pool
    # Older SGLang
    pool = getattr(engine, "mem_pool", None)
    if pool is not None:
        return pool
    return None


def _pool_total(pool: Optional[object]) -> int:
    """Return the total token capacity of *pool*, or 0 if unknown."""
    if pool is None:
        return 0
    size = getattr(pool, "size", None)
    if size is not None:
        return _safe_int(size, 0)
    return 0


def _pool_free(pool: object) -> int:
    """Return free token slots from *pool* (0 on failure)."""
    # Preferred: callable get_available_size()
    fn = getattr(pool, "get_available_size", None)
    if callable(fn):
        return _safe_int(fn(), 0)
    # Fallback: plain attribute
    return _safe_int(getattr(pool, "available", None), 0)


# ---------------------------------------------------------------------------
# Internal helpers — poll_fn construction
# ---------------------------------------------------------------------------


def _make_raw_poll_fn(
    engine: object,
    pool: Optional[object],
) -> Callable[[], tuple[int, int]]:
    """Return the raw (unsmoothed) ``(used, total)`` poll function.

    Tries token pool → scheduler stats → null fallback.
    """
    if pool is not None:
        def _poll_pool() -> tuple[int, int]:
            total = _pool_total(pool)
            if total == 0:
                return 0, 1
            free = _pool_free(pool)
            return total - free, total

        return _poll_pool

    # Scheduler stats fallback
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None and callable(getattr(scheduler, "get_stats", None)):
        def _poll_stats() -> tuple[int, int]:
            stats = scheduler.get_stats()
            total = _safe_int(getattr(stats, "num_total_tokens", None), 1) or 1
            used = _safe_int(getattr(stats, "num_used_tokens", None), 0)
            return used, total

        return _poll_stats

    logger.warning(
        "[memory-guard] SGLang: token pool not found. "
        "KVCacheMonitor will report 0 utilization. "
        "Verify you are using a supported SGLang version."
    )

    def _null_poll() -> tuple[int, int]:
        return 0, 1

    return _null_poll


def _make_smoothed_poll_fn(
    raw_poll_fn: Callable[[], tuple[int, int]],
    window: int = 3,
) -> Callable[[], tuple[int, int]]:
    """Wrap *raw_poll_fn* with a rolling-maximum over *window* readings.

    SGLang's RadixAttention prefix-cache can evict large blocks of KV
    memory at once, causing utilization to drop suddenly.  Without smoothing,
    such a drop would reset the cooldown and delay the next shed-load signal.

    The rolling max suppresses these transient drops: the signal fires as
    long as *any* of the last *window* readings was above the threshold.
    Genuine pressure relief (all *window* readings below threshold) allows
    the signal to clear naturally.

    Args:
        raw_poll_fn: The underlying ``(used, total)`` callable.
        window:      Number of readings to keep for the rolling max (default 3).

    Returns:
        A new ``poll_fn`` that returns ``(smoothed_used, total)`` where
        ``smoothed_used / total == max(last *window* utilizations)``.
    """
    recent: collections.deque[float] = collections.deque(maxlen=window)

    def smoothed() -> tuple[int, int]:
        used, total = raw_poll_fn()
        util = used / total if total > 0 else 0.0
        recent.append(util)
        max_util = max(recent)
        return int(max_util * total), total

    return smoothed


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_int(value: object, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
