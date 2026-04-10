"""SGLang adapter for memory-guard.

Provides:
    guard_sglang — introspect a running SGLang engine, run preflight_inference()
                   to find a safe max_running_requests, and wire a KVCacheMonitor
                   to the engine's KV cache memory pool.

Design contract (ADR 003)
--------------------------
``guard_sglang`` constructs and returns a ``KVCacheMonitor``.  It never calls
into the engine except through the returned monitor's ``poll_fn``.  The
``poll_fn`` reads ``used_tokens`` and ``total_tokens`` from the token memory
pool but never writes to it.  All load-shedding decisions are delegated to the
caller via ``on_warning`` and ``on_shed_load`` callbacks.

SGLang KV cache metrics
------------------------
SGLang's ``TokenizerManager`` and the internal ``ModelRunner`` expose KV block
occupancy through the radix attention cache.  The preferred path is:

    engine.token_to_kv_pool.get_available_size()   # free tokens
    engine.token_to_kv_pool.size                   # total tokens

If the pool is not directly accessible (older SGLang releases, pre-built
containers), the adapter falls back to scheduler statistics:

    engine.scheduler.get_stats().num_used_tokens
    engine.scheduler.get_stats().num_total_tokens

If neither path is available, the monitor returns zero utilization (safe but
invisible) and logs a warning so operators know to upgrade or configure a
custom ``poll_fn``.

Supported SGLang object types
-------------------------------
- ``sglang.srt.server.Runtime``         — the high-level runtime object; the
                                          adapter follows ``runtime.engine``
                                          to reach the inner engine.
- ``sglang.srt.managers.TokenizerManager`` — the async serving manager; holds
                                              ``scheduler`` and access to the
                                              KV pool via the scheduler's
                                              ``tp_worker``.
- ``sglang.srt.server_args.ServerArgs`` is NOT accepted — this adapter works
                                         only with already-running engines.

Usage::

    import sglang as sgl
    from memory_guard.adapters.sglang import guard_sglang

    runtime = sgl.Runtime(model_path="meta-llama/Meta-Llama-3-8B", ...)

    safe, monitor = guard_sglang(
        runtime,
        on_shed_load=lambda u: load_balancer.set_weight("primary", 0),
    )

    print(safe)                         # InferenceSafeConfig
    print(safe.max_num_seqs)            # Pass to SGLang --max-running-requests
    print(safe.gpu_memory_utilization)  # Pass to SGLang --mem-fraction-static

    with monitor.session():
        runtime.wait()
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ..constants import MONITOR_POLL_INTERVAL, SAFETY_RATIO_DEFAULT
from ..estimator import InferenceServingEstimate, estimate_serving_memory
from ..guard import InferenceSafeConfig
from ..inference_monitor import KVCacheMonitor
from .base import optional_import

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def guard_sglang(
    engine: object,
    available_mb: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    on_warning: Optional[Callable[[float], None]] = None,
    on_shed_load: Optional[Callable[[float], None]] = None,
    poll_interval: float = MONITOR_POLL_INTERVAL,
    cooldown_seconds: float = 30.0,
    safety_ratio: float = SAFETY_RATIO_DEFAULT,
    min_num_seqs: int = 1,
) -> tuple[InferenceSafeConfig, KVCacheMonitor]:
    """Preflight check and KV cache monitor for a SGLang engine.

    Reads the model architecture from the engine's server args / model config,
    runs a binary search to find the largest ``max_running_requests`` that fits
    in the GPU memory budget, and returns a wired-but-unstarted
    ``KVCacheMonitor``.

    The monitor polls the token KV pool from a daemon thread and fires
    ``on_warning`` / ``on_shed_load`` callbacks — it never mutates the engine
    (ADR 003).

    Args:
        engine:            A ``sglang.srt.server.Runtime``,
                           ``sglang.srt.managers.TokenizerManager``, or any
                           SGLang engine object with accessible server args.
        available_mb:      Available GPU memory in MB.  Auto-detected from
                           CUDA if not provided.
        max_num_seqs:      Max concurrent requests for the preflight check.
                           Defaults to ``server_args.max_running_requests``
                           (or 256 if not accessible).
        max_seq_len:       Max sequence length.  Defaults to
                           ``server_args.context_length``.
        on_warning:        Callback fired at ≥ 80 % KV cache utilization.
                           Receives the utilization float (0.0–1.0).
        on_shed_load:      Callback fired at ≥ 92 % KV cache utilization.
                           Receives the utilization float (0.0–1.0).
        poll_interval:     Seconds between KV pool polls (default 5 s).
        cooldown_seconds:  Minimum seconds between repeated callback firings.
        safety_ratio:      Fraction of available memory used as the budget.
        min_num_seqs:      Binary-search floor — never reduce below this.

    Returns:
        ``(InferenceSafeConfig, KVCacheMonitor)`` — the preflight result and
        an **unstarted** monitor.  Start it with ``monitor.start()`` or use
        ``with monitor.session(): ...``.

    Raises:
        ImportError: if SGLang is not installed
            (``pip install ml-memguard[sglang]``).
    """
    optional_import("sglang", "sglang")

    inner = _get_inner_engine(engine)
    info = _extract_model_info(inner)

    if max_seq_len is not None:
        info["max_seq_len"] = max_seq_len

    if max_num_seqs is None:
        server_args = _get_server_args(inner)
        max_num_seqs = _safe_int(
            getattr(server_args, "max_running_requests", None), default=256
        )

    if available_mb is None:
        from ..platforms import get_available_memory_mb
        available_mb = get_available_memory_mb()

    budget_mb = available_mb * safety_ratio

    safe_config = _run_preflight(
        info=info,
        max_num_seqs=max_num_seqs,
        min_num_seqs=min_num_seqs,
        available_mb=available_mb,
        budget_mb=budget_mb,
    )

    poll_fn = _make_poll_fn(inner)

    monitor = KVCacheMonitor(
        poll_fn=poll_fn,
        poll_interval=poll_interval,
        on_warning=on_warning,
        on_shed_load=on_shed_load,
        cooldown_seconds=cooldown_seconds,
    )

    return safe_config, monitor


# ---------------------------------------------------------------------------
# Internal helpers — engine introspection
# ---------------------------------------------------------------------------


def _get_inner_engine(engine: object) -> object:
    """Unwrap a SGLang Runtime or similar wrapper to reach the inner engine.

    - ``sglang.Runtime`` exposes the engine as ``runtime.engine``
    - ``TokenizerManager`` is returned directly
    - Bare engine objects are returned as-is
    """
    # sglang.Runtime wraps the engine as .engine
    if hasattr(engine, "engine") and not hasattr(engine, "server_args"):
        return engine.engine
    return engine


def _get_server_args(engine: object) -> object:
    """Return the server_args object (or an empty MagicMock-like fallback)."""
    return getattr(engine, "server_args", None) or _FallbackArgs()


class _FallbackArgs:
    """Provides None for any attribute access — used when server_args is absent."""

    def __getattr__(self, name: str) -> None:
        return None


def _extract_model_info(engine: object) -> dict:
    """Read model architecture metadata from a SGLang engine.

    SGLang stores model configuration in ``engine.server_args`` (a
    ``ServerArgs`` dataclass) and in the model runner's HuggingFace config.
    Falls back to conservative defaults for any missing field.

    Returns a dict with keys matching ``estimate_serving_memory()`` kwargs.
    """
    server_args = _get_server_args(engine)

    # --- HF config (if accessible via model runner) -------------------------
    hf = None
    model_runner = _deep_get(engine, "tp_worker", "model_runner")
    if model_runner is None:
        model_runner = _deep_get(engine, "scheduler", "tp_worker", "model_runner")
    if model_runner is not None:
        hf = getattr(
            getattr(model_runner, "model", None),
            "config",
            None,
        )

    # --- attention geometry -------------------------------------------------
    num_heads: int = _safe_int(getattr(hf, "num_attention_heads", None), 32) if hf else 32
    num_kv_heads: int = (
        _safe_int(getattr(hf, "num_key_value_heads", None), num_heads) if hf else num_heads
    )
    num_layers: int = _safe_int(getattr(hf, "num_hidden_layers", None), 32) if hf else 32
    hidden_size: int = _safe_int(getattr(hf, "hidden_size", None), 4096) if hf else 4096
    head_dim: int = hidden_size // num_heads if num_heads > 0 else 128

    # --- sequence length ----------------------------------------------------
    # SGLang ServerArgs: context_length or max_total_tokens
    max_seq_len: int = _safe_int(
        getattr(server_args, "context_length", None)
        or getattr(server_args, "max_total_tokens", None),
        default=8192,
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

    # --- parameter count ----------------------------------------------------
    num_params: int = _safe_int(getattr(hf, "num_parameters", None), 0) if hf else 0
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


def _make_poll_fn(engine: object) -> Callable[[], tuple[int, int]]:
    """Return a zero-argument callable that reads ``(used_tokens, total_tokens)``.

    Tries three paths in order of preference:

    1. ``engine.token_to_kv_pool`` — preferred; a direct KV block pool.
    2. ``engine.scheduler.get_stats()`` — scheduler-level stats object.
    3. Null fallback (returns ``(0, 1)`` and logs a warning once).
    """
    # Path 1: direct token_to_kv_pool
    pool = getattr(engine, "token_to_kv_pool", None)
    if pool is not None and callable(getattr(pool, "get_available_size", None)):
        def _poll_pool() -> tuple[int, int]:
            total: int = _safe_int(getattr(pool, "size", None), 1) or 1
            free: int = _safe_int(pool.get_available_size(), 0)
            used: int = total - free
            return used, total

        return _poll_pool

    # Path 2: scheduler stats
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None and callable(getattr(scheduler, "get_stats", None)):
        def _poll_stats() -> tuple[int, int]:
            stats = scheduler.get_stats()
            total = _safe_int(getattr(stats, "num_total_tokens", None), 1) or 1
            used = _safe_int(getattr(stats, "num_used_tokens", None), 0)
            return used, total

        return _poll_stats

    # Fallback
    logger.warning(
        "[memory-guard] SGLang KV pool not found on engine. "
        "KVCacheMonitor will return 0 utilization. "
        "Verify you are using a supported SGLang version or pass a custom poll_fn."
    )

    def _null_poll() -> tuple[int, int]:
        return 0, 1

    return _null_poll


# ---------------------------------------------------------------------------
# Internal helpers — preflight binary search
# ---------------------------------------------------------------------------


def _run_preflight(
    info: dict,
    max_num_seqs: int,
    min_num_seqs: int,
    available_mb: float,
    budget_mb: float,
) -> InferenceSafeConfig:
    """Binary-search for the largest ``max_num_seqs`` within *budget_mb*."""
    _kw = {k: v for k, v in info.items() if k != "max_seq_len"}
    _kw["max_seq_len"] = info["max_seq_len"]

    est = estimate_serving_memory(max_num_seqs=max_num_seqs, **_kw)
    if est.fits_in(budget_mb):
        gpu_util = min(0.95, est.total_mb / available_mb) if available_mb > 0 else 0.90
        return InferenceSafeConfig(
            max_num_seqs=max_num_seqs,
            max_seq_len=info["max_seq_len"],
            gpu_memory_utilization=round(gpu_util, 4),
            estimate=est,
            budget_mb=budget_mb,
            available_mb=available_mb,
            fits=True,
            changes=[],
        )

    logger.warning(
        "[memory-guard] SGLang preflight: %d seqs × %d tokens = %.0f MB "
        "exceeds budget %.0f MB. Binary-searching for safe max_running_requests...",
        max_num_seqs, info["max_seq_len"], est.total_mb, budget_mb,
    )

    lo, hi = min_num_seqs, max_num_seqs
    safe_seqs: Optional[int] = None
    safe_est: Optional[InferenceServingEstimate] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = estimate_serving_memory(max_num_seqs=mid, **_kw)
        if candidate.fits_in(budget_mb):
            safe_seqs = mid
            safe_est = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    fits = safe_seqs is not None
    if not fits:
        safe_seqs = min_num_seqs
        safe_est = estimate_serving_memory(max_num_seqs=min_num_seqs, **_kw)

    gpu_util = min(0.95, safe_est.total_mb / available_mb) if available_mb > 0 else 0.90
    changes = [f"max_running_requests reduced {max_num_seqs} → {safe_seqs}"]

    logger.warning(
        "[memory-guard] SGLang safe max_running_requests: %d (%.0f MB). "
        "Suggested: --max-running-requests=%d --mem-fraction-static=%.4f",
        safe_seqs, safe_est.total_mb, safe_seqs, gpu_util,
    )

    return InferenceSafeConfig(
        max_num_seqs=safe_seqs,
        max_seq_len=info["max_seq_len"],
        gpu_memory_utilization=round(gpu_util, 4),
        estimate=safe_est,
        budget_mb=budget_mb,
        available_mb=available_mb,
        fits=fits,
        changes=changes,
    )


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


def _deep_get(obj: object, *attrs: str) -> object:
    """Traverse a chain of attributes, returning None on any missing step."""
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj
