"""Tests for the SGLang adapter (memory_guard.adapters.sglang).

All tests use MagicMock to simulate SGLang objects — SGLang is NOT required.

Covers:
  - guard_sglang returns InferenceSafeConfig with safe.monitor set
  - token pool back-calculation of max_num_seqs
  - gpu_memory_utilization refined from actual KV MB
  - rolling-max smoothing of poll_fn (3-reading window)
  - token pool paths: token_to_kv_pool (≥0.3.0) and mem_pool (older)
  - scheduler stats fallback when no pool found
  - null fallback when no pool or scheduler stats
  - preflight_overrides pass through to guard.preflight_inference
  - custom guard parameter
  - engine unwrapping (sglang.Runtime wrapper)
  - architecture extraction from server_args
  - ImportError when SGLang not installed
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memory_guard.adapters.sglang import (
    _NoArgs,
    _extract_model_info,
    _find_token_pool,
    _get_inner_engine,
    _get_server_args,
    _make_raw_poll_fn,
    _make_smoothed_poll_fn,
    _pool_free,
    _pool_total,
    _safe_int,
    guard_sglang,
)
from memory_guard.guard import InferenceSafeConfig, MemoryGuard
from memory_guard.monitoring.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    context_length=4096,
    dtype="float16",
    quantization=None,
    num_heads=8,
    num_kv_heads=2,
    num_layers=4,
    hidden_size=512,
    total_token_slots=8192,
    free_token_slots=4096,
    pool_type="token_to_kv_pool",  # "token_to_kv_pool", "mem_pool", or None
) -> MagicMock:
    """Build a MagicMock that looks like a SGLang inner engine.

    Uses spec restriction so _get_inner_engine treats it as a bare inner engine
    (no unwrapping needed).
    """
    server_args = MagicMock()
    server_args.context_length = context_length
    server_args.dtype = dtype
    server_args.quantization = quantization
    server_args.max_running_requests = 64

    # HF config accessible via tp_worker.model_runner.model.config
    hf_config = MagicMock()
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_kv_heads
    hf_config.num_hidden_layers = num_layers
    hf_config.hidden_size = hidden_size
    hf_config.num_parameters = 0  # triggers estimate formula

    model = MagicMock()
    model.config = hf_config

    model_runner = MagicMock()
    model_runner.model = model

    tp_worker = MagicMock()
    tp_worker.model_runner = model_runner

    if pool_type == "token_to_kv_pool":
        pool = MagicMock()
        pool.size = total_token_slots
        pool.get_available_size = MagicMock(return_value=free_token_slots)

        engine = MagicMock(spec=["server_args", "tp_worker", "token_to_kv_pool"])
        engine.server_args = server_args
        engine.tp_worker = tp_worker
        engine.token_to_kv_pool = pool

    elif pool_type == "mem_pool":
        pool = MagicMock(spec=["size", "available"])
        pool.size = total_token_slots
        pool.available = free_token_slots

        engine = MagicMock(spec=["server_args", "tp_worker", "mem_pool"])
        engine.server_args = server_args
        engine.tp_worker = tp_worker
        engine.mem_pool = pool

    else:  # no pool
        engine = MagicMock(spec=["server_args", "tp_worker"])
        engine.server_args = server_args
        engine.tp_worker = tp_worker

    return engine


def _call_guard_sglang(engine, **kwargs):
    """Call guard_sglang with sglang import mocked out.

    Creates a real MemoryGuard with a pinned available_mb so results
    are deterministic regardless of actual system memory.
    """
    guard = MemoryGuard.auto(enable_calibration=False)
    with patch.object(type(guard), "available_mb", new_callable=lambda: property(lambda self: 500_000)):
        with patch.object(type(guard), "budget_mb", new_callable=lambda: property(lambda self: 400_000)):
            with patch("memory_guard.adapters.sglang.optional_import"):
                return guard_sglang(engine, guard=guard, **kwargs)


# ---------------------------------------------------------------------------
# _get_inner_engine
# ---------------------------------------------------------------------------

class TestGetInnerEngine:
    def test_bare_engine_returned_directly(self):
        engine = MagicMock(spec=["server_args"])
        assert _get_inner_engine(engine) is engine

    def test_runtime_wrapper_unwrapped_via_engine_attr(self):
        """sglang.Runtime has .engine but NOT .server_args — adapter unwraps it."""
        inner = MagicMock(spec=["server_args"])
        runtime = MagicMock(spec=["engine"])
        runtime.engine = inner
        assert _get_inner_engine(runtime) is inner

    def test_engine_with_both_engine_and_server_args_not_unwrapped(self):
        """If object has both 'engine' and 'server_args', treat it as the inner engine."""
        engine = MagicMock(spec=["engine", "server_args"])
        assert _get_inner_engine(engine) is engine


# ---------------------------------------------------------------------------
# _get_server_args / _NoArgs
# ---------------------------------------------------------------------------

class TestGetServerArgs:
    def test_returns_server_args_attr_when_present(self):
        sa = MagicMock()
        engine = MagicMock(spec=["server_args"])
        engine.server_args = sa
        assert _get_server_args(engine) is sa

    def test_returns_no_args_sentinel_when_absent(self):
        engine = MagicMock(spec=[])
        result = _get_server_args(engine)
        assert isinstance(result, _NoArgs)

    def test_no_args_sentinel_returns_none_for_any_attr(self):
        na = _NoArgs()
        assert na.context_length is None
        assert na.max_running_requests is None
        assert na.anything_at_all is None


# ---------------------------------------------------------------------------
# _find_token_pool
# ---------------------------------------------------------------------------

class TestFindTokenPool:
    def test_token_to_kv_pool_preferred(self):
        engine = _make_engine(pool_type="token_to_kv_pool")
        pool = _find_token_pool(engine)
        assert pool is engine.token_to_kv_pool

    def test_mem_pool_fallback(self):
        engine = _make_engine(pool_type="mem_pool")
        pool = _find_token_pool(engine)
        assert pool is engine.mem_pool

    def test_returns_none_when_no_pool(self):
        engine = _make_engine(pool_type=None)
        assert _find_token_pool(engine) is None


# ---------------------------------------------------------------------------
# _pool_total / _pool_free
# ---------------------------------------------------------------------------

class TestPoolAccessors:
    def test_pool_total_reads_size_attr(self):
        pool = MagicMock()
        pool.size = 1024
        assert _pool_total(pool) == 1024

    def test_pool_total_returns_zero_on_none(self):
        assert _pool_total(None) == 0

    def test_pool_free_prefers_callable_get_available_size(self):
        pool = MagicMock()
        pool.get_available_size = MagicMock(return_value=512)
        assert _pool_free(pool) == 512

    def test_pool_free_falls_back_to_available_attr(self):
        pool = MagicMock(spec=["available"])
        pool.available = 300
        assert _pool_free(pool) == 300


# ---------------------------------------------------------------------------
# Architecture extraction
# ---------------------------------------------------------------------------

class TestExtractModelInfo:
    def test_context_length_used_for_max_seq_len(self):
        engine = _make_engine(context_length=8192)
        info = _extract_model_info(engine)
        assert info["max_seq_len"] == 8192

    def test_dtype_float16_gives_2_bytes(self):
        engine = _make_engine(dtype="float16")
        assert _extract_model_info(engine)["dtype_bytes"] == 2

    def test_dtype_float32_gives_4_bytes(self):
        engine = _make_engine(dtype="float32")
        assert _extract_model_info(engine)["dtype_bytes"] == 4

    def test_dtype_int8_gives_1_byte(self):
        engine = _make_engine(dtype="int8")
        assert _extract_model_info(engine)["dtype_bytes"] == 1

    def test_quantization_awq_gives_4_bits(self):
        engine = _make_engine(quantization="awq")
        assert _extract_model_info(engine)["model_bits"] == 4

    def test_quantization_gptq_gives_4_bits(self):
        engine = _make_engine(quantization="gptq")
        assert _extract_model_info(engine)["model_bits"] == 4

    def test_quantization_none_gives_16_bits(self):
        engine = _make_engine(quantization=None)
        assert _extract_model_info(engine)["model_bits"] == 16

    def test_gqa_kv_heads_from_hf_config(self):
        engine = _make_engine(num_heads=8, num_kv_heads=2)
        info = _extract_model_info(engine)
        assert info["num_kv_heads"] == 2

    def test_head_dim_hidden_over_num_heads(self):
        engine = _make_engine(num_heads=8, hidden_size=512)
        info = _extract_model_info(engine)
        assert info["head_dim"] == 64  # 512 // 8

    def test_defaults_when_no_server_args(self):
        engine = MagicMock(spec=[])
        info = _extract_model_info(engine)
        assert info["max_seq_len"] == 8192   # hardcoded fallback
        assert info["dtype_bytes"] == 2      # fp16 fallback

    def test_num_params_estimated_when_zero(self):
        engine = _make_engine()
        info = _extract_model_info(engine)
        assert info["model_params"] > 0  # estimated from 12 × H² × L


# ---------------------------------------------------------------------------
# Rolling-max smoothing
# ---------------------------------------------------------------------------

class TestMakeSmoothedPollFn:
    def test_single_reading_returns_raw(self):
        raw = lambda: (60, 100)
        smoothed = _make_smoothed_poll_fn(raw, window=3)
        used, total = smoothed()
        assert total == 100
        assert used == 60

    def test_rolling_max_suppresses_drop(self):
        """After a spike, a subsequent lower reading is suppressed by rolling max."""
        readings = iter([(90, 100), (20, 100), (20, 100)])
        raw = lambda: next(readings)
        smoothed = _make_smoothed_poll_fn(raw, window=3)

        smoothed()                    # deque: [0.9]
        used, total = smoothed()      # deque: [0.9, 0.2]; max=0.9
        assert total == 100
        assert used == 90             # rolling max holds 90

    def test_genuine_recovery_clears_signal(self):
        """All window readings below threshold lets the signal fall through."""
        readings = iter([(90, 100), (10, 100), (10, 100), (10, 100)])
        raw = lambda: next(readings)
        smoothed = _make_smoothed_poll_fn(raw, window=3)

        smoothed()   # deque: [0.9]
        smoothed()   # deque: [0.9, 0.1]
        smoothed()   # deque: [0.9, 0.1, 0.1]
        used, total = smoothed()  # deque: [0.1, 0.1, 0.1]; max=0.1
        assert used == 10

    def test_window_1_is_effectively_unsmoothed(self):
        """Window of 1 should return raw value with no lag."""
        readings = iter([(90, 100), (5, 100)])
        raw = lambda: next(readings)
        smoothed = _make_smoothed_poll_fn(raw, window=1)
        smoothed()
        used, total = smoothed()
        assert used == 5

    def test_total_propagated_unchanged(self):
        raw = lambda: (50, 200)
        smoothed = _make_smoothed_poll_fn(raw, window=3)
        _, total = smoothed()
        assert total == 200


# ---------------------------------------------------------------------------
# _make_raw_poll_fn
# ---------------------------------------------------------------------------

class TestMakeRawPollFn:
    def test_pool_path_returns_used_total(self):
        engine = _make_engine(total_token_slots=100, free_token_slots=30)
        pool = _find_token_pool(engine)
        poll = _make_raw_poll_fn(engine, pool)
        used, total = poll()
        assert total == 100
        assert used == 70   # 100 - 30

    def test_scheduler_stats_fallback_when_no_pool(self):
        stats = MagicMock()
        stats.num_total_tokens = 200
        stats.num_used_tokens = 150

        scheduler = MagicMock()
        scheduler.get_stats = MagicMock(return_value=stats)

        engine = MagicMock(spec=["scheduler"])
        engine.scheduler = scheduler

        poll = _make_raw_poll_fn(engine, pool=None)
        used, total = poll()
        assert total == 200
        assert used == 150

    def test_null_fallback_when_no_pool_or_scheduler(self):
        engine = MagicMock(spec=[])
        poll = _make_raw_poll_fn(engine, pool=None)
        used, total = poll()
        assert used == 0
        assert total >= 1


# ---------------------------------------------------------------------------
# guard_sglang — core behaviour
# ---------------------------------------------------------------------------

class TestGuardSglang:
    def test_returns_inference_safe_config(self):
        engine = _make_engine()
        safe = _call_guard_sglang(engine)
        assert isinstance(safe, InferenceSafeConfig)

    def test_monitor_attached_to_safe_config(self):
        engine = _make_engine()
        safe = _call_guard_sglang(engine)
        assert isinstance(safe.monitor, KVCacheMonitor)

    def test_monitor_not_started_on_return(self):
        engine = _make_engine()
        safe = _call_guard_sglang(engine)
        assert safe.monitor.is_running is False

    def test_token_pool_derives_max_num_seqs(self):
        # 8192 total slots / 4096 max_seq_len = 2
        engine = _make_engine(total_token_slots=8192, context_length=4096)
        safe = _call_guard_sglang(engine)
        assert safe.max_num_seqs == 2

    def test_preflight_override_max_num_seqs_respected(self):
        engine = _make_engine(total_token_slots=8192, context_length=4096)
        safe = _call_guard_sglang(engine, max_num_seqs=5)
        assert safe.max_num_seqs == 5

    def test_poll_fn_wired_to_token_pool(self):
        engine = _make_engine(total_token_slots=100, free_token_slots=30)
        safe = _call_guard_sglang(engine)
        # Single reading: used=70 of 100; rolling max over 1 = 70
        used, total = safe.monitor.poll_fn()
        assert total == 100
        assert used == 70

    def test_mem_pool_fallback_wired_correctly(self):
        """Older SGLang uses mem_pool; adapter should fall back and wire it."""
        engine = _make_engine(
            total_token_slots=200, free_token_slots=100,
            pool_type="mem_pool",
        )
        safe = _call_guard_sglang(engine)
        used, total = safe.monitor.poll_fn()
        assert total == 200
        assert used == 100

    def test_gpu_memory_utilization_refined_from_pool(self):
        engine = _make_engine(
            total_token_slots=100,
            num_heads=8, num_kv_heads=2, num_layers=4,
            hidden_size=512, dtype="float16",
        )
        safe = _call_guard_sglang(engine)
        assert 0.0 <= safe.gpu_memory_utilization <= 0.95

    def test_import_error_when_sglang_not_installed(self):
        engine = _make_engine()
        with patch(
            "memory_guard.adapters.sglang.optional_import",
            side_effect=ImportError("sglang required"),
        ):
            with pytest.raises(ImportError, match="sglang"):
                guard_sglang(engine)

    def test_custom_guard_is_used(self):
        engine = _make_engine()
        custom_guard = MemoryGuard.auto(safety_ratio=0.50, enable_calibration=False)
        with patch.object(
            type(custom_guard), "available_mb",
            new_callable=lambda: property(lambda self: 500_000),
        ):
            with patch.object(
                type(custom_guard), "budget_mb",
                new_callable=lambda: property(lambda self: 250_000),
            ):
                with patch("memory_guard.adapters.sglang.optional_import"):
                    safe = guard_sglang(engine, guard=custom_guard)
        assert safe.budget_mb == 250_000

    def test_guard_sglang_importable_from_top_level(self):
        from memory_guard import guard_sglang as gs
        assert callable(gs)

    def test_runtime_wrapper_unwrapped_before_introspection(self):
        """sglang.Runtime wraps the engine via .engine; adapter unwraps correctly."""
        inner = _make_engine()
        runtime = MagicMock(spec=["engine"])
        runtime.engine = inner
        safe = _call_guard_sglang(runtime)
        assert isinstance(safe, InferenceSafeConfig)


# ---------------------------------------------------------------------------
# _safe_int utility
# ---------------------------------------------------------------------------

class TestSafeInt:
    def test_converts_int(self):
        assert _safe_int(42) == 42

    def test_converts_string(self):
        assert _safe_int("128") == 128

    def test_returns_default_on_none(self):
        assert _safe_int(None, 99) == 99

    def test_returns_default_on_type_error(self):
        assert _safe_int(object(), 7) == 7
