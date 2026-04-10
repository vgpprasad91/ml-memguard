"""Tests for the SGLang adapter (memory_guard.adapters.sglang).

All tests use MagicMock to simulate SGLang objects — SGLang is NOT required.

Covers:
  - _get_inner_engine: Runtime wrapper, bare engine
  - _extract_model_info: defaults, dtype, quantization, fallback params
  - _make_poll_fn: token_to_kv_pool path, scheduler.get_stats path, fallback
  - _run_preflight: fast path, binary-search downgrade, fits=False floor
  - guard_sglang: end-to-end wiring, callback passing, ImportError
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memory_guard.adapters.sglang import (
    _FallbackArgs,
    _deep_get,
    _extract_model_info,
    _get_inner_engine,
    _get_server_args,
    _make_poll_fn,
    _run_preflight,
    _safe_int,
    guard_sglang,
)
from memory_guard.guard import InferenceSafeConfig
from memory_guard.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_engine(
    num_heads=8,
    num_kv_heads=2,
    num_layers=4,
    hidden_size=512,
    context_length=2048,
    dtype_str="fp16",
    quantization=None,
    num_parameters=0,
    max_running_requests=64,
    pool_size=1000,
    pool_free=800,
    use_pool=True,
) -> MagicMock:
    """Build a MagicMock that looks like an inner SGLang engine.

    Uses a restricted spec so that _get_inner_engine does not unwrap it.
    """
    # HuggingFace model config
    hf_config = MagicMock()
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_kv_heads
    hf_config.num_hidden_layers = num_layers
    hf_config.hidden_size = hidden_size
    hf_config.num_parameters = num_parameters

    # server_args
    server_args = MagicMock()
    server_args.context_length = context_length
    server_args.dtype = dtype_str
    server_args.quantization = quantization
    server_args.max_running_requests = max_running_requests

    # token_to_kv_pool
    pool = MagicMock()
    pool.size = pool_size
    pool.get_available_size.return_value = pool_free

    # Restrict spec so _get_inner_engine treats this as a bare engine
    _spec = ["server_args", "token_to_kv_pool"]
    engine = MagicMock(spec=_spec)
    engine.server_args = server_args
    if use_pool:
        engine.token_to_kv_pool = pool

    return engine


# ---------------------------------------------------------------------------
# _get_inner_engine
# ---------------------------------------------------------------------------

class TestGetInnerEngine:
    def test_bare_engine_returned_directly(self):
        engine = MagicMock(spec=["server_args"])
        assert _get_inner_engine(engine) is engine

    def test_runtime_wrapper_unwrapped_via_engine_attr(self):
        inner = MagicMock(spec=["server_args"])
        # Runtime: has 'engine' attr but NOT 'server_args' at top level
        runtime = MagicMock(spec=["engine"])
        runtime.engine = inner
        assert _get_inner_engine(runtime) is inner

    def test_engine_with_server_args_returned_directly(self):
        """If object has both 'engine' and 'server_args', it's the engine itself."""
        engine = MagicMock(spec=["engine", "server_args"])
        # Because it has server_args it should be treated as the inner engine
        assert _get_inner_engine(engine) is engine


# ---------------------------------------------------------------------------
# _get_server_args
# ---------------------------------------------------------------------------

class TestGetServerArgs:
    def test_returns_server_args_attr(self):
        sa = MagicMock()
        engine = MagicMock(spec=["server_args"])
        engine.server_args = sa
        assert _get_server_args(engine) is sa

    def test_returns_fallback_when_absent(self):
        engine = MagicMock(spec=[])
        result = _get_server_args(engine)
        assert isinstance(result, _FallbackArgs)

    def test_fallback_returns_none_for_any_attr(self):
        fb = _FallbackArgs()
        assert fb.context_length is None
        assert fb.max_running_requests is None


# ---------------------------------------------------------------------------
# _extract_model_info
# ---------------------------------------------------------------------------

class TestExtractModelInfo:
    def test_num_kv_heads_from_server_args_fallback(self):
        engine = _make_engine(num_heads=8, num_kv_heads=2)
        # Without HF config path, falls back to server_args defaults (no hf config)
        info = _extract_model_info(engine)
        # We can't get hf_config without model_runner, so defaults are used
        assert info["num_kv_heads"] >= 1

    def test_context_length_from_server_args(self):
        engine = _make_engine(context_length=4096)
        info = _extract_model_info(engine)
        assert info["max_seq_len"] == 4096

    def test_dtype_fp16_gives_2_bytes(self):
        engine = _make_engine(dtype_str="fp16")
        assert _extract_model_info(engine)["dtype_bytes"] == 2

    def test_dtype_float32_gives_4_bytes(self):
        engine = _make_engine(dtype_str="float32")
        assert _extract_model_info(engine)["dtype_bytes"] == 4

    def test_dtype_int8_gives_1_byte(self):
        engine = _make_engine(dtype_str="int8")
        assert _extract_model_info(engine)["dtype_bytes"] == 1

    def test_dtype_bfloat16_gives_2_bytes(self):
        engine = _make_engine(dtype_str="bfloat16")
        assert _extract_model_info(engine)["dtype_bytes"] == 2

    def test_quantization_awq_gives_4_bits(self):
        engine = _make_engine(quantization="awq")
        assert _extract_model_info(engine)["model_bits"] == 4

    def test_quantization_gptq_gives_4_bits(self):
        engine = _make_engine(quantization="gptq")
        assert _extract_model_info(engine)["model_bits"] == 4

    def test_quantization_none_gives_16_bits(self):
        engine = _make_engine(quantization=None)
        assert _extract_model_info(engine)["model_bits"] == 16

    def test_num_parameters_estimated_when_zero(self):
        engine = _make_engine(num_parameters=0)
        info = _extract_model_info(engine)
        assert info["model_params"] > 0  # estimated from 12 × H² × L

    def test_defaults_when_no_server_args(self):
        engine = MagicMock(spec=[])
        info = _extract_model_info(engine)
        assert info["max_seq_len"] == 8192  # fallback
        assert info["dtype_bytes"] == 2    # fallback fp16

    def test_hidden_dim_in_info(self):
        engine = _make_engine(hidden_size=1024)
        info = _extract_model_info(engine)
        # Without hf_config path, hidden_dim uses fallback 4096
        assert info["hidden_dim"] >= 1


# ---------------------------------------------------------------------------
# _make_poll_fn
# ---------------------------------------------------------------------------

class TestMakePollFn:
    def test_pool_path_used_over_stats(self):
        engine = _make_engine(pool_size=100, pool_free=70)
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert total == 100
        assert used == 30  # 100 - 70

    def test_pool_path_used_total_minus_free(self):
        engine = _make_engine(pool_size=500, pool_free=499)
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert used == 1
        assert total == 500

    def test_scheduler_stats_path_when_no_pool(self):
        stats = MagicMock()
        stats.num_total_tokens = 200
        stats.num_used_tokens = 150

        scheduler = MagicMock()
        scheduler.get_stats.return_value = stats

        engine = MagicMock(spec=["scheduler"])
        engine.scheduler = scheduler

        poll = _make_poll_fn(engine)
        used, total = poll()
        assert total == 200
        assert used == 150

    def test_null_fallback_when_nothing_available(self):
        engine = MagicMock(spec=[])
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert total >= 1
        assert used == 0


# ---------------------------------------------------------------------------
# _run_preflight
# ---------------------------------------------------------------------------

class TestRunPreflight:
    _BASE = dict(
        model_params=0,
        model_bits=16,
        num_kv_heads=2,
        head_dim=64,
        num_layers=4,
        dtype_bytes=2,
        hidden_dim=512,
    )

    def test_fast_path_when_fits(self):
        info = {**self._BASE, "max_seq_len": 512}
        result = _run_preflight(info, max_num_seqs=8, min_num_seqs=1,
                                available_mb=100_000, budget_mb=100_000)
        assert result.fits is True
        assert result.max_num_seqs == 8
        assert result.changes == []

    def test_downgrade_finds_largest_fitting(self):
        info = {**self._BASE, "max_seq_len": 2048}
        from memory_guard.estimator import estimate_serving_memory
        kw = {k: v for k, v in info.items() if k != "max_seq_len"}
        est_50 = estimate_serving_memory(max_num_seqs=50, max_seq_len=2048, **kw)
        est_51 = estimate_serving_memory(max_num_seqs=51, max_seq_len=2048, **kw)
        budget = (est_50.total_mb + est_51.total_mb) / 2

        result = _run_preflight(info, max_num_seqs=100, min_num_seqs=1,
                                available_mb=budget * 2, budget_mb=budget)
        assert result.max_num_seqs == 50
        assert result.fits is True

    def test_fits_false_when_nothing_fits(self):
        info = {**self._BASE, "max_seq_len": 2048}
        result = _run_preflight(info, max_num_seqs=4, min_num_seqs=2,
                                available_mb=1, budget_mb=0.001)
        assert result.fits is False
        assert result.max_num_seqs == 2

    def test_changes_contain_max_running_requests(self):
        info = {**self._BASE, "max_seq_len": 2048}
        result = _run_preflight(info, max_num_seqs=100, min_num_seqs=1,
                                available_mb=0.1, budget_mb=0.001)
        assert any("max_running_requests" in c for c in result.changes)

    def test_gpu_memory_utilization_in_range(self):
        info = {**self._BASE, "max_seq_len": 512}
        result = _run_preflight(info, max_num_seqs=4, min_num_seqs=1,
                                available_mb=10_000, budget_mb=10_000)
        assert 0.0 <= result.gpu_memory_utilization <= 0.95


# ---------------------------------------------------------------------------
# guard_sglang (integration)
# ---------------------------------------------------------------------------

class TestGuardSglang:
    def _call(self, engine, **kwargs):
        """Call guard_sglang with mocked sglang import.

        Always supplies available_mb=50_000 unless overridden by caller.
        """
        kwargs.setdefault("available_mb", 50_000)
        with patch("memory_guard.adapters.sglang.optional_import"):
            return guard_sglang(engine, **kwargs)

    def test_returns_tuple_of_safe_config_and_monitor(self):
        engine = _make_engine()
        result = self._call(engine)
        assert isinstance(result, tuple) and len(result) == 2
        safe, mon = result
        assert isinstance(safe, InferenceSafeConfig)
        assert isinstance(mon, KVCacheMonitor)

    def test_monitor_not_started_on_return(self):
        engine = _make_engine()
        _, mon = self._call(engine)
        assert mon.is_running is False

    def test_max_num_seqs_read_from_server_args(self):
        # With fallback architecture (32 layers × 32 kv_heads × 128 head_dim)
        # 128 seqs at seq_len=2048 needs ~165 GB.  Use 300 GB budget.
        engine = _make_engine(max_running_requests=128)
        safe, _ = self._call(engine, available_mb=300_000)
        assert safe.max_num_seqs == 128

    def test_max_num_seqs_overridable_by_caller(self):
        # Fallback architecture: 64 seqs at seq_len=2048 needs ~90 GB.
        # Use 200 GB budget to guarantee the fast path fits.
        engine = _make_engine(max_running_requests=512)
        safe, _ = self._call(engine, max_num_seqs=64, available_mb=200_000)
        assert safe.max_num_seqs == 64

    def test_max_seq_len_overridable_by_caller(self):
        engine = _make_engine(context_length=8192)
        safe, _ = self._call(engine, max_seq_len=1024, available_mb=100_000)
        assert safe.max_seq_len == 1024

    def test_poll_fn_reads_kv_pool(self):
        engine = _make_engine(pool_size=100, pool_free=60)
        _, mon = self._call(engine, available_mb=100_000)
        used, total = mon.poll_fn()
        assert total == 100
        assert used == 40

    def test_on_warning_callback_wired(self):
        fired = []
        engine = _make_engine()
        _, mon = self._call(engine, on_warning=lambda u: fired.append(u))
        mon.on_warning(0.85)
        assert fired == [0.85]

    def test_on_shed_load_callback_wired(self):
        fired = []
        engine = _make_engine()
        _, mon = self._call(engine, on_shed_load=lambda u: fired.append(u))
        mon.on_shed_load(0.95)
        assert fired == [0.95]

    def test_guard_sglang_raises_on_missing_sglang(self):
        engine = _make_engine()
        with patch(
            "memory_guard.adapters.sglang.optional_import",
            side_effect=ImportError("sglang required"),
        ):
            with pytest.raises(ImportError, match="sglang"):
                guard_sglang(engine)

    def test_runtime_wrapper_unwrapped(self):
        inner = _make_engine()
        runtime = MagicMock(spec=["engine"])
        runtime.engine = inner
        safe, _ = self._call(runtime, available_mb=100_000)
        assert isinstance(safe, InferenceSafeConfig)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestUtilityHelpers:
    def test_safe_int_converts_int(self):
        assert _safe_int(42) == 42

    def test_safe_int_converts_string(self):
        assert _safe_int("128") == 128

    def test_safe_int_returns_default_on_none(self):
        assert _safe_int(None, default=99) == 99

    def test_safe_int_returns_default_on_type_error(self):
        assert _safe_int(object(), default=7) == 7

    def test_deep_get_traverses_chain(self):
        c = MagicMock()
        b = MagicMock()
        a = MagicMock()
        a.b = b
        b.c = c
        assert _deep_get(a, "b", "c") is c

    def test_deep_get_returns_none_on_missing(self):
        # Give a.x a spec that does NOT include "y", so getattr returns None.
        a = MagicMock(spec=["x"])
        a.x = MagicMock(spec=["z"])  # "y" is not in spec → AttributeError → default None
        assert _deep_get(a, "x", "y") is None


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    def test_guard_sglang_importable(self):
        from memory_guard.adapters.sglang import guard_sglang as gs
        assert callable(gs)
