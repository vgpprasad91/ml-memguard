"""Tests for the vLLM adapter (memory_guard.adapters.vllm).

All tests use MagicMock to simulate vLLM objects — vLLM is NOT required.

Covers:
  - guard_vllm returns InferenceSafeConfig with safe.monitor set
  - cache_config.num_gpu_blocks back-calculation of max_num_seqs
  - gpu_memory_utilization refined from actual KV MB
  - poll_fn reads block_manager free/total blocks
  - preflight_overrides pass through to guard.preflight_inference
  - custom guard parameter
  - LLM / AsyncLLMEngine wrapper unwrapping
  - architecture extraction (GQA, quantization, dtype)
  - ImportError when vLLM not installed
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.adapters.vllm import (
    _extract_model_info,
    _get_llm_engine,
    _make_poll_fn,
    guard_vllm,
)
from memory_guard.guard import InferenceSafeConfig, MemoryGuard
from memory_guard.monitoring.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(
    num_heads=8,
    num_kv_heads=2,
    num_layers=4,
    hidden_size=512,
    max_model_len=2048,
    dtype_str="torch.float16",
    quantization=None,
    num_parameters=0,
    max_num_seqs=64,
    num_gpu_blocks=200,
    block_size=16,
    free_blocks=160,
    total_blocks=200,
) -> MagicMock:
    """Build a MagicMock that looks like a vllm.LLM wrapper.

    The LLM has a ``llm_engine`` attribute pointing to a spec-restricted engine
    so that _get_llm_engine unwraps it correctly.
    """
    hf_config = MagicMock()
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_kv_heads
    hf_config.num_hidden_layers = num_layers
    hf_config.hidden_size = hidden_size
    hf_config.num_parameters = num_parameters

    model_config = MagicMock()
    model_config.hf_config = hf_config
    model_config.max_model_len = max_model_len
    model_config.dtype = dtype_str
    model_config.quantization = quantization

    cache_config = MagicMock()
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.block_size = block_size

    scheduler_config = MagicMock()
    scheduler_config.max_num_seqs = max_num_seqs

    block_manager = MagicMock()
    block_manager.get_num_free_gpu_blocks.return_value = free_blocks
    block_manager.get_num_total_gpu_blocks.return_value = total_blocks

    scheduler = MagicMock()
    scheduler.block_manager = block_manager

    # Engine: restrict spec so it's not mistaken for an LLM wrapper
    engine = MagicMock(
        spec=["model_config", "cache_config", "scheduler_config", "scheduler"]
    )
    engine.model_config = model_config
    engine.cache_config = cache_config
    engine.scheduler_config = scheduler_config
    engine.scheduler = scheduler

    # LLM wrapper: has llm_engine
    llm = MagicMock()
    llm.llm_engine = engine
    return llm


def _call_guard_vllm(llm, **kwargs):
    """Call guard_vllm with vllm import mocked out.

    Creates a real MemoryGuard with a pinned available_mb so results
    are deterministic regardless of actual system memory.
    """
    guard = MemoryGuard.auto(enable_calibration=False)
    # Pin available_mb so preflight is predictable
    with patch.object(type(guard), "available_mb", new_callable=lambda: property(lambda self: 500_000)):
        with patch.object(type(guard), "budget_mb", new_callable=lambda: property(lambda self: 400_000)):
            with patch("memory_guard.adapters.vllm.optional_import"):
                return guard_vllm(llm, guard=guard, **kwargs)


# ---------------------------------------------------------------------------
# _get_llm_engine
# ---------------------------------------------------------------------------

class TestGetLLMEngine:
    def test_llm_returns_llm_engine_attr(self):
        engine = MagicMock()
        llm = MagicMock()
        llm.llm_engine = engine
        assert _get_llm_engine(llm) is engine

    def test_async_llm_engine_returns_engine_attr(self):
        inner = MagicMock()
        wrapper = MagicMock(spec=["engine"])
        wrapper.engine = inner
        assert _get_llm_engine(wrapper) is inner

    def test_bare_engine_returned_directly(self):
        engine = MagicMock(spec=[])
        assert _get_llm_engine(engine) is engine


# ---------------------------------------------------------------------------
# Architecture extraction
# ---------------------------------------------------------------------------

class TestExtractModelInfo:
    def test_gqa_kv_heads_read_from_hf_config(self):
        llm = _make_llm(num_heads=8, num_kv_heads=2)
        info = _extract_model_info(llm.llm_engine)
        assert info["num_kv_heads"] == 2

    def test_head_dim_hidden_over_num_heads(self):
        llm = _make_llm(num_heads=8, hidden_size=512)
        info = _extract_model_info(llm.llm_engine)
        assert info["head_dim"] == 64  # 512 // 8

    def test_dtype_float16_gives_2_bytes(self):
        llm = _make_llm(dtype_str="torch.float16")
        assert _extract_model_info(llm.llm_engine)["dtype_bytes"] == 2

    def test_dtype_float32_gives_4_bytes(self):
        llm = _make_llm(dtype_str="torch.float32")
        assert _extract_model_info(llm.llm_engine)["dtype_bytes"] == 4

    def test_quantization_awq_gives_4_bits(self):
        llm = _make_llm(quantization="awq")
        assert _extract_model_info(llm.llm_engine)["model_bits"] == 4

    def test_quantization_none_gives_16_bits(self):
        llm = _make_llm(quantization=None)
        assert _extract_model_info(llm.llm_engine)["model_bits"] == 16


# ---------------------------------------------------------------------------
# poll_fn
# ---------------------------------------------------------------------------

class TestMakePollFn:
    def test_used_equals_total_minus_free(self):
        llm = _make_llm(free_blocks=70, total_blocks=100)
        poll = _make_poll_fn(llm.llm_engine)
        used, total = poll()
        assert total == 100
        assert used == 30

    def test_missing_block_manager_returns_zero_util(self):
        engine = MagicMock(spec=["scheduler"])
        engine.scheduler = MagicMock(spec=[])  # no block_manager
        poll = _make_poll_fn(engine)
        used, total = poll()
        assert used == 0
        assert total >= 1


# ---------------------------------------------------------------------------
# guard_vllm — core behaviour
# ---------------------------------------------------------------------------

class TestGuardVllm:
    def test_returns_inference_safe_config(self):
        llm = _make_llm()
        safe = _call_guard_vllm(llm)
        assert isinstance(safe, InferenceSafeConfig)

    def test_monitor_attached_to_safe_config(self):
        llm = _make_llm()
        safe = _call_guard_vllm(llm)
        assert isinstance(safe.monitor, KVCacheMonitor)

    def test_monitor_not_started_on_return(self):
        llm = _make_llm()
        safe = _call_guard_vllm(llm)
        assert safe.monitor.is_running is False

    def test_cache_config_derives_max_num_seqs(self):
        # 200 blocks, block_size=16, max_seq_len=2048
        # blocks_per_seq = ceil(2048/16) = 128
        # actual_max_seqs = 200 // 128 = 1
        llm = _make_llm(num_gpu_blocks=200, block_size=16, max_model_len=2048)
        safe = _call_guard_vllm(llm)
        expected_max_seqs = 200 // math.ceil(2048 / 16)  # = 1
        assert safe.max_num_seqs == expected_max_seqs

    def test_no_cache_config_falls_back_to_scheduler_config(self):
        llm = _make_llm(num_gpu_blocks=0, max_num_seqs=48)
        safe = _call_guard_vllm(llm)
        # With 0 gpu_blocks, preflight uses scheduler_config.max_num_seqs=48
        assert safe.max_num_seqs == 48

    def test_preflight_override_max_num_seqs_respected(self):
        llm = _make_llm(num_gpu_blocks=1000, block_size=16, max_model_len=512)
        # Override should take precedence over back-calculation
        safe = _call_guard_vllm(llm, max_num_seqs=10)
        assert safe.max_num_seqs == 10

    def test_poll_fn_wired_to_block_manager(self):
        llm = _make_llm(free_blocks=80, total_blocks=100)
        safe = _call_guard_vllm(llm)
        used, total = safe.monitor.poll_fn()
        assert total == 100
        assert used == 20

    def test_gpu_memory_utilization_refined_from_blocks(self):
        # With known blocks we can compute actual KV MB; the field should be
        # the ratio of that to available_mb, capped at 0.95.
        llm = _make_llm(
            num_gpu_blocks=100, block_size=16,
            num_heads=8, num_kv_heads=2, num_layers=4,
            hidden_size=512, dtype_str="torch.float16",
        )
        safe = _call_guard_vllm(llm)
        # actual_kv_mb = 100 * 16 * 2 * 4 * 2 * 64 * 2 / (1024^2) = 3.2MB / 500_000MB
        # So gpu_memory_utilization ≈ tiny positive value < 0.95
        assert 0.0 <= safe.gpu_memory_utilization <= 0.95

    def test_import_error_when_vllm_not_installed(self):
        llm = _make_llm()
        with patch(
            "memory_guard.adapters.vllm.optional_import",
            side_effect=ImportError("vllm required"),
        ):
            with pytest.raises(ImportError, match="vllm"):
                guard_vllm(llm)

    def test_custom_guard_is_used(self):
        llm = _make_llm()
        custom_guard = MemoryGuard.auto(safety_ratio=0.50, enable_calibration=False)
        with patch.object(
            type(custom_guard), "available_mb",
            new_callable=lambda: property(lambda self: 500_000),
        ):
            with patch.object(
                type(custom_guard), "budget_mb",
                new_callable=lambda: property(lambda self: 250_000),
            ):
                with patch("memory_guard.adapters.vllm.optional_import"):
                    safe = guard_vllm(llm, guard=custom_guard)
        assert safe.budget_mb == 250_000

    def test_guard_vllm_importable_from_top_level(self):
        from memory_guard import guard_vllm as gv
        assert callable(gv)
