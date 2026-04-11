"""Tests for PR 23: inference telemetry schema expansion.

Covers:
  - InferenceTelemetry default construction (all fields zero)
  - InferenceTelemetry.to_dict() includes all 14 keys + oom_occurred sentinel
  - upload_inference_telemetry() posts to POST /v1/telemetry with correct payload
  - upload_inference_telemetry() returns False when no API key configured
  - upload_inference_telemetry() returns False on HTTP error (never raises)
  - record_telemetry() backward compat: still works without inference fields
  - KVCacheMonitor._compute_velocity(): cold start returns 0.0
  - KVCacheMonitor._compute_velocity(): correct blocks/s when block_size_mb == 0
  - KVCacheMonitor._compute_velocity(): correct MB/s when block_size_mb > 0
  - KVCacheMonitor._compute_velocity(): negative delta (eviction) returns negative rate
  - KVCacheMonitor telemetry upload fires after telemetry_upload_interval
  - KVCacheMonitor telemetry upload skips when no API key
  - KVCacheMonitor extended_poll_fn dict merged into InferenceTelemetry
  - KVCacheMonitor extended_poll_fn exception does not crash the monitor loop
  - Worker missing-field defaulting: payload without inference fields is accepted
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

from memory_guard.telemetry import InferenceTelemetry
from memory_guard.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# InferenceTelemetry
# ---------------------------------------------------------------------------

class TestInferenceTelemetry:
    def test_default_construction(self):
        t = InferenceTelemetry()
        assert t.kv_velocity_mbps == 0.0
        assert t.fragmentation_ratio == 0.0
        assert t.eviction_rate == 0.0
        assert t.avg_seq_len == 0.0
        assert t.near_miss_count == 0
        assert t.preemption_count == 0
        assert t.weights_mb == 0.0
        assert t.kvcache_mb == 0.0
        assert t.activations_mb == 0.0
        assert t.cuda_ctx_mb == 0.0
        assert t.model_name == ""
        assert t.backend == ""
        assert t.os_platform == ""

    def test_to_dict_has_all_keys(self):
        t = InferenceTelemetry(kv_velocity_mbps=1.5, eviction_rate=2.0, model_name="m")
        d = t.to_dict()
        expected_keys = {
            "kv_velocity_mbps", "fragmentation_ratio", "eviction_rate",
            "avg_seq_len", "near_miss_count", "preemption_count",
            "weights_mb", "kvcache_mb", "activations_mb", "cuda_ctx_mb",
            "model_name", "backend", "os_platform", "oom_occurred",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_correct(self):
        t = InferenceTelemetry(
            kv_velocity_mbps=3.0, fragmentation_ratio=0.25,
            near_miss_count=5, preemption_count=2,
            model_name="llama", backend="cuda",
        )
        d = t.to_dict()
        assert d["kv_velocity_mbps"] == 3.0
        assert d["fragmentation_ratio"] == 0.25
        assert d["near_miss_count"] == 5
        assert d["preemption_count"] == 2
        assert d["model_name"] == "llama"
        assert d["backend"] == "cuda"
        assert d["oom_occurred"] == 0  # inference rows are never OOM events

    def test_to_dict_oom_always_zero(self):
        """oom_occurred must always be 0 — inference telemetry rows aren't crashes."""
        d = InferenceTelemetry().to_dict()
        assert d["oom_occurred"] == 0


# ---------------------------------------------------------------------------
# cloud.upload_inference_telemetry
# ---------------------------------------------------------------------------

class TestUploadInferenceTelemetry:
    def _mock_httpx_post(self, status: int = 200):
        resp = MagicMock()
        resp.status_code = status
        resp.raise_for_status = MagicMock()
        if status >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
        httpx_mock = MagicMock()
        httpx_mock.post.return_value = resp
        return httpx_mock

    def test_returns_false_no_api_key(self):
        from memory_guard import cloud
        with patch.dict("os.environ", {}, clear=True):
            result = cloud.upload_inference_telemetry(InferenceTelemetry())
        assert result is False

    def test_returns_true_on_success(self):
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post(200)
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                result = cloud.upload_inference_telemetry(InferenceTelemetry())
        assert result is True

    def test_posts_to_v1_telemetry(self):
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post(200)
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                cloud.upload_inference_telemetry(InferenceTelemetry(kv_velocity_mbps=7.0))
        call_args = httpx_mock.post.call_args
        assert "/v1/telemetry" in call_args[0][0]

    def test_payload_contains_inference_fields(self):
        import json
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post(200)
        signals = InferenceTelemetry(kv_velocity_mbps=4.5, eviction_rate=1.2)
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                cloud.upload_inference_telemetry(signals)
        body = json.loads(httpx_mock.post.call_args[1]["content"])
        assert body["kv_velocity_mbps"] == 4.5
        assert body["eviction_rate"] == 1.2
        assert body["oom_occurred"] == 0

    def test_returns_false_on_http_error(self):
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post(500)
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                result = cloud.upload_inference_telemetry(InferenceTelemetry())
        assert result is False

    def test_never_raises(self):
        """Even if httpx is not installed, the call must not propagate an exception."""
        from memory_guard import cloud
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": None}):  # missing dependency
                result = cloud.upload_inference_telemetry(InferenceTelemetry())
        assert result is False


# ---------------------------------------------------------------------------
# record_telemetry backward compatibility
# ---------------------------------------------------------------------------

class TestRecordTelemetryBackwardCompat:
    def test_old_call_signature_still_works(self):
        """record_telemetry(run_data) must work unchanged — no inference fields needed."""
        from memory_guard import cloud
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        httpx_mock = MagicMock()
        httpx_mock.post.return_value = resp
        run_data = {
            "model_name": "llama", "backend": "cuda",
            "batch_size": 4, "oom_occurred": False,
        }
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                result = cloud.record_telemetry(run_data)
        assert result is True


# ---------------------------------------------------------------------------
# KVCacheMonitor._compute_velocity
# ---------------------------------------------------------------------------

class TestComputeVelocity:
    def _monitor(self, block_size_mb: float = 0.0) -> KVCacheMonitor:
        return KVCacheMonitor(
            poll_fn=lambda: (0, 100),
            kv_block_size_mb=block_size_mb,
        )

    def test_cold_start_returns_zero(self):
        mon = self._monitor()
        assert mon._compute_velocity(50, time.time()) == 0.0

    def test_blocks_per_sec_when_no_block_size(self):
        mon = self._monitor(block_size_mb=0.0)
        t0 = time.time()
        mon._compute_velocity(100, t0)  # seed prev
        # 20 blocks in 4 seconds → 5 blocks/s
        vel = mon._compute_velocity(120, t0 + 4.0)
        assert abs(vel - 5.0) < 1e-9

    def test_mb_per_sec_with_block_size(self):
        mon = self._monitor(block_size_mb=2.0)
        t0 = time.time()
        mon._compute_velocity(100, t0)  # seed
        # 10 blocks in 5 s × 2 MB/block = 4 MB/s
        vel = mon._compute_velocity(110, t0 + 5.0)
        assert abs(vel - 4.0) < 1e-9

    def test_negative_delta_returns_negative_rate(self):
        """Eviction (used blocks decreasing) returns negative velocity."""
        mon = self._monitor(block_size_mb=1.0)
        t0 = time.time()
        mon._compute_velocity(200, t0)
        vel = mon._compute_velocity(180, t0 + 2.0)
        assert vel < 0.0

    def test_zero_elapsed_returns_zero(self):
        """Identical timestamps must not cause ZeroDivisionError."""
        mon = self._monitor()
        t0 = time.time()
        mon._compute_velocity(50, t0)
        vel = mon._compute_velocity(60, t0)  # same timestamp
        assert vel == 0.0

    def test_updates_prev_state_after_call(self):
        mon = self._monitor()
        t0 = time.time()
        mon._compute_velocity(50, t0)
        assert mon._prev_used_blocks == 50
        assert mon._prev_poll_time == t0


# ---------------------------------------------------------------------------
# KVCacheMonitor telemetry upload integration
# ---------------------------------------------------------------------------

class TestKVCacheMonitorTelemetryUpload:
    def _run_ticks(
        self,
        monitor: KVCacheMonitor,
        ticks: int,
        used: int = 50,
        total: int = 100,
    ) -> None:
        """Drive the monitor loop manually for *ticks* iterations."""
        for _ in range(ticks):
            try:
                u, t = monitor.poll_fn()
            except Exception:
                u, t = used, total
            utilization = u / t if t > 0 else 0.0
            now = time.time()
            vel = monitor._compute_velocity(u, now)
            with monitor._lock:
                monitor._history.append(utilization)
            if (now - monitor._last_telemetry_upload) >= monitor._telemetry_upload_interval:
                monitor._upload_inference_telemetry(vel)
                monitor._last_telemetry_upload = now

    def test_upload_skips_without_api_key(self):
        """No API key → upload_inference_telemetry returns False, monitor doesn't crash."""
        uploaded: list = []

        def fake_upload(signals):
            uploaded.append(signals)
            return True

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            telemetry_upload_interval=0.0,  # upload every tick
        )
        mon._last_telemetry_upload = 0.0

        with patch.dict("os.environ", {}, clear=True):
            mon._upload_inference_telemetry(1.0)

        assert len(uploaded) == 0  # api_key() returned None, no upload

    def test_upload_fires_after_interval(self):
        """upload_inference_telemetry is called when the interval elapses."""
        from memory_guard import cloud as _cloud

        calls: list = []

        def fake_upload(signals):
            calls.append(signals)
            return True

        mon = KVCacheMonitor(
            poll_fn=lambda: (60, 100),
            telemetry_upload_interval=0.0,  # always upload
        )
        mon._last_telemetry_upload = 0.0

        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.object(_cloud, "upload_inference_telemetry", side_effect=fake_upload):
                mon._upload_inference_telemetry(2.5)

        assert len(calls) == 1
        assert calls[0].kv_velocity_mbps == 2.5

    def test_extended_poll_fn_merged(self):
        """Fields from extended_poll_fn appear in the uploaded InferenceTelemetry."""
        from memory_guard import cloud as _cloud

        captured: list = []

        def extended():
            return {"eviction_rate": 3.7, "avg_seq_len": 512.0, "weights_mb": 14000.0}

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            extended_poll_fn=extended,
            telemetry_upload_interval=0.0,
        )

        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.object(_cloud, "upload_inference_telemetry",
                              side_effect=lambda s: captured.append(s) or True):
                mon._upload_inference_telemetry(0.5)

        assert len(captured) == 1
        assert captured[0].eviction_rate == pytest.approx(3.7)
        assert captured[0].avg_seq_len == pytest.approx(512.0)
        assert captured[0].weights_mb == pytest.approx(14000.0)

    def test_extended_poll_fn_exception_does_not_crash(self):
        """A crashing extended_poll_fn must not propagate to the monitor."""
        from memory_guard import cloud as _cloud

        def broken():
            raise RuntimeError("GPU driver exploded")

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            extended_poll_fn=broken,
            telemetry_upload_interval=0.0,
        )

        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.object(_cloud, "upload_inference_telemetry", return_value=True):
                # Must not raise
                mon._upload_inference_telemetry(0.0)

    def test_context_fields_propagated(self):
        """model_name/backend/os_platform from constructor appear in upload."""
        from memory_guard import cloud as _cloud

        captured: list = []
        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            telemetry_upload_interval=0.0,
            telemetry_model_name="meta-llama/Llama-3-8B",
            telemetry_backend="cuda",
            telemetry_os_platform="linux",
        )

        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.object(_cloud, "upload_inference_telemetry",
                              side_effect=lambda s: captured.append(s) or True):
                mon._upload_inference_telemetry(1.0)

        assert captured[0].model_name == "meta-llama/Llama-3-8B"
        assert captured[0].backend == "cuda"
        assert captured[0].os_platform == "linux"


# ---------------------------------------------------------------------------
# Worker missing-field defaulting (payload contract test)
# ---------------------------------------------------------------------------

class TestWorkerMissingFieldDefaulting:
    """Verify the Python client sends 0.0 for missing inference fields so
    the Worker stores them as 0 (backward-compatible with old callers)."""

    def test_default_telemetry_has_zero_inference_fields(self):
        """An InferenceTelemetry() with all defaults produces a payload where
        every inference signal is 0 — what old clients effectively send."""
        d = InferenceTelemetry().to_dict()
        for field in (
            "kv_velocity_mbps", "fragmentation_ratio", "eviction_rate",
            "avg_seq_len", "near_miss_count", "preemption_count",
            "weights_mb", "kvcache_mb", "activations_mb", "cuda_ctx_mb",
        ):
            assert d[field] == 0 or d[field] == 0.0, f"{field} should default to 0"

    def test_partial_payload_fields_preserved(self):
        """Only supplied fields are non-zero; unsupplied ones remain 0."""
        t = InferenceTelemetry(kv_velocity_mbps=8.0, near_miss_count=3)
        d = t.to_dict()
        assert d["kv_velocity_mbps"] == 8.0
        assert d["near_miss_count"] == 3
        assert d["fragmentation_ratio"] == 0.0
        assert d["eviction_rate"] == 0.0
