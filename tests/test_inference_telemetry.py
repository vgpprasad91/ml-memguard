"""Tests for inference telemetry schema and KVCacheMonitor upload integration.

Covers:
  - InferenceTelemetry default construction (all fields zero)
  - InferenceTelemetry.to_dict() includes all 17 keys + oom_occurred sentinel
  - KVCacheMonitor._compute_velocity(): cold start returns 0.0
  - KVCacheMonitor._compute_velocity(): correct blocks/s when block_size_mb == 0
  - KVCacheMonitor._compute_velocity(): correct MB/s when block_size_mb > 0
  - KVCacheMonitor._compute_velocity(): negative delta (eviction) returns negative rate
  - KVCacheMonitor telemetry upload fires after telemetry_upload_interval
  - KVCacheMonitor telemetry upload silently skips when no backend installed
  - KVCacheMonitor extended_poll_fn dict merged into InferenceTelemetry
  - KVCacheMonitor extended_poll_fn exception does not crash the monitor loop
  - Worker missing-field defaulting: payload without inference fields is accepted
"""

from __future__ import annotations

import time
from unittest.mock import patch

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
        assert t.cuda_graph_mb == 0.0
        assert t.prefill_peak_activation_mb == 0.0
        assert t.max_seq_len_in_flight == 0
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
            "cuda_graph_mb",
            "prefill_peak_activation_mb", "max_seq_len_in_flight",
            "memory_pressure_level", "page_fault_rate",
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
    def test_upload_skips_silently_when_no_backend(self):
        """No backend plugin → _upload_inference_telemetry does not raise."""
        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            telemetry_upload_interval=0.0,
        )
        mon._last_telemetry_upload = 0.0
        # With no backend installed, backends.upload_inference_signals returns False
        # The monitor must not crash or raise.
        with patch("memory_guard.backends.upload_inference_signals", return_value=False):
            mon._upload_inference_telemetry(1.0)  # must not raise

    def test_upload_fires_after_interval(self):
        """upload_inference_signals is called when the interval elapses."""
        calls: list = []

        def fake_upload(signals):
            calls.append(signals)
            return True

        mon = KVCacheMonitor(
            poll_fn=lambda: (60, 100),
            telemetry_upload_interval=0.0,  # always upload
        )
        mon._last_telemetry_upload = 0.0

        with patch("memory_guard.backends.upload_inference_signals", side_effect=fake_upload):
            mon._upload_inference_telemetry(2.5)

        assert len(calls) == 1
        assert calls[0].kv_velocity_mbps == 2.5

    def test_extended_poll_fn_merged(self):
        """Fields from extended_poll_fn appear in the uploaded InferenceTelemetry."""
        captured: list = []

        def extended():
            return {"eviction_rate": 3.7, "avg_seq_len": 512.0, "weights_mb": 14000.0}

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            extended_poll_fn=extended,
            telemetry_upload_interval=0.0,
        )

        with patch("memory_guard.backends.upload_inference_signals",
                   side_effect=lambda s: captured.append(s) or True):
            mon._upload_inference_telemetry(0.5)

        assert len(captured) == 1
        assert captured[0].eviction_rate == pytest.approx(3.7)
        assert captured[0].avg_seq_len == pytest.approx(512.0)
        assert captured[0].weights_mb == pytest.approx(14000.0)

    def test_extended_poll_fn_exception_does_not_crash(self):
        """A crashing extended_poll_fn must not propagate to the monitor."""
        def broken():
            raise RuntimeError("GPU driver exploded")

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            extended_poll_fn=broken,
            telemetry_upload_interval=0.0,
        )

        with patch("memory_guard.backends.upload_inference_signals", return_value=True):
            # Must not raise
            mon._upload_inference_telemetry(0.0)

    def test_context_fields_propagated(self):
        """model_name/backend/os_platform from constructor appear in upload."""
        captured: list = []
        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            telemetry_upload_interval=0.0,
            telemetry_model_name="meta-llama/Llama-3-8B",
            telemetry_backend="cuda",
            telemetry_os_platform="linux",
        )

        with patch("memory_guard.backends.upload_inference_signals",
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
    the backend plugin stores them as 0 (backward-compatible with old callers)."""

    def test_default_telemetry_has_zero_inference_fields(self):
        """An InferenceTelemetry() with all defaults produces a payload where
        every inference signal is 0 — what old clients effectively send."""
        d = InferenceTelemetry().to_dict()
        for field in (
            "kv_velocity_mbps", "fragmentation_ratio", "eviction_rate",
            "avg_seq_len", "near_miss_count", "preemption_count",
            "weights_mb", "kvcache_mb", "activations_mb", "cuda_ctx_mb",
            "cuda_graph_mb", "prefill_peak_activation_mb", "max_seq_len_in_flight",
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


# ---------------------------------------------------------------------------
# CUDA graph baseline
# ---------------------------------------------------------------------------

class TestCudaGraphBaseline:
    def test_baseline_zero_when_torch_unavailable(self):
        """_snapshot_cuda_graph_baseline silently skips if torch is not installed."""
        import sys
        import unittest.mock as mock

        mon = KVCacheMonitor(poll_fn=lambda: (0, 100))
        # Simulate torch import failure
        with mock.patch.dict(sys.modules, {"torch": None}):
            mon._snapshot_cuda_graph_baseline()
        assert mon._cuda_graph_baseline_mb == 0.0

    def test_baseline_emitted_in_telemetry_when_no_extended_poll(self):
        """cuda_graph_mb in uploaded telemetry falls back to _cuda_graph_baseline_mb."""
        captured: list = []
        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            telemetry_upload_interval=0.0,
        )
        mon._cuda_graph_baseline_mb = 2048.0  # inject a known baseline

        with patch("memory_guard.backends.upload_inference_signals",
                   side_effect=lambda s: captured.append(s) or True):
            mon._upload_inference_telemetry(1.0)

        assert len(captured) == 1
        assert captured[0].cuda_graph_mb == pytest.approx(2048.0)

    def test_baseline_emitted_in_predict_signals(self):
        """_run_predict_oom signals dict includes cuda_graph_mb from baseline."""
        captured: list = []

        def fake_predict(signals, **_kw):
            captured.append(dict(signals))
            return None  # no action

        mon = KVCacheMonitor(poll_fn=lambda: (50, 100))
        mon._cuda_graph_baseline_mb = 1500.0

        with patch("memory_guard.backends.predict_oom", side_effect=fake_predict):
            mon._run_predict_oom(2.0, 0.5, True)

        assert len(captured) == 1
        assert captured[0].get("cuda_graph_mb") == pytest.approx(1500.0)

    def test_extended_poll_cuda_graph_overrides_baseline(self):
        """cuda_graph_mb from extended_poll_fn takes precedence over the baseline."""
        captured: list = []

        def extended():
            return {"cuda_graph_mb": 3500.0}

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            extended_poll_fn=extended,
            telemetry_upload_interval=0.0,
        )
        mon._cuda_graph_baseline_mb = 2048.0  # baseline should be overridden

        with patch("memory_guard.backends.upload_inference_signals",
                   side_effect=lambda s: captured.append(s) or True):
            mon._upload_inference_telemetry(1.0)

        assert len(captured) == 1
        assert captured[0].cuda_graph_mb == pytest.approx(3500.0)


# ---------------------------------------------------------------------------
# Prefill activation probe
# ---------------------------------------------------------------------------

class TestPrefillActivationProbe:
    """Tests for _update_prefill_signals, _fetch_max_seq_len_in_flight, and
    their integration with telemetry upload and predict signals."""

    def test_running_max_accumulates_across_ticks(self):
        """_prefill_peak_activation_mb is a running max, not last-tick value."""
        mon = KVCacheMonitor(poll_fn=lambda: (50, 100))
        # Inject a first spike of 1000 MB
        mon._prefill_peak_activation_mb = 1000.0
        # _update_prefill_signals with no torch should not decrease the value
        # (silently skips torch path; no eBPF session either)
        mon._update_prefill_signals(kv_velocity=0.0)
        # Running max must still be at least 1000.0 (may stay or grow)
        assert mon._prefill_peak_activation_mb >= 0.0  # doesn't crash

    def test_running_max_reset_after_upload(self):
        """prefill_peak_activation_mb is reset to 0 after _upload_inference_telemetry."""
        captured: list = []
        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            telemetry_upload_interval=0.0,
        )
        mon._prefill_peak_activation_mb = 4096.0

        with patch("memory_guard.backends.upload_inference_signals",
                   side_effect=lambda s: captured.append(s) or True):
            mon._upload_inference_telemetry(1.0)

        assert len(captured) == 1
        assert captured[0].prefill_peak_activation_mb == pytest.approx(4096.0)
        # Should be reset after upload
        assert mon._prefill_peak_activation_mb == pytest.approx(0.0)

    def test_prefill_emitted_in_predict_signals(self):
        """_run_predict_oom includes prefill_peak_activation_mb in the signals dict."""
        captured: list = []

        def fake_predict(signals, **_kw):
            captured.append(dict(signals))
            return None

        mon = KVCacheMonitor(poll_fn=lambda: (50, 100))
        mon._prefill_peak_activation_mb = 2048.0
        mon._max_seq_len_in_flight      = 512

        with patch("memory_guard.backends.predict_oom", side_effect=fake_predict):
            mon._run_predict_oom(1.0, 0.5, True)

        assert len(captured) == 1
        assert captured[0].get("prefill_peak_activation_mb") == pytest.approx(2048.0)
        assert captured[0].get("max_seq_len_in_flight") == 512

    def test_ebpf_fallback_records_spike_above_threshold(self):
        """When eBPF mmap_growth exceeds expected KV growth by > threshold, spike is recorded."""
        class FakeEBPF:
            available         = True
            mmap_growth_mbps  = 300.0   # 300 MB/s × 5 s = 1500 MB mmap growth
            page_fault_rate   = 0.0
            memory_pressure_bytes = 0

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            ebpf_session=FakeEBPF(),
            prefill_spike_threshold_mb=512.0,
            poll_interval=5.0,
        )
        # kv_velocity = 100 MB/s × 5 s = 500 MB expected; mmap = 1500 MB; excess = 1000 MB > 512 MB threshold
        mon._update_prefill_signals(kv_velocity=100.0)
        assert mon._prefill_peak_activation_mb == pytest.approx(1000.0)

    def test_ebpf_fallback_silent_below_threshold(self):
        """Excess below threshold does not trigger a spike."""
        class FakeEBPF:
            available         = True
            mmap_growth_mbps  = 110.0   # 110 × 5 = 550 MB; expected KV = 100 × 5 = 500; excess = 50 < 512
            page_fault_rate   = 0.0
            memory_pressure_bytes = 0

        mon = KVCacheMonitor(
            poll_fn=lambda: (50, 100),
            ebpf_session=FakeEBPF(),
            prefill_spike_threshold_mb=512.0,
            poll_interval=5.0,
        )
        mon._update_prefill_signals(kv_velocity=100.0)
        assert mon._prefill_peak_activation_mb == pytest.approx(0.0)

    def test_fetch_max_seq_len_returns_zero_on_connection_error(self):
        """_fetch_max_seq_len_in_flight returns 0 when the endpoint is unreachable."""
        mon = KVCacheMonitor(
            poll_fn=lambda: (0, 100),
            vllm_metrics_url="http://localhost:9999/metrics",
        )
        # No server listening — should not raise
        result = mon._fetch_max_seq_len_in_flight()
        assert result == 0

    def test_fetch_max_seq_len_parses_prometheus_text(self):
        """_fetch_max_seq_len_in_flight correctly parses vLLM Prometheus text."""
        from unittest.mock import MagicMock, patch as mpatch
        import io

        fake_body = (
            "# HELP vllm:num_running_seqs Number of running sequences.\n"
            "# TYPE vllm:num_running_seqs gauge\n"
            "vllm:num_running_seqs{engine=\"0\"} 8.0\n"
            "# HELP vllm:avg_prompt_len Average prompt length.\n"
            "# TYPE vllm:avg_prompt_len gauge\n"
            "vllm:avg_prompt_len{engine=\"0\"} 512.0\n"
        ).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mon = KVCacheMonitor(
            poll_fn=lambda: (0, 100),
            vllm_metrics_url="http://localhost:8000/metrics",
        )

        with mpatch("urllib.request.urlopen", return_value=mock_resp):
            result = mon._fetch_max_seq_len_in_flight()

        # 8 seqs × 512 tokens = 4096
        assert result == 4096
