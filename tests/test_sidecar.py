"""Tests for PR 31: sidecar HTTP server and VLLMMetricsPollFn.

Covers:
  - VLLMMetricsPollFn._parse_kv_cache_perc: gpu_cache_usage_perc fraction
  - VLLMMetricsPollFn._parse_kv_cache_perc: kv_cache_usage_perc fallback name
  - VLLMMetricsPollFn._parse_kv_cache_perc: percentage-scale value (>1.0) divided by 100
  - VLLMMetricsPollFn._parse_kv_cache_perc: metric with label set
  - VLLMMetricsPollFn._parse_kv_cache_perc: absent metric returns 0.0
  - VLLMMetricsPollFn._parse_kv_cache_perc: comment lines ignored
  - VLLMMetricsPollFn.__call__: network error returns (0, 100)
  - VLLMMetricsPollFn.__call__: returns (round(util*100), 100)
  - KVCacheMonitor.last_oom_probability: defaults to 0.0
  - KVCacheMonitor.last_oom_probability: reset to 0.0 on start()
  - KVCacheMonitor.last_oom_probability: updated after successful predict_oom
  - MemGuardSidecar._handle_readyz: 200 when probability <= threshold
  - MemGuardSidecar._handle_readyz: 503 when probability > threshold
  - MemGuardSidecar._handle_readyz: body contains oom_probability and threshold
  - MemGuardSidecar._handle_readyz: exactly at threshold returns 200 (not-yet-exceeded)
  - HTTP GET /healthz returns 200 and {"status": "ok"}
  - HTTP GET /readyz returns 200 when monitor probability is low
  - HTTP GET /readyz returns 503 when monitor probability is high
  - HTTP GET unknown path returns 404
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.monitoring.inference_monitor import KVCacheMonitor
from memory_guard.deployment.sidecar import MemGuardSidecar, VLLMMetricsPollFn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs: Any) -> KVCacheMonitor:
    defaults: Dict[str, Any] = dict(poll_fn=lambda: (0, 100))
    defaults.update(kwargs)
    return KVCacheMonitor(**defaults)


def _fetch(url: str, timeout: float = 3.0) -> tuple[int, Dict]:
    """Fetch JSON from url, return (status_code, body_dict)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# VLLMMetricsPollFn._parse_kv_cache_perc
# ---------------------------------------------------------------------------

class TestParseKvCachePerc:
    def test_gpu_cache_usage_perc_fraction(self):
        text = 'vllm:gpu_cache_usage_perc{model_name="llama"} 0.45\n'
        assert VLLMMetricsPollFn._parse_kv_cache_perc(text) == pytest.approx(0.45)

    def test_kv_cache_usage_perc_fallback_name(self):
        text = "vllm:kv_cache_usage_perc 0.62\n"
        assert VLLMMetricsPollFn._parse_kv_cache_perc(text) == pytest.approx(0.62)

    def test_percentage_scale_divided_by_100(self):
        # Some vLLM builds report 0-100 rather than 0-1
        text = "vllm:gpu_cache_usage_perc 73.5\n"
        assert VLLMMetricsPollFn._parse_kv_cache_perc(text) == pytest.approx(0.735)

    def test_metric_with_label_set(self):
        text = (
            '# HELP vllm:gpu_cache_usage_perc KV cache usage\n'
            '# TYPE vllm:gpu_cache_usage_perc gauge\n'
            'vllm:gpu_cache_usage_perc{model_name="opt",revision="main"} 0.88\n'
        )
        assert VLLMMetricsPollFn._parse_kv_cache_perc(text) == pytest.approx(0.88)

    def test_absent_metric_returns_zero(self):
        text = "some_other_metric 1.0\n"
        assert VLLMMetricsPollFn._parse_kv_cache_perc(text) == pytest.approx(0.0)

    def test_empty_text_returns_zero(self):
        assert VLLMMetricsPollFn._parse_kv_cache_perc("") == pytest.approx(0.0)

    def test_comment_lines_ignored(self):
        text = (
            "# vllm:gpu_cache_usage_perc 0.99\n"   # commented out
            "vllm:gpu_cache_usage_perc 0.55\n"
        )
        assert VLLMMetricsPollFn._parse_kv_cache_perc(text) == pytest.approx(0.55)

    def test_gpu_cache_usage_perc_preferred_over_kv(self):
        # When both present, first match wins (gpu variant listed first in _METRIC_NAMES)
        text = (
            "vllm:gpu_cache_usage_perc 0.30\n"
            "vllm:kv_cache_usage_perc 0.90\n"
        )
        assert VLLMMetricsPollFn._parse_kv_cache_perc(text) == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# VLLMMetricsPollFn.__call__
# ---------------------------------------------------------------------------

class TestVLLMMetricsPollFnCall:
    def test_network_error_returns_zero_tuple(self):
        pfn = VLLMMetricsPollFn("http://127.0.0.1:19999")  # nothing listening
        used, total = pfn()
        assert used == 0
        assert total == 100

    def test_returns_scaled_used_total(self):
        pfn = VLLMMetricsPollFn("http://ignored")
        metrics_text = "vllm:gpu_cache_usage_perc 0.75\n"

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = metrics_text.encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            used, total = pfn()

        assert total == 100
        assert used == 75  # round(0.75 * 100)

    def test_zero_utilization_returns_zero_used(self):
        pfn = VLLMMetricsPollFn("http://ignored")
        metrics_text = "vllm:gpu_cache_usage_perc 0.0\n"

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = metrics_text.encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            used, total = pfn()

        assert used == 0
        assert total == 100


# ---------------------------------------------------------------------------
# KVCacheMonitor.last_oom_probability
# ---------------------------------------------------------------------------

class TestLastOomProbability:
    def test_defaults_to_zero(self):
        mon = _make_monitor()
        assert mon.last_oom_probability == pytest.approx(0.0)

    def test_reset_to_zero_on_start(self):
        mon = _make_monitor()
        mon._last_oom_probability = 0.85
        mon.start()
        mon.stop()
        assert mon.last_oom_probability == pytest.approx(0.0)

    def test_updated_after_predict_oom(self):
        mon = _make_monitor()
        result = {
            "oom_probability": 0.77,
            "action": "shed_load",
            "horizon_seconds": 60,
            "confidence": 0.9,
            "model_source": "ml",
        }
        with patch("memory_guard.integrations.predict_oom", return_value=result):
            mon._run_predict_oom(kv_velocity=2.0, utilization=0.85, shed_ready=True)
        assert mon.last_oom_probability == pytest.approx(0.77)

    def test_unchanged_when_predict_oom_returns_none(self):
        mon = _make_monitor()
        mon._last_oom_probability = 0.5
        with patch("memory_guard.integrations.predict_oom", return_value=None):
            mon._run_predict_oom(kv_velocity=1.0, utilization=0.5, shed_ready=False)
        # Should not change when backend returns None
        assert mon.last_oom_probability == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MemGuardSidecar._handle_readyz logic (no HTTP layer)
# ---------------------------------------------------------------------------

class TestHandleReadyz:
    def _sidecar(self, prob: float, threshold: float = 0.70) -> MemGuardSidecar:
        mon = _make_monitor()
        mon._last_oom_probability = prob
        return MemGuardSidecar(mon, threshold=threshold)

    def test_200_when_below_threshold(self):
        s = self._sidecar(0.50)
        code, body = s._handle_readyz()
        assert code == 200

    def test_503_when_above_threshold(self):
        s = self._sidecar(0.80)
        code, body = s._handle_readyz()
        assert code == 503

    def test_200_at_exactly_threshold(self):
        # threshold is exclusive: prob > threshold → 503
        s = self._sidecar(0.70, threshold=0.70)
        code, _ = s._handle_readyz()
        assert code == 200

    def test_503_just_above_threshold(self):
        s = self._sidecar(0.701, threshold=0.70)
        code, _ = s._handle_readyz()
        assert code == 503

    def test_body_contains_oom_probability(self):
        s = self._sidecar(0.42)
        _, body = s._handle_readyz()
        assert "oom_probability" in body
        assert body["oom_probability"] == pytest.approx(0.42, abs=1e-3)

    def test_body_contains_threshold(self):
        s = self._sidecar(0.42, threshold=0.65)
        _, body = s._handle_readyz()
        assert body["threshold"] == pytest.approx(0.65)

    def test_200_body_status_ready(self):
        s = self._sidecar(0.10)
        _, body = s._handle_readyz()
        assert body["status"] == "ready"

    def test_503_body_status_not_ready(self):
        s = self._sidecar(0.95)
        _, body = s._handle_readyz()
        assert body["status"] == "not_ready"


# ---------------------------------------------------------------------------
# HTTP endpoint integration tests (real server on random port)
# ---------------------------------------------------------------------------

def _start_sidecar(monitor: KVCacheMonitor, threshold: float = 0.70) -> tuple[MemGuardSidecar, int]:
    """Start a sidecar on a random OS-assigned port; return (sidecar, port)."""
    import socket
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    sidecar = MemGuardSidecar(monitor, threshold=threshold)
    sidecar.start(host="127.0.0.1", port=port, block=False)
    # Brief wait for server thread to come up
    for _ in range(20):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5)
            break
        except Exception:
            time.sleep(0.05)
    return sidecar, port


class TestHTTPEndpoints:
    def test_healthz_returns_200(self):
        mon = _make_monitor()
        sidecar, port = _start_sidecar(mon)
        try:
            code, body = _fetch(f"http://127.0.0.1:{port}/healthz")
            assert code == 200
            assert body["status"] == "ok"
        finally:
            sidecar.stop()

    def test_readyz_200_low_probability(self):
        mon = _make_monitor()
        mon._last_oom_probability = 0.10
        sidecar, port = _start_sidecar(mon)
        try:
            code, body = _fetch(f"http://127.0.0.1:{port}/readyz")
            assert code == 200
            assert body["status"] == "ready"
        finally:
            sidecar.stop()

    def test_readyz_503_high_probability(self):
        mon = _make_monitor()
        mon._last_oom_probability = 0.85
        sidecar, port = _start_sidecar(mon)
        try:
            code, body = _fetch(f"http://127.0.0.1:{port}/readyz")
            assert code == 503
            assert body["status"] == "not_ready"
        finally:
            sidecar.stop()

    def test_unknown_path_returns_404(self):
        mon = _make_monitor()
        sidecar, port = _start_sidecar(mon)
        try:
            code, _ = _fetch(f"http://127.0.0.1:{port}/not-a-path")
            assert code == 404
        finally:
            sidecar.stop()

    def test_readyz_reflects_updated_probability(self):
        mon = _make_monitor()
        mon._last_oom_probability = 0.10
        sidecar, port = _start_sidecar(mon)
        try:
            code, _ = _fetch(f"http://127.0.0.1:{port}/readyz")
            assert code == 200

            # Simulate OOM probability rising above threshold
            mon._last_oom_probability = 0.90
            code, body = _fetch(f"http://127.0.0.1:{port}/readyz")
            assert code == 503
            assert body["oom_probability"] == pytest.approx(0.9, abs=0.001)
        finally:
            sidecar.stop()
