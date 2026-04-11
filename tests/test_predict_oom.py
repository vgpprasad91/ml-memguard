"""Tests for PR 24: POST /v1/predict — predictive OOM endpoint.

Covers:
  - predict_oom() returns None when no API key configured
  - predict_oom() returns None on timeout (httpx.TimeoutException)
  - predict_oom() returns None on HTTP error (never raises)
  - predict_oom() returns None when httpx is unavailable (never raises)
  - predict_oom() returns dict on success with expected keys
  - predict_oom() uses 50 ms timeout (timeout=0.05 in httpx.post call)
  - predict_oom() includes model_name and backend in payload
  - predict_oom() merges caller signals dict into payload
  - Worker scoring: safe conditions → oom_probability < 0.5
  - Worker scoring: dangerous conditions → oom_probability > 0.7
  - Worker scoring: critical conditions → oom_probability > 0.92
  - Worker scoring: action mapping none / shed_load / restart
  - Worker scoring: horizon_seconds decreases as probability increases
  - Worker scoring: baseline_adj < 0 lowers probability (lenient user)
  - Worker scoring: baseline_adj > 0 raises probability (aggressive user)
  - KVCacheMonitor: on_shed_load fires when predict_oom returns p > 0.7
  - KVCacheMonitor: restart fires when predict_oom returns p > 0.92
  - KVCacheMonitor: neither fires when predict_oom returns p <= 0.7
  - KVCacheMonitor: no action when predict_oom returns None (fallback)
  - KVCacheMonitor: no double shed_load within cooldown
  - KVCacheMonitor: predictive restart respects 120 s cooldown
  - KVCacheMonitor: predict skipped when no API key
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Python mirror of the Worker scoring function (same coefficients)
# ---------------------------------------------------------------------------

_LR = {
    "BIAS":          -1.5,
    "VELOCITY":       0.20,
    "FRAGMENTATION":  3.00,
    "EVICTION":       0.50,
    "AVG_SEQ_LEN":    0.0005,
    "NEAR_MISS":      0.20,
    "PREEMPTION":     0.10,
}


def _sigmoid(z: float) -> float:
    EPS = 1e-15
    raw = 1.0 / (1.0 + math.exp(-z))
    return max(EPS, min(1.0 - EPS, raw))


def _score(
    velocity: float = 0.0,
    fragmentation: float = 0.0,
    eviction_rate: float = 0.0,
    avg_seq_len: float = 0.0,
    near_miss_count: float = 0.0,
    preemption_count: float = 0.0,
    baseline_adj: float = 0.0,
) -> float:
    z = (
        _LR["BIAS"]
        + _LR["VELOCITY"]      * velocity
        + _LR["FRAGMENTATION"] * fragmentation
        + _LR["EVICTION"]      * eviction_rate
        + _LR["AVG_SEQ_LEN"]   * avg_seq_len
        + _LR["NEAR_MISS"]     * near_miss_count
        + _LR["PREEMPTION"]    * preemption_count
        + baseline_adj
    )
    return _sigmoid(z)


def _action(p: float) -> str:
    if p > 0.92: return "restart"
    if p > 0.70: return "shed_load"
    return "none"


def _horizon(p: float) -> int:
    if p > 0.92: return 30
    if p > 0.70: return 60
    if p > 0.50: return 120
    return 300


# ---------------------------------------------------------------------------
# predict_oom — cloud.py
# ---------------------------------------------------------------------------

class TestPredictOom:
    def _mock_httpx_post(self, payload: dict, status: int = 200):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        if status >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
        else:
            resp.raise_for_status = MagicMock()
        httpx_mock = MagicMock()
        httpx_mock.post.return_value = resp
        return httpx_mock

    def test_returns_none_no_api_key(self):
        from memory_guard import cloud
        with patch.dict("os.environ", {}, clear=True):
            assert cloud.predict_oom({}) is None

    def test_returns_none_on_timeout(self):
        from memory_guard import cloud
        import httpx as _httpx  # real module for the exception type

        class FakeHttpx:
            TimeoutException = _httpx.TimeoutException

            @staticmethod
            def post(*a, **kw):
                raise _httpx.TimeoutException("timeout")

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": FakeHttpx}):
                result = cloud.predict_oom({})
        assert result is None

    def test_returns_none_on_http_error(self):
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post({}, status=503)
        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                result = cloud.predict_oom({})
        assert result is None

    def test_never_raises(self):
        from memory_guard import cloud
        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": None}):
                result = cloud.predict_oom({})
        assert result is None

    def test_returns_dict_on_success(self):
        from memory_guard import cloud
        payload = {
            "oom_probability": 0.85,
            "action": "shed_load",
            "horizon_seconds": 60,
            "confidence": 0.6,
        }
        httpx_mock = self._mock_httpx_post(payload, status=200)
        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                result = cloud.predict_oom({"kv_velocity_mbps": 3.0})
        assert result is not None
        assert result["oom_probability"] == pytest.approx(0.85)
        assert result["action"] == "shed_load"
        assert result["horizon_seconds"] == 60

    def test_50ms_timeout_enforced(self):
        """httpx.post must be called with timeout=0.05."""
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post(
            {"oom_probability": 0.1, "action": "none", "horizon_seconds": 300, "confidence": 0.3}
        )
        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                cloud.predict_oom({})
        _, kwargs = httpx_mock.post.call_args
        assert kwargs.get("timeout") == pytest.approx(0.05)

    def test_payload_includes_model_and_backend(self):
        import json
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post(
            {"oom_probability": 0.1, "action": "none", "horizon_seconds": 300, "confidence": 0.3}
        )
        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                cloud.predict_oom(
                    {"kv_velocity_mbps": 1.0},
                    model_name="llama",
                    backend="cuda",
                )
        body = json.loads(httpx_mock.post.call_args[1]["content"])
        assert body["model_name"] == "llama"
        assert body["backend"] == "cuda"

    def test_payload_merges_signals_dict(self):
        import json
        from memory_guard import cloud
        httpx_mock = self._mock_httpx_post(
            {"oom_probability": 0.1, "action": "none", "horizon_seconds": 300, "confidence": 0.3}
        )
        signals = {"kv_velocity_mbps": 4.2, "eviction_rate": 3.0}
        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                cloud.predict_oom(signals)
        body = json.loads(httpx_mock.post.call_args[1]["content"])
        assert body["kv_velocity_mbps"] == pytest.approx(4.2)
        assert body["eviction_rate"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Worker scoring function (Python mirror tests)
# ---------------------------------------------------------------------------

class TestScoringFunction:
    """Mirror the Worker's logistic regression in Python to verify the logic.

    These tests serve as a specification: when the Worker coefficients change,
    these tests must be updated to match so the Python client and Worker stay
    in sync.
    """

    def test_safe_conditions_low_probability(self):
        """Normal serving state → p < 0.5 → action 'none'."""
        p = _score(velocity=0.5, fragmentation=0.05, eviction_rate=0.0)
        assert p < 0.5
        assert _action(p) == "none"

    def test_moderate_risk_shed_load(self):
        """High velocity + moderate fragmentation → 0.70 < p ≤ 0.92 → 'shed_load'."""
        p = _score(velocity=5.0, fragmentation=0.4, eviction_rate=2.0)
        assert 0.70 < p <= 0.92
        assert _action(p) == "shed_load"

    def test_critical_conditions_restart(self):
        """All three primary signals in danger zone → p > 0.92 → 'restart'."""
        p = _score(velocity=8.0, fragmentation=0.6, eviction_rate=4.0)
        assert p > 0.92
        assert _action(p) == "restart"

    def test_action_none_at_low_p(self):
        assert _action(0.3) == "none"
        assert _action(0.70) == "none"   # boundary: not strictly >

    def test_action_shed_load_range(self):
        assert _action(0.701) == "shed_load"
        assert _action(0.92) == "shed_load"  # boundary: not strictly >

    def test_action_restart_range(self):
        assert _action(0.921) == "restart"
        assert _action(0.999) == "restart"

    def test_horizon_decreases_with_probability(self):
        """Higher probability → shorter predicted horizon."""
        assert _horizon(0.3)  > _horizon(0.6)
        assert _horizon(0.6)  > _horizon(0.75)
        assert _horizon(0.75) > _horizon(0.95)

    def test_lenient_baseline_adj_lowers_probability(self):
        """User with clean history (adj = -0.3) gets lower p than neutral."""
        p_neutral = _score(velocity=3.0, fragmentation=0.3, eviction_rate=1.0)
        p_lenient  = _score(velocity=3.0, fragmentation=0.3, eviction_rate=1.0,
                            baseline_adj=-0.3)
        assert p_lenient < p_neutral

    def test_aggressive_baseline_adj_raises_probability(self):
        """User with frequent near-misses (adj = +0.2) gets higher p."""
        p_neutral   = _score(velocity=3.0, fragmentation=0.3, eviction_rate=1.0)
        p_aggressive = _score(velocity=3.0, fragmentation=0.3, eviction_rate=1.0,
                              baseline_adj=0.2)
        assert p_aggressive > p_neutral

    def test_near_miss_contributes_positively(self):
        p_zero = _score(near_miss_count=0)
        p_high = _score(near_miss_count=10)
        assert p_high > p_zero

    def test_probability_bounded_0_1(self):
        """Sigmoid output must always be in (0, 1)."""
        for vel in (0, 1, 10, 100):
            for frag in (0.0, 0.5, 1.0):
                p = _score(velocity=vel, fragmentation=frag, eviction_rate=vel)
                assert 0.0 < p < 1.0


# ---------------------------------------------------------------------------
# KVCacheMonitor — _run_predict_oom integration
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> KVCacheMonitor:
    defaults = dict(poll_fn=lambda: (50, 100))
    defaults.update(kwargs)
    return KVCacheMonitor(**defaults)


def _make_prediction(p: float) -> dict:
    return {
        "oom_probability": p,
        "action": _action(p),
        "horizon_seconds": _horizon(p),
        "confidence": 0.6,
    }


class TestKVCacheMonitorPredictIntegration:
    def test_shed_load_fires_when_p_gt_07(self):
        shed_calls: list = []
        mon = _make_monitor(on_shed_load=lambda u: shed_calls.append(u))

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom",
                              return_value=_make_prediction(0.80)):
                mon._run_predict_oom(kv_velocity=2.0, utilization=0.75,
                                     shed_ready=True)

        assert len(shed_calls) == 1

    def test_restart_fires_when_p_gt_092(self):
        restart_calls: list = []
        mon = _make_monitor(restart_callback=lambda: restart_calls.append(1))

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom",
                              return_value=_make_prediction(0.95)):
                mon._run_predict_oom(kv_velocity=6.0, utilization=0.88,
                                     shed_ready=True)

        assert len(restart_calls) == 1

    def test_no_action_when_p_at_or_below_07(self):
        shed_calls: list = []
        restart_calls: list = []
        mon = _make_monitor(
            on_shed_load=lambda u: shed_calls.append(u),
            restart_callback=lambda: restart_calls.append(1),
        )

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom",
                              return_value=_make_prediction(0.45)):
                mon._run_predict_oom(kv_velocity=0.5, utilization=0.50,
                                     shed_ready=True)

        assert shed_calls == []
        assert restart_calls == []

    def test_fallback_when_predict_oom_returns_none(self):
        """None result → no action taken by predict path; fallback to reactive."""
        shed_calls: list = []
        mon = _make_monitor(on_shed_load=lambda u: shed_calls.append(u))

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom", return_value=None):
                mon._run_predict_oom(kv_velocity=5.0, utilization=0.88,
                                     shed_ready=True)

        assert shed_calls == []  # predictive path did nothing; reactive still handles it

    def test_no_double_shed_load_within_cooldown(self):
        """shed_ready=False blocks a second predictive shed_load fire."""
        shed_calls: list = []
        mon = _make_monitor(on_shed_load=lambda u: shed_calls.append(u))

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom",
                              return_value=_make_prediction(0.80)):
                # shed_ready=False means cooldown hasn't elapsed
                mon._run_predict_oom(kv_velocity=2.0, utilization=0.75,
                                     shed_ready=False)

        assert shed_calls == []

    def test_predictive_restart_respects_120s_cooldown(self):
        """Restart callback must not fire again within the 120 s cooldown."""
        restart_calls: list = []
        mon = _make_monitor(restart_callback=lambda: restart_calls.append(1))
        # Simulate a restart that happened 5 seconds ago (within cooldown)
        mon._last_predictive_restart = time.time() - 5.0

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom",
                              return_value=_make_prediction(0.97)):
                mon._run_predict_oom(kv_velocity=9.0, utilization=0.95,
                                     shed_ready=True)

        assert restart_calls == []  # cooldown blocked second restart

    def test_predict_skipped_when_no_api_key(self):
        """No API key → predict_oom never called."""
        shed_calls: list = []
        mon = _make_monitor(on_shed_load=lambda u: shed_calls.append(u))

        with patch.dict("os.environ", {}, clear=True):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom") as mock_predict:
                mon._run_predict_oom(kv_velocity=5.0, utilization=0.88,
                                     shed_ready=True)
                mock_predict.assert_not_called()

        assert shed_calls == []

    def test_extended_poll_fn_signals_sent_to_predict(self):
        """extended_poll_fn dict is merged into the predict_oom signals payload."""
        captured_signals: list = []

        def fake_predict(signals, **kw):
            captured_signals.append(signals.copy())
            return _make_prediction(0.30)

        mon = _make_monitor(
            extended_poll_fn=lambda: {"eviction_rate": 5.5, "fragmentation_ratio": 0.7}
        )

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom", side_effect=fake_predict):
                mon._run_predict_oom(kv_velocity=3.0, utilization=0.70,
                                     shed_ready=True)

        assert captured_signals[0]["eviction_rate"] == pytest.approx(5.5)
        assert captured_signals[0]["fragmentation_ratio"] == pytest.approx(0.7)

    def test_exception_in_predict_does_not_crash_monitor(self):
        """Any exception in _run_predict_oom must be swallowed."""
        mon = _make_monitor()

        with patch.dict("os.environ", {"MEMGUARD_BACKEND_KEY": "test-key-abcdefgh"}):
            from memory_guard import cloud as _cloud
            with patch.object(_cloud, "predict_oom",
                              side_effect=RuntimeError("network exploded")):
                # Must not raise
                mon._run_predict_oom(kv_velocity=3.0, utilization=0.70,
                                     shed_ready=True)
