"""Tests for predictive OOM logic.

Covers:
  - Predictor scoring: safe conditions → oom_probability < 0.5
  - Predictor scoring: dangerous conditions → oom_probability > 0.7
  - Predictor scoring: critical conditions → oom_probability > 0.92
  - Predictor scoring: action mapping none / shed_load / restart
  - Predictor scoring: horizon_seconds decreases as probability increases
  - Predictor scoring: baseline_adj < 0 lowers probability (lenient user)
  - Predictor scoring: baseline_adj > 0 raises probability (aggressive user)
  - KVCacheMonitor: on_shed_load fires when predict_oom returns p > 0.7
  - KVCacheMonitor: restart fires when predict_oom returns p > 0.92
  - KVCacheMonitor: neither fires when predict_oom returns p <= 0.7
  - KVCacheMonitor: no action when predict_oom returns None (fallback)
  - KVCacheMonitor: no double shed_load within cooldown
  - KVCacheMonitor: predictive restart respects 120 s cooldown
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.monitoring.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Python mirror of the predictor scoring function (same coefficients)
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
# Predictor scoring function (Python mirror tests)
# ---------------------------------------------------------------------------

class TestScoringFunction:
    """Mirror the reference logistic regression in Python to verify the logic.

    These tests serve as a specification: when the reference coefficients
    change, these tests must be updated to match so the Python client and
    predictor stay in sync.
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

        with patch("memory_guard.integrations.predict_oom",
                   return_value=_make_prediction(0.80)):
            mon._run_predict_oom(kv_velocity=2.0, utilization=0.75,
                                 shed_ready=True)

        assert len(shed_calls) == 1

    def test_restart_fires_when_p_gt_092(self):
        restart_calls: list = []
        mon = _make_monitor(restart_callback=lambda: restart_calls.append(1))

        with patch("memory_guard.integrations.predict_oom",
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

        with patch("memory_guard.integrations.predict_oom",
                   return_value=_make_prediction(0.45)):
            mon._run_predict_oom(kv_velocity=0.5, utilization=0.50,
                                 shed_ready=True)

        assert shed_calls == []
        assert restart_calls == []

    def test_fallback_when_predict_oom_returns_none(self):
        """None result → no action taken by predict path; fallback to reactive."""
        shed_calls: list = []
        mon = _make_monitor(on_shed_load=lambda u: shed_calls.append(u))

        with patch("memory_guard.integrations.predict_oom", return_value=None):
            mon._run_predict_oom(kv_velocity=5.0, utilization=0.88,
                                 shed_ready=True)

        assert shed_calls == []  # predictive path did nothing; reactive still handles it

    def test_no_double_shed_load_within_cooldown(self):
        """shed_ready=False blocks a second predictive shed_load fire."""
        shed_calls: list = []
        mon = _make_monitor(on_shed_load=lambda u: shed_calls.append(u))

        with patch("memory_guard.integrations.predict_oom",
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

        with patch("memory_guard.integrations.predict_oom",
                   return_value=_make_prediction(0.97)):
            mon._run_predict_oom(kv_velocity=9.0, utilization=0.95,
                                 shed_ready=True)

        assert restart_calls == []  # cooldown blocked second restart

    def test_extended_poll_fn_signals_sent_to_predict(self):
        """extended_poll_fn dict is merged into the predict_oom signals payload."""
        captured_signals: list = []

        def fake_predict(signals, **kw):
            captured_signals.append(signals.copy())
            return _make_prediction(0.30)

        mon = _make_monitor(
            extended_poll_fn=lambda: {"eviction_rate": 5.5, "fragmentation_ratio": 0.7}
        )

        with patch("memory_guard.integrations.predict_oom", side_effect=fake_predict):
            mon._run_predict_oom(kv_velocity=3.0, utilization=0.70,
                                 shed_ready=True)

        assert captured_signals[0]["eviction_rate"] == pytest.approx(5.5)
        assert captured_signals[0]["fragmentation_ratio"] == pytest.approx(0.7)

    def test_exception_in_predict_does_not_crash_monitor(self):
        """Any exception in _run_predict_oom must be swallowed."""
        mon = _make_monitor()

        with patch("memory_guard.integrations.predict_oom",
                   side_effect=RuntimeError("network exploded")):
            # Must not raise
            mon._run_predict_oom(kv_velocity=3.0, utilization=0.70,
                                 shed_ready=True)
