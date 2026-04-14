"""Tests for PR 29: model_source field and ML/LR logging in _run_predict_oom.

Covers:
  - _run_predict_oom emits a debug log line containing model_source for "ml"
  - _run_predict_oom emits a debug log line containing model_source for "lr_fallback"
  - Log line for restart action includes model_source
  - Log line for shed_load action includes model_source
  - model_source "unknown" (field absent) does not crash _run_predict_oom
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.monitoring.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> KVCacheMonitor:
    """Return a KVCacheMonitor wired with a dummy poll_fn."""
    defaults = dict(
        poll_fn=lambda: (0, 100),
        on_warning=None,
        on_shed_load=MagicMock(),
    )
    defaults.update(kwargs)
    return KVCacheMonitor(**defaults)


def _predict_result(
    p: float,
    action: str,
    model_source: str,
) -> Dict[str, Any]:
    """Build a fake predict response."""
    return {
        "oom_probability": p,
        "action":          action,
        "horizon_seconds": 60,
        "confidence":      0.9,
        "model_source":    model_source,
    }


# ---------------------------------------------------------------------------
# _run_predict_oom logging
# ---------------------------------------------------------------------------

class TestRunPredictOomLogging:
    """Verify that _run_predict_oom emits a debug line with model_source."""

    def _run(
        self,
        p: float,
        model_source: str,
        shed_ready: bool = True,
        has_restart: bool = False,
    ) -> list[str]:
        """
        Run _run_predict_oom with a mocked predict_oom and capture emitted logs.
        Returns list of strings that were passed to self._emit_log.
        """
        emitted: list[str] = []

        restart_cb = MagicMock() if has_restart else None
        mon = _make_monitor(restart_callback=restart_cb)
        mon._emit_log = lambda msg: emitted.append(msg)   # type: ignore[method-assign]

        result = _predict_result(p, "shed_load" if p > 0.70 else "none", model_source)

        with patch("memory_guard.integrations.predict_oom", return_value=result):
            mon._run_predict_oom(
                kv_velocity=2.0,
                utilization=0.85,
                shed_ready=shed_ready,
            )
        return emitted

    def test_debug_log_emitted_for_ml_source(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="memory_guard.monitoring.inference_monitor"):
            with patch("memory_guard.integrations.predict_oom",
                       return_value=_predict_result(0.5, "none", "ml")):
                mon = _make_monitor()
                mon._run_predict_oom(
                    kv_velocity=1.0, utilization=0.5, shed_ready=False
                )
        assert any("ml" in r.message for r in caplog.records)

    def test_debug_log_emitted_for_lr_fallback_source(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="memory_guard.monitoring.inference_monitor"):
            with patch("memory_guard.integrations.predict_oom",
                       return_value=_predict_result(0.5, "none", "lr_fallback")):
                mon = _make_monitor()
                mon._run_predict_oom(
                    kv_velocity=1.0, utilization=0.5, shed_ready=False
                )
        assert any("lr_fallback" in r.message for r in caplog.records)

    def test_shed_load_emit_log_includes_model_source_ml(self):
        logs = self._run(p=0.80, model_source="ml", shed_ready=True)
        assert any("ml" in m for m in logs), f"Expected [ml] in emit_log: {logs}"

    def test_shed_load_emit_log_includes_model_source_lr_fallback(self):
        logs = self._run(p=0.80, model_source="lr_fallback", shed_ready=True)
        assert any("lr_fallback" in m for m in logs)

    def test_restart_emit_log_includes_model_source(self):
        logs = self._run(p=0.95, model_source="ml", has_restart=True)
        assert any("ml" in m for m in logs)

    def test_missing_model_source_does_not_crash(self):
        """When backend omits model_source, _run_predict_oom should not raise."""
        mon = _make_monitor()
        result = {
            "oom_probability": 0.5,
            "action":          "none",
            "horizon_seconds": 60,
            "confidence":      0.9,
            # model_source intentionally absent — backward compat
        }
        with patch("memory_guard.integrations.predict_oom", return_value=result):
            mon._run_predict_oom(kv_velocity=1.0, utilization=0.5, shed_ready=False)
        # No exception raised — test passes implicitly

    def test_model_source_unknown_logs_unknown(self, caplog):
        """Absent model_source defaults to 'unknown' in the log."""
        result = {
            "oom_probability": 0.5,
            "action":          "none",
            "horizon_seconds": 60,
            "confidence":      0.9,
        }
        with caplog.at_level(logging.DEBUG, logger="memory_guard.monitoring.inference_monitor"):
            with patch("memory_guard.integrations.predict_oom", return_value=result):
                mon = _make_monitor()
                mon._run_predict_oom(
                    kv_velocity=1.0, utilization=0.5, shed_ready=False
                )
        assert any("unknown" in r.message for r in caplog.records)
