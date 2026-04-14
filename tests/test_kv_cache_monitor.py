"""Tests for KVCacheMonitor.

Covers:
  - Lifecycle: start / stop / is_running / context manager / double start
  - Metrics: current_utilization, utilization_history, formula, deque maxlen
  - Callbacks: on_warning (80%), on_shed_load (92%), priority, cooldown
  - Resilience: crashing poll_fn, crashing callbacks, zero total_blocks
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from memory_guard.monitoring.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAST = 0.01   # poll_interval for tests — fast enough to get 2+ polls in 50 ms
_WAIT = 0.12   # sleep time to let ≥2 poll cycles complete


def _make_monitor(poll_fn=None, **kwargs) -> KVCacheMonitor:
    """Return a KVCacheMonitor with a fast poll interval."""
    if poll_fn is None:
        poll_fn = lambda: (0, 100)
    return KVCacheMonitor(poll_fn=poll_fn, poll_interval=_FAST, **kwargs)


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestKVCacheMonitorLifecycle:
    def test_is_running_false_before_start(self):
        mon = _make_monitor()
        assert mon.is_running is False

    def test_is_running_true_after_start(self):
        mon = _make_monitor()
        mon.start()
        try:
            assert mon.is_running is True
        finally:
            mon.stop()

    def test_is_running_false_after_stop(self):
        mon = _make_monitor()
        mon.start()
        mon.stop()
        assert mon.is_running is False

    def test_context_manager_starts_and_stops(self):
        mon = _make_monitor()
        with mon.session() as entered:
            assert entered is mon
            assert mon.is_running is True
        assert mon.is_running is False

    def test_context_manager_returns_monitor_instance(self):
        mon = _make_monitor()
        with mon.session() as entered:
            assert isinstance(entered, KVCacheMonitor)

    def test_double_start_stops_previous_thread(self):
        mon = _make_monitor()
        mon.start()
        first_thread = mon._thread
        mon.start()
        try:
            assert mon._thread is not first_thread
            assert mon.is_running is True
        finally:
            mon.stop()

    def test_stop_before_start_does_not_raise(self):
        mon = _make_monitor()
        mon.stop()  # Should be a no-op

    def test_history_cleared_on_restart(self):
        """Restarting discards readings from the previous session."""
        mon = _make_monitor(poll_fn=lambda: (50, 100))
        with mon.session():
            time.sleep(_WAIT)
        assert 0.5 in mon.utilization_history

        # Restart with a different poll_fn — start() clears history first
        mon.poll_fn = lambda: (90, 100)
        mon.start()
        time.sleep(_WAIT)
        mon.stop()

        assert 0.5 not in mon.utilization_history
        assert 0.9 in mon.utilization_history


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestKVCacheMonitorMetrics:
    def test_current_utilization_zero_before_any_poll(self):
        mon = _make_monitor()
        assert mon.current_utilization == 0.0

    def test_utilization_history_empty_before_start(self):
        mon = _make_monitor()
        assert mon.utilization_history == []

    def test_utilization_formula_used_over_total(self):
        """utilization = used / total — verify exact value."""
        poll_fn = lambda: (75, 100)
        mon = _make_monitor(poll_fn=poll_fn)
        with mon.session():
            time.sleep(_WAIT)
        assert abs(mon.current_utilization - 0.75) < 1e-9

    def test_zero_total_blocks_gives_zero_utilization(self):
        poll_fn = lambda: (0, 0)
        mon = _make_monitor(poll_fn=poll_fn)
        with mon.session():
            time.sleep(_WAIT)
        assert mon.current_utilization == 0.0

    def test_history_accumulates_readings(self):
        poll_fn = lambda: (50, 100)
        mon = _make_monitor(poll_fn=poll_fn)
        with mon.session():
            time.sleep(_WAIT)
        assert len(mon.utilization_history) >= 2

    def test_history_max_size_respected(self):
        poll_fn = lambda: (50, 100)
        mon = KVCacheMonitor(poll_fn=poll_fn, poll_interval=_FAST, history_size=5)
        with mon.session():
            time.sleep(0.15)   # enough polls to exceed history_size=5
        assert len(mon.utilization_history) <= 5

    def test_utilization_history_returns_copy(self):
        poll_fn = lambda: (50, 100)
        mon = _make_monitor(poll_fn=poll_fn)
        with mon.session():
            time.sleep(_WAIT)
        h1 = mon.utilization_history
        h2 = mon.utilization_history
        assert h1 == h2
        assert h1 is not h2  # separate list objects


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------

class TestKVCacheMonitorCallbacks:
    def test_on_warning_fires_at_80_percent(self):
        fired = []
        poll_fn = lambda: (80, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_warning=lambda u: fired.append(u),
            cooldown_seconds=0,
        )
        with mon.session():
            time.sleep(_WAIT)
        assert len(fired) > 0
        assert all(abs(u - 0.80) < 1e-9 for u in fired)

    def test_on_warning_not_fired_below_80_percent(self):
        fired = []
        poll_fn = lambda: (79, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_warning=lambda u: fired.append(u),
            cooldown_seconds=0,
        )
        with mon.session():
            time.sleep(_WAIT)
        assert fired == []

    def test_on_shed_load_fires_at_92_percent(self):
        fired = []
        poll_fn = lambda: (92, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_shed_load=lambda u: fired.append(u),
            cooldown_seconds=0,
        )
        with mon.session():
            time.sleep(_WAIT)
        assert len(fired) > 0
        assert all(abs(u - 0.92) < 1e-9 for u in fired)

    def test_on_shed_load_not_fired_below_92_percent(self):
        fired = []
        poll_fn = lambda: (91, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_shed_load=lambda u: fired.append(u),
            cooldown_seconds=0,
        )
        with mon.session():
            time.sleep(_WAIT)
        assert fired == []

    def test_shed_load_takes_priority_over_warning_at_92_percent(self):
        """At 92 %, only on_shed_load fires — not on_warning."""
        warning_fired = []
        shed_fired = []
        poll_fn = lambda: (92, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_warning=lambda u: warning_fired.append(u),
            on_shed_load=lambda u: shed_fired.append(u),
            cooldown_seconds=0,
        )
        with mon.session():
            time.sleep(_WAIT)
        assert len(shed_fired) > 0
        assert warning_fired == []

    def test_cooldown_limits_callback_frequency(self):
        """With cooldown=1000s, callback fires at most once."""
        fired = []
        poll_fn = lambda: (90, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_warning=lambda u: fired.append(u),
            cooldown_seconds=1000,
        )
        with mon.session():
            time.sleep(_WAIT)
        assert len(fired) <= 1

    def test_crashing_poll_fn_does_not_kill_thread(self):
        """Exceptions from poll_fn are caught; thread keeps running."""
        calls = [0]

        def bad_poll():
            calls[0] += 1
            raise RuntimeError("intentional poll error")

        mon = _make_monitor(poll_fn=bad_poll)
        with mon.session():
            time.sleep(_WAIT)
            # Assert inside the with block — stop() hasn't run yet
            assert mon.is_running is True
            assert calls[0] >= 2

    def test_crashing_on_warning_callback_does_not_kill_thread(self):
        """Exceptions from on_warning are caught; thread keeps running."""
        def bad_warning(_):
            raise RuntimeError("intentional callback error")

        poll_fn = lambda: (85, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_warning=bad_warning,
            cooldown_seconds=0,
        )
        with mon.session():
            time.sleep(_WAIT)
            assert mon.is_running is True

    def test_crashing_on_shed_load_callback_does_not_kill_thread(self):
        """Exceptions from on_shed_load are caught; thread keeps running."""
        def bad_shed(_):
            raise RuntimeError("intentional callback error")

        poll_fn = lambda: (95, 100)
        mon = _make_monitor(
            poll_fn=poll_fn,
            on_shed_load=bad_shed,
            cooldown_seconds=0,
        )
        with mon.session():
            time.sleep(_WAIT)
            assert mon.is_running is True


# ---------------------------------------------------------------------------
# Export / import test
# ---------------------------------------------------------------------------

class TestKVCacheMonitorExport:
    def test_importable_from_top_level_package(self):
        from memory_guard import KVCacheMonitor as KVM  # noqa: F401
        assert KVM is KVCacheMonitor

    def test_threshold_constants_accessible(self):
        assert KVCacheMonitor.THRESHOLD_WARNING == pytest.approx(0.80)
        assert KVCacheMonitor.THRESHOLD_SHED_LOAD == pytest.approx(0.92)
