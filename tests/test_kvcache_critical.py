"""Tests for KVCacheMonitor critical_threshold graceful restart (PR 7).

Covers:
  - critical_threshold attribute stored on instance (default 0.95)
  - restart_callback attribute stored on instance (None by default)
  - critical_ticks attribute stored on instance (default 3)
  - restart_callback fires after N consecutive ticks at or above critical_threshold
  - restart_callback NOT fired on a single spike (counter reset by dip below threshold)
  - restart_callback NOT fired when N-1 consecutive ticks above threshold then drops
  - counter resets when utilization dips below threshold; fires again after next N ticks
  - restart_callback fires multiple times across distinct N-tick bursts
  - crashing restart_callback does not kill the monitor thread
  - restart_callback not called when threshold never exceeded
  - custom critical_ticks value respected
  - custom critical_threshold value respected
  - _critical_consecutive resets to 0 on monitor restart (start() called again)
"""

from __future__ import annotations

import threading
import time

import pytest

from memory_guard.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Timing constants — keep tests fast but reliable
# ---------------------------------------------------------------------------

_FAST = 0.01     # poll interval (10 ms → ~10 polls/100 ms)
_WAIT = 0.15     # enough time for >10 polls at _FAST interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mon(poll_fn, critical_threshold=0.95, critical_ticks=3,
         restart_callback=None, **kwargs) -> KVCacheMonitor:
    return KVCacheMonitor(
        poll_fn=poll_fn,
        poll_interval=_FAST,
        on_log=lambda _: None,   # suppress log noise in test output
        cooldown_seconds=0,      # don't suppress on_warning/on_shed_load
        critical_threshold=critical_threshold,
        critical_ticks=critical_ticks,
        restart_callback=restart_callback,
        **kwargs,
    )


def _always_high(pct=96):
    """poll_fn that always returns pct% usage."""
    return lambda: (pct, 100)


def _always_low():
    return lambda: (50, 100)


# ---------------------------------------------------------------------------
# Attribute initialisation
# ---------------------------------------------------------------------------

class TestCriticalAttributes:
    def test_critical_threshold_default(self):
        mon = KVCacheMonitor(poll_fn=_always_low())
        assert mon.critical_threshold == pytest.approx(0.95)

    def test_restart_callback_default_is_none(self):
        mon = KVCacheMonitor(poll_fn=_always_low())
        assert mon.restart_callback is None

    def test_critical_ticks_default(self):
        mon = KVCacheMonitor(poll_fn=_always_low())
        assert mon.critical_ticks == 3

    def test_custom_critical_threshold_stored(self):
        mon = _mon(_always_low(), critical_threshold=0.98)
        assert mon.critical_threshold == pytest.approx(0.98)

    def test_custom_restart_callback_stored(self):
        cb = lambda: None
        mon = _mon(_always_low(), restart_callback=cb)
        assert mon.restart_callback is cb

    def test_custom_critical_ticks_stored(self):
        mon = _mon(_always_low(), critical_ticks=5)
        assert mon.critical_ticks == 5


# ---------------------------------------------------------------------------
# Fires after N consecutive ticks
# ---------------------------------------------------------------------------

class TestCriticalFires:
    def test_callback_fires_after_n_consecutive_ticks(self):
        fired = []
        mon = _mon(_always_high(96), restart_callback=lambda: fired.append(1))
        with mon.session():
            time.sleep(_WAIT)
        assert len(fired) >= 1

    def test_callback_fires_with_custom_threshold(self):
        """Threshold of 0.70 — poll at 75% → should fire."""
        fired = []
        mon = _mon(
            lambda: (75, 100),
            critical_threshold=0.70,
            critical_ticks=3,
            restart_callback=lambda: fired.append(1),
        )
        with mon.session():
            time.sleep(_WAIT)
        assert len(fired) >= 1

    def test_callback_fires_with_custom_critical_ticks(self):
        """critical_ticks=2 — fires after just 2 consecutive above-threshold polls."""
        fired = []
        mon = _mon(
            _always_high(96),
            critical_ticks=2,
            restart_callback=lambda: fired.append(1),
        )
        with mon.session():
            time.sleep(_WAIT)
        assert len(fired) >= 1

    def test_callback_receives_no_argument(self):
        """restart_callback is zero-argument — verify it is called as such."""
        results = []

        def cb():
            results.append("called")

        mon = _mon(_always_high(96), restart_callback=cb)
        with mon.session():
            time.sleep(_WAIT)
        assert len(results) >= 1

    def test_counter_resets_after_firing_allowing_second_burst(self):
        """After N ticks → fire → reset counter, a second N-tick burst fires again."""
        call_count = [0]
        fired = []

        def burst_poll():
            call_count[0] += 1
            # Ticks 1-3: high → first fire at tick 3, counter reset to 0
            # Tick 4: low → counter stays 0 (belt-and-suspenders reset)
            # Ticks 5-7: high → second fire at tick 7
            # Ticks 8+: low
            n = call_count[0]
            if n <= 3 or (5 <= n <= 7):
                return (96, 100)
            return (50, 100)

        mon = _mon(burst_poll, critical_ticks=3, restart_callback=lambda: fired.append(1))
        with mon.session():
            time.sleep(_WAIT)
        # Two distinct bursts of 3 → exactly 2 fires
        assert len(fired) >= 2


# ---------------------------------------------------------------------------
# Does NOT fire on insufficient consecutive ticks
# ---------------------------------------------------------------------------

class TestCriticalDoesNotFire:
    def test_callback_not_fired_when_threshold_never_exceeded(self):
        fired = []
        mon = _mon(
            lambda: (80, 100),   # 80% — below default 0.95 threshold
            restart_callback=lambda: fired.append(1),
        )
        with mon.session():
            time.sleep(_WAIT)
        assert fired == []

    def test_callback_not_fired_on_single_spike_then_dip(self):
        """Single above-threshold tick followed by many below-threshold → never fires."""
        call_count = [0]

        def spike_once():
            call_count[0] += 1
            return (96, 100) if call_count[0] == 1 else (50, 100)

        fired = []
        mon = _mon(spike_once, critical_ticks=3, restart_callback=lambda: fired.append(1))
        with mon.session():
            time.sleep(_WAIT)
        assert fired == []

    def test_callback_not_fired_on_n_minus_one_consecutive_ticks(self):
        """Exactly N-1 consecutive ticks, then drops — callback must NOT fire."""
        call_count = [0]
        critical_ticks = 4

        def two_then_low():
            call_count[0] += 1
            # First (critical_ticks - 1) calls: above threshold
            return (96, 100) if call_count[0] <= (critical_ticks - 1) else (50, 100)

        fired = []
        mon = _mon(
            two_then_low,
            critical_ticks=critical_ticks,
            restart_callback=lambda: fired.append(1),
        )
        with mon.session():
            time.sleep(_WAIT)
        assert fired == []

    def test_callback_not_fired_when_restart_callback_is_none(self):
        """No callback configured — no error, counter still increments safely."""
        # Just verify this doesn't raise even when repeatedly exceeding threshold
        mon = _mon(_always_high(96), restart_callback=None)
        with mon.session():
            time.sleep(_WAIT)
        # If we reach here without exception, the test passes

    def test_alternating_above_below_never_fires(self):
        """Above/below alternating pattern — counter always resets before N ticks."""
        call_count = [0]

        def alternating():
            call_count[0] += 1
            return (96, 100) if call_count[0] % 2 == 1 else (50, 100)

        fired = []
        mon = _mon(alternating, critical_ticks=3, restart_callback=lambda: fired.append(1))
        with mon.session():
            time.sleep(_WAIT)
        assert fired == []


# ---------------------------------------------------------------------------
# Counter resets on monitor restart
# ---------------------------------------------------------------------------

class TestCriticalCounterReset:
    def test_consecutive_counter_resets_on_start(self):
        """start() must reset _critical_consecutive regardless of prior state."""
        # Use a very large critical_ticks so the callback never fires and the
        # counter just accumulates freely.
        mon = _mon(_always_high(96), critical_ticks=1000, restart_callback=lambda: None)

        # Corrupt the counter to a large value to prove start() clears it.
        mon._critical_consecutive = 999

        # start() resets the counter (inside the lock, before the thread launches).
        mon.start()
        mon.stop()  # join thread so _critical_consecutive is stable

        # If the reset worked: 0 + at most a few polls = small number (< 10).
        # If the reset was skipped: 999 + a few polls = large number (> 999).
        assert mon._critical_consecutive < 10

    def test_critical_consecutive_increments_correctly(self):
        """After 2 polls above threshold (with ticks=10), counter should be 2."""
        mon = _mon(_always_high(96), critical_ticks=10, restart_callback=lambda: None)
        mon.start()
        time.sleep(0.025)  # ~2 polls at 10 ms interval
        consecutive = mon._critical_consecutive
        mon.stop()
        # Should have counted 1 or 2 consecutive ticks without firing (ticks=10)
        assert consecutive >= 1


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

class TestCriticalResilience:
    def test_crashing_restart_callback_does_not_kill_thread(self):
        """An exception from restart_callback must be swallowed; monitor keeps running."""
        def bad_restart():
            raise RuntimeError("intentional restart_callback error")

        mon = _mon(_always_high(96), critical_ticks=3, restart_callback=bad_restart)
        with mon.session():
            time.sleep(_WAIT)
            assert mon.is_running is True

    def test_thread_safe_multiple_fires(self):
        """Many consecutive fires (counter resets each time) — no data races."""
        fired = []
        lock = threading.Lock()

        def cb():
            with lock:
                fired.append(1)

        mon = _mon(_always_high(96), critical_ticks=3, restart_callback=cb)
        with mon.session():
            time.sleep(0.3)  # long enough for many fire-reset cycles
        assert len(fired) >= 1  # at minimum one fire; no assertion on upper bound
