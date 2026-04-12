"""Rolling-window rate calculator for probe metrics.

Used internally by :class:`PageFaultProbe` (fault_rate_per_s) and
:class:`MmapGrowthProbe` (growth_rate_mbps) to compute rates over a
sliding time window without retaining unbounded event history.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple


class _RollingWindow:
    """Accumulate event values and compute rate over a sliding time window.

    Parameters
    ----------
    window_s:
        Width of the sliding window in seconds (default: 5.0).

    Usage
    -----
    ::

        w = _RollingWindow(window_s=5.0)
        w.add(1.0)               # count a fault (value = 1)
        w.add(1024 * 1024)       # count 1 MiB allocated (value = bytes)
        rate = w.rate()          # faults/s  or  bytes/s
    """

    def __init__(self, window_s: float = 5.0) -> None:
        self._window_s: float = window_s
        # Deque of (monotonic_timestamp, value)
        self._events: Deque[Tuple[float, float]] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, value: float = 1.0, ts: Optional[float] = None) -> None:
        """Record one event with the given *value* at time *ts*.

        Parameters
        ----------
        value:
            Numeric value to accumulate (e.g. 1 for a fault count, or
            ``alloc_bytes`` for a memory growth event).
        ts:
            Monotonic timestamp (seconds).  Defaults to ``time.monotonic()``.
        """
        now = ts if ts is not None else time.monotonic()
        self._events.append((now, value))
        self._expire(now)

    def rate(self, now: Optional[float] = None) -> float:
        """Return ``sum(values) / elapsed_seconds`` over the rolling window.

        Returns ``0.0`` when fewer than two events are in the window (cannot
        compute a meaningful rate from a single point in time).

        Parameters
        ----------
        now:
            Reference timestamp for expiry.  Defaults to ``time.monotonic()``.
        """
        now = now if now is not None else time.monotonic()
        self._expire(now)
        if len(self._events) < 2:
            return 0.0
        elapsed = self._events[-1][0] - self._events[0][0]
        if elapsed <= 0.0:
            return 0.0
        total = sum(v for _, v in self._events)
        return total / elapsed

    def count(self) -> int:
        """Number of events currently retained in the window."""
        return len(self._events)

    def reset(self) -> None:
        """Discard all stored events."""
        self._events.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _expire(self, now: float) -> None:
        """Drop events older than ``window_s`` from the front of the deque."""
        cutoff = now - self._window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
