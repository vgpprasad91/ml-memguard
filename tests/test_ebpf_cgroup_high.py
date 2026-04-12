"""Tests for PR 54: CgroupMemoryHighProbe — cgroup memory.high BPF probe.

Covers:
  CgroupMemoryHighProbe
  ---------------------
   1. load() raises OSError on non-Linux platforms
   2. BPF event parsing: all fields populated from _CgroupMemHighEvent struct
   3. pressure_bytes is correctly calculated (actual - high)
   4. on_high callback is dispatched for every event
   5. on_oom_imminent is NOT dispatched when pressure_bytes < threshold
   6. on_oom_imminent is NOT dispatched when pressure_bytes == threshold (exclusive)
   7. on_oom_imminent IS dispatched when pressure_bytes > threshold
   8. cgroup_filter passes events whose cgroup_id matches the prefix
   9. cgroup_filter silently drops events whose cgroup_id does not match
  10. mock ring buffer drain: multiple events dispatched in arrival order
"""

from __future__ import annotations

import ctypes
import sys
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.ebpf.probes.cgroup_memory_high import (
    CgroupMemoryHighProbe,
    _CgroupMemHighEvent,
    _DEFAULT_OOM_IMMINENT_THRESHOLD_MB,
)
from memory_guard.ebpf._event import EVENT_MEMORY_HIGH, MemguardBPFEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_event(
    timestamp_ns:   int   = 1_000_000_000,
    pressure_bytes: int   = 1024 * 1024,        # 1 MiB
    pid:            int   = 42,
    cgroup_id:      bytes = b"/kubepods/pod-abc/container-xyz",
) -> _CgroupMemHighEvent:
    """Build a _CgroupMemHighEvent that looks like a real perf-buffer payload.

    Direct assignment to a ``ctypes.c_char * N`` field is the correct way to
    set string data — memmove via the field accessor does not write through.
    """
    raw = _CgroupMemHighEvent()
    raw.timestamp_ns   = timestamp_ns
    raw.pressure_bytes = pressure_bytes
    raw.pid            = pid
    raw._pad           = 0
    # c_char * 128 assignment truncates to 127 bytes + NUL automatically
    raw.cgroup_id      = cgroup_id[:127]
    return raw


def _dispatch_raw(probe: CgroupMemoryHighProbe, raw: _CgroupMemHighEvent) -> None:
    """Call the probe's internal _dispatch method directly (bypass BPF layer)."""
    probe._dispatch(raw)


# ===========================================================================
# 1. Platform check
# ===========================================================================

class TestLoadPlatformGuard:

    def test_load_raises_os_error_on_non_linux(self):
        """load() raises OSError on macOS / Windows."""
        probe = CgroupMemoryHighProbe()
        with patch("memory_guard.ebpf.probes.cgroup_memory_high.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with pytest.raises(OSError, match="Linux-only"):
                probe.load()


# ===========================================================================
# 2. BPF event parsing
# ===========================================================================

class TestEventParsing:

    def test_all_fields_populated_from_raw_struct(self):
        """_dispatch() populates every MemguardBPFEvent field correctly."""
        received: List[MemguardBPFEvent] = []
        probe = CgroupMemoryHighProbe(on_high=received.append)

        raw = _make_raw_event(
            timestamp_ns   = 999_000_000,
            pressure_bytes = 2 * 1024 * 1024,   # 2 MiB
            pid            = 77,
            cgroup_id      = b"/kubepods/burstable/pod-xyz",
        )
        _dispatch_raw(probe, raw)

        assert len(received) == 1
        e = received[0]
        assert e.ts_ns          == 999_000_000
        assert e.pressure_bytes == 2 * 1024 * 1024
        assert e.pid            == 77
        assert e.cgroup_id      == "/kubepods/burstable/pod-xyz"
        assert e.event_type     == EVENT_MEMORY_HIGH


# ===========================================================================
# 3. pressure_bytes calculation
# ===========================================================================

class TestPressureBytesCalculation:

    def test_pressure_bytes_equals_actual_minus_high(self):
        """pressure_bytes in the dispatched event equals the struct field value
        (the BPF C program pre-calculates actual - high; Python just passes it)."""
        received: List[MemguardBPFEvent] = []
        probe = CgroupMemoryHighProbe(on_high=received.append)

        # Simulate: actual = 1.5 GiB, high = 1 GiB → pressure = 512 MiB
        expected_pressure = 512 * 1024 * 1024
        raw = _make_raw_event(pressure_bytes=expected_pressure)
        _dispatch_raw(probe, raw)

        assert received[0].pressure_bytes == expected_pressure


# ===========================================================================
# 4–7. Callback dispatch
# ===========================================================================

class TestCallbackDispatch:

    def test_on_high_called_for_every_event(self):
        """on_high fires regardless of pressure_bytes magnitude."""
        calls: List[MemguardBPFEvent] = []
        probe = CgroupMemoryHighProbe(
            on_high=calls.append,
            oom_imminent_threshold_mb=512.0,
        )
        # Low pressure — well below OOM threshold
        _dispatch_raw(probe, _make_raw_event(pressure_bytes=1))
        assert len(calls) == 1

    def test_on_oom_imminent_not_fired_below_threshold(self):
        """on_oom_imminent is NOT called when pressure_bytes < threshold."""
        oom_calls: List[MemguardBPFEvent] = []
        threshold_mb = 512.0
        threshold_b  = int(threshold_mb * 1024 * 1024)

        probe = CgroupMemoryHighProbe(
            on_oom_imminent=oom_calls.append,
            oom_imminent_threshold_mb=threshold_mb,
        )
        # pressure just below threshold
        _dispatch_raw(probe, _make_raw_event(pressure_bytes=threshold_b - 1))
        assert oom_calls == []

    def test_on_oom_imminent_not_fired_at_exact_threshold(self):
        """on_oom_imminent uses strict greater-than — equal does NOT fire it."""
        oom_calls: List[MemguardBPFEvent] = []
        threshold_mb = 256.0
        threshold_b  = int(threshold_mb * 1024 * 1024)

        probe = CgroupMemoryHighProbe(
            on_oom_imminent=oom_calls.append,
            oom_imminent_threshold_mb=threshold_mb,
        )
        _dispatch_raw(probe, _make_raw_event(pressure_bytes=threshold_b))
        assert oom_calls == []

    def test_on_oom_imminent_fired_above_threshold(self):
        """on_oom_imminent IS called when pressure_bytes > threshold."""
        oom_calls: List[MemguardBPFEvent] = []
        threshold_mb = 256.0
        threshold_b  = int(threshold_mb * 1024 * 1024)

        probe = CgroupMemoryHighProbe(
            on_oom_imminent=oom_calls.append,
            oom_imminent_threshold_mb=threshold_mb,
        )
        _dispatch_raw(probe, _make_raw_event(pressure_bytes=threshold_b + 1))
        assert len(oom_calls) == 1
        assert oom_calls[0].pressure_bytes == threshold_b + 1


# ===========================================================================
# 8–9. cgroup_filter
# ===========================================================================

class TestCgroupFilter:

    def test_filter_passes_matching_cgroup(self):
        """Events whose cgroup_id starts with cgroup_filter are dispatched."""
        received: List[MemguardBPFEvent] = []
        probe = CgroupMemoryHighProbe(
            on_high=received.append,
            cgroup_filter="/kubepods/",
        )
        _dispatch_raw(
            probe,
            _make_raw_event(cgroup_id=b"/kubepods/burstable/pod-abc"),
        )
        assert len(received) == 1

    def test_filter_drops_non_matching_cgroup(self):
        """Events whose cgroup_id does NOT match the prefix are silently dropped."""
        received: List[MemguardBPFEvent] = []
        probe = CgroupMemoryHighProbe(
            on_high=received.append,
            cgroup_filter="/kubepods/",
        )
        # system.slice does not match /kubepods/ prefix
        _dispatch_raw(
            probe,
            _make_raw_event(cgroup_id=b"/system.slice/docker.service"),
        )
        assert received == []


# ===========================================================================
# 10. Mock ring buffer drain — multiple events in order
# ===========================================================================

class TestRingBufferDrain:

    def test_multiple_events_dispatched_in_arrival_order(self):
        """Draining a sequence of events preserves arrival order."""
        received: List[MemguardBPFEvent] = []
        probe = CgroupMemoryHighProbe(on_high=received.append)

        events = [
            _make_raw_event(timestamp_ns=1_000, pressure_bytes=100 * 1024),
            _make_raw_event(timestamp_ns=2_000, pressure_bytes=200 * 1024),
            _make_raw_event(timestamp_ns=3_000, pressure_bytes=300 * 1024),
        ]
        for raw in events:
            _dispatch_raw(probe, raw)

        assert len(received) == 3
        assert [e.ts_ns for e in received]          == [1_000, 2_000, 3_000]
        assert [e.pressure_bytes for e in received] == [
            100 * 1024, 200 * 1024, 300 * 1024
        ]
