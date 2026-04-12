"""Tests for PR 55: PageFaultProbe and MmapGrowthProbe.

Covers:
  PageFaultProbe
  --------------
   1. BPF event parsing: all fields populated from _PageFaultEvent struct
   2. PID filter passes allowlisted PID
   3. PID filter drops non-allowlisted PID
   4. fault_rate_per_s rolling window: correct rate computed

  MmapGrowthProbe
  ---------------
   5. BPF event parsing: all fields populated for mmap subtype
   6. BPF event parsing: brk subtype set correctly in extra dict
   7. PID filter drops non-allowlisted PID
   8. growth_rate_mbps rolling window: correct MB/s computed

  _RollingWindow (shared internals)
  ---------------------------------
   9. rate() returns 0.0 with fewer than two events

  Integration / teardown
  ----------------------
  10. Silent-kill simulation: rapid fault injection drives high fault_rate_per_s
  11. teardown: detach() safe when never loaded (no OSError)
  12. load() raises OSError on non-Linux (macOS / Windows)
"""

from __future__ import annotations

import ctypes
import sys
import time
from typing import List
from unittest.mock import patch

import pytest

from memory_guard.ebpf.probes.page_fault import PageFaultProbe, _PageFaultEvent
from memory_guard.ebpf.probes.mmap_growth import MmapGrowthProbe, _MmapEvent, _SUBTYPE_BRK, _SUBTYPE_MMAP
from memory_guard.ebpf.probes._rolling_window import _RollingWindow
from memory_guard.ebpf._event import EVENT_PAGE_FAULT, EVENT_MMAP_GROWTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fault_event(
    timestamp_ns:  int = 1_000_000_000,
    fault_address: int = 0x7FFF_0000,
    error_code:    int = 4,             # PF_USER bit
    pid:           int = 42,
) -> _PageFaultEvent:
    raw = _PageFaultEvent()
    raw.timestamp_ns  = timestamp_ns
    raw.fault_address = fault_address
    raw.error_code    = error_code
    raw.pid           = pid
    return raw


def _make_mmap_event(
    timestamp_ns:  int = 2_000_000_000,
    alloc_bytes:   int = 4 * 1024 * 1024,   # 4 MiB
    pid:           int = 99,
    event_subtype: int = _SUBTYPE_MMAP,
) -> _MmapEvent:
    raw = _MmapEvent()
    raw.timestamp_ns  = timestamp_ns
    raw.alloc_bytes   = alloc_bytes
    raw.pid           = pid
    raw.event_subtype = event_subtype
    return raw


# ===========================================================================
# 1. PageFaultProbe — event parsing
# ===========================================================================

class TestPageFaultParsing:

    def test_all_fields_populated(self):
        """_dispatch() fills every MemguardBPFEvent field correctly."""
        received = []
        probe = PageFaultProbe(on_fault=received.append)

        raw = _make_fault_event(
            timestamp_ns  = 999_123_456,
            fault_address = 0xDEAD_BEEF,
            error_code    = 7,
            pid           = 1234,
        )
        probe._dispatch(raw)

        assert len(received) == 1
        e = received[0]
        assert e.ts_ns              == 999_123_456
        assert e.event_type         == EVENT_PAGE_FAULT
        assert e.pressure_bytes     == 0
        assert e.pid                == 1234
        assert e.extra["fault_address"] == 0xDEAD_BEEF
        assert e.extra["error_code"]    == 7


# ===========================================================================
# 2–3. PageFaultProbe — PID filtering
# ===========================================================================

class TestPageFaultPidFilter:

    def test_passes_allowlisted_pid(self):
        """Events for a PID in the allowlist are dispatched."""
        received = []
        probe = PageFaultProbe(
            on_fault=received.append,
            pid_allowlist={1234},
        )
        probe._dispatch(_make_fault_event(pid=1234))
        assert len(received) == 1

    def test_drops_non_allowlisted_pid(self):
        """Events for a PID NOT in the allowlist are silently dropped."""
        received = []
        probe = PageFaultProbe(
            on_fault=received.append,
            pid_allowlist={1234},
        )
        probe._dispatch(_make_fault_event(pid=9999))
        assert received == []


# ===========================================================================
# 4. PageFaultProbe — fault_rate_per_s rolling window
# ===========================================================================

class TestFaultRateRollingWindow:

    def test_fault_rate_per_s_computed_correctly(self):
        """10 faults over ~1 s → rate ≈ 10 f/s."""
        probe = PageFaultProbe(rate_window_s=10.0)
        t0 = time.monotonic()
        for i in range(11):  # 11 events → 10 intervals of 0.1 s → 1.0 s elapsed
            probe._window.add(1.0, ts=t0 + i * 0.1)
        # sum=11, elapsed=1.0s, rate=11/1.0=11.0  (events still within window)
        rate = probe.fault_rate_per_s
        assert 9.0 <= rate <= 15.0


# ===========================================================================
# 5–6. MmapGrowthProbe — event parsing
# ===========================================================================

class TestMmapParsing:

    def test_all_fields_populated_mmap_subtype(self):
        """_dispatch() fills every MemguardBPFEvent field for an mmap event."""
        received = []
        probe = MmapGrowthProbe(on_growth=received.append)

        raw = _make_mmap_event(
            timestamp_ns  = 555_000_000,
            alloc_bytes   = 8 * 1024 * 1024,   # 8 MiB
            pid           = 77,
            event_subtype = _SUBTYPE_MMAP,
        )
        probe._dispatch(raw)

        assert len(received) == 1
        e = received[0]
        assert e.ts_ns          == 555_000_000
        assert e.event_type     == EVENT_MMAP_GROWTH
        assert e.pressure_bytes == 8 * 1024 * 1024
        assert e.pid            == 77
        assert e.extra["subtype"] == "mmap"

    def test_brk_subtype_set_in_extra(self):
        """Events with event_subtype=1 have extra['subtype'] == 'brk'."""
        received = []
        probe = MmapGrowthProbe(on_growth=received.append)
        probe._dispatch(_make_mmap_event(event_subtype=_SUBTYPE_BRK))
        assert received[0].extra["subtype"] == "brk"


# ===========================================================================
# 7. MmapGrowthProbe — PID filtering
# ===========================================================================

class TestMmapPidFilter:

    def test_drops_non_allowlisted_pid(self):
        """Mmap events for a PID not in the allowlist are dropped."""
        received = []
        probe = MmapGrowthProbe(
            on_growth=received.append,
            pid_allowlist={1234},
        )
        probe._dispatch(_make_mmap_event(pid=9999))
        assert received == []


# ===========================================================================
# 8. MmapGrowthProbe — growth_rate_mbps rolling window
# ===========================================================================

class TestGrowthRateRollingWindow:

    def test_growth_rate_mbps_computed_correctly(self):
        """5 × 100 MiB over ~2 s → ~250 MiB/s."""
        probe = MmapGrowthProbe(rate_window_s=10.0)
        t0 = time.monotonic()
        alloc = 100 * 1024 * 1024   # 100 MiB
        for i in range(5):
            probe._window.add(float(alloc), ts=t0 + i * 0.5)
        # sum = 500 MiB = 524_288_000 bytes, elapsed = 4×0.5 = 2.0 s
        # rate() = 262_144_000 bytes/s  →  growth_rate_mbps = 250.0
        rate = probe.growth_rate_mbps
        assert 200.0 <= rate <= 300.0


# ===========================================================================
# 9. _RollingWindow — edge case
# ===========================================================================

class TestRollingWindowEdge:

    def test_rate_zero_with_fewer_than_two_events(self):
        """rate() returns 0.0 when 0 or 1 events are in the window."""
        w = _RollingWindow(window_s=5.0)
        assert w.rate() == 0.0
        w.add(1.0, ts=100.0)
        assert w.rate(now=100.0) == 0.0


# ===========================================================================
# 10. Silent-kill simulation
# ===========================================================================

class TestSilentKillSimulation:

    def test_rapid_fault_injection_drives_high_rate(self):
        """Simulate a swap storm: 50 faults in 0.5 s → fault_rate_per_s > 50."""
        received = []
        probe = PageFaultProbe(
            on_fault=received.append,
            rate_window_s=2.0,
        )
        t0 = 5_000.0
        for i in range(51):   # 51 events, 50 intervals × 10 ms = 500 ms
            probe._window.add(1.0, ts=t0 + i * 0.01)
            if i < 51:
                probe._dispatch(_make_fault_event(pid=42, timestamp_ns=int((t0 + i * 0.01) * 1e9)))

        # rate = 51 / (50 × 0.01) ≈ 102 f/s — well above the 50 f/s threshold
        assert probe.fault_rate_per_s > 50.0
        assert len(received) == 51


# ===========================================================================
# 11. Teardown safety
# ===========================================================================

class TestTeardown:

    def test_detach_safe_when_never_loaded(self):
        """detach() on an unloaded probe must not raise."""
        probe = PageFaultProbe()
        probe.detach()          # must be a no-op
        assert not probe.is_loaded

    def test_mmap_detach_safe_when_never_loaded(self):
        """MmapGrowthProbe.detach() on an unloaded probe must not raise."""
        probe = MmapGrowthProbe()
        probe.detach()
        assert not probe.is_loaded


# ===========================================================================
# 12. Platform guard
# ===========================================================================

class TestLoadPlatformGuard:

    def test_page_fault_load_raises_on_non_linux(self):
        """PageFaultProbe.load() raises OSError on macOS / Windows."""
        probe = PageFaultProbe()
        with patch("memory_guard.ebpf.probes.page_fault.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with pytest.raises(OSError, match="Linux-only"):
                probe.load()

    def test_mmap_load_raises_on_non_linux(self):
        """MmapGrowthProbe.load() raises OSError on macOS / Windows."""
        probe = MmapGrowthProbe()
        with patch("memory_guard.ebpf.probes.mmap_growth.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with pytest.raises(OSError, match="Linux-only"):
                probe.load()
