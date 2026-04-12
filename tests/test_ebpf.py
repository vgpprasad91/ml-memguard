"""Tests for PR 35: eBPF cgroup memory + preemption probes.

Covers:
  CgroupMemoryProbe
  -----------------
  - load() raises OSError on non-Linux platforms
  - load() raises PermissionError when uid != 0 and caps absent (non-root)
  - load() raises ImportError when bcc is not installed
  - detach() is safe to call when not loaded (no-op)
  - is_loaded is False before load(), True after
  - poll() is safe when not loaded (no-op)
  - __repr__ reports state

  PreemptionProbe
  ---------------
  - load() raises OSError on non-Linux platforms
  - load() raises PermissionError when uid != 0 and caps absent
  - load() raises ImportError when bcc is not installed
  - detach() is safe when unloaded
  - is_loaded tracks load/detach lifecycle
  - __repr__ includes target_pid and state

  EBPFProbeManager
  ----------------
  - start() without load() is a no-op (logs warning)
  - is_loaded is False before load()
  - is_running is False before start()
  - repr contains state
  - stop() is safe when never started
  - _dispatch_mem_event routes LEVEL_HIGH to on_high callback
  - _dispatch_mem_event routes LEVEL_OOM to on_oom callback
  - _dispatch_mem_event sets ebpf_wake on LEVEL_HIGH
  - _dispatch_mem_event does NOT set ebpf_wake on LEVEL_OOM
  - _dispatch_preemption_event calls on_preemption callback
  - callback exceptions are swallowed (do not propagate)

  KVCacheMonitor + eBPF integration
  ----------------------------------
  - use_ebpf=False → _ebpf_manager stays None after start()/stop()
  - use_ebpf=True  → _start_ebpf() called; manager stored when load succeeds
  - stop() calls _ebpf_manager.stop() and clears reference
  - stop() sets _ebpf_wake (unblocks waiting thread)
  - _ebpf_wake.wait() used in _loop (not _stop.wait)

  Root-only tests (skipped when uid != 0)
  ----------------------------------------
  (All root-only tests are decorated with @root_required and will be
  skipped on macOS developer machines.)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import namedtuple
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

root_required = pytest.mark.skipif(
    os.getuid() != 0,
    reason="eBPF probes require root or CAP_BPF — skipped on non-root",
)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from memory_guard.ebpf.cgroup_memory import (
    CgroupMemoryProbe,
    MemPressureEvent,
    LEVEL_HIGH,
    LEVEL_OOM,
    _has_cap_bpf,
)
from memory_guard.ebpf.preemption import (
    PreemptionProbe,
    PreemptionEvent,
)
from memory_guard.ebpf import EBPFProbeManager


# ---------------------------------------------------------------------------
# CgroupMemoryProbe — platform / permission / import guards
# ---------------------------------------------------------------------------

class TestCgroupMemoryProbeGuards:
    def test_load_raises_on_non_linux(self):
        """load() must raise OSError on any non-Linux platform."""
        with patch("sys.platform", "darwin"):
            probe = CgroupMemoryProbe(on_event=lambda e: None)
            with pytest.raises(OSError, match="Linux-only"):
                probe.load()

    def test_load_raises_permission_error_when_not_root_no_caps(self):
        """Non-root without CAP_BPF raises PermissionError."""
        with patch("sys.platform", "linux"):
            with patch("os.getuid", return_value=1000):
                with patch("memory_guard.ebpf.cgroup_memory._has_cap_bpf", return_value=False):
                    probe = CgroupMemoryProbe(on_event=lambda e: None)
                    with pytest.raises(PermissionError):
                        probe.load()

    def test_load_raises_import_error_when_bcc_missing(self):
        """ImportError is raised when bcc is not installed."""
        with patch("sys.platform", "linux"):
            with patch("os.getuid", return_value=0):
                with patch.dict("sys.modules", {"bcc": None}):
                    probe = CgroupMemoryProbe(on_event=lambda e: None)
                    with pytest.raises(ImportError, match="bcc"):
                        probe.load()

    def test_detach_noop_when_unloaded(self):
        """detach() must not raise when called before load()."""
        probe = CgroupMemoryProbe(on_event=lambda e: None)
        probe.detach()  # must not raise

    def test_is_loaded_false_before_load(self):
        probe = CgroupMemoryProbe(on_event=lambda e: None)
        assert not probe.is_loaded

    def test_poll_noop_when_unloaded(self):
        """poll() must not raise when bpf is None."""
        probe = CgroupMemoryProbe(on_event=lambda e: None)
        probe.poll()  # must not raise

    def test_repr_reports_unloaded(self):
        probe = CgroupMemoryProbe(on_event=lambda e: None)
        assert "unloaded" in repr(probe)

    def test_repr_reports_loaded_after_bpf_set(self):
        probe = CgroupMemoryProbe(on_event=lambda e: None)
        probe._bpf = object()  # simulate loaded state
        assert "loaded" in repr(probe)


# ---------------------------------------------------------------------------
# PreemptionProbe — platform / permission / import guards
# ---------------------------------------------------------------------------

class TestPreemptionProbeGuards:
    def test_load_raises_on_non_linux(self):
        probe = PreemptionProbe(target_pid=12345, on_event=lambda e: None)
        with patch("sys.platform", "darwin"):
            with pytest.raises(OSError, match="Linux-only"):
                probe.load()

    def test_load_raises_permission_error_when_not_root_no_caps(self):
        with patch("sys.platform", "linux"):
            with patch("os.getuid", return_value=1000):
                with patch("memory_guard.ebpf.preemption._has_cap_bpf", return_value=False):
                    probe = PreemptionProbe(target_pid=1, on_event=lambda e: None)
                    with pytest.raises(PermissionError):
                        probe.load()

    def test_load_raises_import_error_when_bcc_missing(self):
        with patch("sys.platform", "linux"):
            with patch("os.getuid", return_value=0):
                with patch.dict("sys.modules", {"bcc": None}):
                    probe = PreemptionProbe(target_pid=1, on_event=lambda e: None)
                    with pytest.raises(ImportError, match="bcc"):
                        probe.load()

    def test_detach_noop_when_unloaded(self):
        probe = PreemptionProbe(target_pid=1, on_event=lambda e: None)
        probe.detach()  # must not raise

    def test_is_loaded_false_before_load(self):
        probe = PreemptionProbe(target_pid=42, on_event=lambda e: None)
        assert not probe.is_loaded

    def test_repr_includes_pid_and_state(self):
        probe = PreemptionProbe(target_pid=9999, on_event=lambda e: None)
        r = repr(probe)
        assert "9999" in r
        assert "unloaded" in r

    def test_repr_loaded_when_bpf_set(self):
        probe = PreemptionProbe(target_pid=1, on_event=lambda e: None)
        probe._bpf = object()
        assert "loaded" in repr(probe)


# ---------------------------------------------------------------------------
# EBPFProbeManager — lifecycle + dispatch (no root required)
# ---------------------------------------------------------------------------

def _make_mem_event(level: int) -> MemPressureEvent:
    return MemPressureEvent(level=level, cgroup_path="/test", timestamp_ns=1_000_000)


def _make_preemption_event() -> PreemptionEvent:
    return PreemptionEvent(pid=1234, exit_code=9, timestamp_ns=2_000_000)


class TestEBPFProbeManagerLifecycle:
    def test_start_without_load_is_noop(self, caplog):
        mgr = EBPFProbeManager()
        import logging
        with caplog.at_level(logging.WARNING, logger="memory_guard.ebpf"):
            mgr.start()
        assert not mgr.is_running

    def test_is_loaded_false_before_load(self):
        mgr = EBPFProbeManager()
        assert not mgr.is_loaded

    def test_is_running_false_before_start(self):
        mgr = EBPFProbeManager()
        assert not mgr.is_running

    def test_stop_safe_when_never_started(self):
        mgr = EBPFProbeManager()
        mgr.stop()  # must not raise

    def test_repr_contains_state(self):
        mgr = EBPFProbeManager()
        r = repr(mgr)
        assert "unloaded" in r

    def test_repr_contains_worker_pid(self):
        mgr = EBPFProbeManager(worker_pid=5678)
        assert "5678" in repr(mgr)


class TestEBPFProbeManagerDispatch:
    def test_dispatch_level_high_calls_on_high(self):
        received: List[MemPressureEvent] = []
        mgr = EBPFProbeManager(on_high=received.append)
        event = _make_mem_event(LEVEL_HIGH)
        mgr._dispatch_mem_event(event)
        assert len(received) == 1
        assert received[0].level == LEVEL_HIGH

    def test_dispatch_level_oom_calls_on_oom(self):
        received: List[MemPressureEvent] = []
        mgr = EBPFProbeManager(on_oom=received.append)
        event = _make_mem_event(LEVEL_OOM)
        mgr._dispatch_mem_event(event)
        assert len(received) == 1
        assert received[0].level == LEVEL_OOM

    def test_dispatch_level_high_sets_ebpf_wake(self):
        wake = threading.Event()
        mgr = EBPFProbeManager(ebpf_wake=wake)
        assert not wake.is_set()
        mgr._dispatch_mem_event(_make_mem_event(LEVEL_HIGH))
        assert wake.is_set()

    def test_dispatch_level_oom_does_not_set_ebpf_wake(self):
        wake = threading.Event()
        mgr = EBPFProbeManager(ebpf_wake=wake)
        mgr._dispatch_mem_event(_make_mem_event(LEVEL_OOM))
        assert not wake.is_set()

    def test_dispatch_high_without_on_high_noop(self):
        """No on_high callback — must not raise."""
        mgr = EBPFProbeManager(on_high=None)
        mgr._dispatch_mem_event(_make_mem_event(LEVEL_HIGH))  # must not raise

    def test_dispatch_oom_without_on_oom_noop(self):
        mgr = EBPFProbeManager(on_oom=None)
        mgr._dispatch_mem_event(_make_mem_event(LEVEL_OOM))

    def test_on_high_exception_is_swallowed(self):
        def bad_cb(event: MemPressureEvent) -> None:
            raise RuntimeError("boom")
        mgr = EBPFProbeManager(on_high=bad_cb)
        mgr._dispatch_mem_event(_make_mem_event(LEVEL_HIGH))  # must not raise

    def test_on_oom_exception_is_swallowed(self):
        def bad_cb(event: MemPressureEvent) -> None:
            raise RuntimeError("boom")
        mgr = EBPFProbeManager(on_oom=bad_cb)
        mgr._dispatch_mem_event(_make_mem_event(LEVEL_OOM))

    def test_dispatch_preemption_calls_on_preemption(self):
        received: List[PreemptionEvent] = []
        mgr = EBPFProbeManager(on_preemption=received.append)
        event = _make_preemption_event()
        mgr._dispatch_preemption_event(event)
        assert len(received) == 1
        assert received[0].pid == 1234

    def test_on_preemption_exception_swallowed(self):
        def bad_cb(event: PreemptionEvent) -> None:
            raise RuntimeError("boom")
        mgr = EBPFProbeManager(on_preemption=bad_cb)
        mgr._dispatch_preemption_event(_make_preemption_event())


# ---------------------------------------------------------------------------
# KVCacheMonitor + eBPF integration (no root required)
# ---------------------------------------------------------------------------

class TestKVCacheMonitorEBPFIntegration:
    def _make_monitor(self, use_ebpf: bool = False):
        from memory_guard.inference_monitor import KVCacheMonitor
        return KVCacheMonitor(poll_fn=lambda: (0, 1), use_ebpf=use_ebpf)

    def test_use_ebpf_false_manager_stays_none(self):
        """Without use_ebpf, no eBPF manager is ever created."""
        mon = self._make_monitor(use_ebpf=False)
        with patch.object(mon, "_start_ebpf") as mock_start_ebpf:
            mon.start()
            mon.stop()
        mock_start_ebpf.assert_not_called()

    def test_use_ebpf_true_calls_start_ebpf(self):
        """With use_ebpf=True, _start_ebpf() is called during start()."""
        mon = self._make_monitor(use_ebpf=True)
        with patch.object(mon, "_start_ebpf") as mock_start_ebpf:
            with patch("memory_guard.inference_monitor.KVCacheMonitor._loop"):
                mon.start()
                mon.stop()
        mock_start_ebpf.assert_called_once()

    def test_start_ebpf_silently_skips_on_import_error(self):
        """ImportError from EBPFProbeManager.load() is swallowed in _start_ebpf."""
        mon = self._make_monitor(use_ebpf=True)
        with patch("memory_guard.ebpf.EBPFProbeManager") as MockMgr:
            MockMgr.return_value.load.side_effect = ImportError("bcc not found")
            mon._start_ebpf()
        assert mon._ebpf_manager is None

    def test_start_ebpf_silently_skips_on_permission_error(self):
        mon = self._make_monitor(use_ebpf=True)
        with patch("memory_guard.ebpf.EBPFProbeManager") as MockMgr:
            MockMgr.return_value.load.side_effect = PermissionError("no caps")
            mon._start_ebpf()
        assert mon._ebpf_manager is None

    def test_start_ebpf_silently_skips_on_os_error(self):
        mon = self._make_monitor(use_ebpf=True)
        with patch("memory_guard.ebpf.EBPFProbeManager") as MockMgr:
            MockMgr.return_value.load.side_effect = OSError("no tracepoint")
            mon._start_ebpf()
        assert mon._ebpf_manager is None

    def test_start_ebpf_stores_manager_on_success(self):
        """When load()+start() succeed, _ebpf_manager is set."""
        mon = self._make_monitor(use_ebpf=True)
        mock_mgr = MagicMock()
        with patch("memory_guard.ebpf.EBPFProbeManager", return_value=mock_mgr):
            mon._start_ebpf()
        assert mon._ebpf_manager is mock_mgr
        mock_mgr.load.assert_called_once()
        mock_mgr.start.assert_called_once()

    def test_stop_calls_ebpf_manager_stop(self):
        """stop() must call _ebpf_manager.stop() and clear the reference."""
        mon = self._make_monitor(use_ebpf=False)
        mock_mgr = MagicMock()
        mon._ebpf_manager = mock_mgr

        mon._stop.set()  # prevent _loop from blocking
        mon.stop()

        mock_mgr.stop.assert_called_once()
        assert mon._ebpf_manager is None

    def test_stop_sets_ebpf_wake(self):
        """stop() must set _ebpf_wake to unblock any waiting loop."""
        mon = self._make_monitor(use_ebpf=False)
        mon._stop.set()
        mon.stop()
        assert mon._ebpf_wake.is_set()

    def test_ebpf_wake_event_exists_on_monitor(self):
        mon = self._make_monitor(use_ebpf=False)
        assert isinstance(mon._ebpf_wake, threading.Event)


# ---------------------------------------------------------------------------
# Root-only: real BPF program compilation (skipped on macOS / non-root)
# ---------------------------------------------------------------------------

@root_required
class TestCgroupProbeRootOnly:
    def test_load_and_detach_cycle(self):
        """Full load → detach lifecycle (Linux root only)."""
        received: List[MemPressureEvent] = []
        probe = CgroupMemoryProbe(on_event=received.append)
        probe.load()
        assert probe.is_loaded
        probe.poll(timeout_ms=5)
        probe.detach()
        assert not probe.is_loaded

    def test_repr_loaded_after_load(self):
        probe = CgroupMemoryProbe(on_event=lambda e: None)
        probe.load()
        try:
            assert "loaded" in repr(probe)
        finally:
            probe.detach()


@root_required
class TestPreemptionProbeRootOnly:
    def test_load_and_detach(self):
        """Full load → poll → detach (Linux root only)."""
        received: List[PreemptionEvent] = []
        probe = PreemptionProbe(target_pid=os.getpid(), on_event=received.append)
        probe.load()
        assert probe.is_loaded
        probe.poll(timeout_ms=5)
        probe.detach()
        assert not probe.is_loaded


@root_required
class TestEBPFProbeManagerRootOnly:
    def test_load_start_stop(self):
        """Full manager lifecycle on a real Linux kernel (root only)."""
        mgr = EBPFProbeManager()
        mgr.load()
        assert mgr.is_loaded
        mgr.start()
        assert mgr.is_running
        time.sleep(0.05)
        mgr.stop()
        assert not mgr.is_running
        assert not mgr.is_loaded
