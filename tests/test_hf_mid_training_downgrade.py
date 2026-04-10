"""Tests for PR 3 — mid-training batch-size downgrade logic.

Covers the two-phase approach:
  on_step_begin  — detects monitor drop, records pending, does NOT mutate args
  on_epoch_begin — applies pending downgrade, scales gradient_accumulation_steps

All tests are self-contained and use SimpleNamespace / MagicMock to avoid any
dependency on transformers or torch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.adapters.huggingface import MemoryGuardCallback


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


def _args(batch_size: int = 8, grad_accum: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
    )


def _state(step: int = 0, epoch: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(global_step=step, epoch=epoch)


def _control() -> SimpleNamespace:
    return SimpleNamespace(should_log=False)


def _make_monitor(current_batch_size: int = 8, pressure: float = 0.50) -> MagicMock:
    m = MagicMock()
    m.current_batch_size = current_batch_size
    m.pressure_history = [pressure]
    return m


def _make_guard(monitor: MagicMock) -> MagicMock:
    """Build a guard whose monitor() context manager yields *monitor*."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=monitor)
    session.__exit__ = MagicMock(return_value=False)

    guard = MagicMock()
    guard.monitor.return_value = session
    return guard


def _started_callback(monitor: MagicMock, *, batch_size: int = 8) -> MemoryGuardCallback:
    """Return a MemoryGuardCallback that has already had on_train_begin called."""
    guard = _make_guard(monitor)
    cb = MemoryGuardCallback(guard=guard)
    cb.on_train_begin(_args(batch_size), _state(), _control())
    # Point _monitor directly at the mock so tests can mutate current_batch_size
    cb._monitor = monitor
    return cb


# ---------------------------------------------------------------------------
# on_step_begin — detection, pending state, no immediate mutation
# ---------------------------------------------------------------------------


class TestOnStepBeginDetection:
    def test_noop_when_monitor_matches_current_batch_size(self):
        mon = _make_monitor(current_batch_size=8)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        cb.on_step_begin(args, _state(), _control())

        assert cb._pending_batch_size is None
        assert args.per_device_train_batch_size == 8  # unchanged

    def test_noop_when_monitor_above_current_batch_size(self):
        """Monitor returning a higher value must not trigger a pending downgrade."""
        mon = _make_monitor(current_batch_size=16)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        cb.on_step_begin(args, _state(), _control())

        assert cb._pending_batch_size is None

    def test_records_pending_when_monitor_drops(self):
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        cb.on_step_begin(args, _state(), _control())

        assert cb._pending_batch_size == 4

    def test_does_not_mutate_args_immediately(self):
        """The actual batch_size change must NOT happen during on_step_begin."""
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        cb.on_step_begin(args, _state(), _control())

        assert args.per_device_train_batch_size == 8  # untouched
        assert args.gradient_accumulation_steps == 1  # untouched

    def test_sets_control_should_log_on_first_detection(self):
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)
        ctrl = _control()

        cb.on_step_begin(args, _state(), ctrl)

        assert ctrl.should_log is True

    def test_does_not_set_should_log_on_repeat_steps(self):
        """should_log is only set on the first detection, not every step."""
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        # First step — sets pending
        ctrl1 = _control()
        cb.on_step_begin(args, _state(step=1), ctrl1)
        assert ctrl1.should_log is True

        # Second step — pending already set, should_log must NOT be re-set
        ctrl2 = _control()
        ctrl2.should_log = False  # explicitly False
        cb.on_step_begin(args, _state(step=2), ctrl2)
        assert ctrl2.should_log is False

    def test_takes_minimum_on_further_drop(self):
        """If the monitor drops again before the epoch boundary, take the lower value."""
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        cb.on_step_begin(args, _state(step=1), _control())
        assert cb._pending_batch_size == 4

        mon.current_batch_size = 2  # monitor drops further
        cb.on_step_begin(args, _state(step=2), _control())
        assert cb._pending_batch_size == 2  # updated to minimum

    def test_noop_when_monitor_is_none(self):
        """Calling on_step_begin before on_train_begin must not raise."""
        guard = MagicMock()
        cb = MemoryGuardCallback(guard=guard)  # no on_train_begin

        cb.on_step_begin(_args(), _state(), _control())  # must not raise
        assert cb._pending_batch_size is None

    def test_logs_warning_on_first_detection(self):
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)

        with patch("memory_guard.adapters.huggingface.logger") as mock_log:
            cb.on_step_begin(_args(batch_size=8), _state(), _control())
            mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# on_epoch_begin — applies pending downgrade atomically
# ---------------------------------------------------------------------------


class TestOnEpochBeginApply:
    def test_applies_batch_size_at_epoch_boundary(self):
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8, grad_accum=1)

        cb.on_step_begin(args, _state(), _control())      # records pending=4
        cb.on_epoch_begin(args, _state(epoch=2.0), _control())

        assert args.per_device_train_batch_size == 4

    def test_scales_gradient_accumulation_by_ratio(self):
        """8→4 halves batch, so grad_accum must double to preserve effective batch."""
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8, grad_accum=2)

        cb.on_step_begin(args, _state(), _control())
        cb.on_epoch_begin(args, _state(epoch=2.0), _control())

        assert args.gradient_accumulation_steps == 4  # 2 × (8//4) = 4

    def test_preserves_effective_batch_size(self):
        """effective_batch = per_device_batch × grad_accum must not change."""
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8, grad_accum=2)
        effective_before = args.per_device_train_batch_size * args.gradient_accumulation_steps

        cb.on_step_begin(args, _state(), _control())
        cb.on_epoch_begin(args, _state(epoch=2.0), _control())

        effective_after = args.per_device_train_batch_size * args.gradient_accumulation_steps
        assert effective_after == effective_before

    def test_clears_pending_after_applying(self):
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        cb.on_step_begin(args, _state(), _control())
        assert cb._pending_batch_size is not None

        cb.on_epoch_begin(args, _state(epoch=2.0), _control())
        assert cb._pending_batch_size is None

    def test_noop_when_no_pending(self):
        mon = _make_monitor(current_batch_size=8)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8, grad_accum=1)

        cb.on_epoch_begin(args, _state(epoch=2.0), _control())

        assert args.per_device_train_batch_size == 8
        assert args.gradient_accumulation_steps == 1

    def test_skips_when_args_already_at_target(self):
        """If something else already lowered args.per_device_train_batch_size, skip."""
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8, grad_accum=1)

        cb.on_step_begin(args, _state(), _control())
        args.per_device_train_batch_size = 4  # externally updated before epoch boundary

        cb.on_epoch_begin(args, _state(epoch=2.0), _control())

        assert args.gradient_accumulation_steps == 1  # must not double-scale

    def test_4x_downgrade_scales_grad_accum_4x(self):
        """8→2 is a 4× ratio, so grad_accum scales ×4."""
        mon = _make_monitor(current_batch_size=2)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8, grad_accum=1)

        cb.on_step_begin(args, _state(), _control())
        cb.on_epoch_begin(args, _state(epoch=2.0), _control())

        assert args.per_device_train_batch_size == 2
        assert args.gradient_accumulation_steps == 4

    def test_logs_applied_downgrade(self):
        mon = _make_monitor(current_batch_size=4)
        cb = _started_callback(mon, batch_size=8)
        args = _args(batch_size=8)

        cb.on_step_begin(args, _state(), _control())
        with patch("memory_guard.adapters.huggingface.logger") as mock_log:
            cb.on_epoch_begin(args, _state(epoch=2.0), _control())
            mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# Full lifecycle — drive a simulated two-epoch training run
# ---------------------------------------------------------------------------


class TestFullLifecycleTwoEpochs:
    def test_downgrade_applied_at_epoch_boundary(self):
        """Simulate: epoch 1 steps trigger drop, epoch 2 begin applies it."""
        mon = _make_monitor(current_batch_size=8)
        guard = _make_guard(mon)
        cb = MemoryGuardCallback(guard=guard)
        cb._monitor = mon

        args = _args(batch_size=8, grad_accum=1)
        state = _state(step=0, epoch=1.0)
        ctrl = _control()

        # --- on_train_begin ---
        cb.on_train_begin(args, state, ctrl)
        cb._monitor = mon  # re-pin after on_train_begin resets it

        # --- epoch 1: step 5 — monitor drops to 4 ---
        state.global_step = 5
        mon.current_batch_size = 4
        cb.on_step_begin(args, state, ctrl)

        # Args must be unchanged after step
        assert args.per_device_train_batch_size == 8
        assert args.gradient_accumulation_steps == 1
        assert cb._pending_batch_size == 4

        # --- epoch 2 begins — apply pending ---
        state.epoch = 2.0
        cb.on_epoch_begin(args, state, ctrl)

        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 2  # 1 × (8//4)
        assert cb._pending_batch_size is None

        # --- epoch 2: steps — monitor stays at 4, no further pending ---
        mon.current_batch_size = 4  # matches new current
        cb.on_step_begin(args, state, ctrl)
        assert cb._pending_batch_size is None

    def test_two_successive_drops_take_minimum(self):
        """Monitor drops 8→4 at step 2, then 4→2 at step 5; epoch applies 8→2."""
        mon = _make_monitor(current_batch_size=8)
        guard = _make_guard(mon)
        cb = MemoryGuardCallback(guard=guard)
        cb._monitor = mon

        args = _args(batch_size=8, grad_accum=1)

        cb.on_train_begin(args, _state(), _control())
        cb._monitor = mon

        # Step 2: drop to 4
        mon.current_batch_size = 4
        cb.on_step_begin(args, _state(step=2), _control())
        assert cb._pending_batch_size == 4

        # Step 5: monitor drops further to 2
        mon.current_batch_size = 2
        cb.on_step_begin(args, _state(step=5), _control())
        assert cb._pending_batch_size == 2  # minimum wins

        # Epoch boundary applies 8→2 (ratio=4, grad_accum ×4)
        cb.on_epoch_begin(args, _state(epoch=2.0), _control())
        assert args.per_device_train_batch_size == 2
        assert args.gradient_accumulation_steps == 4
        assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 8

    def test_on_train_end_clears_pending(self):
        mon = _make_monitor(current_batch_size=4)
        guard = _make_guard(mon)
        cb = MemoryGuardCallback(guard=guard)
        cb._monitor = mon

        args = _args(batch_size=8)
        cb.on_train_begin(args, _state(), _control())
        cb._monitor = mon
        cb.on_step_begin(args, _state(), _control())
        assert cb._pending_batch_size is not None

        cb.on_train_end(args, _state(), _control())
        assert cb._pending_batch_size is None

    def test_on_train_begin_resets_pending_for_reuse(self):
        """Reusing a callback across Trainer runs must not carry stale state."""
        mon = _make_monitor(current_batch_size=4)
        guard = _make_guard(mon)
        cb = MemoryGuardCallback(guard=guard)
        cb._monitor = mon
        cb._pending_batch_size = 2  # stale leftover

        args = _args(batch_size=8)
        cb.on_train_begin(args, _state(), _control())

        assert cb._pending_batch_size is None
