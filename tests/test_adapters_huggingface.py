"""Tests for memory_guard.adapters.huggingface.

Covers:
    - guard_trainer: introspects model, runs preflight, mutates trainer.args,
      appends MemoryGuardCallback to callback_handler
    - MemoryGuardCallback lifecycle: on_train_begin starts monitor,
      on_train_end stops monitor and calls record_result, on_log warns on
      high pressure
    - Lazy __getattr__ in memory_guard.__init__ exposes both symbols without
      requiring torch/transformers at import time
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.adapters.huggingface import MemoryGuardCallback, guard_trainer
from memory_guard.monitoring.monitor import RuntimeMonitor


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

_ARGS = SimpleNamespace(per_device_train_batch_size=4)
_STATE = SimpleNamespace()
_CONTROL = SimpleNamespace()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_safe_config(
    batch_size: int = 4,
    grad_accumulation: int = 1,
    grad_checkpoint: bool = False,
    seq_length: int = 2048,
    lora_rank: int = 8,
    lora_layers: int = 16,
    fits: bool = True,
) -> object:
    """Build a SafeConfig without touching the real MemoryGuard."""
    from memory_guard.guard import SafeConfig

    estimate = MagicMock()
    estimate.total_mb = 10_000.0
    return SafeConfig(
        batch_size=batch_size,
        seq_length=seq_length,
        lora_rank=lora_rank,
        lora_layers=lora_layers,
        grad_checkpoint=grad_checkpoint,
        grad_accumulation=grad_accumulation,
        estimate=estimate,
        budget_mb=12_000.0,
        available_mb=16_000.0,
        changes=[],
        fits=fits,
    )


def _make_mock_guard(safe_config=None):
    """Return (mock_guard, safe_config).

    The guard's monitor() returns a proper context-manager stub whose
    __enter__ yields a mock RuntimeMonitor with an empty pressure_history.
    """
    if safe_config is None:
        safe_config = _make_safe_config()

    mock_monitor = MagicMock()
    mock_monitor.pressure_history = []

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_monitor)
    mock_session.__exit__ = MagicMock(return_value=False)

    guard = MagicMock()
    guard.preflight.return_value = safe_config
    guard.monitor.return_value = mock_session

    return guard, safe_config


def _make_mock_trainer(batch_size: int = 8):
    """Build a minimal Trainer-shaped namespace usable by guard_trainer."""
    config = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        quantization_config=None,
    )
    param = MagicMock()
    param.numel.return_value = 7_000_000_000

    model = MagicMock()
    model.config = config
    model.dtype = "torch.float16"
    model.parameters.return_value = [param]

    args = SimpleNamespace(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
    )
    callback_handler = SimpleNamespace(callbacks=[])

    return SimpleNamespace(model=model, args=args, callback_handler=callback_handler)


# ---------------------------------------------------------------------------
# guard_trainer tests
# ---------------------------------------------------------------------------


class TestGuardTrainer:
    def test_callback_appended_to_handler(self):
        trainer = _make_mock_trainer()
        guard, _ = _make_mock_guard()
        guard_trainer(trainer, guard=guard)

        assert len(trainer.callback_handler.callbacks) == 1
        assert isinstance(trainer.callback_handler.callbacks[0], MemoryGuardCallback)

    def test_args_batch_size_written(self):
        safe = _make_safe_config(batch_size=2)
        trainer = _make_mock_trainer(batch_size=8)
        guard, _ = _make_mock_guard(safe)
        guard_trainer(trainer, guard=guard)

        assert trainer.args.per_device_train_batch_size == 2

    def test_args_gradient_accumulation_written(self):
        safe = _make_safe_config(grad_accumulation=4)
        trainer = _make_mock_trainer()
        guard, _ = _make_mock_guard(safe)
        guard_trainer(trainer, guard=guard)

        assert trainer.args.gradient_accumulation_steps == 4

    def test_args_gradient_checkpointing_written(self):
        safe = _make_safe_config(grad_checkpoint=True)
        trainer = _make_mock_trainer()
        guard, _ = _make_mock_guard(safe)
        guard_trainer(trainer, guard=guard)

        assert trainer.args.gradient_checkpointing is True

    def test_returns_safe_config(self):
        trainer = _make_mock_trainer()
        guard, safe = _make_mock_guard()
        result = guard_trainer(trainer, guard=guard)

        assert result is safe

    def test_preflight_receives_introspected_values(self):
        trainer = _make_mock_trainer()
        guard, _ = _make_mock_guard()
        guard_trainer(trainer, guard=guard)

        guard.preflight.assert_called_once()
        kw = guard.preflight.call_args[1]
        assert kw["model_params"] == 7_000_000_000
        assert kw["model_bits"] == 16   # float16 → 16 bits
        assert kw["hidden_dim"] == 4096
        assert kw["num_heads"] == 32
        assert kw["num_layers"] == 32

    def test_preflight_overrides_forwarded(self):
        trainer = _make_mock_trainer()
        guard, _ = _make_mock_guard()
        guard_trainer(trainer, guard=guard, batch_size=1, lora_rank=64)

        kw = guard.preflight.call_args[1]
        assert kw["batch_size"] == 1
        assert kw["lora_rank"] == 64

    def test_auto_creates_guard_when_none(self):
        """guard_trainer instantiates MemoryGuard.auto() when guard=None."""
        trainer = _make_mock_trainer()
        mock_guard_instance = MagicMock()
        mock_guard_instance.preflight.return_value = _make_safe_config()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_guard_instance.monitor.return_value = mock_session

        with patch(
            "memory_guard.adapters.huggingface.MemoryGuard"
        ) as MockGuard:
            MockGuard.auto.return_value = mock_guard_instance
            guard_trainer(trainer, guard=None)

        MockGuard.auto.assert_called_once()


# ---------------------------------------------------------------------------
# MemoryGuardCallback lifecycle tests
# ---------------------------------------------------------------------------


class TestMemoryGuardCallbackLifecycle:
    def test_on_train_begin_starts_monitor_with_batch_size(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_train_begin(_ARGS, _STATE, _CONTROL)

        guard.monitor.assert_called_once_with(batch_size=4)

    def test_on_train_begin_enters_monitor_context(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_train_begin(_ARGS, _STATE, _CONTROL)

        guard.monitor.return_value.__enter__.assert_called_once()

    def test_on_train_end_exits_monitor_session(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_train_begin(_ARGS, _STATE, _CONTROL)
        cb.on_train_end(_ARGS, _STATE, _CONTROL)

        guard.monitor.return_value.__exit__.assert_called_once_with(None, None, None)

    def test_on_train_end_calls_record_result(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_train_begin(_ARGS, _STATE, _CONTROL)
        cb.on_train_end(_ARGS, _STATE, _CONTROL)

        guard.record_result.assert_called_once()

    def test_on_train_end_without_begin_does_not_raise(self):
        """on_train_end must be safe even if called before on_train_begin."""
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_train_end(_ARGS, _STATE, _CONTROL)  # no prior on_train_begin

        guard.record_result.assert_called_once()

    def test_on_train_end_clears_monitor_references(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_train_begin(_ARGS, _STATE, _CONTROL)
        assert cb._monitor is not None

        cb.on_train_end(_ARGS, _STATE, _CONTROL)
        assert cb._monitor is None
        assert cb._monitor_session is None

    def test_on_log_no_crash_when_monitor_not_started(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_log(_ARGS, _STATE, _CONTROL, logs={})  # must not raise

    def test_on_log_warns_when_pressure_above_threshold(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)
        cb.on_train_begin(_ARGS, _STATE, _CONTROL)

        # Inject pressure above the WARNING threshold
        cb._monitor.pressure_history = [RuntimeMonitor.THRESHOLD_WARNING + 0.05]

        with patch("memory_guard.adapters.huggingface.logger") as mock_logger:
            cb.on_log(_ARGS, _STATE, _CONTROL, logs={})
            mock_logger.warning.assert_called_once()

    def test_on_log_no_warning_when_pressure_below_threshold(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)
        cb.on_train_begin(_ARGS, _STATE, _CONTROL)

        # Inject pressure well below WARNING threshold
        cb._monitor.pressure_history = [RuntimeMonitor.THRESHOLD_WARNING * 0.5]

        with patch("memory_guard.adapters.huggingface.logger") as mock_logger:
            cb.on_log(_ARGS, _STATE, _CONTROL, logs={})
            mock_logger.warning.assert_not_called()

    def test_on_log_no_warning_when_pressure_history_empty(self):
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)
        cb.on_train_begin(_ARGS, _STATE, _CONTROL)

        cb._monitor.pressure_history = []

        with patch("memory_guard.adapters.huggingface.logger") as mock_logger:
            cb.on_log(_ARGS, _STATE, _CONTROL, logs={})
            mock_logger.warning.assert_not_called()

    def test_on_train_end_idempotent(self):
        """Calling on_train_end twice must not raise or double-record."""
        guard, _ = _make_mock_guard()
        cb = MemoryGuardCallback(guard=guard)

        cb.on_train_begin(_ARGS, _STATE, _CONTROL)
        cb.on_train_end(_ARGS, _STATE, _CONTROL)
        cb.on_train_end(_ARGS, _STATE, _CONTROL)  # second call — must not raise

        assert guard.record_result.call_count == 2  # once per on_train_end


# ---------------------------------------------------------------------------
# Lazy __getattr__ in memory_guard package
# ---------------------------------------------------------------------------


class TestLazyPackageExports:
    def test_package_importable_without_hf_or_torch(self):
        """Core package must load without transformers/torch."""
        import memory_guard
        assert hasattr(memory_guard, "__version__")

    def test_lazy_getattr_resolves_callback_class(self):
        import memory_guard
        assert memory_guard.MemoryGuardCallback is MemoryGuardCallback

    def test_lazy_getattr_resolves_guard_trainer(self):
        import memory_guard
        assert memory_guard.guard_trainer is guard_trainer

    def test_unknown_attr_raises_attribute_error(self):
        import memory_guard
        with pytest.raises(AttributeError):
            _ = memory_guard.does_not_exist_xyz
