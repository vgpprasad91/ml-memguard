"""HuggingFace Transformers adapter for memory-guard.

Provides:
    MemoryGuardCallback   — TrainerCallback that starts/stops the memory monitor
    guard_trainer         — one-call setup: introspect → preflight → patch args → add callback

The module is safe to import on a torch-free machine.  ``transformers`` is
resolved lazily via a try/except at class-definition time; if absent the class
falls back to ``object`` as its base so that duck-typing still works.  The
heavy side of things (``guard.monitor``, ``guard.record_result``) is only
executed when the callbacks actually fire.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..guard import MemoryGuard, SafeConfig
from .base import introspect_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy base class — TrainerCallback when available, plain object otherwise.
# This lets the module be imported without transformers installed.
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:
    _TrainerCallbackBase = object  # type: ignore[assignment,misc]


class MemoryGuardCallback(_TrainerCallbackBase):  # type: ignore[misc]
    """HuggingFace ``TrainerCallback`` that monitors memory during training.

    Lifecycle:
        ``on_train_begin`` → starts ``guard.monitor(per_device_train_batch_size)``
        ``on_log``         → surfaces sustained pressure warnings through Trainer's logger
        ``on_train_end``   → stops the monitor; calls ``guard.record_result()``

    Batch size is **not** mutated mid-training (that is PR 3).

    Usage via :func:`guard_trainer` (recommended)::

        safe = guard_trainer(trainer, guard=guard)

    Or manually::

        trainer.add_callback(MemoryGuardCallback(guard=guard))
    """

    def __init__(self, guard: Any) -> None:
        self._guard = guard
        self._monitor_session: Any = None  # _MonitorSession context manager
        self._monitor: Any = None          # RuntimeMonitor (after __enter__)

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Start the background memory monitor."""
        batch_size: int = getattr(args, "per_device_train_batch_size", 1)
        self._monitor_session = self._guard.monitor(batch_size=batch_size)
        self._monitor = self._monitor_session.__enter__()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Surface sustained memory pressure through Trainer's logger."""
        if self._monitor is None:
            return
        history = getattr(self._monitor, "pressure_history", [])
        if not history:
            return
        latest = history[-1]
        from ..monitor import RuntimeMonitor
        if latest >= RuntimeMonitor.THRESHOLD_WARNING:
            logger.warning(
                "[memory-guard] Memory pressure at %.0f%% — "
                "consider reducing batch size or enabling gradient checkpointing.",
                latest * 100,
            )

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Stop the monitor and record peak memory for auto-calibration."""
        if self._monitor_session is not None:
            try:
                self._monitor_session.__exit__(None, None, None)
            except Exception:
                logger.debug("Monitor stop raised during on_train_end", exc_info=True)
            finally:
                self._monitor_session = None
                self._monitor = None

        # guard.record_result() auto-detects torch.cuda.max_memory_allocated()
        # and MLX peak memory; pass nothing and let it handle both backends.
        try:
            self._guard.record_result()
        except Exception:
            logger.debug("record_result raised in on_train_end", exc_info=True)


# ---------------------------------------------------------------------------
# guard_trainer — one-call setup
# ---------------------------------------------------------------------------

def guard_trainer(
    trainer: Any,
    guard: Optional[Any] = None,
    **preflight_overrides: Any,
) -> SafeConfig:
    """Attach memory-guard to a HuggingFace ``Trainer`` in one call.

    Steps:

    1. Introspects ``trainer.model`` to read architecture metadata.
    2. Runs ``guard.preflight()`` (auto-downgrades if the config doesn't fit).
    3. Writes the safe config back into ``trainer.args``:

       * ``per_device_train_batch_size``
       * ``gradient_accumulation_steps``
       * ``gradient_checkpointing``

    4. Appends a :class:`MemoryGuardCallback` to ``trainer.callback_handler``.

    Args:
        trainer: A ``transformers.Trainer`` instance (or any object with
            ``.model``, ``.args``, and ``.callback_handler.callbacks``).
        guard: A :class:`~memory_guard.MemoryGuard` instance.  ``None`` triggers
            ``MemoryGuard.auto()``.
        **preflight_overrides: Forwarded verbatim to ``guard.preflight()``
            (e.g. ``batch_size``, ``seq_length``, ``lora_rank``).

    Returns:
        The :class:`~memory_guard.SafeConfig` produced by preflight so callers
        can inspect what was changed.
    """
    if guard is None:
        guard = MemoryGuard.auto()

    model_info = introspect_model(trainer.model)

    preflight_kwargs: dict = dict(
        model_params=model_info["num_parameters"],
        model_bits=model_info["model_bits"],
        hidden_dim=model_info["hidden_size"],
        num_heads=model_info["num_attention_heads"],
        num_layers=model_info["num_hidden_layers"],
    )
    preflight_kwargs.update(preflight_overrides)

    safe = guard.preflight(**preflight_kwargs)

    # Write safe config back into trainer.args
    trainer.args.per_device_train_batch_size = safe.batch_size
    trainer.args.gradient_accumulation_steps = safe.grad_accumulation
    trainer.args.gradient_checkpointing = safe.grad_checkpoint

    # Append callback
    callback = MemoryGuardCallback(guard=guard)
    trainer.callback_handler.callbacks.append(callback)

    logger.info(
        "[memory-guard] guard_trainer: batch_size=%d, grad_accum=%d, grad_ckpt=%s",
        safe.batch_size,
        safe.grad_accumulation,
        safe.grad_checkpoint,
    )

    return safe
