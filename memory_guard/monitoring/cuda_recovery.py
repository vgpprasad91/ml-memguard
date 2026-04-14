"""CUDA OOM recovery — catch OutOfMemoryError and retry with smaller batch.

Similar to HuggingFace accelerate's auto_find_batch_size but as a
standalone, framework-agnostic utility.

Only active on CUDA devices. On Apple Silicon, use the proactive
estimator + runtime monitor instead (no OOM exceptions on Metal).
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CUDAOOMRecovery:
    """Wrap training steps with automatic CUDA OOM recovery.

    Catches torch.cuda.OutOfMemoryError, halves batch size, clears
    cache, and retries. Tracks OOM count for diagnostics.

    Usage:
        recovery = CUDAOOMRecovery(initial_batch_size=8)

        for epoch in range(num_epochs):
            for batch in dataloader:
                result = recovery.step(
                    train_fn, batch,
                    batch_size=recovery.current_batch_size,
                )

        print(f"Final batch size: {recovery.current_batch_size}")
        print(f"OOM events: {recovery.oom_count}")
    """

    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_retries: int = 5,
        on_oom: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Args:
            initial_batch_size: Starting batch size.
            min_batch_size: Will not go below this.
            max_retries: Max consecutive OOM retries per step.
            on_oom: Callback(old_batch_size, new_batch_size) on each OOM.
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_retries = max_retries
        self.on_oom = on_oom
        self.oom_count = 0
        self._torch = None

    def _get_torch(self):
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                raise RuntimeError(
                    "CUDAOOMRecovery requires PyTorch. "
                    "Install with: pip install torch"
                )
        return self._torch

    def step(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute fn with OOM recovery.

        If fn raises CUDA OOM, halve batch_size, clear cache, retry.
        The `batch_size` kwarg is automatically set to current_batch_size.

        Returns:
            Whatever fn returns on success.

        Raises:
            RuntimeError: If batch_size reaches min and still OOM.
        """
        torch = self._get_torch()

        for attempt in range(self.max_retries):
            try:
                call_kwargs = {**kwargs, "batch_size": self.current_batch_size}
                return fn(*args, **call_kwargs)

            except torch.cuda.OutOfMemoryError:
                self.oom_count += 1
                old_bs = self.current_batch_size
                self.current_batch_size = max(
                    self.min_batch_size, self.current_batch_size // 2
                )

                logger.warning(
                    f"CUDA OOM #{self.oom_count}: "
                    f"batch_size {old_bs} -> {self.current_batch_size}"
                )

                if self.on_oom:
                    self.on_oom(old_bs, self.current_batch_size)

                # Clear CUDA cache
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()

                # If we couldn't reduce (already at min), raise.
                # But if we just halved TO min, allow one retry at min.
                if old_bs == self.current_batch_size:
                    raise RuntimeError(
                        f"CUDA OOM at min batch_size={self.min_batch_size}. "
                        f"Model may be too large for this GPU. "
                        f"Try: smaller model, quantization, or gradient checkpointing."
                    )

        raise RuntimeError(f"Exhausted {self.max_retries} OOM retries")

    def find_max_batch_size(
        self,
        probe_fn: Callable,
        start: int = 1,
        max_batch: int = 128,
        steps_per_probe: int = 3,
    ) -> int:
        """Binary search for maximum batch size (like Lightning BatchSizeFinder).

        Args:
            probe_fn: Function(batch_size) that runs a training step.
            start: Starting batch size for search.
            max_batch: Upper bound.
            steps_per_probe: Steps to run at each size (more = more reliable).

        Returns:
            Maximum batch size that doesn't OOM.
        """
        torch = self._get_torch()

        # Phase 1: Power scaling — double until OOM
        bs = start
        last_good = start
        oom_at = None  # The batch size that OOM'd

        while bs <= max_batch:
            torch.cuda.empty_cache()
            try:
                for _ in range(steps_per_probe):
                    probe_fn(batch_size=bs)
                last_good = bs
                bs *= 2
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_at = bs
                break

        # If no OOM occurred (all sizes fit) or OOM was on the first doubling
        if oom_at is None or last_good >= max_batch:
            return min(last_good, max_batch)

        # Phase 2: Binary search between last_good and oom_at
        lo, hi = last_good, min(oom_at, max_batch)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            torch.cuda.empty_cache()
            try:
                for _ in range(steps_per_probe):
                    probe_fn(batch_size=mid)
                lo = mid
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hi = mid

        return lo
