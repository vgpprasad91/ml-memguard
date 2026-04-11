"""MemoryGuard — the main entry point.

Combines proactive estimation, auto-downgrade, and runtime monitoring
into a single, easy-to-use API.

    guard = MemoryGuard.auto()
    safe = guard.preflight(model_params=9e9, ...)
    with guard.monitor(safe.batch_size) as mon:
        for step in range(1000):
            train_step(batch_size=mon.current_batch_size)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .bandit import BanditPolicy
    from .bandit_state import ConfigAction, StateKey
    from .inference_monitor import KVCacheMonitor

from .downgrade import DowngradeResult, auto_downgrade
from .estimator import InferenceServingEstimate, MemoryEstimate, estimate_serving_memory, estimate_training_memory
from .monitor import RuntimeMonitor
from .platforms import Backend, PlatformInfo, detect_platform, get_available_memory_mb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate grids for the RL bandit
# ---------------------------------------------------------------------------

#: Batch sizes offered to the bandit as candidates in preflight().
_BANDIT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

#: LoRA ranks offered to the bandit as candidates in preflight().
_BANDIT_LORA_RANKS: tuple[int, ...] = (0, 4, 8, 16, 32, 64)

#: Concurrent sequence counts offered to the bandit in preflight_inference().
_BANDIT_NUM_SEQS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@dataclass
class SafeConfig:
    """Memory-safe training configuration returned by preflight()."""
    batch_size: int
    seq_length: int
    lora_rank: int
    lora_layers: int
    grad_checkpoint: bool
    grad_accumulation: int
    estimate: MemoryEstimate
    budget_mb: float
    available_mb: float
    changes: list[str]
    fits: bool

    def __str__(self):
        status = "FITS" if self.fits else "DOES NOT FIT"
        lines = [
            f"SafeConfig ({status}):",
            f"  batch_size:       {self.batch_size}",
            f"  seq_length:       {self.seq_length}",
            f"  lora_rank:        {self.lora_rank}",
            f"  lora_layers:      {self.lora_layers}",
            f"  grad_checkpoint:  {self.grad_checkpoint}",
            f"  grad_accumulation:{self.grad_accumulation}",
            f"  estimated memory: {self.estimate.total_mb:.0f} MB",
            f"  budget:           {self.budget_mb:.0f} MB",
            f"  available:        {self.available_mb:.0f} MB",
        ]
        if self.changes:
            lines.append(f"  changes applied:  {len(self.changes)}")
            for c in self.changes:
                lines.append(f"    - {c}")
        return "\n".join(lines)


@dataclass
class InferenceSafeConfig:
    """Memory-safe serving configuration returned by preflight_inference().

    Pass max_num_seqs to vLLM's ``--max-num-seqs`` (or SGLang's equivalent)
    and use gpu_memory_utilization as the ``--gpu-memory-utilization`` hint.
    """
    max_num_seqs: int
    max_seq_len: int
    gpu_memory_utilization: float    # Suggested vLLM / SGLang parameter (0–1)
    estimate: InferenceServingEstimate
    budget_mb: float
    available_mb: float
    fits: bool
    changes: list[str]
    monitor: Optional[KVCacheMonitor] = field(
        default=None, compare=False, repr=False
    )

    def __str__(self) -> str:
        status = "FITS" if self.fits else "DOES NOT FIT"
        lines = [
            f"InferenceSafeConfig ({status}):",
            f"  max_num_seqs:            {self.max_num_seqs}",
            f"  max_seq_len:             {self.max_seq_len}",
            f"  gpu_memory_utilization:  {self.gpu_memory_utilization:.2f}",
            f"  estimated memory:        {self.estimate.total_mb:.0f} MB",
            f"  budget:                  {self.budget_mb:.0f} MB",
            f"  available:               {self.available_mb:.0f} MB",
        ]
        if self.changes:
            lines.append(f"  changes applied:  {len(self.changes)}")
            for c in self.changes:
                lines.append(f"    - {c}")
        return "\n".join(lines)


class MemoryGuard:
    """Cross-platform memory guard for ML training.

    Combines three layers of protection:
    1. Proactive estimation (preflight) — before training starts
    2. Auto-downgrade — iteratively reduces config to fit budget
    3. Runtime monitoring — background thread polls memory pressure

    Usage:
        # Auto-detect platform
        guard = MemoryGuard.auto()

        # Or specify
        guard = MemoryGuard(safety_ratio=0.75)

        # Pre-flight check
        safe = guard.preflight(
            model_params=9_000_000_000, model_bits=4,
            hidden_dim=4096, num_heads=32, num_layers=32,
            batch_size=4, seq_length=2048,
            lora_rank=32, lora_layers=16,
        )
        print(safe)  # Shows adjusted config

        # Runtime monitoring
        with guard.monitor(safe.batch_size) as mon:
            for step in range(num_steps):
                actual_bs = mon.current_batch_size  # May decrease mid-training
                train_step(batch_size=actual_bs)
    """

    def __init__(
        self,
        platform_info: Optional[PlatformInfo] = None,
        safety_ratio: float = 0.80,  # See constants.SAFETY_RATIO_DEFAULT
        enable_calibration: bool = True,
        enable_bandit: bool = True,
    ):
        """
        Args:
            platform_info: Detected platform, or None for auto-detect.
            safety_ratio: Use at most this fraction of available memory.
                         0.80 = leave 20% headroom (recommended).
                         0.90 = aggressive, higher risk of pressure.
                         0.70 = conservative, for shared machines.
            enable_calibration: If True, apply learned correction factors
                              from past training runs to improve accuracy.
            enable_bandit: If True, load the RL bandit policy from disk and
                          use it to propose configs in preflight().  Falls back
                          to the existing binary-search path on cold start or
                          exploration.  Disable to reproduce v0.3 behaviour
                          exactly.
        """
        self.platform = platform_info or detect_platform()
        self.safety_ratio = safety_ratio
        self.enable_calibration = enable_calibration
        self.enable_bandit = enable_bandit

        self._calibration_store = None
        if enable_calibration:
            from .calibration import CalibrationStore
            self._calibration_store = CalibrationStore()

        self._last_estimate_mb: Optional[float] = None  # For post-training recording
        self._policy: Optional[BanditPolicy] = None
        self._last_action: Optional[ConfigAction] = None
        self._last_state_key: Optional[StateKey] = None

        if enable_bandit:
            from .bandit import BanditPolicy as _BP
            self._policy = _BP.load()

    @classmethod
    def auto(
        cls,
        safety_ratio: float = 0.80,
        enable_calibration: bool = True,
        enable_bandit: bool = True,
    ) -> "MemoryGuard":
        """Create a MemoryGuard with auto-detected platform."""
        return cls(
            safety_ratio=safety_ratio,
            enable_calibration=enable_calibration,
            enable_bandit=enable_bandit,
        )

    @property
    def available_mb(self) -> float:
        """Currently available memory in MB."""
        return get_available_memory_mb(self.platform.backend)

    @property
    def budget_mb(self) -> float:
        """Memory budget (available × safety_ratio + partial swap credit).

        Swap/compressor headroom is included at 50% credit because swap
        is much slower than RAM and causes performance degradation.
        Using it avoids crashes but at a throughput cost.
        """
        ram_budget = self.available_mb * self.safety_ratio
        from .constants import SWAP_CREDIT_RATIO
        swap_credit = self.platform.swap_available_mb * SWAP_CREDIT_RATIO
        return ram_budget + swap_credit

    def estimate(self, **kwargs) -> MemoryEstimate:
        """Estimate training memory without auto-downgrade.

        Pass same kwargs as estimate_training_memory().
        """
        return estimate_training_memory(**kwargs)

    def preflight(
        self,
        model_params: int,
        model_bits: int = 4,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_layers: int = 32,
        batch_size: int = 4,
        seq_length: int = 2048,
        lora_rank: int = 8,
        lora_layers: int = 16,
        optimizer: str = "adam",
        grad_checkpoint: bool = False,
        grad_accumulation: int = 1,
        flash_attention: Optional[bool] = None,
        lazy_evaluation: Optional[bool] = None,
    ) -> SafeConfig:
        """Run pre-flight memory check and auto-downgrade if needed.

        Returns a SafeConfig with memory-safe parameters.
        If the original config fits, it's returned unchanged.
        If not, parameters are iteratively reduced.

        The RL bandit policy (if loaded) is consulted first.  It may propose
        a ``(batch_size, lora_rank)`` pair that it has learned works well for
        the current device/model.  The proposal is validated by the estimator
        (safety net is preserved) before being accepted.  On cold start or
        exploration the existing binary-search path runs as before.

        flash_attention and lazy_evaluation are auto-detected from
        platform if not explicitly set.
        """
        # Auto-detect framework features from platform
        if flash_attention is None:
            flash_attention = True  # Default on for all modern frameworks
        if lazy_evaluation is None:
            # MLX uses lazy evaluation on Apple Silicon only.
            # Intel Macs don't support MLX training (no Metal ML compute).
            lazy_evaluation = self.platform.backend == Backend.APPLE_SILICON

        available = self.available_mb
        budget = self.budget_mb  # Includes swap credit

        from .estimator import ModelSpec, TrainSpec

        model_spec = ModelSpec(
            params=model_params, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers, bits=model_bits,
        )

        # ---- RL bandit: consult policy before binary-search ----------------
        from .bandit_state import ConfigAction, StateKey
        state_key = StateKey.from_values(
            available_mb=available,
            backend=self.platform.backend.value,
            model_params=model_params,
            model_bits=model_bits,
        )
        self._last_state_key = state_key

        candidates = [
            ConfigAction(
                batch_size=bs,
                lora_rank=lr,
                seq_length=seq_length,
                max_num_seqs=0,
            )
            for bs in _BANDIT_BATCH_SIZES
            for lr in _BANDIT_LORA_RANKS
        ]

        if self._policy is not None:
            policy_action = self._policy.select_action(state_key, candidates)
            if policy_action is not None:
                # Validate the policy's recommendation with the estimator
                p_train_spec = TrainSpec(
                    batch_size=policy_action.batch_size,
                    seq_length=policy_action.seq_length,
                    lora_rank=policy_action.lora_rank,
                    lora_layers=lora_layers,
                    optimizer=optimizer,
                    grad_checkpoint=grad_checkpoint,
                    grad_accumulation=grad_accumulation,
                    flash_attention=flash_attention,
                    lazy_evaluation=lazy_evaluation,
                )
                p_est = estimate_training_memory(model=model_spec, train=p_train_spec)
                p_eff_mb = p_est.total_mb
                if self.enable_calibration and self._calibration_store:
                    from .calibration import apply_calibration
                    p_corr, _ = apply_calibration(
                        p_est.total_mb,
                        backend=self.platform.backend.value,
                        store=self._calibration_store,
                    )
                    p_eff_mb = p_corr

                if p_eff_mb <= budget:
                    self._last_action = policy_action
                    self._last_estimate_mb = p_eff_mb
                    logger.debug(
                        "[memory-guard] Bandit → batch_size=%d lora_rank=%d "
                        "(%.0fMB ≤ budget %.0fMB).",
                        policy_action.batch_size, policy_action.lora_rank,
                        p_eff_mb, budget,
                    )
                    return SafeConfig(
                        batch_size=policy_action.batch_size,
                        seq_length=policy_action.seq_length,
                        lora_rank=policy_action.lora_rank,
                        lora_layers=lora_layers,
                        grad_checkpoint=grad_checkpoint,
                        grad_accumulation=grad_accumulation,
                        estimate=p_est,
                        budget_mb=budget,
                        available_mb=available,
                        changes=[],
                        fits=True,
                    )
                logger.debug(
                    "[memory-guard] Bandit action failed validation "
                    "(%.0fMB > budget %.0fMB). Falling back to binary search.",
                    p_eff_mb, budget,
                )
        # ---- end RL bandit -------------------------------------------------

        train_spec = TrainSpec(
            batch_size=batch_size, seq_length=seq_length,
            lora_rank=lora_rank, lora_layers=lora_layers,
            optimizer=optimizer, grad_checkpoint=grad_checkpoint,
            grad_accumulation=grad_accumulation,
            flash_attention=flash_attention,
            lazy_evaluation=lazy_evaluation,
        )

        # Initial estimate (formula-based)
        est = estimate_training_memory(model=model_spec, train=train_spec)

        # Apply auto-calibration correction if available
        # Apply calibration as a separate comparison value — don't mutate
        # est.total_mb, or __str__() will show component sum != total.
        effective_mb = est.total_mb
        if self.enable_calibration and self._calibration_store:
            from .calibration import apply_calibration
            corrected_mb, factor = apply_calibration(
                est.total_mb, backend=self.platform.backend.value,
                store=self._calibration_store,
            )
            if factor != 1.0:
                logger.info(
                    f"Calibration correction: {est.total_mb:.0f}MB × "
                    f"{factor:.3f} = {corrected_mb:.0f}MB "
                    f"(based on {self._calibration_store.num_points} past runs)"
                )
                effective_mb = corrected_mb

        self._last_estimate_mb = effective_mb

        if effective_mb <= budget:
            # Fits! Return original config.
            self._last_action = ConfigAction(
                batch_size=batch_size,
                lora_rank=lora_rank,
                seq_length=seq_length,
                max_num_seqs=0,
            )
            return SafeConfig(
                batch_size=batch_size, seq_length=seq_length,
                lora_rank=lora_rank, lora_layers=lora_layers,
                grad_checkpoint=grad_checkpoint,
                grad_accumulation=grad_accumulation,
                estimate=est, budget_mb=budget, available_mb=available,
                changes=[], fits=True,
            )

        # Doesn't fit — auto-downgrade
        logger.warning(
            f"Estimated {effective_mb:.0f}MB exceeds budget {budget:.0f}MB "
            f"({available:.0f}MB × {self.safety_ratio:.0%}). Auto-downgrading..."
        )

        result = auto_downgrade(
            budget_mb=budget,
            model_params=model_params, model_bits=model_bits,
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
            batch_size=batch_size, seq_length=seq_length,
            lora_rank=lora_rank, lora_layers=lora_layers,
            grad_checkpoint=grad_checkpoint,
            grad_accumulation=grad_accumulation,
            optimizer=optimizer,
            flash_attention=flash_attention,
            lazy_evaluation=lazy_evaluation,
        )

        self._last_action = ConfigAction(
            batch_size=result.batch_size,
            lora_rank=result.lora_rank,
            seq_length=result.seq_length,
            max_num_seqs=0,
        )
        return SafeConfig(
            batch_size=result.batch_size, seq_length=result.seq_length,
            lora_rank=result.lora_rank, lora_layers=result.lora_layers,
            grad_checkpoint=result.grad_checkpoint,
            grad_accumulation=result.grad_accumulation,
            estimate=result.final_estimate, budget_mb=budget,
            available_mb=available, changes=result.changes, fits=result.fits,
        )

    def monitor(
        self,
        batch_size: int,
        poll_interval: float = 5.0,
        max_downgrades: int = 3,
        **kwargs,
    ) -> RuntimeMonitor:
        """Create a RuntimeMonitor (use as context manager).

        Usage:
            with guard.monitor(batch_size=4) as mon:
                for step in training_loop:
                    train_step(batch_size=mon.current_batch_size)
        """
        mon = RuntimeMonitor(
            poll_interval=poll_interval,
            backend=self.platform.backend,
            max_downgrades=max_downgrades,
            memory_limit_mb=self.budget_mb,
            **kwargs,
        )
        return mon.session(batch_size)

    def preflight_inference(
        self,
        model_params: int,
        model_bits: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        num_layers: int = 32,
        max_num_seqs: int = 256,
        max_seq_len: int = 8192,
        dtype_bytes: int = 2,
        hidden_dim: int = 0,
        min_num_seqs: int = 1,
    ) -> InferenceSafeConfig:
        """Run pre-flight memory check for a serving deployment.

        Binary-searches for the largest ``max_num_seqs`` that fits within
        the memory budget.  Never mutates a running engine — call this
        before starting vLLM or SGLang to determine safe launch parameters.

        The RL bandit policy (if loaded) is consulted first.  It may propose
        a ``max_num_seqs`` it has learned is safe for the current device/model.
        The proposal is validated by the estimator before being accepted.  On
        cold start or exploration the existing binary-search path runs as before.

        The KV cache formula used is the ceiling:
            2 × num_layers × num_kv_heads × head_dim × max_seq_len × max_num_seqs × dtype_bytes

        Args:
            model_params:   Total parameter count.
            model_bits:     Weight quantization precision (4, 8, 16, 32).
            num_kv_heads:   GQA-aware KV head count (e.g. 8 for Llama-3-8B).
            head_dim:       Attention head dimension (hidden_dim // num_heads).
            num_layers:     Number of transformer layers.
            max_num_seqs:   Requested max concurrent requests.  Reduced if needed.
            max_seq_len:    Maximum sequence length (prompt + generation).
            dtype_bytes:    KV cache element width (2=fp16/bf16, 4=fp32, 1=int8).
            hidden_dim:     When > 0, include decode-phase activation buffers.
            min_num_seqs:   Never reduce max_num_seqs below this floor.

        Returns:
            InferenceSafeConfig with the largest fitting max_num_seqs, a
            suggested gpu_memory_utilization, and fits=False when even
            min_num_seqs exceeds the budget (engine may OOM under peak load).
        """
        available = self.available_mb
        budget = self.budget_mb

        _kw = dict(
            model_params=model_params, model_bits=model_bits,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            num_layers=num_layers, max_seq_len=max_seq_len,
            dtype_bytes=dtype_bytes, hidden_dim=hidden_dim,
        )

        # ---- RL bandit: consult policy before binary-search ----------------
        from .bandit_state import ConfigAction, StateKey
        state_key = StateKey.from_values(
            available_mb=available,
            backend=self.platform.backend.value,
            model_params=model_params,
            model_bits=model_bits,
        )
        self._last_state_key = state_key

        candidates = [
            ConfigAction(
                batch_size=1,
                lora_rank=0,
                seq_length=max_seq_len,
                max_num_seqs=n,
            )
            for n in _BANDIT_NUM_SEQS
            if n <= max_num_seqs
        ]

        if self._policy is not None and candidates:
            policy_action = self._policy.select_action(state_key, candidates)
            if policy_action is not None:
                p_est = estimate_serving_memory(
                    max_num_seqs=policy_action.max_num_seqs, **_kw
                )
                if p_est.fits_in(budget):
                    gpu_util = (
                        min(0.95, p_est.total_mb / available)
                        if available > 0 else 0.90
                    )
                    self._last_action = policy_action
                    self._last_estimate_mb = p_est.total_mb
                    logger.debug(
                        "[memory-guard] Bandit → max_num_seqs=%d (%.0fMB ≤ %.0fMB).",
                        policy_action.max_num_seqs, p_est.total_mb, budget,
                    )
                    return InferenceSafeConfig(
                        max_num_seqs=policy_action.max_num_seqs,
                        max_seq_len=policy_action.seq_length,
                        gpu_memory_utilization=round(gpu_util, 4),
                        estimate=p_est,
                        budget_mb=budget,
                        available_mb=available,
                        fits=True,
                        changes=[],
                    )
                logger.debug(
                    "[memory-guard] Bandit inference action failed validation "
                    "(%.0fMB > budget %.0fMB). Falling back to binary search.",
                    p_est.total_mb, budget,
                )
        # ---- end RL bandit -------------------------------------------------

        # Fast path — requested config already fits
        est = estimate_serving_memory(max_num_seqs=max_num_seqs, **_kw)
        if est.fits_in(budget):
            gpu_util = min(0.95, est.total_mb / available) if available > 0 else 0.90
            self._last_action = ConfigAction(
                batch_size=1,
                lora_rank=0,
                seq_length=max_seq_len,
                max_num_seqs=max_num_seqs,
            )
            self._last_estimate_mb = est.total_mb
            return InferenceSafeConfig(
                max_num_seqs=max_num_seqs, max_seq_len=max_seq_len,
                gpu_memory_utilization=round(gpu_util, 4),
                estimate=est, budget_mb=budget, available_mb=available,
                fits=True, changes=[],
            )

        logger.warning(
            "[memory-guard] %d seqs × %d tokens requires %.0f MB, "
            "exceeds budget %.0f MB. Binary-searching for safe max_num_seqs...",
            max_num_seqs, max_seq_len, est.total_mb, budget,
        )

        # Binary search: largest value in [min_num_seqs, max_num_seqs] that fits
        lo, hi = min_num_seqs, max_num_seqs
        safe_num_seqs: Optional[int] = None
        safe_est: Optional[InferenceServingEstimate] = None

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = estimate_serving_memory(max_num_seqs=mid, **_kw)
            if candidate.fits_in(budget):
                safe_num_seqs = mid
                safe_est = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        fits = safe_num_seqs is not None
        if not fits:
            # Even min_num_seqs doesn't fit — return it with fits=False
            safe_num_seqs = min_num_seqs
            safe_est = estimate_serving_memory(max_num_seqs=min_num_seqs, **_kw)

        gpu_util = min(0.95, safe_est.total_mb / available) if available > 0 else 0.90
        changes = [f"max_num_seqs reduced {max_num_seqs} → {safe_num_seqs}"]

        logger.warning(
            "[memory-guard] Safe max_num_seqs: %d (%.0f MB). "
            "Suggested: vllm --max-num-seqs=%d --gpu-memory-utilization=%.4f",
            safe_num_seqs, safe_est.total_mb, safe_num_seqs, gpu_util,
        )

        self._last_action = ConfigAction(
            batch_size=1,
            lora_rank=0,
            seq_length=max_seq_len,
            max_num_seqs=safe_num_seqs,
        )
        self._last_estimate_mb = safe_est.total_mb

        return InferenceSafeConfig(
            max_num_seqs=safe_num_seqs, max_seq_len=max_seq_len,
            gpu_memory_utilization=round(gpu_util, 4),
            estimate=safe_est, budget_mb=budget, available_mb=available,
            fits=fits, changes=changes,
        )

    def record_result(
        self,
        actual_peak_mb: Optional[float] = None,
        model_name: str = "",
        oom_occurred: bool = False,
        policy_update: bool = True,
        **kwargs,
    ):
        """Record actual peak memory after training for auto-calibration.

        Call this after training completes. If actual_peak_mb is None,
        attempts to read from mx.metal.get_peak_memory() or
        torch.cuda.max_memory_allocated().

        Over time, this builds a calibration dataset that corrects
        the formula's output to match real-world measurements.

        When ``policy_update=True`` (the default) and a bandit policy is
        loaded, the Q-table is updated with the reward from this run and the
        policy is saved to disk.  Set ``oom_occurred=True`` when the run
        ended with an OOM error so the bandit learns to avoid that config.

        Args:
            actual_peak_mb: Measured peak memory (MB).  Auto-detected if None.
            model_name:     Human-readable model identifier for logging.
            oom_occurred:   True if the run ended with an OOM error.
            policy_update:  Whether to update the bandit Q-table.
            **kwargs:       Forwarded to record_training_result().
        """
        if not self.enable_calibration or not self._calibration_store:
            return

        # Auto-detect actual peak if not provided
        if actual_peak_mb is None:
            from .platforms import get_mlx_peak_memory_mb
            actual_peak_mb = get_mlx_peak_memory_mb()

        if actual_peak_mb is None:
            try:
                import torch
                if torch.cuda.is_available():
                    actual_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except ImportError:
                pass

        if actual_peak_mb is None or actual_peak_mb <= 0:
            logger.debug("Could not determine actual peak memory for calibration")
            return

        if self._last_estimate_mb is None or self._last_estimate_mb <= 0:
            logger.debug("No estimate available to calibrate against")
            return

        from .calibration import record_training_result
        reward = record_training_result(
            estimated_mb=self._last_estimate_mb,
            actual_peak_mb=actual_peak_mb,
            model_name=model_name,
            backend=self.platform.backend.value,
            store=self._calibration_store,
            budget_mb=self.budget_mb,
            oom_occurred=oom_occurred,
            **kwargs,
        )

        # Update the bandit policy with the outcome of this run
        if (
            policy_update
            and self._policy is not None
            and self._last_state_key is not None
            and self._last_action is not None
        ):
            self._policy.update(
                self._last_state_key, self._last_action, reward.combined
            )
            self._policy.save()
            logger.debug(
                "[memory-guard] Bandit policy updated: reward=%.3f "
                "(states=%d, updates=%d).",
                reward.combined,
                self._policy.num_states,
                self._policy.num_updates,
            )
