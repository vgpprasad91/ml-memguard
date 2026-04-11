"""Reward signal for the RL contextual bandit (v0.4).

Bridges the existing calibration system (which records *what happened*)
to the RL policy (which needs a scalar reward to update Q-values).

Components
----------
RewardSignal
    Three-field dataclass: ``outcome`` (binary ±1), ``efficiency_bonus``
    (continuous 0–1 calibration-accuracy score), and ``combined``
    (weighted sum ready to pass to the Q-update rule).

compute_reward(estimated_mb, actual_peak_mb, budget_mb, oom_occurred)
    Pure function — no I/O, no side effects.  Computes all three fields
    from a single training-run observation.

Design notes
------------
Outcome (+1 / -1)
    A clean completion is +1; an OOM event is -1.  This is the dominant
    signal: an OOM is strictly worse than any clean run, regardless of
    memory efficiency.

Efficiency bonus (0–1)
    Measures how accurately the estimator predicted actual peak memory,
    scaled by the budget so that large-memory devices don't dominate:

        efficiency_bonus = 1 - abs(estimated_mb - actual_peak_mb) / budget_mb

    Clamped to [0, 1].  A perfect estimate gives 1.0; a run where the
    estimate was off by the full budget gives 0.0.  If ``budget_mb`` is
    zero or unknown, the bonus defaults to 0.0 (no efficiency signal).
    On OOM the bonus is always 0.0 — a failed run carries no efficiency
    information worth rewarding.

Combined reward
    weighted sum:  OUTCOME_WEIGHT × outcome + EFFICIENCY_WEIGHT × efficiency_bonus

    With the default weights (0.6 / 0.4), range is [-0.6, 1.0]:
        - Perfect clean run with perfect estimate : +1.0
        - Clean run, estimate off by full budget  : +0.6
        - OOM, any estimate                       : -0.6

    Weights are module-level constants so callers can override them for
    experiments without subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Weighting constants
# ---------------------------------------------------------------------------

OUTCOME_WEIGHT: float = 0.6
"""Weight applied to the binary outcome (+1 / -1) in the combined reward."""

EFFICIENCY_WEIGHT: float = 0.4
"""Weight applied to the efficiency bonus (0–1) in the combined reward."""

# Outcome sentinels
OUTCOME_SUCCESS: float = 1.0
OUTCOME_OOM: float = -1.0


# ---------------------------------------------------------------------------
# RewardSignal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardSignal:
    """Scalar reward emitted after a single training or serving run.

    Frozen so it can be stored in logs and passed around without mutation.

    Attributes:
        outcome:          +1.0 (clean completion) or -1.0 (OOM).
        efficiency_bonus: Calibration-accuracy score, 0.0–1.0.
                          0.0 when budget_mb is unknown or OOM occurred.
        combined:         OUTCOME_WEIGHT × outcome + EFFICIENCY_WEIGHT × efficiency_bonus.
                          Range: [-0.6, 1.0] with default weights.
    """

    outcome: float
    efficiency_bonus: float
    combined: float

    @property
    def is_oom(self) -> bool:
        """True when the run ended with an OOM event."""
        return self.outcome < 0.0


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


def compute_reward(
    estimated_mb: float,
    actual_peak_mb: float,
    budget_mb: float,
    oom_occurred: bool,
    outcome_weight: float = OUTCOME_WEIGHT,
    efficiency_weight: float = EFFICIENCY_WEIGHT,
) -> RewardSignal:
    """Compute a ``RewardSignal`` from a single training-run observation.

    Pure function — no I/O, no side effects.

    Args:
        estimated_mb:     Memory estimate produced by the estimator before
                          the run (in MB).
        actual_peak_mb:   Actual peak memory observed during the run (in MB).
                          Obtained from ``mx.metal.get_peak_memory()`` or
                          ``torch.cuda.max_memory_allocated()``.
        budget_mb:        Memory budget used for the preflight check (in MB).
                          Pass 0.0 when unknown; the efficiency bonus will
                          be 0.0 in that case.
        oom_occurred:     True if the run ended with an OOM error.
        outcome_weight:   Weight for the binary outcome component.
                          Defaults to ``OUTCOME_WEIGHT`` (0.6).
        efficiency_weight: Weight for the efficiency bonus component.
                           Defaults to ``EFFICIENCY_WEIGHT`` (0.4).

    Returns:
        A frozen ``RewardSignal``.

    Examples:
        >>> sig = compute_reward(5000, 4800, 8000, oom_occurred=False)
        >>> sig.outcome
        1.0
        >>> sig.efficiency_bonus   # 1 - |5000-4800| / 8000 = 0.975
        0.975
        >>> sig.combined           # 0.6*1.0 + 0.4*0.975 = 0.99
        0.99

        >>> oom = compute_reward(5000, 9000, 8000, oom_occurred=True)
        >>> oom.outcome
        -1.0
        >>> oom.efficiency_bonus
        0.0
        >>> oom.combined
        -0.6
    """
    outcome = OUTCOME_OOM if oom_occurred else OUTCOME_SUCCESS

    # Efficiency bonus: 0 on OOM or unknown budget
    if oom_occurred or budget_mb <= 0.0:
        efficiency_bonus = 0.0
    else:
        raw = 1.0 - abs(estimated_mb - actual_peak_mb) / budget_mb
        efficiency_bonus = max(0.0, min(1.0, raw))

    combined = outcome_weight * outcome + efficiency_weight * efficiency_bonus

    return RewardSignal(
        outcome=outcome,
        efficiency_bonus=efficiency_bonus,
        combined=combined,
    )
