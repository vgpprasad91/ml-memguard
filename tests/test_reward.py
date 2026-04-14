"""Tests for memory_guard.reward — reward signal for the RL bandit.

All tests are pure (no I/O, no disk writes).  compute_reward is a
deterministic pure function so every assertion uses exact values.

Covers:
  - RewardSignal: frozen, fields, is_oom property
  - compute_reward: clean run, OOM run, perfect estimate, edge cases
  - Efficiency bonus clamping (never below 0 or above 1)
  - Unknown / zero budget_mb defaults efficiency_bonus to 0.0
  - Combined weighted sum correctness
  - Custom weights passed through
  - record_training_result: returns RewardSignal, no-break on existing callers
"""

from __future__ import annotations

import pytest

from memory_guard.adaptation.reward import (
    EFFICIENCY_WEIGHT,
    OUTCOME_OOM,
    OUTCOME_SUCCESS,
    OUTCOME_WEIGHT,
    RewardSignal,
    compute_reward,
)


# ---------------------------------------------------------------------------
# RewardSignal dataclass
# ---------------------------------------------------------------------------

class TestRewardSignal:
    def test_fields_stored(self):
        sig = RewardSignal(outcome=1.0, efficiency_bonus=0.8, combined=0.92)
        assert sig.outcome == 1.0
        assert sig.efficiency_bonus == 0.8
        assert sig.combined == pytest.approx(0.92)

    def test_frozen_cannot_be_mutated(self):
        sig = RewardSignal(outcome=1.0, efficiency_bonus=0.5, combined=0.8)
        with pytest.raises((AttributeError, TypeError)):
            sig.outcome = -1.0  # type: ignore[misc]

    def test_is_oom_false_for_success(self):
        sig = RewardSignal(outcome=1.0, efficiency_bonus=0.5, combined=0.8)
        assert sig.is_oom is False

    def test_is_oom_true_for_oom(self):
        sig = RewardSignal(outcome=-1.0, efficiency_bonus=0.0, combined=-0.6)
        assert sig.is_oom is True


# ---------------------------------------------------------------------------
# compute_reward — clean completion
# ---------------------------------------------------------------------------

class TestComputeRewardCleanRun:
    def test_outcome_is_positive_one(self):
        sig = compute_reward(5_000, 4_800, 8_000, oom_occurred=False)
        assert sig.outcome == pytest.approx(OUTCOME_SUCCESS)

    def test_perfect_estimate_gives_bonus_one(self):
        # estimated == actual → |diff| = 0 → bonus = 1.0
        sig = compute_reward(5_000, 5_000, 8_000, oom_occurred=False)
        assert sig.efficiency_bonus == pytest.approx(1.0)

    def test_efficiency_bonus_formula(self):
        # 1 - |5000 - 4800| / 8000 = 1 - 200/8000 = 0.975
        sig = compute_reward(5_000, 4_800, 8_000, oom_occurred=False)
        assert sig.efficiency_bonus == pytest.approx(0.975)

    def test_combined_weighted_sum(self):
        # outcome=1.0, bonus=0.975 → 0.6*1.0 + 0.4*0.975 = 0.99
        sig = compute_reward(5_000, 4_800, 8_000, oom_occurred=False)
        expected = OUTCOME_WEIGHT * 1.0 + EFFICIENCY_WEIGHT * 0.975
        assert sig.combined == pytest.approx(expected)

    def test_large_estimate_error_clamps_bonus_to_zero(self):
        # |10000 - 1000| / 8000 = 1.125 → 1 - 1.125 = -0.125 → clamped to 0.0
        sig = compute_reward(10_000, 1_000, 8_000, oom_occurred=False)
        assert sig.efficiency_bonus == pytest.approx(0.0)

    def test_combined_never_below_outcome_weight_times_outcome(self):
        # worst clean-run combined = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        sig = compute_reward(10_000, 1_000, 8_000, oom_occurred=False)
        assert sig.combined == pytest.approx(OUTCOME_WEIGHT * 1.0)


# ---------------------------------------------------------------------------
# compute_reward — OOM
# ---------------------------------------------------------------------------

class TestComputeRewardOOM:
    def test_outcome_is_negative_one(self):
        sig = compute_reward(5_000, 9_000, 8_000, oom_occurred=True)
        assert sig.outcome == pytest.approx(OUTCOME_OOM)

    def test_efficiency_bonus_is_zero_on_oom(self):
        # OOM always gives 0 efficiency bonus regardless of estimate accuracy
        sig = compute_reward(5_000, 5_001, 8_000, oom_occurred=True)
        assert sig.efficiency_bonus == pytest.approx(0.0)

    def test_combined_on_oom(self):
        # 0.6 * (-1.0) + 0.4 * 0.0 = -0.6
        sig = compute_reward(5_000, 9_000, 8_000, oom_occurred=True)
        assert sig.combined == pytest.approx(OUTCOME_WEIGHT * OUTCOME_OOM)

    def test_is_oom_property(self):
        sig = compute_reward(5_000, 9_000, 8_000, oom_occurred=True)
        assert sig.is_oom is True


# ---------------------------------------------------------------------------
# compute_reward — edge cases
# ---------------------------------------------------------------------------

class TestComputeRewardEdgeCases:
    def test_zero_budget_gives_zero_bonus(self):
        sig = compute_reward(5_000, 4_800, budget_mb=0.0, oom_occurred=False)
        assert sig.efficiency_bonus == pytest.approx(0.0)

    def test_negative_budget_gives_zero_bonus(self):
        sig = compute_reward(5_000, 4_800, budget_mb=-1.0, oom_occurred=False)
        assert sig.efficiency_bonus == pytest.approx(0.0)

    def test_efficiency_bonus_clamped_above_one(self):
        # Shouldn't be possible with the formula but guard against it
        sig = compute_reward(5_000, 5_000, budget_mb=0.001, oom_occurred=False)
        assert sig.efficiency_bonus <= 1.0

    def test_efficiency_bonus_never_negative(self):
        sig = compute_reward(0, 100_000, budget_mb=1_000, oom_occurred=False)
        assert sig.efficiency_bonus >= 0.0

    def test_custom_weights_applied(self):
        sig = compute_reward(
            5_000, 5_000, 8_000, oom_occurred=False,
            outcome_weight=0.7, efficiency_weight=0.3,
        )
        # perfect estimate: bonus=1.0 → combined = 0.7*1.0 + 0.3*1.0 = 1.0
        assert sig.combined == pytest.approx(1.0)

    def test_returns_reward_signal_type(self):
        sig = compute_reward(5_000, 4_800, 8_000, oom_occurred=False)
        assert isinstance(sig, RewardSignal)


# ---------------------------------------------------------------------------
# record_training_result integration
# ---------------------------------------------------------------------------

class TestRecordTrainingResultReturnsReward:
    """record_training_result must return a RewardSignal without breaking
    existing callers that ignore the return value."""

    def test_returns_reward_signal(self, tmp_path):
        from memory_guard.adaptation.calibration import CalibrationStore, record_training_result
        store = CalibrationStore(path=tmp_path / "cal.json")
        result = record_training_result(
            estimated_mb=5_000,
            actual_peak_mb=4_800,
            budget_mb=8_000,
            oom_occurred=False,
            store=store,
        )
        assert isinstance(result, RewardSignal)

    def test_clean_run_returns_positive_outcome(self, tmp_path):
        from memory_guard.adaptation.calibration import CalibrationStore, record_training_result
        store = CalibrationStore(path=tmp_path / "cal.json")
        result = record_training_result(
            estimated_mb=5_000,
            actual_peak_mb=4_800,
            budget_mb=8_000,
            oom_occurred=False,
            store=store,
        )
        assert result.outcome == pytest.approx(1.0)

    def test_oom_returns_negative_outcome(self, tmp_path):
        from memory_guard.adaptation.calibration import CalibrationStore, record_training_result
        store = CalibrationStore(path=tmp_path / "cal.json")
        result = record_training_result(
            estimated_mb=5_000,
            actual_peak_mb=9_000,
            budget_mb=8_000,
            oom_occurred=True,
            store=store,
        )
        assert result.outcome == pytest.approx(-1.0)

    def test_no_budget_defaults_work(self, tmp_path):
        """Existing callers omitting budget_mb/oom_occurred still work."""
        from memory_guard.adaptation.calibration import CalibrationStore, record_training_result
        store = CalibrationStore(path=tmp_path / "cal.json")
        result = record_training_result(
            estimated_mb=5_000,
            actual_peak_mb=4_800,
            store=store,
        )
        # budget_mb=0.0 default → efficiency_bonus=0.0
        assert isinstance(result, RewardSignal)
        assert result.efficiency_bonus == pytest.approx(0.0)

    def test_calibration_point_still_recorded(self, tmp_path):
        """The store must still grow — calibration side-effect not dropped."""
        from memory_guard.adaptation.calibration import CalibrationStore, record_training_result
        store = CalibrationStore(path=tmp_path / "cal.json")
        assert store.num_points == 0
        record_training_result(
            estimated_mb=5_000,
            actual_peak_mb=4_800,
            store=store,
        )
        assert store.num_points == 1

    def test_top_level_importable(self):
        from memory_guard import RewardSignal as RS, compute_reward as cr
        assert callable(cr)
        assert RS is RewardSignal
