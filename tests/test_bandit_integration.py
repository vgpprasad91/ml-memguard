"""Integration tests — RL bandit wired into MemoryGuard (v0.4 PR 4).

All tests that touch disk use tmp_path so they never write to
~/.memory-guard.  Platform-dependent calls (available_mb, budget_mb,
etc.) are patched via monkeypatch so the suite runs hermetically on any
machine and class-level side-effects are always rolled back.

Covers:
  - MemoryGuard.__init__: enable_bandit=True loads policy, False skips it
  - MemoryGuard.auto(): enable_bandit parameter passed through
  - preflight(): cold-start → existing binary-search path unchanged
  - preflight(): bandit proposes action → estimator validates → returns it
  - preflight(): bandit action fails validation → falls back to binary search
  - preflight(): _last_action/_last_state_key always set after the call
  - preflight(): enable_bandit=False → policy is None, _last_action still set
  - preflight_inference(): cold-start → existing path unchanged
  - preflight_inference(): bandit proposes max_num_seqs → returns it
  - preflight_inference(): _last_action/_last_state_key always set
  - record_result(): updates policy Q-table and saves to disk
  - record_result(): policy_update=False skips Q-table update
  - record_result(): no _last_action set → policy update skipped
  - record_result(): oom_occurred=True → negative reward stored
  - record_result(): calibration point still recorded regardless of bandit
  - Full round-trip: preflight → record_result → policy persisted
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_guard.adaptation.bandit import BanditPolicy
from memory_guard.adaptation.bandit_state import ConfigAction, StateKey
from memory_guard.guard import MemoryGuard


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_MODEL_PARAMS = 7_000_000_000   # 7B
_MODEL_BITS = 16
_AVAILABLE_MB = 40_000.0        # ~40 GB (lands in 24_48gb tier)
_BUDGET_MB = _AVAILABLE_MB * 0.80  # 32 000 MB


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_guard(tmp_path: Path, *, enable_bandit: bool = True, **kwargs) -> MemoryGuard:
    """Return a MemoryGuard with all disk I/O pinned to tmp_path."""
    from memory_guard.adaptation.calibration import CalibrationStore
    store = CalibrationStore(path=tmp_path / "cal.json")
    guard = MemoryGuard(enable_bandit=enable_bandit, enable_calibration=True, **kwargs)
    guard._calibration_store = store
    if enable_bandit:
        guard._policy = BanditPolicy.load(tmp_path / "rl_policy.json")
    return guard


def _patch_memory(monkeypatch, guard: MemoryGuard, available: float = _AVAILABLE_MB) -> None:
    """Patch available_mb and budget_mb via monkeypatch so they auto-restore."""
    monkeypatch.setattr(type(guard), "available_mb", property(lambda self: available))
    monkeypatch.setattr(type(guard), "budget_mb", property(lambda self: available * 0.80))


def _default_preflight_kwargs() -> dict:
    return dict(
        model_params=_MODEL_PARAMS,
        model_bits=_MODEL_BITS,
        hidden_dim=4096,
        num_heads=32,
        num_layers=32,
        batch_size=4,
        seq_length=2048,
        lora_rank=16,
        lora_layers=16,
    )


def _default_inference_kwargs() -> dict:
    return dict(
        model_params=_MODEL_PARAMS,
        model_bits=_MODEL_BITS,
        num_kv_heads=8,
        head_dim=128,
        num_layers=32,
        max_num_seqs=64,
        max_seq_len=4096,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_enable_bandit_true_creates_policy(self, tmp_path):
        guard = _make_guard(tmp_path, enable_bandit=True)
        assert guard._policy is not None
        assert isinstance(guard._policy, BanditPolicy)

    def test_enable_bandit_false_no_policy(self, tmp_path):
        guard = _make_guard(tmp_path, enable_bandit=False)
        assert guard._policy is None

    def test_last_action_starts_none(self, tmp_path):
        guard = _make_guard(tmp_path)
        assert guard._last_action is None

    def test_last_state_key_starts_none(self, tmp_path):
        guard = _make_guard(tmp_path)
        assert guard._last_state_key is None

    def test_auto_passes_enable_bandit_false(self):
        guard = MemoryGuard.auto(enable_bandit=False)
        assert guard._policy is None

    def test_auto_passes_enable_bandit_true(self):
        guard = MemoryGuard.auto(enable_bandit=True)
        assert guard.enable_bandit is True


# ---------------------------------------------------------------------------
# preflight(): cold start — existing path unchanged
# ---------------------------------------------------------------------------

class TestPreflightColdStart:
    def test_cold_start_returns_safe_config(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        safe = guard.preflight(**_default_preflight_kwargs())
        from memory_guard.guard import SafeConfig
        assert isinstance(safe, SafeConfig)

    def test_cold_start_sets_last_state_key(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_state_key is not None
        assert isinstance(guard._last_state_key, StateKey)

    def test_cold_start_sets_last_action(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_action is not None
        assert isinstance(guard._last_action, ConfigAction)

    def test_no_policy_sets_last_action(self, tmp_path, monkeypatch):
        """enable_bandit=False still tracks _last_action from existing path."""
        guard = _make_guard(tmp_path, enable_bandit=False)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_state_key is not None
        assert guard._last_action is not None

    def test_last_action_batch_size_positive(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_action.batch_size >= 1

    def test_last_action_seq_length_matches_arg(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_action.seq_length == 2048

    def test_last_action_max_num_seqs_is_zero_for_training(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_action.max_num_seqs == 0


# ---------------------------------------------------------------------------
# preflight(): bandit proposes action
# ---------------------------------------------------------------------------

class TestPreflightBanditProposal:
    def _trained_policy(self, tmp_path: Path) -> BanditPolicy:
        """Return policy with high Q for batch_size=4, lora_rank=16 (epsilon=0)."""
        p = BanditPolicy(epsilon=0.0)
        sk = StateKey.from_values(
            available_mb=_AVAILABLE_MB,
            backend="cpu",
            model_params=_MODEL_PARAMS,
            model_bits=_MODEL_BITS,
        )
        action = ConfigAction(batch_size=4, lora_rank=16, seq_length=2048, max_num_seqs=0)
        p.update(sk, action, reward=1.0, alpha=1.0)
        p.save(tmp_path / "rl_policy.json")
        return p

    def test_bandit_path_returns_safe_config(self, tmp_path, monkeypatch):
        self._trained_policy(tmp_path)
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        safe = guard.preflight(**_default_preflight_kwargs())
        from memory_guard.guard import SafeConfig
        assert isinstance(safe, SafeConfig)

    def test_bandit_path_sets_last_action(self, tmp_path, monkeypatch):
        self._trained_policy(tmp_path)
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_action is not None
        assert isinstance(guard._last_action, ConfigAction)

    def test_bandit_path_sets_last_state_key(self, tmp_path, monkeypatch):
        self._trained_policy(tmp_path)
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_state_key is not None

    def test_bandit_action_seq_length_matches_requested(self, tmp_path, monkeypatch):
        """Bandit candidates are generated at the caller's seq_length."""
        self._trained_policy(tmp_path)
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        assert guard._last_action.seq_length == 2048


# ---------------------------------------------------------------------------
# preflight(): bandit action fails validation (tiny budget)
# ---------------------------------------------------------------------------

class TestPreflightBanditValidationFail:
    def test_fallback_when_policy_over_recommends(self, tmp_path, monkeypatch):
        """Bandit recommends batch_size=32 on a tiny-memory machine → fallback."""
        p = BanditPolicy(epsilon=0.0)
        sk = StateKey.from_values(
            available_mb=200.0,
            backend="cpu",
            model_params=_MODEL_PARAMS,
            model_bits=_MODEL_BITS,
        )
        action = ConfigAction(batch_size=32, lora_rank=64, seq_length=2048, max_num_seqs=0)
        p.update(sk, action, reward=1.0, alpha=1.0)
        p.save(tmp_path / "rl_policy.json")

        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard, available=500.0)
        safe = guard.preflight(**_default_preflight_kwargs())
        from memory_guard.guard import SafeConfig
        assert isinstance(safe, SafeConfig)


# ---------------------------------------------------------------------------
# preflight_inference(): cold start
# ---------------------------------------------------------------------------

class TestPreflightInferenceColdStart:
    def test_cold_start_returns_inference_safe_config(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        result = guard.preflight_inference(**_default_inference_kwargs())
        from memory_guard.guard import InferenceSafeConfig
        assert isinstance(result, InferenceSafeConfig)

    def test_cold_start_sets_last_state_key(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight_inference(**_default_inference_kwargs())
        assert guard._last_state_key is not None
        assert isinstance(guard._last_state_key, StateKey)

    def test_cold_start_sets_last_action(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight_inference(**_default_inference_kwargs())
        assert guard._last_action is not None
        assert isinstance(guard._last_action, ConfigAction)

    def test_last_action_has_max_num_seqs(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight_inference(**_default_inference_kwargs())
        assert guard._last_action.max_num_seqs >= 1

    def test_last_action_seq_length_matches_max_seq_len(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight_inference(**_default_inference_kwargs())
        assert guard._last_action.seq_length == 4096

    def test_last_estimate_mb_set(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight_inference(**_default_inference_kwargs())
        assert guard._last_estimate_mb is not None
        assert guard._last_estimate_mb > 0


# ---------------------------------------------------------------------------
# preflight_inference(): bandit proposes max_num_seqs
# ---------------------------------------------------------------------------

class TestPreflightInferenceBandit:
    def _trained_policy(self, tmp_path: Path, max_num_seqs: int = 32) -> BanditPolicy:
        p = BanditPolicy(epsilon=0.0)
        sk = StateKey.from_values(
            available_mb=_AVAILABLE_MB,
            backend="cpu",
            model_params=_MODEL_PARAMS,
            model_bits=_MODEL_BITS,
        )
        action = ConfigAction(
            batch_size=1, lora_rank=0, seq_length=4096, max_num_seqs=max_num_seqs,
        )
        p.update(sk, action, reward=0.9, alpha=1.0)
        p.save(tmp_path / "rl_policy.json")
        return p

    def test_bandit_inference_returns_config(self, tmp_path, monkeypatch):
        self._trained_policy(tmp_path, max_num_seqs=32)
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        result = guard.preflight_inference(**_default_inference_kwargs())
        from memory_guard.guard import InferenceSafeConfig
        assert isinstance(result, InferenceSafeConfig)

    def test_bandit_inference_sets_last_action(self, tmp_path, monkeypatch):
        self._trained_policy(tmp_path)
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight_inference(**_default_inference_kwargs())
        assert guard._last_action is not None

    def test_bandit_inference_sets_last_state_key(self, tmp_path, monkeypatch):
        self._trained_policy(tmp_path)
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight_inference(**_default_inference_kwargs())
        assert guard._last_state_key is not None


# ---------------------------------------------------------------------------
# record_result(): policy update
# ---------------------------------------------------------------------------

class TestRecordResultPolicyUpdate:
    def _setup(self, tmp_path: Path, monkeypatch):
        """Return guard after preflight (state tracking is set)."""
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        return guard

    def test_policy_updated_after_record_result(self, tmp_path, monkeypatch):
        guard = self._setup(tmp_path, monkeypatch)
        updates_before = guard._policy.num_updates
        guard.record_result(actual_peak_mb=20_000.0)
        assert guard._policy.num_updates == updates_before + 1

    def test_policy_update_false_skips_update(self, tmp_path, monkeypatch):
        guard = self._setup(tmp_path, monkeypatch)
        updates_before = guard._policy.num_updates
        guard.record_result(actual_peak_mb=20_000.0, policy_update=False)
        assert guard._policy.num_updates == updates_before

    def test_no_last_action_skips_policy_update(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard._last_estimate_mb = 20_000.0
        # Don't call preflight — _last_action stays None
        assert guard._last_action is None
        updates_before = guard._policy.num_updates
        guard.record_result(actual_peak_mb=18_000.0)
        assert guard._policy.num_updates == updates_before

    def test_no_last_state_key_skips_policy_update(self, tmp_path, monkeypatch):
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)
        guard._last_estimate_mb = 20_000.0
        guard._last_action = ConfigAction(batch_size=4, lora_rank=16, seq_length=2048)
        # _last_state_key still None
        updates_before = guard._policy.num_updates
        guard.record_result(actual_peak_mb=18_000.0)
        assert guard._policy.num_updates == updates_before

    def test_oom_gives_negative_reward(self, tmp_path, monkeypatch):
        guard = self._setup(tmp_path, monkeypatch)
        state_key = guard._last_state_key
        action = guard._last_action
        assert state_key is not None
        assert action is not None
        guard.record_result(actual_peak_mb=50_000.0, oom_occurred=True)
        q = guard._policy.q_value(state_key, action)
        assert q < 0.0

    def test_calibration_still_recorded(self, tmp_path, monkeypatch):
        guard = self._setup(tmp_path, monkeypatch)
        points_before = guard._calibration_store.num_points
        guard.record_result(actual_peak_mb=20_000.0)
        assert guard._calibration_store.num_points == points_before + 1

    def test_enable_bandit_false_no_policy_update(self, tmp_path, monkeypatch):
        """When bandit is off, record_result still runs calibration but doesn't crash."""
        guard = _make_guard(tmp_path, enable_bandit=False)
        _patch_memory(monkeypatch, guard)
        guard.preflight(**_default_preflight_kwargs())
        # Should not raise even with policy=None
        guard.record_result(actual_peak_mb=20_000.0)


# ---------------------------------------------------------------------------
# Full round-trip
# ---------------------------------------------------------------------------

class TestFullRoundTrip:
    def test_preflight_record_result_policy_updated(self, tmp_path, monkeypatch):
        """preflight → record_result → Q-table has an entry."""
        policy_path = tmp_path / "rl_policy.json"

        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)

        guard.preflight(**_default_preflight_kwargs())
        updates_before = guard._policy.num_updates

        guard.record_result(actual_peak_mb=22_000.0)

        assert guard._policy.num_updates == updates_before + 1

        # Save and verify file structure
        guard._policy.save(policy_path)
        data = json.loads(policy_path.read_text())
        assert "q_table" in data
        assert data["version"] == "0.4.0"

    def test_policy_survives_reload(self, tmp_path, monkeypatch):
        """After save, a freshly loaded policy sees the same Q-values."""
        policy_path = tmp_path / "rl_policy.json"

        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)

        guard.preflight(**_default_preflight_kwargs())

        state_key = guard._last_state_key
        action = guard._last_action

        # Apply one explicit update and save
        guard._policy.update(state_key, action, reward=0.8)
        guard._policy.save(policy_path)

        # Reload and compare Q-values
        p2 = BanditPolicy.load(policy_path)
        assert p2.q_value(state_key, action) == pytest.approx(
            guard._policy.q_value(state_key, action)
        )

    def test_state_key_is_deterministic_across_calls(self, tmp_path, monkeypatch):
        """Same device/model → same StateKey on consecutive preflight calls."""
        guard = _make_guard(tmp_path)
        _patch_memory(monkeypatch, guard)

        guard.preflight(**_default_preflight_kwargs())
        sk1 = guard._last_state_key

        guard.preflight(**_default_preflight_kwargs())
        sk2 = guard._last_state_key

        assert sk1 == sk2
