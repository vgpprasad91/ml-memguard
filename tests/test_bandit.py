"""Tests for memory_guard.bandit — BanditPolicy tabular Q-learner.

All tests that write to disk use tmp_path (pytest fixture) so they never
touch ~/.memory-guard.

Covers:
  - Initialisation defaults
  - select_action: cold start, empty candidates, epsilon=1 / epsilon=0
  - select_action: argmax over candidates
  - update: Q-formula correctness, new state initialisation
  - update: epsilon decay and floor
  - update: num_updates counter, custom alpha
  - Persistence: save/load round-trip (Q-values and epsilon)
  - Persistence: load from absent file → fresh policy
  - Persistence: load from corrupted JSON → fresh policy
  - Persistence: save creates parent directories
  - Persistence: saved file contains version field
  - Device isolation: different StateKeys don't contaminate each other
  - Serialisation helpers: round-trip for StateKey and ConfigAction strings
"""

from __future__ import annotations

import json
import math

import pytest

from memory_guard.bandit import (
    DEFAULT_POLICY_PATH,
    BanditPolicy,
    _action_to_str,
    _state_key_to_str,
    _str_to_action,
    _str_to_state_key,
)
from memory_guard.bandit_state import (
    ConfigAction,
    DeviceFingerprint,
    ModelFingerprint,
    StateKey,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_state(memory_tier="8_24gb", backend="cuda", os_platform="linux",
                param_class="7_13b", bits=16) -> StateKey:
    return StateKey(
        device=DeviceFingerprint(memory_tier=memory_tier, backend=backend, os_platform=os_platform),
        model=ModelFingerprint(param_class=param_class, bits=bits),
    )


def _make_action(batch_size=4, lora_rank=16, seq_length=2048, max_num_seqs=0) -> ConfigAction:
    return ConfigAction(
        batch_size=batch_size, lora_rank=lora_rank,
        seq_length=seq_length, max_num_seqs=max_num_seqs,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_epsilon(self):
        p = BanditPolicy()
        assert p.epsilon == pytest.approx(0.9)

    def test_default_epsilon_decay(self):
        p = BanditPolicy()
        assert p.epsilon_decay == pytest.approx(0.995)

    def test_default_epsilon_floor(self):
        p = BanditPolicy()
        assert p.epsilon_floor == pytest.approx(0.05)

    def test_default_alpha(self):
        p = BanditPolicy()
        assert p.alpha == pytest.approx(0.1)

    def test_q_table_starts_empty(self):
        p = BanditPolicy()
        assert p.num_states == 0

    def test_num_updates_starts_zero(self):
        p = BanditPolicy()
        assert p.num_updates == 0


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------

class TestSelectAction:
    def test_empty_candidates_returns_none(self):
        p = BanditPolicy()
        sk = _make_state()
        assert p.select_action(sk, [], epsilon=0.0) is None

    def test_cold_start_returns_none(self):
        p = BanditPolicy()
        sk = _make_state()
        candidates = [_make_action(batch_size=4)]
        assert p.select_action(sk, candidates, epsilon=0.0) is None

    def test_epsilon_one_always_returns_none(self):
        p = BanditPolicy()
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0)
        for _ in range(20):
            assert p.select_action(sk, [action], epsilon=1.0) is None

    def test_epsilon_zero_returns_argmax(self):
        p = BanditPolicy()
        sk = _make_state()
        a_low = _make_action(batch_size=2)
        a_high = _make_action(batch_size=8)
        p.update(sk, a_low, reward=0.3)
        p.update(sk, a_high, reward=0.9)
        result = p.select_action(sk, [a_low, a_high], epsilon=0.0)
        assert result == a_high

    def test_argmax_picks_highest_q_candidate(self):
        p = BanditPolicy()
        sk = _make_state()
        actions = [_make_action(batch_size=i) for i in range(1, 6)]
        rewards = [0.1, 0.4, 0.9, 0.6, 0.2]
        for a, r in zip(actions, rewards):
            p.update(sk, a, reward=r)
        chosen = p.select_action(sk, actions, epsilon=0.0)
        assert chosen == actions[2]  # batch_size=3 has reward 0.9 → highest Q

    def test_unseen_candidates_default_to_zero(self):
        p = BanditPolicy()
        sk = _make_state()
        known = _make_action(batch_size=4)
        unknown = _make_action(batch_size=8)
        # give known a negative Q
        p.update(sk, known, reward=-1.0, alpha=1.0)
        # unknown defaults to 0.0, so it should win
        result = p.select_action(sk, [known, unknown], epsilon=0.0)
        assert result == unknown

    def test_epsilon_override_respected(self):
        """Passing epsilon kwarg overrides self.epsilon."""
        p = BanditPolicy(epsilon=0.0)  # policy prefers to exploit
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0)
        # force exploration via override
        result = p.select_action(sk, [action], epsilon=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# update — Q-formula and metadata
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_q_formula_from_zero(self):
        # Q_new = 0 + alpha * (reward - 0) = alpha * reward
        p = BanditPolicy(alpha=0.1)
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0)
        assert p.q_value(sk, action) == pytest.approx(0.1)

    def test_q_formula_second_update(self):
        # After first update Q=0.1; second: Q = 0.1 + 0.1*(1.0-0.1) = 0.19
        p = BanditPolicy(alpha=0.1)
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0)
        p.update(sk, action, reward=1.0)
        assert p.q_value(sk, action) == pytest.approx(0.19)

    def test_new_state_initialised_to_zero(self):
        p = BanditPolicy()
        sk = _make_state()
        action = _make_action()
        assert p.q_value(sk, action) == pytest.approx(0.0)

    def test_increments_num_updates(self):
        p = BanditPolicy()
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0)
        p.update(sk, action, reward=1.0)
        assert p.num_updates == 2

    def test_epsilon_decays_after_update(self):
        p = BanditPolicy(epsilon=0.9, epsilon_decay=0.995)
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0)
        assert p.epsilon == pytest.approx(0.9 * 0.995)

    def test_epsilon_floored(self):
        p = BanditPolicy(epsilon=0.05, epsilon_decay=0.5, epsilon_floor=0.05)
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0)
        # 0.05 * 0.5 = 0.025 < floor → stays at 0.05
        assert p.epsilon == pytest.approx(0.05)

    def test_custom_alpha_overrides_default(self):
        p = BanditPolicy(alpha=0.1)
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0, alpha=0.5)
        # Q = 0 + 0.5 * (1.0 - 0) = 0.5
        assert p.q_value(sk, action) == pytest.approx(0.5)

    def test_repeated_updates_converge_toward_reward(self):
        p = BanditPolicy(alpha=0.1)
        sk = _make_state()
        action = _make_action()
        for _ in range(200):
            p.update(sk, action, reward=1.0)
        # Should converge close to 1.0 (geometric series: 1 - 0.9^200 ≈ 1.0)
        assert p.q_value(sk, action) > 0.99


# ---------------------------------------------------------------------------
# Device isolation
# ---------------------------------------------------------------------------

class TestDeviceIsolation:
    def test_different_state_keys_dont_contaminate(self):
        p = BanditPolicy()
        sk_a100 = _make_state(memory_tier="80plus_gb", backend="cuda")
        sk_m2 = _make_state(memory_tier="24_48gb", backend="apple_silicon",
                             os_platform="darwin")
        action = _make_action()

        p.update(sk_a100, action, reward=1.0, alpha=1.0)
        # M2 entry should still be 0.0
        assert p.q_value(sk_m2, action) == pytest.approx(0.0)

    def test_select_action_on_untrained_device_returns_none(self):
        p = BanditPolicy()
        sk_a100 = _make_state(memory_tier="80plus_gb", backend="cuda")
        sk_m2 = _make_state(memory_tier="24_48gb", backend="apple_silicon",
                             os_platform="darwin")
        action = _make_action()
        p.update(sk_a100, action, reward=1.0)
        # M2 has never been seen — cold start
        assert p.select_action(sk_m2, [action], epsilon=0.0) is None


# ---------------------------------------------------------------------------
# Persistence — save / load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_round_trip_q_values(self, tmp_path):
        p = BanditPolicy()
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=1.0, alpha=1.0)

        path = tmp_path / "rl_policy.json"
        p.save(path)

        p2 = BanditPolicy.load(path)
        assert p2.q_value(sk, action) == pytest.approx(p.q_value(sk, action))

    def test_round_trip_epsilon(self, tmp_path):
        p = BanditPolicy(epsilon=0.42)
        path = tmp_path / "rl_policy.json"
        p.save(path)
        p2 = BanditPolicy.load(path)
        assert p2.epsilon == pytest.approx(0.42)

    def test_round_trip_num_updates(self, tmp_path):
        p = BanditPolicy()
        sk = _make_state()
        action = _make_action()
        p.update(sk, action, reward=0.5)
        p.update(sk, action, reward=0.5)
        path = tmp_path / "rl_policy.json"
        p.save(path)
        p2 = BanditPolicy.load(path)
        assert p2.num_updates == 2

    def test_load_absent_file_returns_fresh_policy(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        p = BanditPolicy.load(path)
        assert p.num_states == 0
        assert p.num_updates == 0

    def test_load_corrupted_json_returns_fresh_policy(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ not valid json }")
        p = BanditPolicy.load(path)
        assert p.num_states == 0

    def test_load_missing_q_table_key_returns_fresh(self, tmp_path):
        path = tmp_path / "no_qtable.json"
        path.write_text(json.dumps({"version": "0.4.0", "epsilon": 0.5}))
        p = BanditPolicy.load(path)
        assert p.num_states == 0

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "rl_policy.json"
        p = BanditPolicy()
        p.save(nested)
        assert nested.exists()

    def test_saved_file_contains_version_field(self, tmp_path):
        path = tmp_path / "rl_policy.json"
        BanditPolicy().save(path)
        data = json.loads(path.read_text())
        assert "version" in data
        assert data["version"] == "0.4.0"

    def test_saved_file_is_valid_json(self, tmp_path):
        p = BanditPolicy()
        sk = _make_state()
        p.update(sk, _make_action(), reward=0.8)
        path = tmp_path / "rl_policy.json"
        p.save(path)
        data = json.loads(path.read_text())
        assert "q_table" in data


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

class TestSerialisationHelpers:
    def test_state_key_round_trip(self):
        sk = _make_state()
        assert _str_to_state_key(_state_key_to_str(sk)) == sk

    def test_action_round_trip(self):
        action = _make_action(batch_size=8, lora_rank=32, seq_length=4096, max_num_seqs=64)
        assert _str_to_action(_action_to_str(action)) == action

    def test_state_key_str_contains_all_fields(self):
        sk = _make_state(memory_tier="sub_8gb", backend="rocm",
                          os_platform="linux", param_class="1_7b", bits=4)
        s = _state_key_to_str(sk)
        assert "sub_8gb" in s
        assert "rocm" in s
        assert "linux" in s
        assert "1_7b" in s
        assert "4" in s

    def test_invalid_state_key_str_raises(self):
        with pytest.raises(ValueError):
            _str_to_state_key("only|three|parts")

    def test_invalid_action_str_raises(self):
        with pytest.raises(ValueError):
            _str_to_action("4|16|2048")  # missing max_num_seqs
