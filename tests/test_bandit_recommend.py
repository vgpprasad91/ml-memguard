"""Tests for BanditPolicy.recommend, recommend_conservative, and confidence.

PR 5 — Unit tests for the bandit recovery-path API.

Covers:
  - recommend(): returns argmax action across Q-table rows
  - recommend(): returns None for an unknown state key (cold-start)
  - recommend(): returns None when the Q-table row exists but is empty
  - recommend(): deterministic — same result on repeated calls, no epsilon
  - recommend_conservative(): applies 15% margin to batch_size and max_num_seqs
  - recommend_conservative(): leaves lora_rank and seq_length unchanged
  - recommend_conservative(): respects floor=1 for batch_size
  - recommend_conservative(): respects floor=0 for max_num_seqs
  - recommend_conservative(): propagates None when state key is unseen
  - confidence: 0.0 at zero updates
  - confidence: scales linearly up to MIN_UPDATES_FOR_CONFIDENCE
  - confidence: clamps to 1.0 at exactly MIN_UPDATES_FOR_CONFIDENCE updates
  - confidence: does not exceed 1.0 beyond the threshold
"""

from __future__ import annotations

import pytest

from memory_guard.adaptation.bandit import BanditPolicy, MIN_UPDATES_FOR_CONFIDENCE
from memory_guard.adaptation.bandit_state import (
    ConfigAction,
    DeviceFingerprint,
    ModelFingerprint,
    StateKey,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(
    memory_tier: str = "8_24gb",
    backend: str = "cuda",
    os_platform: str = "linux",
    param_class: str = "7_13b",
    bits: int = 16,
) -> StateKey:
    return StateKey(
        device=DeviceFingerprint(
            memory_tier=memory_tier,
            backend=backend,
            os_platform=os_platform,
        ),
        model=ModelFingerprint(param_class=param_class, bits=bits),
    )


def _make_action(
    batch_size: int = 4,
    lora_rank: int = 16,
    seq_length: int = 2048,
    max_num_seqs: int = 64,
) -> ConfigAction:
    return ConfigAction(
        batch_size=batch_size,
        lora_rank=lora_rank,
        seq_length=seq_length,
        max_num_seqs=max_num_seqs,
    )


def _policy_with_updates(sk: StateKey, rewards: dict) -> BanditPolicy:
    """Return a policy with Q-values seeded via update() at alpha=1.0."""
    p = BanditPolicy(alpha=1.0)
    for action, reward in rewards.items():
        p.update(sk, action, reward=reward, alpha=1.0)
    return p


# ---------------------------------------------------------------------------
# recommend()
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_returns_argmax_action(self):
        sk = _make_state()
        a_low = _make_action(batch_size=2)
        a_mid = _make_action(batch_size=4)
        a_high = _make_action(batch_size=8)
        p = _policy_with_updates(sk, {a_low: 0.2, a_mid: 0.5, a_high: 0.9})
        assert p.recommend(sk) == a_high

    def test_returns_argmax_with_multiple_candidates(self):
        sk = _make_state()
        actions = [_make_action(batch_size=i, max_num_seqs=32 * i) for i in range(1, 6)]
        rewards = [0.1, 0.7, 0.4, 0.9, 0.3]
        p = BanditPolicy(alpha=1.0)
        for a, r in zip(actions, rewards):
            p.update(sk, a, reward=r, alpha=1.0)
        # actions[3] has reward 0.9 — highest
        assert p.recommend(sk) == actions[3]

    def test_cold_start_returns_none(self):
        p = BanditPolicy()
        sk = _make_state()
        assert p.recommend(sk) is None

    def test_unseen_state_key_returns_none(self):
        sk_seen = _make_state(backend="cuda")
        sk_unseen = _make_state(backend="rocm")
        p = BanditPolicy(alpha=1.0)
        p.update(sk_seen, _make_action(), reward=0.8, alpha=1.0)
        assert p.recommend(sk_unseen) is None

    def test_deterministic_no_epsilon(self):
        """recommend() must return the same action on every call (no randomness)."""
        sk = _make_state()
        a1 = _make_action(batch_size=4)
        a2 = _make_action(batch_size=8)
        p = _policy_with_updates(sk, {a1: 0.3, a2: 0.8})
        results = {p.recommend(sk) for _ in range(50)}
        assert len(results) == 1  # always the same action

    def test_single_action_in_table_is_returned(self):
        sk = _make_state()
        action = _make_action()
        p = BanditPolicy(alpha=1.0)
        p.update(sk, action, reward=0.5, alpha=1.0)
        assert p.recommend(sk) == action

    def test_different_state_keys_are_independent(self):
        sk_a = _make_state(backend="cuda")
        sk_b = _make_state(backend="apple_silicon", os_platform="darwin")
        a_high = _make_action(batch_size=8)
        a_low = _make_action(batch_size=2)
        p = BanditPolicy(alpha=1.0)
        p.update(sk_a, a_high, reward=0.9, alpha=1.0)
        p.update(sk_b, a_low, reward=0.2, alpha=1.0)
        assert p.recommend(sk_a) == a_high
        assert p.recommend(sk_b) == a_low


# ---------------------------------------------------------------------------
# recommend_conservative()
# ---------------------------------------------------------------------------

class TestRecommendConservative:
    def test_cold_start_returns_none(self):
        p = BanditPolicy()
        assert p.recommend_conservative(_make_state()) is None

    def test_applies_margin_to_batch_size(self):
        sk = _make_state()
        action = _make_action(batch_size=20, max_num_seqs=100)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.15)
        # floor(20 * 0.85) = floor(17.0) = 17
        assert result is not None
        assert result.batch_size == 17

    def test_applies_margin_to_max_num_seqs(self):
        sk = _make_state()
        action = _make_action(batch_size=8, max_num_seqs=100)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.15)
        # floor(100 * 0.85) = 85
        assert result is not None
        assert result.max_num_seqs == 85

    def test_lora_rank_unchanged(self):
        sk = _make_state()
        action = _make_action(lora_rank=32)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.15)
        assert result is not None
        assert result.lora_rank == 32

    def test_seq_length_unchanged(self):
        sk = _make_state()
        action = _make_action(seq_length=4096)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.15)
        assert result is not None
        assert result.seq_length == 4096

    def test_batch_size_floor_is_one(self):
        """batch_size must never fall below 1 regardless of margin."""
        sk = _make_state()
        # batch_size=1; shrink by 50% would give 0, but floor clamps to 1
        action = _make_action(batch_size=1, max_num_seqs=10)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.50)
        assert result is not None
        assert result.batch_size == 1

    def test_max_num_seqs_floor_is_zero(self):
        """max_num_seqs can go to 0 (not applicable for training configs)."""
        sk = _make_state()
        action = _make_action(batch_size=4, max_num_seqs=1)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.50)
        assert result is not None
        assert result.max_num_seqs == 0

    def test_zero_margin_returns_best_unchanged(self):
        sk = _make_state()
        action = _make_action(batch_size=8, max_num_seqs=64)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.0)
        assert result is not None
        assert result.batch_size == 8
        assert result.max_num_seqs == 64

    def test_custom_margin_respected(self):
        sk = _make_state()
        action = _make_action(batch_size=100, max_num_seqs=200)
        p = _policy_with_updates(sk, {action: 0.9})
        result = p.recommend_conservative(sk, margin=0.30)
        # floor(100 * 0.70) = 70; floor(200 * 0.70) = 140
        assert result is not None
        assert result.batch_size == 70
        assert result.max_num_seqs == 140


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_zero_at_init(self):
        p = BanditPolicy()
        assert p.confidence == pytest.approx(0.0)

    def test_scales_linearly_before_threshold(self):
        p = BanditPolicy(alpha=1.0)
        sk = _make_state()
        action = _make_action()
        half = MIN_UPDATES_FOR_CONFIDENCE // 2
        for _ in range(half):
            p.update(sk, action, reward=0.5, alpha=1.0)
        assert p.confidence == pytest.approx(half / MIN_UPDATES_FOR_CONFIDENCE)

    def test_reaches_one_at_threshold(self):
        p = BanditPolicy(alpha=1.0)
        sk = _make_state()
        action = _make_action()
        for _ in range(MIN_UPDATES_FOR_CONFIDENCE):
            p.update(sk, action, reward=0.5, alpha=1.0)
        assert p.confidence == pytest.approx(1.0)

    def test_does_not_exceed_one_beyond_threshold(self):
        p = BanditPolicy(alpha=1.0)
        sk = _make_state()
        action = _make_action()
        for _ in range(MIN_UPDATES_FOR_CONFIDENCE * 3):
            p.update(sk, action, reward=0.5, alpha=1.0)
        assert p.confidence == pytest.approx(1.0)
        assert p.confidence <= 1.0

    def test_increments_after_each_update(self):
        p = BanditPolicy()
        sk = _make_state()
        action = _make_action()
        prev = p.confidence
        for i in range(1, 6):
            p.update(sk, action, reward=0.5)
            assert p.confidence > prev
            prev = p.confidence

    def test_independent_of_state_key_count(self):
        """confidence counts total updates, not distinct states."""
        p = BanditPolicy(alpha=1.0)
        sk_a = _make_state(backend="cuda")
        sk_b = _make_state(backend="rocm")
        action = _make_action()
        half = MIN_UPDATES_FOR_CONFIDENCE // 2
        for _ in range(half):
            p.update(sk_a, action, reward=0.5, alpha=1.0)
        for _ in range(half):
            p.update(sk_b, action, reward=0.5, alpha=1.0)
        assert p.confidence == pytest.approx(1.0)
