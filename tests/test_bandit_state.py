"""Tests for memory_guard.bandit_state — state & action encoding.

All tests are pure (no I/O, no framework imports).  Every bucketing function
is deterministic, so tests use exact equality rather than ranges.

Covers:
  - bucket_memory: tier boundaries and labels
  - bucket_params: parameter class boundaries and labels
  - bucket_bits: canonical bit-width snapping
  - DeviceFingerprint: construction, hashability, tier assignment
  - ModelFingerprint: construction, hashability, class assignment
  - ConfigAction: frozen, hashable, field validation
  - StateKey: NamedTuple contract, Q-table usage, convenience constructor
"""

from __future__ import annotations

import sys

import pytest

from memory_guard.adaptation.bandit_state import (
    MEMORY_TIER_BOUNDARIES_MB,
    MEMORY_TIER_LABELS,
    PARAM_CLASS_BOUNDARIES,
    PARAM_CLASS_LABELS,
    ConfigAction,
    DeviceFingerprint,
    ModelFingerprint,
    StateKey,
    bucket_bits,
    bucket_memory,
    bucket_params,
)


# ---------------------------------------------------------------------------
# bucket_memory
# ---------------------------------------------------------------------------

class TestBucketMemory:
    def test_below_8gb_is_sub_8gb(self):
        assert bucket_memory(4_096.0) == "sub_8gb"

    def test_exactly_at_8gb_boundary_is_8_24gb(self):
        # boundary is exclusive on the left: < 8192 → sub_8gb; ≥ 8192 → 8_24gb
        assert bucket_memory(8_192.0) == "8_24gb"

    def test_mid_8_24gb(self):
        assert bucket_memory(16_000.0) == "8_24gb"

    def test_exactly_at_24gb_boundary_is_24_48gb(self):
        assert bucket_memory(24_576.0) == "24_48gb"

    def test_mid_24_48gb(self):
        assert bucket_memory(40_960.0) == "24_48gb"

    def test_exactly_at_48gb_boundary_is_48_80gb(self):
        assert bucket_memory(49_152.0) == "48_80gb"

    def test_mid_48_80gb(self):
        assert bucket_memory(65_536.0) == "48_80gb"

    def test_exactly_at_80gb_boundary_is_80plus(self):
        assert bucket_memory(81_920.0) == "80plus_gb"

    def test_very_large_is_80plus(self):
        assert bucket_memory(200_000.0) == "80plus_gb"

    def test_zero_is_sub_8gb(self):
        assert bucket_memory(0.0) == "sub_8gb"

    def test_labels_count_matches_boundaries(self):
        assert len(MEMORY_TIER_LABELS) == len(MEMORY_TIER_BOUNDARIES_MB) + 1


# ---------------------------------------------------------------------------
# bucket_params
# ---------------------------------------------------------------------------

class TestBucketParams:
    def test_500m_is_sub_1b(self):
        assert bucket_params(500e6) == "sub_1b"

    def test_exactly_1b_is_1_7b(self):
        assert bucket_params(1e9) == "1_7b"

    def test_3b_is_1_7b(self):
        assert bucket_params(3e9) == "1_7b"

    def test_exactly_7b_is_7_13b(self):
        assert bucket_params(7e9) == "7_13b"

    def test_8b_is_7_13b(self):
        assert bucket_params(8e9) == "7_13b"

    def test_exactly_13b_is_13_35b(self):
        assert bucket_params(13e9) == "13_35b"

    def test_20b_is_13_35b(self):
        assert bucket_params(20e9) == "13_35b"

    def test_exactly_35b_is_35plus(self):
        assert bucket_params(35e9) == "35plus_b"

    def test_70b_is_35plus(self):
        assert bucket_params(70e9) == "35plus_b"

    def test_labels_count_matches_boundaries(self):
        assert len(PARAM_CLASS_LABELS) == len(PARAM_CLASS_BOUNDARIES) + 1


# ---------------------------------------------------------------------------
# bucket_bits
# ---------------------------------------------------------------------------

class TestBucketBits:
    def test_4_stays_4(self):
        assert bucket_bits(4) == 4

    def test_8_stays_8(self):
        assert bucket_bits(8) == 8

    def test_16_stays_16(self):
        assert bucket_bits(16) == 16

    def test_32_stays_32(self):
        assert bucket_bits(32) == 32

    def test_3_snaps_to_4(self):
        assert bucket_bits(3) == 4

    def test_6_snaps_to_4(self):
        # |6-4|=2 == |6-8|=2; equidistant — min() returns the first canonical (4)
        assert bucket_bits(6) == 4

    def test_5_snaps_to_4(self):
        # |5-4|=1 < |5-8|=3
        assert bucket_bits(5) == 4

    def test_12_snaps_to_8(self):
        assert bucket_bits(12) == 8

    def test_24_snaps_to_16(self):
        # |24-16|=8 == |24-32|=8; equidistant — min() returns the first canonical (16)
        assert bucket_bits(24) == 16

    def test_1_snaps_to_4(self):
        assert bucket_bits(1) == 4

    def test_64_snaps_to_32(self):
        assert bucket_bits(64) == 32


# ---------------------------------------------------------------------------
# DeviceFingerprint
# ---------------------------------------------------------------------------

class TestDeviceFingerprint:
    def test_from_values_basic(self):
        fp = DeviceFingerprint.from_values(
            available_mb=16_000, backend="cuda", os_platform="linux"
        )
        assert fp.memory_tier == "8_24gb"
        assert fp.backend == "cuda"
        assert fp.os_platform == "linux"

    def test_auto_os_platform_detected(self):
        fp = DeviceFingerprint.from_values(available_mb=4_000, backend="apple_silicon")
        assert fp.os_platform == sys.platform

    def test_frozen_cannot_be_mutated(self):
        fp = DeviceFingerprint.from_values(available_mb=4_000, backend="cuda")
        with pytest.raises((AttributeError, TypeError)):
            fp.backend = "rocm"  # type: ignore[misc]

    def test_hashable_usable_as_dict_key(self):
        fp = DeviceFingerprint.from_values(available_mb=4_000, backend="cuda")
        d = {fp: 42}
        assert d[fp] == 42

    def test_equal_fingerprints_same_hash(self):
        fp1 = DeviceFingerprint.from_values(4_096, "cuda", "linux")
        fp2 = DeviceFingerprint.from_values(4_096, "cuda", "linux")
        assert fp1 == fp2
        assert hash(fp1) == hash(fp2)

    def test_different_memory_tier_not_equal(self):
        fp1 = DeviceFingerprint.from_values(4_096, "cuda", "linux")
        fp2 = DeviceFingerprint.from_values(16_000, "cuda", "linux")
        assert fp1 != fp2

    def test_backend_string_preserved(self):
        fp = DeviceFingerprint.from_values(4_000, "apple_silicon", "darwin")
        assert fp.backend == "apple_silicon"


# ---------------------------------------------------------------------------
# ModelFingerprint
# ---------------------------------------------------------------------------

class TestModelFingerprint:
    def test_from_values_basic(self):
        fp = ModelFingerprint.from_values(model_params=8e9, model_bits=16)
        assert fp.param_class == "7_13b"
        assert fp.bits == 16

    def test_bits_normalised(self):
        # |6-4|=2 == |6-8|=2; equidistant — snaps to first canonical (4)
        fp = ModelFingerprint.from_values(model_params=3e9, model_bits=6)
        assert fp.bits == 4

    def test_frozen_cannot_be_mutated(self):
        fp = ModelFingerprint.from_values(8e9, 16)
        with pytest.raises((AttributeError, TypeError)):
            fp.bits = 4  # type: ignore[misc]

    def test_hashable_usable_as_dict_key(self):
        fp = ModelFingerprint.from_values(8e9, 16)
        d = {fp: "llama3"}
        assert d[fp] == "llama3"

    def test_equal_fingerprints_same_hash(self):
        fp1 = ModelFingerprint.from_values(8e9, 16)
        fp2 = ModelFingerprint.from_values(9e9, 16)  # same class: 7_13b
        assert fp1 == fp2

    def test_different_bits_not_equal(self):
        fp1 = ModelFingerprint.from_values(8e9, 16)
        fp2 = ModelFingerprint.from_values(8e9, 4)
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# ConfigAction
# ---------------------------------------------------------------------------

class TestConfigAction:
    def test_basic_construction(self):
        action = ConfigAction(batch_size=4, lora_rank=16, seq_length=2048)
        assert action.batch_size == 4
        assert action.lora_rank == 16
        assert action.seq_length == 2048
        assert action.max_num_seqs == 0  # default

    def test_with_max_num_seqs(self):
        action = ConfigAction(batch_size=1, lora_rank=0, seq_length=8192, max_num_seqs=64)
        assert action.max_num_seqs == 64

    def test_frozen_cannot_be_mutated(self):
        action = ConfigAction(batch_size=4, lora_rank=16, seq_length=2048)
        with pytest.raises((AttributeError, TypeError)):
            action.batch_size = 8  # type: ignore[misc]

    def test_hashable_usable_as_dict_key(self):
        action = ConfigAction(batch_size=4, lora_rank=16, seq_length=2048)
        q = {action: 0.5}
        assert q[action] == 0.5

    def test_equal_actions_same_hash(self):
        a1 = ConfigAction(4, 16, 2048)
        a2 = ConfigAction(4, 16, 2048)
        assert a1 == a2
        assert hash(a1) == hash(a2)

    def test_different_seq_length_not_equal(self):
        a1 = ConfigAction(4, 16, 2048)
        a2 = ConfigAction(4, 16, 4096)
        assert a1 != a2

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            ConfigAction(batch_size=0, lora_rank=16, seq_length=2048)

    def test_invalid_seq_length_raises(self):
        with pytest.raises(ValueError, match="seq_length"):
            ConfigAction(batch_size=4, lora_rank=16, seq_length=0)

    def test_negative_lora_rank_raises(self):
        with pytest.raises(ValueError, match="lora_rank"):
            ConfigAction(batch_size=4, lora_rank=-1, seq_length=2048)

    def test_negative_max_num_seqs_raises(self):
        with pytest.raises(ValueError, match="max_num_seqs"):
            ConfigAction(batch_size=4, lora_rank=0, seq_length=2048, max_num_seqs=-1)

    def test_zero_lora_rank_valid(self):
        # lora_rank=0 means full fine-tune or not applicable
        action = ConfigAction(batch_size=4, lora_rank=0, seq_length=2048)
        assert action.lora_rank == 0


# ---------------------------------------------------------------------------
# StateKey
# ---------------------------------------------------------------------------

class TestStateKey:
    def _make_key(self, avail_mb=16_000, backend="cuda", params=8e9, bits=16):
        return StateKey.from_values(
            available_mb=avail_mb,
            backend=backend,
            model_params=params,
            model_bits=bits,
            os_platform="linux",
        )

    def test_from_values_returns_state_key(self):
        key = self._make_key()
        assert isinstance(key, StateKey)

    def test_device_and_model_accessible(self):
        key = self._make_key()
        assert isinstance(key.device, DeviceFingerprint)
        assert isinstance(key.model, ModelFingerprint)

    def test_hashable_usable_as_dict_key(self):
        key = self._make_key()
        q_table = {key: {}}
        assert key in q_table

    def test_equal_keys_same_hash(self):
        k1 = self._make_key()
        k2 = self._make_key()
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_different_backend_not_equal(self):
        k1 = self._make_key(backend="cuda")
        k2 = self._make_key(backend="apple_silicon")
        assert k1 != k2

    def test_q_table_workflow(self):
        """StateKey + ConfigAction can be used together as a Q-table."""
        key = self._make_key()
        action = ConfigAction(batch_size=4, lora_rank=16, seq_length=2048)
        q_table: dict[StateKey, dict[ConfigAction, float]] = {}
        q_table.setdefault(key, {})[action] = 0.7
        assert q_table[key][action] == pytest.approx(0.7)

    def test_namedtuple_unpacking(self):
        key = self._make_key()
        device, model = key
        assert device == key.device
        assert model == key.model
