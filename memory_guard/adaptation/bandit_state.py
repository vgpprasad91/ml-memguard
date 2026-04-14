"""State and action encoding for the RL contextual bandit (v0.4).

Provides the pure data layer — no policy logic, no I/O.

Components
----------
DeviceFingerprint
    Hashable device identity: bucketed available memory × backend × platform.
    Two runs on the same machine with similar free memory hash to the same key
    so the Q-table generalises across sessions without requiring exact equality.

ModelFingerprint
    Hashable model identity: bucketed parameter count × quantisation bits.
    A 7B model at fp16 and a 7B model at 4-bit are distinct actions spaces
    (different KV cache and weight footprints) so they get separate Q-entries.

ConfigAction
    Frozen, hashable discrete action: (batch_size, lora_rank, seq_length,
    max_num_seqs).  ``max_num_seqs=0`` signals "not applicable" for
    training-only configs.

StateKey
    Named tuple of (DeviceFingerprint, ModelFingerprint) — the Q-table
    row key.  Kept as a NamedTuple (not a dataclass) so it is natively
    hashable and usable as a dict key without extra machinery.

Bucket boundaries
-----------------
All bucket boundaries are module-level constants so callers can inspect them
and tests can reason about edge cases without magic numbers.

    Memory tiers   :  <8 GB · 8–24 GB · 24–48 GB · 48–80 GB · 80+ GB
    Parameter classes: ≤1 B · 1–7 B · 7–13 B · 13–35 B · 35+ B
    Bits (normalised): 4 · 8 · 16 · 32
"""

from __future__ import annotations

import platform as _platform_mod
import sys
from dataclasses import dataclass
from typing import NamedTuple, Optional

# ---------------------------------------------------------------------------
# Bucket boundary constants (MB / parameter counts)
# ---------------------------------------------------------------------------

#: Available-memory tier boundaries in MB.
MEMORY_TIER_BOUNDARIES_MB: tuple[float, ...] = (
    8_192.0,    # 8 GB
    24_576.0,   # 24 GB
    49_152.0,   # 48 GB
    81_920.0,   # 80 GB
)

#: Memory tier labels, aligned with MEMORY_TIER_BOUNDARIES_MB.
MEMORY_TIER_LABELS: tuple[str, ...] = (
    "sub_8gb",
    "8_24gb",
    "24_48gb",
    "48_80gb",
    "80plus_gb",
)

#: Parameter-count class boundaries (raw parameter counts, not billions).
PARAM_CLASS_BOUNDARIES: tuple[float, ...] = (
    1e9,    # 1 B
    7e9,    # 7 B
    13e9,   # 13 B
    35e9,   # 35 B
)

#: Parameter class labels, aligned with PARAM_CLASS_BOUNDARIES.
PARAM_CLASS_LABELS: tuple[str, ...] = (
    "sub_1b",
    "1_7b",
    "7_13b",
    "13_35b",
    "35plus_b",
)

#: Recognised quantisation bit-widths (everything else snaps to the nearest).
_CANONICAL_BITS: tuple[int, ...] = (4, 8, 16, 32)


# ---------------------------------------------------------------------------
# Pure bucketing functions
# ---------------------------------------------------------------------------


def bucket_memory(available_mb: float) -> str:
    """Return the memory-tier label for *available_mb*.

    Args:
        available_mb: Free / budgeted GPU or unified memory in megabytes.

    Returns:
        One of ``MEMORY_TIER_LABELS`` — always a non-empty string.

    Examples:
        >>> bucket_memory(4_096)
        'sub_8gb'
        >>> bucket_memory(16_000)
        '8_24gb'
        >>> bucket_memory(40_960)
        '24_48gb'
        >>> bucket_memory(65_536)
        '48_80gb'
        >>> bucket_memory(100_000)
        '80plus_gb'
    """
    for boundary, label in zip(MEMORY_TIER_BOUNDARIES_MB, MEMORY_TIER_LABELS):
        if available_mb < boundary:
            return label
    return MEMORY_TIER_LABELS[-1]


def bucket_params(model_params: float) -> str:
    """Return the parameter-class label for *model_params*.

    Args:
        model_params: Raw parameter count (not in billions).

    Returns:
        One of ``PARAM_CLASS_LABELS`` — always a non-empty string.

    Examples:
        >>> bucket_params(500e6)
        'sub_1b'
        >>> bucket_params(3e9)
        '1_7b'
        >>> bucket_params(8e9)
        '7_13b'
        >>> bucket_params(20e9)
        '13_35b'
        >>> bucket_params(70e9)
        '35plus_b'
    """
    for boundary, label in zip(PARAM_CLASS_BOUNDARIES, PARAM_CLASS_LABELS):
        if model_params < boundary:
            return label
    return PARAM_CLASS_LABELS[-1]


def bucket_bits(model_bits: int) -> int:
    """Normalise *model_bits* to the nearest canonical bit-width.

    Canonical widths: 4, 8, 16, 32.  Any value outside this set snaps
    to the nearest canonical.  Values ≤ 4 snap to 4; values ≥ 32 snap
    to 32.

    Args:
        model_bits: Raw quantisation bits (e.g. 3, 4, 6, 8, 16, 32).

    Returns:
        One of ``{4, 8, 16, 32}``.

    Examples:
        >>> bucket_bits(4)
        4
        >>> bucket_bits(6)
        8
        >>> bucket_bits(16)
        16
        >>> bucket_bits(3)
        4
    """
    return min(_CANONICAL_BITS, key=lambda b: abs(b - model_bits))


# ---------------------------------------------------------------------------
# DeviceFingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceFingerprint:
    """Hashable identity for a device + memory configuration.

    Two sessions on the same machine with available_mb in the same tier
    produce the same fingerprint, so the Q-table generalises across
    sessions without requiring exact memory equality.

    Attributes:
        memory_tier: Bucketed available memory (one of MEMORY_TIER_LABELS).
        backend:     String form of the Backend enum value, e.g. "cuda",
                     "apple_silicon".  Falls back to "unknown".
        os_platform: ``sys.platform`` value: "darwin", "linux", "win32", etc.
    """

    memory_tier: str
    backend: str
    os_platform: str

    @classmethod
    def from_values(
        cls,
        available_mb: float,
        backend: str,
        os_platform: Optional[str] = None,
    ) -> "DeviceFingerprint":
        """Construct a fingerprint from raw values.

        Args:
            available_mb: Free / budgeted memory in MB.
            backend:      Backend enum value string or any identifier string.
            os_platform:  ``sys.platform`` string; auto-detected if None.

        Returns:
            A frozen ``DeviceFingerprint``.
        """
        return cls(
            memory_tier=bucket_memory(available_mb),
            backend=str(backend),
            os_platform=os_platform if os_platform is not None else sys.platform,
        )


# ---------------------------------------------------------------------------
# ModelFingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFingerprint:
    """Hashable identity for a model architecture.

    Attributes:
        param_class: Bucketed parameter count (one of PARAM_CLASS_LABELS).
        bits:        Normalised quantisation bit-width (4 / 8 / 16 / 32).
    """

    param_class: str
    bits: int

    @classmethod
    def from_values(
        cls,
        model_params: float,
        model_bits: int,
    ) -> "ModelFingerprint":
        """Construct a fingerprint from raw values.

        Args:
            model_params: Raw parameter count (not in billions).
            model_bits:   Quantisation bit-width; normalised to 4/8/16/32.

        Returns:
            A frozen ``ModelFingerprint``.
        """
        return cls(
            param_class=bucket_params(model_params),
            bits=bucket_bits(model_bits),
        )


# ---------------------------------------------------------------------------
# ConfigAction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigAction:
    """A discrete configuration action the policy can recommend.

    Frozen and hashable so it can be used as a Q-table key.

    Attributes:
        batch_size:    Per-device training batch size.
        lora_rank:     LoRA rank (0 = full fine-tune or not applicable).
        seq_length:    Sequence length in tokens.
        max_num_seqs:  Max concurrent inference sequences (0 = not applicable
                       for training-only configs).
    """

    batch_size: int
    lora_rank: int
    seq_length: int
    max_num_seqs: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {self.batch_size}")
        if self.lora_rank < 0:
            raise ValueError(f"lora_rank must be ≥ 0, got {self.lora_rank}")
        if self.seq_length < 1:
            raise ValueError(f"seq_length must be ≥ 1, got {self.seq_length}")
        if self.max_num_seqs < 0:
            raise ValueError(f"max_num_seqs must be ≥ 0, got {self.max_num_seqs}")


# ---------------------------------------------------------------------------
# StateKey
# ---------------------------------------------------------------------------


class StateKey(NamedTuple):
    """Q-table lookup key: (DeviceFingerprint, ModelFingerprint).

    Using a NamedTuple (not a dataclass) gives native hashability,
    tuple unpacking, and readable ``repr`` with no extra machinery.

    Example:
        key = StateKey(device=dev_fp, model=mdl_fp)
        q_table[key][action] = 0.0
    """

    device: DeviceFingerprint
    model: ModelFingerprint

    @classmethod
    def from_values(
        cls,
        available_mb: float,
        backend: str,
        model_params: float,
        model_bits: int,
        os_platform: Optional[str] = None,
    ) -> "StateKey":
        """Convenience constructor from raw numeric values.

        Args:
            available_mb:  Free / budgeted memory in MB.
            backend:       Backend string (e.g. ``Backend.CUDA.value``).
            model_params:  Raw parameter count.
            model_bits:    Quantisation bit-width.
            os_platform:   ``sys.platform``; auto-detected if None.

        Returns:
            A ``StateKey`` ready to use as a Q-table key.
        """
        device = DeviceFingerprint.from_values(
            available_mb=available_mb,
            backend=backend,
            os_platform=os_platform,
        )
        model = ModelFingerprint.from_values(
            model_params=model_params,
            model_bits=model_bits,
        )
        return cls(device=device, model=model)
