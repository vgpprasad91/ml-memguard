"""Tabular Q-learner for the RL contextual bandit (v0.4).

Implements ``BanditPolicy`` — a per-device epsilon-greedy Q-table that
learns which ``ConfigAction`` maximises throughput without OOM for a given
``StateKey``.

Design summary
--------------
Q-table
    ``Dict[StateKey, Dict[ConfigAction, float]]``.  Unknown entries default
    to 0.0.  Each ``StateKey`` (device × model fingerprint) has an independent
    row, so a policy trained on an A100 can never contaminate an M2 Max entry.

select_action
    Epsilon-greedy.  Returns ``None`` on:
      - empty candidates list
      - cold start (state not yet seen by this policy)
      - exploration branch (random() < epsilon)
    Otherwise returns the ``ConfigAction`` with the highest Q-value across
    the supplied ``candidates``.  Unseen candidate actions default to 0.0.

update
    Incremental Q-update rule (exponential moving average toward the target):

        Q[s][a] += alpha * (reward - Q[s][a])

    After each update, epsilon decays:

        epsilon = max(epsilon_floor, epsilon * epsilon_decay)

    This ensures exploration never fully stops — new model architectures and
    device configurations will always get some binary-search fallback.

Persistence
    ``save(path)`` serialises the full Q-table as JSON.  ``load(path)``
    deserialises it; returns a fresh policy silently if the file is absent
    or corrupted (cold-start fallback, never raises).  The default path is
    ``~/.memory-guard/rl_policy.json``.  Writes are atomic (tempfile +
    os.replace) so a crash during save never leaves a half-written file.

Serialisation format
    StateKey  → pipe-separated string: ``"memory_tier|backend|os|param_class|bits"``
    ConfigAction → pipe-separated string: ``"batch_size|lora_rank|seq_length|max_num_seqs"``
    These strings become JSON object keys, keeping the file human-readable.
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from .bandit_state import (
    ConfigAction,
    DeviceFingerprint,
    ModelFingerprint,
    StateKey,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_POLICY_PATH: Path = Path.home() / ".memory-guard" / "rl_policy.json"

_FORMAT_VERSION: str = "0.4.0"

_DEFAULT_EPSILON: float = 0.9
_DEFAULT_EPSILON_DECAY: float = 0.995
_DEFAULT_EPSILON_FLOOR: float = 0.05
_DEFAULT_ALPHA: float = 0.1


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _state_key_to_str(key: StateKey) -> str:
    """Stable pipe-separated string representation of a StateKey."""
    d, m = key.device, key.model
    return f"{d.memory_tier}|{d.backend}|{d.os_platform}|{m.param_class}|{m.bits}"


def _str_to_state_key(s: str) -> StateKey:
    """Reconstruct a StateKey from its pipe-separated string form."""
    parts = s.split("|")
    if len(parts) != 5:
        raise ValueError(f"Cannot parse StateKey from {s!r}")
    memory_tier, backend, os_platform, param_class, bits_str = parts
    device = DeviceFingerprint(
        memory_tier=memory_tier,
        backend=backend,
        os_platform=os_platform,
    )
    model = ModelFingerprint(
        param_class=param_class,
        bits=int(bits_str),
    )
    return StateKey(device=device, model=model)


def _action_to_str(action: ConfigAction) -> str:
    """Stable pipe-separated string representation of a ConfigAction."""
    return f"{action.batch_size}|{action.lora_rank}|{action.seq_length}|{action.max_num_seqs}"


def _str_to_action(s: str) -> ConfigAction:
    """Reconstruct a ConfigAction from its pipe-separated string form."""
    parts = s.split("|")
    if len(parts) != 4:
        raise ValueError(f"Cannot parse ConfigAction from {s!r}")
    batch_size, lora_rank, seq_length, max_num_seqs = parts
    return ConfigAction(
        batch_size=int(batch_size),
        lora_rank=int(lora_rank),
        seq_length=int(seq_length),
        max_num_seqs=int(max_num_seqs),
    )


# ---------------------------------------------------------------------------
# Cloud merge helper
# ---------------------------------------------------------------------------

def _merge_cloud_policy(policy: "BanditPolicy") -> None:
    """Merge cloud Q-table entries into *policy* in-place.

    Strategy: weighted average by ``num_updates`` for entries that exist in
    both; cloud-only entries are added directly (free knowledge on cold start).
    Never raises — cloud failures are silently ignored.
    """
    try:
        from .cloud import download_policy
        cloud_data = download_policy()
        if not cloud_data or "q_table" not in cloud_data:
            return

        cloud_updates: int = int(cloud_data.get("num_updates", 0))
        local_updates: int = policy.num_updates
        total: int = cloud_updates + local_updates

        for sk_str, actions_raw in cloud_data["q_table"].items():
            try:
                sk = _str_to_state_key(sk_str)
            except (ValueError, KeyError):
                continue
            if not isinstance(actions_raw, dict):
                continue

            for a_str, cloud_q in actions_raw.items():
                try:
                    action = _str_to_action(a_str)
                    cloud_q = float(cloud_q)
                except (ValueError, KeyError):
                    continue

                local_q = policy._q.get(sk, {}).get(action)
                if local_q is None:
                    # Cloud has experience we don't — adopt it
                    policy._q.setdefault(sk, {})[action] = cloud_q
                elif total > 0:
                    # Weighted average: more updates → more influence
                    w_local = local_updates / total
                    w_cloud = cloud_updates / total
                    policy._q[sk][action] = w_local * local_q + w_cloud * cloud_q

        logger.debug(
            "[memory-guard] Merged cloud policy: %d states now loaded.",
            policy.num_states,
        )
    except Exception as exc:
        logger.debug("[memory-guard] Cloud policy merge skipped: %s", exc)


# ---------------------------------------------------------------------------
# BanditPolicy
# ---------------------------------------------------------------------------


class BanditPolicy:
    """Epsilon-greedy tabular Q-learner for per-device config optimisation.

    Usage::

        policy = BanditPolicy.load()          # load or start fresh
        candidates = [ConfigAction(...), ...]

        action = policy.select_action(state_key, candidates)
        if action is None:
            # cold start or exploration — fall back to binary search
            action = binary_search_preflight(...)

        # ... run training with action ...

        reward = record_training_result(...).combined
        policy.update(state_key, action, reward)
        policy.save()
    """

    def __init__(
        self,
        epsilon: float = _DEFAULT_EPSILON,
        epsilon_decay: float = _DEFAULT_EPSILON_DECAY,
        epsilon_floor: float = _DEFAULT_EPSILON_FLOOR,
        alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        """
        Args:
            epsilon:        Initial exploration rate (0–1).  0.9 = heavy
                            exploration on first encounters.
            epsilon_decay:  Multiplicative decay applied after each update.
                            0.995 ≈ 1 % decay per 200 updates.
            epsilon_floor:  Minimum exploration rate.  Never decays below
                            this value so novel architectures always get
                            some binary-search fallback.
            alpha:          Q-learning rate (0–1).  0.1 gives stable
                            convergence; higher values track reward faster
                            but are noisier.
        """
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_floor: float = epsilon_floor
        self.alpha: float = alpha
        self._q: Dict[StateKey, Dict[ConfigAction, float]] = {}
        self.num_updates: int = 0

    # ------------------------------------------------------------------
    # Q-table access
    # ------------------------------------------------------------------

    def q_value(self, state_key: StateKey, action: ConfigAction) -> float:
        """Return the Q-value for *(state_key, action)*, defaulting to 0.0."""
        return self._q.get(state_key, {}).get(action, 0.0)

    @property
    def num_states(self) -> int:
        """Number of distinct StateKeys seen so far."""
        return len(self._q)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_action(
        self,
        state_key: StateKey,
        candidates: List[ConfigAction],
        epsilon: Optional[float] = None,
    ) -> Optional[ConfigAction]:
        """Choose an action epsilon-greedily.

        Returns ``None`` in any of these cases, prompting the caller to
        fall back to binary-search estimation:

        - ``candidates`` is empty
        - Cold start: ``state_key`` has never been seen by this policy
        - Exploration branch: ``random() < epsilon``

        Returns the ``ConfigAction`` with the highest Q-value (across
        ``candidates``) on the exploitation branch.  Unseen candidates
        default to Q = 0.0.

        Args:
            state_key:   The current (device, model) state.
            candidates:  List of ``ConfigAction`` objects the policy may
                         choose from.  Must be non-empty for exploitation
                         to fire.
            epsilon:     Override the policy's current ``epsilon``; useful
                         for testing or annealing schedules.

        Returns:
            A ``ConfigAction`` from ``candidates``, or ``None``.
        """
        if not candidates:
            return None

        if state_key not in self._q:
            return None  # cold start — no prior experience

        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
            return None  # explore — let caller run binary search

        # Exploit: argmax Q[s][a] over candidates (0.0 for unseen actions)
        q_row = self._q[state_key]
        return max(candidates, key=lambda a: q_row.get(a, 0.0))

    def update(
        self,
        state_key: StateKey,
        action: ConfigAction,
        reward: float,
        alpha: Optional[float] = None,
    ) -> None:
        """Update the Q-table for *(state_key, action)* with *reward*.

        Applies the incremental Q-update rule::

            Q[s][a] += alpha * (reward - Q[s][a])

        Then decays ``epsilon``::

            epsilon = max(epsilon_floor, epsilon * epsilon_decay)

        Args:
            state_key: The (device, model) state observed.
            action:    The ``ConfigAction`` that was executed.
            reward:    Scalar reward (e.g. ``RewardSignal.combined``).
            alpha:     Learning rate override; uses ``self.alpha`` if None.
        """
        lr = self.alpha if alpha is None else alpha

        if state_key not in self._q:
            self._q[state_key] = {}

        q_old = self._q[state_key].get(action, 0.0)
        self._q[state_key][action] = q_old + lr * (reward - q_old)

        # Decay epsilon — exploration rate decreases as experience accumulates
        self.epsilon = max(self.epsilon_floor, self.epsilon * self.epsilon_decay)
        self.num_updates += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """Serialise the policy to *path* as JSON (atomic write).

        Creates parent directories as needed.  Fails silently with a
        warning on any I/O error so a save failure never crashes a
        training run.

        Args:
            path: Destination file.  Defaults to ``DEFAULT_POLICY_PATH``.
        """
        target = Path(path) if path is not None else DEFAULT_POLICY_PATH

        payload: dict = {
            "version": _FORMAT_VERSION,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_floor": self.epsilon_floor,
            "alpha": self.alpha,
            "num_updates": self.num_updates,
            "q_table": {
                _state_key_to_str(sk): {
                    _action_to_str(a): q
                    for a, q in actions.items()
                }
                for sk, actions in self._q.items()
            },
        }

        try:
            parent = target.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)

            fd, tmp_path = tempfile.mkstemp(
                dir=str(parent), suffix=".tmp", prefix=".rl_"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, indent=2)
                os.chmod(tmp_path, 0o600)
                os.replace(tmp_path, str(target))
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception as exc:
            logger.warning("[memory-guard] Could not save RL policy: %s", exc)

        # Cloud sync — fire-and-forget; never blocks or raises
        try:
            from .cloud import upload_policy
            upload_policy(payload)
        except Exception:
            pass

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        **init_kwargs,
    ) -> "BanditPolicy":
        """Deserialise a policy from *path*.

        Returns a fresh ``BanditPolicy(**init_kwargs)`` (cold start) if the
        file is absent, unreadable, or structurally invalid.  Never raises.

        Args:
            path:        Source file.  Defaults to ``DEFAULT_POLICY_PATH``.
            **init_kwargs: Passed to ``__init__`` for a fresh policy on
                           cold start or as defaults overridden by the file.

        Returns:
            A ``BanditPolicy`` with the Q-table and epsilon from the file,
            or a fresh policy if loading fails.
        """
        source = Path(path) if path is not None else DEFAULT_POLICY_PATH

        if not source.exists():
            logger.debug(
                "[memory-guard] No RL policy at %s — starting fresh.", source
            )
            policy = cls(**init_kwargs)
            _merge_cloud_policy(policy)
            return policy

        try:
            with open(source) as f:
                data = json.load(f)

            if not isinstance(data, dict) or "q_table" not in data:
                raise ValueError("Missing 'q_table' key")

            policy = cls(
                epsilon=float(data.get("epsilon", init_kwargs.get("epsilon", _DEFAULT_EPSILON))),
                epsilon_decay=float(data.get("epsilon_decay", init_kwargs.get("epsilon_decay", _DEFAULT_EPSILON_DECAY))),
                epsilon_floor=float(data.get("epsilon_floor", init_kwargs.get("epsilon_floor", _DEFAULT_EPSILON_FLOOR))),
                alpha=float(data.get("alpha", init_kwargs.get("alpha", _DEFAULT_ALPHA))),
            )
            policy.num_updates = int(data.get("num_updates", 0))

            for sk_str, actions_raw in data["q_table"].items():
                try:
                    sk = _str_to_state_key(sk_str)
                except (ValueError, KeyError):
                    logger.debug(
                        "[memory-guard] Skipping unparseable StateKey %r", sk_str
                    )
                    continue

                if not isinstance(actions_raw, dict):
                    continue

                policy._q[sk] = {}
                for a_str, q_val in actions_raw.items():
                    try:
                        action = _str_to_action(a_str)
                        policy._q[sk][action] = float(q_val)
                    except (ValueError, KeyError):
                        logger.debug(
                            "[memory-guard] Skipping unparseable action %r", a_str
                        )

            logger.debug(
                "[memory-guard] Loaded RL policy: %d states, epsilon=%.3f",
                policy.num_states,
                policy.epsilon,
            )
            _merge_cloud_policy(policy)
            return policy

        except Exception as exc:
            logger.warning(
                "[memory-guard] Could not load RL policy from %s (%s) — starting fresh.",
                source,
                exc,
            )
            return cls(**init_kwargs)
