"""Online adaptation — learn and improve estimates over time."""
from .bandit import BanditPolicy, MIN_UPDATES_FOR_CONFIDENCE
from .bandit_state import ConfigAction, DeviceFingerprint, ModelFingerprint, StateKey
from .calibration import CalibrationStore, record_training_result
from .reward import RewardSignal, compute_reward

__all__ = [
    "BanditPolicy",
    "MIN_UPDATES_FOR_CONFIDENCE",
    "ConfigAction",
    "DeviceFingerprint",
    "ModelFingerprint",
    "StateKey",
    "CalibrationStore",
    "record_training_result",
    "RewardSignal",
    "compute_reward",
]
