"""Preflight estimation — predict memory before a job starts."""
from .estimator import (
    InferenceServingEstimate,
    MemoryEstimate,
    ModelSpec,
    TrainSpec,
    FinetuneMethod,
    ModelArch,
    estimate_training_memory,
    estimate_inference_memory,
    estimate_serving_memory,
)
from .downgrade import DowngradeResult, auto_downgrade

__all__ = [
    "InferenceServingEstimate",
    "MemoryEstimate",
    "ModelSpec",
    "TrainSpec",
    "FinetuneMethod",
    "ModelArch",
    "estimate_training_memory",
    "estimate_inference_memory",
    "estimate_serving_memory",
    "DowngradeResult",
    "auto_downgrade",
]
