"""memory_guard.adapters — framework-specific adapter modules.

Each sub-module wraps a training framework (HuggingFace Transformers,
Unsloth, …) and exposes a uniform interface to the rest of memory-guard.
"""

from .base import introspect_model, optional_import

__all__ = ["introspect_model", "optional_import"]
