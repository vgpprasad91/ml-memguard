"""Base adapter utilities: model introspection and optional-dependency helpers.

No framework imports at module level — everything is lazy so that the core
package stays importable without any optional dependencies installed.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict


def optional_import(name: str, extra: str) -> Any:
    """Import *name* or raise a clear install hint.

    Args:
        name:  The Python module to import (e.g. ``"transformers"``).
        extra: The ml-memguard extras key (e.g. ``"hf"`` or ``"unsloth"``).

    Returns:
        The imported module object.

    Raises:
        ImportError: with a ``pip install ml-memguard[<extra>]`` message when
            the module is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"{name!r} is required for this feature. "
            f"Install it with:  pip install ml-memguard[{extra}]"
        ) from exc


def introspect_model(model: Any) -> Dict[str, Any]:
    """Read architecture metadata from a pretrained model.

    Inspects ``model.config`` for architecture hyperparameters and infers
    the effective precision from the quantization config (BitsAndBytes 4/8-bit)
    or from ``model.dtype`` (fp16/bf16 → 16, fp32 → 32).

    No framework import is performed; the function works against any object
    that provides the expected attributes.

    Args:
        model: A HuggingFace-style model with a ``.config`` attribute and a
            ``.parameters()`` iterator whose elements have a ``.numel()``
            method.

    Returns:
        A :class:`dict` with the following keys:

        ============== ======= ================================================
        Key            Type    Description
        ============== ======= ================================================
        hidden_size    int     ``model.config.hidden_size``
        num_attention_heads  int  ``model.config.num_attention_heads``
        num_hidden_layers    int  ``model.config.num_hidden_layers``
        num_key_value_heads  int  ``model.config.num_key_value_heads``, falls
                                  back to ``num_attention_heads`` when absent
        model_bits     int     Effective bits per weight (4, 8, 16, or 32)
        num_parameters int     Total parameter count across all tensors
        ============== ======= ================================================
    """
    config = model.config

    hidden_size: int = config.hidden_size
    num_attention_heads: int = config.num_attention_heads
    num_hidden_layers: int = config.num_hidden_layers
    num_key_value_heads: int = getattr(
        config, "num_key_value_heads", num_attention_heads
    )

    model_bits: int = _infer_bits(model)
    num_parameters: int = sum(p.numel() for p in model.parameters())

    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_key_value_heads,
        "model_bits": model_bits,
        "num_parameters": num_parameters,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_bits(model: Any) -> int:
    """Infer effective weight precision in bits.

    Priority order:
    1. ``model.config.quantization_config`` — BitsAndBytes 4-bit or 8-bit.
    2. ``model.dtype`` string — "float16"/"bfloat16" → 16, everything else → 32.
    """
    config = getattr(model, "config", None)
    if config is not None:
        qc = getattr(config, "quantization_config", None)
        if qc is not None:
            # BitsAndBytes 4-bit (load_in_4bit=True or quant_type nf4/fp4)
            if getattr(qc, "load_in_4bit", False):
                return 4
            quant_type: str = getattr(qc, "quant_type", "")
            if quant_type in ("nf4", "fp4"):
                return 4
            # BitsAndBytes 8-bit
            if getattr(qc, "load_in_8bit", False):
                return 8

    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        dtype_str = str(dtype)
        if "float16" in dtype_str or "bfloat16" in dtype_str:
            return 16

    return 32
