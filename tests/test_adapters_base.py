"""Tests for memory_guard.adapters.base — model introspection and optional_import."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memory_guard.adapters.base import introspect_model, optional_import


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param(numel: int) -> MagicMock:
    p = MagicMock()
    p.numel.return_value = numel
    return p


def _make_model(
    *,
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_hidden_layers: int = 32,
    num_key_value_heads: int | None = 8,
    quantization_config: object = None,
    dtype: str = "torch.float16",
    total_params: int = 7_000_000_000,
) -> MagicMock:
    """Return a MagicMock that looks like a HuggingFace model.

    Uses SimpleNamespace for ``config`` so that attribute absence is real
    (getattr fallback works correctly when num_key_value_heads is omitted).
    """
    config_kwargs: dict = dict(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        quantization_config=quantization_config,
    )
    if num_key_value_heads is not None:
        config_kwargs["num_key_value_heads"] = num_key_value_heads

    model = MagicMock()
    model.config = SimpleNamespace(**config_kwargs)
    model.dtype = dtype
    model.parameters.return_value = [_make_param(total_params)]
    return model


# ---------------------------------------------------------------------------
# introspect_model — happy path
# ---------------------------------------------------------------------------


class TestIntrospectModelHappyPath:
    def test_basic_fields_returned(self) -> None:
        model = _make_model(
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=32,
            num_key_value_heads=8,
            dtype="torch.float16",
            total_params=7_000_000_000,
        )
        result = introspect_model(model)

        assert result["hidden_size"] == 4096
        assert result["num_attention_heads"] == 32
        assert result["num_hidden_layers"] == 32
        assert result["num_key_value_heads"] == 8
        assert result["model_bits"] == 16
        assert result["num_parameters"] == 7_000_000_000

    def test_bfloat16_dtype_gives_16_bits(self) -> None:
        model = _make_model(dtype="torch.bfloat16")
        assert introspect_model(model)["model_bits"] == 16

    def test_float32_dtype_gives_32_bits(self) -> None:
        model = _make_model(dtype="torch.float32")
        assert introspect_model(model)["model_bits"] == 32

    def test_unknown_dtype_defaults_to_32_bits(self) -> None:
        model = _make_model(dtype="torch.float64")
        assert introspect_model(model)["model_bits"] == 32


# ---------------------------------------------------------------------------
# introspect_model — num_key_value_heads fallback
# ---------------------------------------------------------------------------


class TestNumKeyValueHeadsFallback:
    def test_falls_back_to_num_attention_heads_when_absent(self) -> None:
        """If num_key_value_heads is not on config, use num_attention_heads."""
        model = _make_model(num_attention_heads=32, num_key_value_heads=None)
        result = introspect_model(model)
        assert result["num_key_value_heads"] == 32

    def test_explicit_gqa_value_is_preserved(self) -> None:
        """Grouped-query attention with fewer KV heads than query heads."""
        model = _make_model(num_attention_heads=32, num_key_value_heads=8)
        result = introspect_model(model)
        assert result["num_key_value_heads"] == 8


# ---------------------------------------------------------------------------
# introspect_model — quantization detection
# ---------------------------------------------------------------------------


class TestBitsInference:
    def test_bnb_4bit_load_in_4bit_flag(self) -> None:
        qc = SimpleNamespace(load_in_4bit=True, load_in_8bit=False, quant_type="nf4")
        model = _make_model(quantization_config=qc, dtype="torch.float32")
        assert introspect_model(model)["model_bits"] == 4

    def test_bnb_4bit_via_quant_type_nf4(self) -> None:
        qc = SimpleNamespace(load_in_4bit=False, load_in_8bit=False, quant_type="nf4")
        model = _make_model(quantization_config=qc, dtype="torch.float32")
        assert introspect_model(model)["model_bits"] == 4

    def test_bnb_4bit_via_quant_type_fp4(self) -> None:
        qc = SimpleNamespace(load_in_4bit=False, load_in_8bit=False, quant_type="fp4")
        model = _make_model(quantization_config=qc, dtype="torch.float32")
        assert introspect_model(model)["model_bits"] == 4

    def test_bnb_8bit(self) -> None:
        qc = SimpleNamespace(load_in_4bit=False, load_in_8bit=True, quant_type="")
        model = _make_model(quantization_config=qc, dtype="torch.float32")
        assert introspect_model(model)["model_bits"] == 8

    def test_no_quantization_falls_through_to_dtype(self) -> None:
        model = _make_model(quantization_config=None, dtype="torch.float16")
        assert introspect_model(model)["model_bits"] == 16

    def test_quantization_config_none_fp32(self) -> None:
        model = _make_model(quantization_config=None, dtype="torch.float32")
        assert introspect_model(model)["model_bits"] == 32


# ---------------------------------------------------------------------------
# introspect_model — parameter counting
# ---------------------------------------------------------------------------


class TestParameterCounting:
    def test_single_param(self) -> None:
        model = _make_model(total_params=1_000_000)
        assert introspect_model(model)["num_parameters"] == 1_000_000

    def test_multi_param_sum(self) -> None:
        """Verify that all parameter tensors are summed."""
        model = _make_model()
        model.parameters.return_value = [
            _make_param(3_000_000_000),
            _make_param(4_000_000_000),
        ]
        assert introspect_model(model)["num_parameters"] == 7_000_000_000

    def test_zero_params(self) -> None:
        model = _make_model()
        model.parameters.return_value = []
        assert introspect_model(model)["num_parameters"] == 0


# ---------------------------------------------------------------------------
# optional_import
# ---------------------------------------------------------------------------


class TestOptionalImport:
    def test_imports_existing_module(self) -> None:
        mod = optional_import("os", "hf")
        import os

        assert mod is os

    def test_imports_nested_module(self) -> None:
        mod = optional_import("os.path", "hf")
        import os.path

        assert mod is os.path

    def test_raises_import_error_for_missing_module(self) -> None:
        with pytest.raises(ImportError, match="pip install ml-memguard\\[hf\\]"):
            optional_import("_nonexistent_module_xyz", "hf")

    def test_error_message_contains_module_name(self) -> None:
        with pytest.raises(ImportError, match="_nonexistent_module_xyz"):
            optional_import("_nonexistent_module_xyz", "hf")

    def test_error_message_contains_extra_name(self) -> None:
        with pytest.raises(ImportError, match="ml-memguard\\[unsloth\\]"):
            optional_import("_nonexistent_module_xyz", "unsloth")
