# ADR 002 — QLoRA double-quantization: trust quantization_config or require explicit model_bits?

**Status**: Accepted  
**Decided**: 2026-04-10 (implemented in commit 2270202, PR 4)  
**Applies to**: `memory_guard.adapters.unsloth.guard_unsloth_model`,
               `memory_guard.adapters.base._infer_bits`

---

## Context

When a user loads a model with BnB double-quantization
(`bnb_4bit_use_double_quant=True`), two questions arise:

1. Should `introspect_model` trust `model.config.quantization_config` and report
   `model_bits=4`, or should it require the caller to pass `model_bits` explicitly?
2. How should the memory estimator account for the additional ~5 % saving that
   double-quant provides over standard NF4?

### What double-quantization actually saves

Standard NF4 (4-bit NormalFloat):
- Weights: 4 bits per parameter
- Quantization constants (one fp16 value per 64-weight block): ~0.25 bits/param

Double-quant additionally quantizes those constants to 8-bit:
- Saves roughly `(64 × 16 − 64 × 8) / (64 × 4)` ≈ **6.25 % of weight memory**
- In practice, measured savings cluster around 4–7 % depending on block size
  and model architecture.  A conservative 5 % floor is defensible.

Because `guard.preflight()` accepts only integer bits, there is no clean way to
express "3.8-bit effective precision".

### Option A — Require explicit model_bits (reject trust)

Force the caller to pass `model_bits` when using double-quant:

```python
# User must always spell this out:
safe = guard_unsloth_model(model, model_bits=4)
```

If `quantization_config` is present and `model_bits` is not passed, raise or
warn with `"double-quant detected: pass model_bits=4 explicitly"`.

**Advantage**: API is explicit; no hidden correction factors.

**Problems**:
- `model_bits=4` is always correct for BnB 4-bit regardless of double-quant.
  The double-quant correction is a *memory* correction, not a *bits* correction.
  Forcing the user to pass `model_bits=4` explicitly does not communicate that.
- It puts friction on the common Unsloth path where 90 % of users load with
  `load_in_4bit=True, use_double_quant=True` by default — the adapter would
  fail or warn every time.
- The intent of the adapter layer is precisely to remove the need for users to
  spell out introspectable values.

### Option B — Trust quantization_config, apply documented correction (chosen)

`_infer_bits` reads `quantization_config.load_in_4bit` → reports `model_bits=4`.
`guard_unsloth_model` separately detects `bnb_4bit_use_double_quant=True` and
multiplies `num_parameters` by `_DOUBLE_QUANT_CORRECTION = 0.95` before calling
`preflight`:

```python
if _is_double_quant(model):
    num_params = int(num_params * _DOUBLE_QUANT_CORRECTION)
```

The user can still override with `guard_unsloth_model(model, model_bits=4)` to
lock in the bits, or with any other `preflight_overrides` key.

**Advantages**:
- Zero friction on the default Unsloth path.
- `model_bits=4` is reported faithfully — it *is* 4-bit; the correction is a
  proxy for the reduced quantization-constant footprint, not a change in
  bit-width.
- The correction is documented in the module docstring and the ADR (this file);
  it is not a silent fudge factor.
- Auto-calibration (`guard.record_result()`) captures the residual error after
  the first training run and narrows it over subsequent runs.
- If the 5 % estimate is wrong for a specific model, the user can always override
  via `preflight_overrides` (e.g. `model_params=model.num_parameters()`).

**Trade-off**: the correction factor (0.95) is a conservative engineering
estimate, not a measurement.  It is bounded; worst case is a 5 % over-estimate
of weight memory, which is well within the 20 % safety margin of the default
`safety_ratio=0.80`.

## Decision

**Trust `quantization_config`, apply the 5 % parameter-count correction, and
document the override.**

Specifically:
- `_infer_bits` in `adapters/base.py` trusts `load_in_4bit=True` → returns 4.
- `guard_unsloth_model` in `adapters/unsloth.py` calls `_is_double_quant()` and
  applies `_DOUBLE_QUANT_CORRECTION = 0.95` to `num_parameters` when detected.
- `model_bits` in the returned `SafeConfig` is still 4 (the actual quantization
  width), only the effective parameter count used for the memory estimate changes.
- Both the module docstring and the README "Framework Adapters" section document
  the override path for users who need a different value.

## Consequences

- The adapter is frictionless on the default `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
  workflow with zero user configuration.
- `_DOUBLE_QUANT_CORRECTION` is a module-level constant in `unsloth.py`; it is
  the single place to update if future measurement data suggests a better value.
- Auto-calibration will correct residual estimation error automatically after
  3+ training runs on a given device, making the hard-coded constant progressively
  less important.
- `guard_trainer` and the base `introspect_model` are unaffected by this decision;
  the correction is Unsloth-specific and lives only in `guard_unsloth_model`.

## Dependency graph note

These two decisions (ADR 001 and ADR 002) blocked PRs 3 and 4 respectively.
The PR dependency order was:

```
PR 1 (base introspection)
  └─ PR 2 (HF callback, monitor-only)
       └─ PR 3 (HF mid-training downgrade) ← ADR 001
            └─ PR 4 (Unsloth wrapper)       ← ADR 002
                 └─ PR 5 (docs, v0.2.0)
```

PRs 2 and 4 could technically have been parallelised after PR 1, but were kept
sequential so PR 4's `guard_sft_trainer` could import the fully-implemented
`MemoryGuardCallback` (including the downgrade logic from PR 3) rather than a
stub.
