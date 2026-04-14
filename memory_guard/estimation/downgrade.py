"""Auto-downgrade training configuration to fit memory budget.

Iteratively reduces parameters in quality-preserving order:
  1. Enable gradient checkpointing  (reduces activation memory)
  2. Halve batch size               (compensate with grad accumulation)
  3. Halve sequence length           (shorter context)
  4. Halve LoRA rank                 (fewer trainable params)
  5. Halve LoRA layers               (fewer adapted layers)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from .estimator import MemoryEstimate, estimate_training_memory

logger = logging.getLogger(__name__)


@dataclass
class DowngradeResult:
    """Result of auto-downgrade process."""
    batch_size: int
    seq_length: int
    lora_rank: int
    lora_layers: int
    grad_checkpoint: bool
    grad_accumulation: int
    changes: list[str] = field(default_factory=list)
    final_estimate: Optional[MemoryEstimate] = None
    fits: bool = True


def auto_downgrade(
    budget_mb: float,
    model_params: int,
    model_bits: int = 4,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    num_layers: int = 32,
    batch_size: int = 4,
    seq_length: int = 2048,
    lora_rank: int = 32,
    lora_layers: int = 16,
    grad_checkpoint: bool = False,
    grad_accumulation: int = 1,
    optimizer: str = "adam",
    flash_attention: bool = True,
    lazy_evaluation: bool = False,
    max_iterations: int = 20,
    min_batch_size: int = 1,
    min_seq_length: int = 128,
    min_lora_rank: int = 4,
    min_lora_layers: int = 2,
) -> DowngradeResult:
    """Iteratively downgrade config until estimated memory fits budget.

    Args:
        budget_mb: Maximum memory budget in MB.
        model_params: Total model parameters.
        model_bits: Quantization bits.
        hidden_dim, num_heads, num_layers: Model architecture.
        batch_size, seq_length, lora_rank, lora_layers: Training config.
        grad_checkpoint, grad_accumulation, optimizer: Training options.
        max_iterations: Safety limit on downgrade iterations.
        min_*: Minimum values for each parameter.

    Returns:
        DowngradeResult with adjusted config and list of changes.
    """
    changes = []

    def _estimate():
        from .estimator import ModelSpec, TrainSpec
        return estimate_training_memory(
            model=ModelSpec(
                params=model_params, bits=model_bits,
                hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
            ),
            train=TrainSpec(
                batch_size=batch_size, seq_length=seq_length,
                lora_rank=lora_rank, lora_layers=lora_layers,
                optimizer=optimizer, grad_checkpoint=grad_checkpoint,
                grad_accumulation=grad_accumulation,
                flash_attention=flash_attention,
                lazy_evaluation=lazy_evaluation,
            ),
        )

    for _ in range(max_iterations):
        est = _estimate()
        if est.total_mb <= budget_mb:
            return DowngradeResult(
                batch_size=batch_size, seq_length=seq_length,
                lora_rank=lora_rank, lora_layers=lora_layers,
                grad_checkpoint=grad_checkpoint,
                grad_accumulation=grad_accumulation,
                changes=changes, final_estimate=est, fits=True,
            )

        # Step 1: Enable gradient checkpointing
        # Saves 1 - sqrt(lora_layers)/lora_layers of activation memory
        if not grad_checkpoint:
            grad_checkpoint = True
            ckpt_ratio = 1.0 - (math.sqrt(lora_layers) / max(lora_layers, 1))
            savings = est.activations_mb * ckpt_ratio
            changes.append(f"Enable gradient checkpointing (saves ~{savings:.0f}MB activations)")
            continue

        # Step 2: Halve batch size (biggest impact on activations + KV cache)
        if batch_size > min_batch_size:
            old = batch_size
            batch_size = max(min_batch_size, batch_size // 2)
            grad_accumulation *= 2  # Maintain effective batch size
            changes.append(f"Batch size {old} -> {batch_size} (grad_accum -> {grad_accumulation})")
            continue

        # Step 3: Halve sequence length
        if seq_length > min_seq_length:
            old = seq_length
            seq_length = max(min_seq_length, seq_length // 2)
            changes.append(f"Sequence length {old} -> {seq_length}")
            continue

        # Step 4: Halve LoRA rank
        if lora_rank > min_lora_rank:
            old = lora_rank
            lora_rank = max(min_lora_rank, lora_rank // 2)
            changes.append(f"LoRA rank {old} -> {lora_rank}")
            continue

        # Step 5: Halve LoRA layers
        if lora_layers > min_lora_layers:
            old = lora_layers
            lora_layers = max(min_lora_layers, lora_layers // 2)
            changes.append(f"LoRA layers {old} -> {lora_layers}")
            continue

        # Cannot downgrade further
        est = _estimate()
        changes.append(
            f"WARNING: Cannot fit in budget "
            f"({est.total_mb:.0f}MB needed > {budget_mb:.0f}MB available). "
            f"Consider a smaller model or more memory."
        )
        return DowngradeResult(
            batch_size=batch_size, seq_length=seq_length,
            lora_rank=lora_rank, lora_layers=lora_layers,
            grad_checkpoint=grad_checkpoint,
            grad_accumulation=grad_accumulation,
            changes=changes, final_estimate=est, fits=False,
        )

    est = _estimate()
    return DowngradeResult(
        batch_size=batch_size, seq_length=seq_length,
        lora_rank=lora_rank, lora_layers=lora_layers,
        grad_checkpoint=grad_checkpoint,
        grad_accumulation=grad_accumulation,
        changes=changes, final_estimate=est,
        fits=est.total_mb <= budget_mb,
    )
