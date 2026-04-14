"""Proactive memory estimation for ML training.

Calculates peak memory BEFORE training starts. Supports:
  - Standard transformers (Llama, Mistral, Qwen, Phi)
  - Mixture of Experts (Mixtral, DeepSeek, Qwen-MoE)
  - Multi-modal models (vision-language, audio-language)
  - Full fine-tuning, LoRA, QLoRA (double quantization), DoRA
  - Gradient checkpointing, mixed precision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FinetuneMethod(Enum):
    LORA = "lora"
    QLORA = "qlora"         # LoRA on quantized base + NF4 double quant
    DORA = "dora"           # Weight-decomposed LoRA (slightly more memory)
    FULL = "full"           # Full fine-tuning (no adapters)
    FULL_FROZEN = "full_frozen"  # Frozen layers + unfrozen top-N


class ModelArch(Enum):
    DENSE = "dense"         # Standard transformer
    MOE = "moe"             # Mixture of Experts
    MULTIMODAL = "multimodal"  # Vision/audio encoder + LLM


@dataclass
class InferenceServingEstimate:
    """Memory breakdown for serving workloads (vLLM, SGLang, etc.).

    Unlike MemoryEstimate (which is training-focused), this captures the
    serving-specific profile: static model weights plus the KV cache ceiling
    for max_num_seqs concurrent requests.
    """
    model_weights_mb: float = 0.0
    kv_cache_mb: float = 0.0      # 2 × layers × kv_heads × head_dim × max_seq_len × max_num_seqs × dtype_bytes
    activations_mb: float = 0.0   # Decode-phase per-token buffers; 0 when hidden_dim not supplied
    overhead_mb: float = 0.0
    total_mb: float = 0.0

    # Serving parameters kept for reference
    max_num_seqs: int = 0
    max_seq_len: int = 0

    def fits_in(self, budget_mb: float) -> bool:
        """Return True if the total estimate fits within budget_mb."""
        return self.total_mb <= budget_mb

    def __str__(self) -> str:
        lines = ["InferenceServingEstimate:"]
        components: list[tuple[str, float]] = [
            ("Model weights", self.model_weights_mb),
            ("KV cache (ceiling)", self.kv_cache_mb),
        ]
        if self.activations_mb > 0:
            components.append(("Activations (decode)", self.activations_mb))
        components.append(("Framework overhead", self.overhead_mb))
        for name, val in components:
            lines.append(f"  {name + ':':<26}{val:>8.0f} MB")
        lines.append(f"  {'-' * 36}")
        lines.append(f"  {'TOTAL:':<26}{self.total_mb:>8.0f} MB ({self.total_mb / 1024:.1f} GB)")
        lines.append(f"  max_num_seqs:  {self.max_num_seqs}")
        lines.append(f"  max_seq_len:   {self.max_seq_len}")
        return "\n".join(lines)


@dataclass
class MemoryEstimate:
    """Breakdown of estimated peak training memory."""
    model_weights_mb: float = 0.0
    adapter_params_mb: float = 0.0
    optimizer_states_mb: float = 0.0
    activations_mb: float = 0.0
    gradients_mb: float = 0.0
    kv_cache_mb: float = 0.0
    encoder_mb: float = 0.0        # Vision/audio encoder (multi-modal)
    moe_routing_mb: float = 0.0    # MoE gating/dispatch buffers
    quantization_overhead_mb: float = 0.0  # QLoRA scales/zeros
    overhead_mb: float = 0.0
    total_mb: float = 0.0

    def __str__(self):
        lines = ["Memory Estimate:"]
        components = [
            ("Model weights", self.model_weights_mb),
            ("Adapter params", self.adapter_params_mb),
            ("Optimizer states", self.optimizer_states_mb),
            ("Activations", self.activations_mb),
            ("Gradients", self.gradients_mb),
            ("KV cache", self.kv_cache_mb),
        ]
        # Only show non-zero optional components
        if self.encoder_mb > 0:
            components.append(("Encoder (multi-modal)", self.encoder_mb))
        if self.moe_routing_mb > 0:
            components.append(("MoE routing buffers", self.moe_routing_mb))
        if self.quantization_overhead_mb > 0:
            components.append(("Quantization overhead", self.quantization_overhead_mb))
        components.append(("Framework overhead", self.overhead_mb))

        for name, val in components:
            lines.append(f"  {name + ':':<24}{val:>8.0f} MB")
        lines.append(f"  {'-' * 34}")
        lines.append(f"  {'TOTAL:':<24}{self.total_mb:>8.0f} MB ({self.total_mb / 1024:.1f} GB)")
        return "\n".join(lines)

    def fits_in(self, budget_mb: float) -> bool:
        return self.total_mb <= budget_mb


@dataclass
class ModelSpec:
    """Model architecture specification for memory estimation.

    For common models, use ModelSpec.from_name("llama-7b").
    """
    params: int                    # Total parameters
    hidden_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # GQA (None = same as num_heads)
    num_layers: int = 32
    intermediate_dim: Optional[int] = None  # FFN dim (default: 4 * hidden)

    # Quantization
    bits: int = 16                 # Weight precision (4, 8, 16, 32)

    # MoE
    arch: ModelArch = ModelArch.DENSE
    num_experts: int = 1
    num_active_experts: int = 1    # top-k routing

    # Multi-modal
    encoder_params: int = 0        # Vision/audio encoder params
    encoder_bits: int = 16

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_heads

    @property
    def ffn_dim(self) -> int:
        return self.intermediate_dim or (4 * self.hidden_dim)

    @classmethod
    def from_name(cls, name: str) -> "ModelSpec":
        """Create ModelSpec from common model name."""
        presets = {
            "llama-7b": cls(params=7e9, hidden_dim=4096, num_heads=32, num_layers=32),
            "llama-13b": cls(params=13e9, hidden_dim=5120, num_heads=40, num_layers=40),
            "llama-70b": cls(params=70e9, hidden_dim=8192, num_heads=64, num_kv_heads=8, num_layers=80),
            "mistral-7b": cls(params=7e9, hidden_dim=4096, num_heads=32, num_kv_heads=8, num_layers=32),
            "qwen-7b": cls(params=7e9, hidden_dim=4096, num_heads=32, num_layers=32),
            "qwen-9b": cls(params=9e9, hidden_dim=4096, num_heads=32, num_layers=32),  # Qwen3.5-9B
            "phi-3-mini": cls(params=3.8e9, hidden_dim=3072, num_heads=32, num_layers=32),
            "mixtral-8x7b": cls(
                params=47e9, hidden_dim=4096, num_heads=32, num_kv_heads=8, num_layers=32,
                arch=ModelArch.MOE, num_experts=8, num_active_experts=2,
            ),
            "deepseek-moe-16b": cls(
                params=16e9, hidden_dim=2048, num_heads=16, num_layers=28,
                arch=ModelArch.MOE, num_experts=64, num_active_experts=6,
            ),
            "llava-7b": cls(
                params=7e9, hidden_dim=4096, num_heads=32, num_layers=32,
                arch=ModelArch.MULTIMODAL, encoder_params=300e6,
            ),
        }
        key = name.lower().replace("_", "-")
        if key in presets:
            return presets[key]
        raise ValueError(f"Unknown model: {name}. Known: {list(presets.keys())}")


@dataclass
class TrainSpec:
    """Training configuration for memory estimation."""
    method: FinetuneMethod = FinetuneMethod.LORA
    batch_size: int = 1
    seq_length: int = 512
    lora_rank: int = 8
    lora_layers: int = 16
    lora_targets: int = 4          # Projections per layer: Q,K,V,O=4. Set 6 for gate+up_proj.
    optimizer: str = "adam"        # adam, adamw, sgd, adafactor, lion
    grad_checkpoint: bool = False
    grad_accumulation: int = 1
    mixed_precision: bool = True   # Training in fp16/bf16
    qlora_double_quant: bool = False  # NF4 double quantization
    flash_attention: bool = True   # FlashAttention (default on for modern frameworks)
    lazy_evaluation: bool = False  # MLX lazy eval (reduces actual vs estimated)

    @property
    def dtype_bytes(self) -> int:
        return 2 if self.mixed_precision else 4

    @property
    def optimizer_multiplier(self) -> float:
        """Memory multiplier for optimizer states relative to trainable params."""
        return {
            "adam": 3.0,      # params + m + v
            "adamw": 3.0,
            "sgd": 2.0,       # params + momentum
            "adafactor": 1.5,  # row/col factored moments
            "lion": 2.0,      # params + momentum (EMA)
        }.get(self.optimizer, 3.0)


def estimate_training_memory(
    model: Optional[ModelSpec] = None,
    train: Optional[TrainSpec] = None,
    # Legacy API — individual args (used when model/train not provided)
    model_params: int = 0,
    model_bits: int = 16,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    num_layers: int = 32,
    batch_size: int = 1,
    seq_length: int = 512,
    lora_rank: int = 8,
    lora_layers: int = 16,
    optimizer: str = "adam",
    grad_checkpoint: bool = False,
    grad_accumulation: int = 1,
    flash_attention: bool = True,
    lazy_evaluation: bool = False,
    dtype_bytes: int = 2,
    overhead_ratio: float = 0.25,  # Must match constants.OVERHEAD_RATIO_TRAINING
) -> MemoryEstimate:
    """Estimate peak training memory in MB.

    Two calling conventions:
        # New: structured specs
        est = estimate_training_memory(
            model=ModelSpec.from_name("qwen-9b"),
            train=TrainSpec(batch_size=2, lora_rank=16),
        )

        # Legacy: individual args (backward compatible)
        est = estimate_training_memory(
            model_params=9e9, model_bits=4,
            batch_size=2, lora_rank=16,
        )
    """
    # Normalize to specs
    if model is None:
        model = ModelSpec(
            params=model_params, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            bits=model_bits,
        )
    if train is None:
        train = TrainSpec(
            batch_size=batch_size, seq_length=seq_length,
            lora_rank=lora_rank, lora_layers=lora_layers,
            optimizer=optimizer, grad_checkpoint=grad_checkpoint,
            grad_accumulation=grad_accumulation,
            flash_attention=flash_attention,
            lazy_evaluation=lazy_evaluation,
        )

    # Input validation
    if model.params < 0:
        raise ValueError(f"model_params must be >= 0, got {model.params}")
    if model.bits not in (2, 3, 4, 8, 16, 32):
        raise ValueError(f"model_bits must be 2/3/4/8/16/32, got {model.bits}")
    if model.num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {model.num_layers}")
    if model.num_heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {model.num_heads}")
    if train.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {train.batch_size}")
    if train.seq_length < 1:
        raise ValueError(f"seq_length must be >= 1, got {train.seq_length}")
    if train.lora_rank < 1 and train.method in (FinetuneMethod.LORA, FinetuneMethod.QLORA, FinetuneMethod.DORA):
        raise ValueError(f"lora_rank must be >= 1, got {train.lora_rank}")

    est = MemoryEstimate()
    MB = 1024 * 1024

    # 1. Model weights
    weight_bytes = model.bits / 8
    est.model_weights_mb = (model.params * weight_bytes) / MB

    # QLoRA double quantization overhead (see constants.QLORA_DOUBLE_QUANT_OVERHEAD_BYTES)
    if train.method == FinetuneMethod.QLORA and train.qlora_double_quant:
        from ..constants import QLORA_DOUBLE_QUANT_OVERHEAD_BYTES
        est.quantization_overhead_mb = (model.params * QLORA_DOUBLE_QUANT_OVERHEAD_BYTES) / MB

    # 2. Multi-modal encoder
    if model.arch == ModelArch.MULTIMODAL and model.encoder_params > 0:
        encoder_bytes = model.encoder_bits / 8
        est.encoder_mb = (model.encoder_params * encoder_bytes) / MB

    # 3. Adapter / trainable parameters
    if train.method == FinetuneMethod.FULL:
        # Full fine-tuning: all params are trainable (in training dtype)
        trainable_params = model.params
        est.adapter_params_mb = (trainable_params * train.dtype_bytes) / MB
    elif train.method == FinetuneMethod.FULL_FROZEN:
        # Top-N layers unfrozen (clamped to actual layer count)
        params_per_layer = model.params / model.num_layers
        unfrozen = min(train.lora_layers, model.num_layers)
        trainable_params = int(params_per_layer * unfrozen)
        est.adapter_params_mb = (trainable_params * train.dtype_bytes) / MB
    elif train.method in (FinetuneMethod.LORA, FinetuneMethod.QLORA, FinetuneMethod.DORA):
        # LoRA: each target projection gets (in × r + r × out) params
        lora_params_per_layer = train.lora_targets * 2 * model.hidden_dim * train.lora_rank
        trainable_params = lora_params_per_layer * train.lora_layers
        est.adapter_params_mb = (trainable_params * train.dtype_bytes) / MB

        # DoRA adds a magnitude vector per adapted layer
        if train.method == FinetuneMethod.DORA:
            dora_extra = train.lora_targets * model.hidden_dim * train.lora_layers * train.dtype_bytes
            est.adapter_params_mb += dora_extra / MB
    else:
        trainable_params = 0

    # 4. Optimizer states
    est.optimizer_states_mb = est.adapter_params_mb * train.optimizer_multiplier

    # 5. Activations
    # Per-projection input buffers based on:
    #   HyC-LoRA (MLSys 2025): https://mlsys.org/virtual/2025/poster/3254
    #   LoRA-FA (ICLR 2025): https://openreview.net/forum?id=RbKThNNFxr
    #
    # Per-layer activation memory components:
    #   - LoRA input buffers: lora_targets × batch × seq × hidden × dtype
    #     (stored at each adapted projection for backprop)
    #   - Attention scores:   batch × heads × seq × seq × dtype
    #     WITH FlashAttention: O(n) not O(n²) — only stores running
    #     statistics and tiling metadata, NOT the full score matrix.
    #     Reduced to batch × heads × seq × dtype (logsumexp stats).
    #   - FFN intermediate:   batch × seq × ffn_dim × dtype
    #   - Layer norm buffers: 2 × batch × seq × hidden × dtype
    #
    # MLX lazy evaluation discount: MLX fuses operations and defers
    # allocations, reducing actual peak vs eager frameworks.
    # Discount factor based on one M4 Max measurement (5.6% error).

    if train.grad_checkpoint:
        effective_layers = max(1, int(train.lora_layers ** 0.5))
    else:
        effective_layers = train.lora_layers

    # LoRA input activation buffers (X stored at each adapted projection)
    lora_input_per_layer = (
        train.lora_targets * train.batch_size * train.seq_length
        * model.hidden_dim * train.dtype_bytes
    )

    # Attention score matrix
    if train.flash_attention:
        # FlashAttention: recomputes scores from tiled Q/K/V blocks in
        # backward pass. Only stores O(n) logsumexp statistics per head,
        # not the O(n²) score matrix.
        # Ref: Dao et al., "FlashAttention: Fast and Memory-Efficient
        # Exact Attention with IO-Awareness", NeurIPS 2022.
        # https://arxiv.org/abs/2205.14135
        attn_scores_per_layer = (
            train.batch_size * model.num_heads * train.seq_length
            * train.dtype_bytes  # O(n) per head, not O(n²)
        )
    else:
        # Standard attention: full materialized score matrix
        attn_scores_per_layer = (
            train.batch_size * model.num_heads * train.seq_length
            * train.seq_length * train.dtype_bytes  # O(n²)
        )

    # FFN intermediate activation
    ffn_per_layer = (
        train.batch_size * train.seq_length * model.ffn_dim * train.dtype_bytes
    )
    # LayerNorm buffers (pre-attention + pre-FFN)
    ln_per_layer = (
        2 * train.batch_size * train.seq_length * model.hidden_dim * train.dtype_bytes
    )

    total_act_per_layer = lora_input_per_layer + attn_scores_per_layer + ffn_per_layer + ln_per_layer

    # MLX lazy evaluation discount: operation fusion and deferred
    # materialization can reduce actual peak vs theoretical worst-case.
    from ..constants import LAZY_EVAL_DISCOUNT
    lazy_discount = LAZY_EVAL_DISCOUNT if train.lazy_evaluation else 1.0

    est.activations_mb = (total_act_per_layer * effective_layers * lazy_discount) / MB

    # MoE: additional activation memory for routing and expert computation
    if model.arch == ModelArch.MOE:
        # Routing: batch × seq × num_experts scores
        routing_mb = (train.batch_size * train.seq_length * model.num_experts * 4) / MB
        # Active expert FFN activations
        expert_act = (train.batch_size * train.seq_length *
                      model.ffn_dim * model.num_active_experts * train.dtype_bytes)
        est.moe_routing_mb = routing_mb + (expert_act * effective_layers) / MB

    # 6. Gradients
    est.gradients_mb = est.adapter_params_mb

    # 7. KV cache
    # GQA-aware: use kv_heads, not num_heads
    kv_per_layer = (2 * train.batch_size * train.seq_length *
                    model.kv_heads * model.head_dim * train.dtype_bytes)
    est.kv_cache_mb = (kv_per_layer * model.num_layers) / MB

    # 8. Overhead
    # Two components: proportional (fragmentation, temp buffers) and
    # fixed (framework runtime, Metal shaders, Python interpreter).
    # The fixed cost is critical for small models where proportional
    # overhead alone under-estimates by 30-40%.
    from ..constants import FIXED_OVERHEAD_MB
    subtotal = (est.model_weights_mb + est.adapter_params_mb +
                est.optimizer_states_mb + est.activations_mb +
                est.gradients_mb + est.kv_cache_mb +
                est.encoder_mb + est.moe_routing_mb +
                est.quantization_overhead_mb)
    est.overhead_mb = subtotal * overhead_ratio + FIXED_OVERHEAD_MB
    est.total_mb = subtotal + est.overhead_mb

    return est


def estimate_inference_memory(
    model: Optional[ModelSpec] = None,
    batch_size: int = 1,
    seq_length: int = 2048,
    # Legacy API
    model_params: int = 0,
    model_bits: int = 16,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    num_layers: int = 32,
    dtype_bytes: int = 2,
) -> MemoryEstimate:
    """Estimate peak inference memory (no gradients/optimizer)."""
    if model is None:
        model = ModelSpec(
            params=model_params, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            bits=model_bits,
        )

    est = MemoryEstimate()
    MB = 1024 * 1024

    est.model_weights_mb = (model.params * (model.bits / 8)) / MB

    if model.arch == ModelArch.MULTIMODAL:
        est.encoder_mb = (model.encoder_params * (model.encoder_bits / 8)) / MB

    kv_per_layer = (2 * batch_size * seq_length *
                    model.kv_heads * model.head_dim * dtype_bytes)
    est.kv_cache_mb = (kv_per_layer * model.num_layers) / MB

    est.activations_mb = (batch_size * seq_length * model.hidden_dim * dtype_bytes) / MB

    subtotal = est.model_weights_mb + est.kv_cache_mb + est.activations_mb + est.encoder_mb
    from ..constants import OVERHEAD_RATIO_INFERENCE, FIXED_OVERHEAD_MB
    est.overhead_mb = subtotal * OVERHEAD_RATIO_INFERENCE + FIXED_OVERHEAD_MB
    est.total_mb = subtotal + est.overhead_mb

    return est


def estimate_serving_memory(
    model_params: int,
    model_bits: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_layers: int = 32,
    max_num_seqs: int = 256,
    max_seq_len: int = 8192,
    dtype_bytes: int = 2,
    hidden_dim: int = 0,
) -> InferenceServingEstimate:
    """Estimate peak memory for a vLLM / SGLang serving deployment.

    Computes the KV cache CEILING (all sequences at max_seq_len simultaneously)
    plus static model weight footprint.  PagedAttention / RadixAttention allocate
    on demand so real-world utilization is typically lower, but
    ``preflight_inference()`` uses this ceiling to guarantee the engine never
    OOMs under peak load.

    Args:
        model_params:  Total parameter count of the model.
        model_bits:    Weight quantization precision (4, 8, 16, 32).
        num_kv_heads:  GQA-aware KV head count.  For MHA models this equals
                       num_attention_heads.  For Llama-3-8B it is 8, not 32.
        head_dim:      Attention head dimension (hidden_dim // num_attention_heads).
        num_layers:    Number of transformer layers.
        max_num_seqs:  Maximum concurrent requests (vLLM ``--max-num-seqs``).
        max_seq_len:   Maximum total sequence length per request (prompt +
                       generation); corresponds to vLLM ``--max-model-len``.
        dtype_bytes:   KV cache element width in bytes.
                       2 = fp16 / bf16 (default), 4 = fp32, 1 = int8 / fp8.
        hidden_dim:    Model hidden dimension.  When > 0, per-token decode-phase
                       activation buffers are included.  When 0 (default) they
                       are omitted — they are typically < 1 % of the KV cache
                       for large max_num_seqs and are absorbed by the overhead
                       factor.

    Returns:
        InferenceServingEstimate with model_weights_mb, kv_cache_mb,
        activations_mb, overhead_mb, and total_mb.
    """
    if model_params < 0:
        raise ValueError(f"model_params must be >= 0, got {model_params}")
    if model_bits not in (2, 3, 4, 8, 16, 32):
        raise ValueError(f"model_bits must be 2/3/4/8/16/32, got {model_bits}")
    if num_kv_heads < 1:
        raise ValueError(f"num_kv_heads must be >= 1, got {num_kv_heads}")
    if head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if max_num_seqs < 1:
        raise ValueError(f"max_num_seqs must be >= 1, got {max_num_seqs}")
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    if dtype_bytes not in (1, 2, 4):
        raise ValueError(f"dtype_bytes must be 1, 2, or 4, got {dtype_bytes}")

    MB = 1024 * 1024
    est = InferenceServingEstimate(max_num_seqs=max_num_seqs, max_seq_len=max_seq_len)

    # 1. Static model weights — loaded once at engine start, never change
    est.model_weights_mb = (model_params * (model_bits / 8)) / MB

    # 2. KV cache ceiling
    #    Two tensors (K and V) per layer, for every KV head, for every
    #    token position in every concurrent sequence.
    #    Formula: 2 × num_layers × num_kv_heads × head_dim
    #             × max_seq_len × max_num_seqs × dtype_bytes
    kv_bytes = (
        2 * num_layers * num_kv_heads * head_dim
        * max_seq_len * max_num_seqs * dtype_bytes
    )
    est.kv_cache_mb = kv_bytes / MB

    # 3. Per-request activation buffers (decode phase only)
    #    Each active request processes one new token per forward pass, so
    #    activation per pass ≈ max_num_seqs × hidden_dim × num_layers × dtype_bytes.
    #    Typically < 1 % of the KV cache for large deployments.
    if hidden_dim > 0:
        act_bytes = max_num_seqs * hidden_dim * num_layers * dtype_bytes
        est.activations_mb = act_bytes / MB

    # 4. Proportional fragmentation overhead + fixed framework runtime cost
    from ..constants import OVERHEAD_RATIO_INFERENCE, FIXED_OVERHEAD_MB
    subtotal = est.model_weights_mb + est.kv_cache_mb + est.activations_mb
    est.overhead_mb = subtotal * OVERHEAD_RATIO_INFERENCE + FIXED_OVERHEAD_MB
    est.total_mb = subtotal + est.overhead_mb

    return est
