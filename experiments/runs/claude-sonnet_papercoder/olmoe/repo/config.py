```python
## config.py
"""Configuration dataclasses for OLMoE reproduction.

This module defines all configuration dataclasses used throughout the OLMoE
reproduction codebase. It has zero internal dependencies and is imported by
all other modules.

Based on: "OLMoE: Open Mixture-of-Experts Language Models"
All values sourced directly from the paper (Appendix B, Table 10, Section 2, 4)
and config.yaml.
"""

import copy
import math
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional


@dataclass
class OLMoEConfig:
    """Model architecture configuration for OLMoE-1B-7B.

    All defaults match the final OLMoE-1B-7B configuration from Table 10
    and Appendix B of the paper.
    """

    # -------------------------------------------------------------------------
    # Architecture (Table 10, Section 2)
    # -------------------------------------------------------------------------
    hidden_dim: int = 2048
    """Model hidden dimension, shared across all layers."""

    num_layers: int = 16
    """Number of transformer blocks."""

    num_heads: int = 16
    """Number of attention heads. head_dim = hidden_dim // num_heads = 128."""

    ffn_dim: int = 1024
    """Per-expert FFN intermediate dimension.
    Fine-grained: 64 experts × 1024 vs 1 expert × 8192 for dense equivalent.
    """

    num_experts: int = 64
    """Total experts per MoE layer (Section 4.1.2)."""

    top_k: int = 8
    """Number of activated experts per token (Section 2, Table 1)."""

    vocab_size: int = 50304
    """GPT-NeoX tokenizer vocabulary size (Table 10)."""

    max_seq_len: int = 4096
    """Maximum sequence length (Table 10)."""

    # -------------------------------------------------------------------------
    # Positional Embedding (Table 10)
    # -------------------------------------------------------------------------
    rope_theta: float = 10000.0
    """RoPE base frequency (Table 10)."""

    # -------------------------------------------------------------------------
    # Normalization (Section 4.2.3, Table 10)
    # -------------------------------------------------------------------------
    layer_norm_type: str = "rmsnorm"
    """Normalization type. RMSNorm, not non-parametric LayerNorm."""

    rms_norm_eps: float = 1.0e-05
    """RMSNorm epsilon (Table 10)."""

    use_qk_norm: bool = True
    """Whether to apply QK-Norm on Q and K projections (Section 4.2.5)."""

    # -------------------------------------------------------------------------
    # Activation and Attention (Table 10)
    # -------------------------------------------------------------------------
    activation: str = "swiglu"
    """Activation function. SwiGLU as in Table 10."""

    attention_variant: str = "full"
    """Attention variant. Full attention, no GQA/MQA (Table 10)."""

    use_bias: bool = False
    """Whether to use biases in Linear layers (Table 10: no biases)."""

    tie_word_embeddings: bool = False
    """Whether to tie input embedding and LM head weights (Table 10: no tying)."""

    # -------------------------------------------------------------------------
    # MoE Routing (Section 2, Section 4.1.4)
    # -------------------------------------------------------------------------
    routing_type: str = "token_choice"
    """Routing algorithm. Dropless token-choice routing (not expert-choice)."""

    use_megablocks: bool = True
    """Whether to use MegaBlocks dMoE for dropless sparse ops.
    Falls back to scatter/gather implementation if False or MegaBlocks unavailable.
    """

    # -------------------------------------------------------------------------
    # Auxiliary Losses (Section 4.1.6, Section 4.1.7, Table 1)
    # -------------------------------------------------------------------------
    lb_loss_weight: float = 0.01
    """Load balancing loss weight alpha (Section 4.1.6)."""

    router_z_loss_weight: float = 0.001
    """Router z-loss weight beta (Section 4.1.7)."""

    # -------------------------------------------------------------------------
    # Initialization (Section 4.2.2, Appendix B)
    # -------------------------------------------------------------------------
    init_distribution: str = "truncated_normal"
    """Weight initialization distribution."""

    init_std: float = 0.02
    """Standard deviation for weight initialization."""

    init_trunc_factor: float = 3.0
    """Truncation factor: clip at ±(init_std * init_trunc_factor) = ±0.06."""

    # -------------------------------------------------------------------------
    # Dropout (not used in final model but kept for ablations)
    # -------------------------------------------------------------------------
    dropout: float = 0.0
    """Dropout probability. Set to 0.0 for OLMoE-1B-7B."""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads}). "
                f"Got head_dim = {self.hidden_dim / self.num_heads}."
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})."
            )
        if self.lb_loss_weight < 0:
            raise ValueError(
                f"lb_loss_weight must be >= 0, got {self.lb_loss_weight}."
            )
        if self.router_z_loss_weight < 0:
            raise ValueError(
                f"router_z_loss_weight must be >= 0, got {self.router_z_loss_weight}."
            )
        if self.init_std <= 0:
            raise ValueError(
                f"init_std must be > 0, got {self.init_std}."
            )
        if self.init_trunc_factor <= 0:
            raise ValueError(
                f"init_trunc_factor must be > 0, got {self.init_trunc_factor}."
            )

    @property
    def head_dim(self) -> int:
        """Attention head dimension: hidden_dim // num_heads = 128."""
        return self.hidden_dim // self.num_heads

    @property
    def init_trunc_val(self) -> float:
        """Truncation value for weight init: init_std * init_trunc_factor = 0.06."""
        return self.init_std * self.init_trunc_factor

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OLMoEConfig":
        """Construct OLMoEConfig from a dictionary.

        Handles both flat dicts and nested YAML structure (with 'model:' key).
        Unknown keys are silently ignored for forward compatibility.

        Args:
            d: Dictionary of configuration values. May be flat or nested
               with a 'model' key containing the model config.

        Returns:
            OLMoEConfig instance with values from the dict.
        """
        # Handle nested YAML structure: {'model': {...}}
        if "model" in d and isinstance(d["model"], dict):
            d = d["model"]

        known_fields = {f.name: f for f in fields(cls)}
        kwargs: Dict[str, Any] = {}

        for key, value in d.items():
            if key in known_fields:
                field_type = known_fields[key].type
                # Cast to the annotated type if needed
                kwargs[key] = _cast_value(value, field_type)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Includes all fields plus computed properties for logging completeness.

        Returns:
            Dictionary representation of the configuration.
        """
        result = asdict(self)
        # Add computed properties
        result["head_dim"] = self.head_dim
        result["init_trunc_val"] = self.init_trunc_val
        return result


@dataclass
class TrainingConfig:
    """Pretraining hyperparameter configuration for OLMoE-1B-7B.

    All defaults match the final OLMoE-1B-7B pretraining configuration
    from Table 10 and Appendix B of the paper.
    """

    # -------------------------------------------------------------------------
    # Optimizer (Table 10, Appendix B)
    # -------------------------------------------------------------------------
    optimizer: str = "adamw"
    """Optimizer type."""

    learning_rate: float = 4.0e-04
    """Peak learning rate (Table 10; paper v2 corrected from 5e-4 to 4e-4)."""

    min_lr: float = 4.0e-05
    """Minimum LR before annealing phase (Table 10)."""

    annealing_min_lr: float = 0.0
    """LR decays to 0 at end of annealing phase (Table 10)."""

    adam_beta1: float = 0.9
    """AdamW beta1 (Table 10)."""

    adam_beta2: float = 0.95
    """AdamW beta2 (Table 10)."""

    adam_eps: float = 1.0e-08
    """AdamW epsilon (Section 4.2.6; changed from OLMo's 1e-5 to 1e-8)."""

    weight_decay: float = 0.1
    """Weight decay applied to ALL parameters including embeddings and RMSNorm.
    This is non-standard but explicitly stated in Sections 4.2.3 and 4.2.4.
    """

    # -------------------------------------------------------------------------
    # LR Schedule (Table 10, Appendix B)
    # -------------------------------------------------------------------------
    lr_schedule: str = "cosine"
    """LR schedule type for main training phase."""

    warmup_steps: int = 2500
    """Number of linear warmup steps (Table 10)."""

    annealing_schedule: str = "linear"
    """LR schedule during annealing phase: linear decay to 0 (Table 10)."""

    # -------------------------------------------------------------------------
    # Token Budget (Section 2, Appendix B)
    # -------------------------------------------------------------------------
    total_tokens: int = 5_133_000_000_000
    """Total pretraining tokens: 5.133T (1.3 epochs of OLMoE-Mix)."""

    annealing_tokens: int = 100_000_000_000
    """Tokens in the annealing phase: final 100B tokens (Appendix B)."""

    # -------------------------------------------------------------------------
    # Batch Configuration (Table 10)
    # -------------------------------------------------------------------------
    seq_len: int = 4096
    """Sequence length (Table 10)."""

    batch_size_samples: int = 1024
    """Number of samples per training step (Table 10)."""

    batch_size_tokens: int = 4_194_304
    """Tokens per training step: ~4M = 1024 × 4096 (Table 10)."""

    # -------------------------------------------------------------------------
    # Gradient (Table 10)
    # -------------------------------------------------------------------------
    grad_clip: float = 1.0
    """Global gradient norm clipping threshold (Table 10)."""

    gradient_reduce_dtype: str = "fp32"
    """Dtype for gradient reduction across devices (Table 10)."""

    optimizer_state_dtype: str = "fp32"
    """Dtype for optimizer states (Table 10)."""

    # -------------------------------------------------------------------------
    # Mixed Precision and Distributed (Appendix B)
    # -------------------------------------------------------------------------
    bf16: bool = True
    """Whether to use BF16 mixed precision training (Appendix B)."""

    fp32_reduce: bool = True
    """Whether to perform gradient reduction in FP32 (Appendix B)."""

    fsdp: bool = True
    """Whether to use PyTorch FSDP with ZeRO (Appendix B)."""

    zero_stage: int = 3
    """ZeRO stage via FSDP (Appendix B: ZeRO-3)."""

    # -------------------------------------------------------------------------
    # Checkpointing and Logging
    # -------------------------------------------------------------------------
    save_every_steps: int = 5000
    """Save intermediate checkpoints every N steps."""

    eval_every_steps: int = 1000
    """Run in-loop evaluation every N steps."""

    log_every_steps: int = 1
    """Log metrics every N steps."""

    output_dir: str = "outputs"
    """Directory for saving checkpoints and logs."""

    wandb_project: str = "olmoe"
    """Weights & Biases project name."""

    run_name: str = "olmoe-1b-7b"
    """Experiment run name for logging."""

    # -------------------------------------------------------------------------
    # Data (Section 2, Appendix B)
    # -------------------------------------------------------------------------
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    """GPT-NeoX tokenizer (vocab_size=50304)."""

    num_epochs: float = 1.3
    """Number of training epochs (Muennighoff et al. 2023)."""

    shuffle_at_epoch_start: bool = True
    """Whether to shuffle data at the start of each epoch."""

    reshuffle_before_annealing: bool = True
    """Whether to reshuffle entire dataset before annealing phase (Appendix B)."""

    # -------------------------------------------------------------------------
    # Computed fields (set in __post_init__)
    # -------------------------------------------------------------------------
    max_steps: int = field(init=False)
    """Total training steps: total_tokens // batch_size_tokens."""

    annealing_steps: int = field(init=False)
    """Annealing phase steps: annealing_tokens // batch_size_tokens."""

    def __post_init__(self) -> None:
        """Compute derived fields and validate configuration."""
        # Compute derived fields
        self.max_steps = self.total_tokens // self.batch_size_tokens
        self.annealing_steps = self.annealing_tokens // self.batch_size_tokens

        # Validate batch size consistency
        expected_batch_tokens = self.batch_size_samples * self.seq_len
        if self.batch_size_tokens != expected_batch_tokens:
            raise ValueError(
                f"batch_size_tokens ({self.batch_size_tokens}) must equal "
                f"batch_size_samples × seq_len = "
                f"{self.batch_size_samples} × {self.seq_len} = "
                f"{expected_batch_tokens}."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}."
            )
        if self.min_lr < 0:
            raise ValueError(
                f"min_lr must be >= 0, got {self.min_lr}."
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}."
            )
        if self.grad_clip <= 0:
            raise ValueError(
                f"grad_clip must be > 0, got {self.grad_clip}."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Construct TrainingConfig from a dictionary.

        Handles both flat dicts and nested YAML structure (with 'pretraining:' key).
        Computed fields (max_steps, annealing_steps) are recomputed in __post_init__
        rather than read from the dict to ensure consistency.

        Args:
            d: Dictionary of configuration values. May be flat or nested
               with a 'pretraining' key containing the training config.

        Returns:
            TrainingConfig instance with values from the dict.
        """
        # Handle nested YAML structure: {'pretraining': {...}}
        if "pretraining" in d and isinstance(d["pretraining"], dict):
            d = d["pretraining"]

        # Fields that are computed in __post_init__ — skip them from dict
        computed_field_names = {"max_steps", "annealing_steps"}

        known_fields = {
            f.name: f for f in fields(cls)
            if f.name not in computed_field_names
        }
        kwargs: Dict[str, Any] = {}

        for key, value in d.items():
            if key in known_fields:
                field_type = known_fields[key].type
                kwargs[key] = _cast_value(value, field_type)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Includes all fields including computed ones for logging completeness.

        Returns:
            Dictionary representation of the configuration.
        """
        result: Dict[str, Any] = {}
        for f in fields(self):
            result[f.name] = getattr(self, f.name)
        return result


@dataclass
class SFTConfig:
    """Instruction tuning (SFT) configuration for OLMoE-1B-7B.

    All defaults match the SFT configuration from Appendix B and Section 4.3.
    """

    # -------------------------------------------------------------------------
    # Optimizer (Appendix B)
    # -------------------------------------------------------------------------
    optimizer: str = "adamw"
    """Optimizer type for SFT."""

    learning_rate: float = 2.0e-05
    """Constant learning rate for SFT (Appendix B)."""

    lr_schedule: str = "constant"
    """LR schedule: constant for SFT (Appendix B)."""

    adam_beta1: float = 0.9
    """AdamW beta1."""

    adam_beta2: float = 0.95
    """AdamW beta2."""

    adam_eps: float = 1.0e-08
    """AdamW epsilon."""

    weight_decay: float = 0.1
    """Weight decay applied to all parameters."""

    # -------------------------------------------------------------------------
    # Training Duration (Appendix B)
    # -------------------------------------------------------------------------
    num_epochs: int = 2
    """Number of SFT training epochs (Appendix B)."""

    # -------------------------------------------------------------------------
    # Batch Configuration (Appendix B)
    # -------------------------------------------------------------------------
    global_batch_size: int = 128
    """Global batch size: 4 nodes × 8 GPUs × per_device=2 × grad_accum=2 (Appendix B)."""

    per_device_batch_size: int = 2
    """Per-device batch size (Appendix B)."""

    gradient_accumulation_steps: int = 2
    """Gradient accumulation steps (Appendix B)."""

    max_seq_len: int = 4096
    """Maximum sequence length; samples longer than this are filtered (Appendix B)."""

    # -------------------------------------------------------------------------
    # Mixed Precision (Appendix B)
    # -------------------------------------------------------------------------
    bf16: bool = True
    """Whether to use BF16 mixed precision."""

    # -------------------------------------------------------------------------
    # Auxiliary Losses (Section 4.3) — CRITICAL: both False for SFT
    # -------------------------------------------------------------------------
    use_lb_loss: bool = False
    """Whether to use load balancing loss during SFT.
    MUST be False: paper Section 4.3 shows this improves performance.
    """

    use_router_z_loss: bool = False
    """Whether to use router z-loss during SFT. Not used during adaptation."""

    # -------------------------------------------------------------------------
    # Loss Aggregation (Appendix B)
    # -------------------------------------------------------------------------
    loss_aggregation: str = "token_level"
    """Loss aggregation method. Token-level (not sample-level) per Appendix B.
    Improves performance on long generative tasks like AlpacaEval.
    """

    # -------------------------------------------------------------------------
    # Checkpoint Selection (Section 4.3)
    # -------------------------------------------------------------------------
    use_post_annealing_checkpoint: bool = True
    """Whether to use post-annealing checkpoint for SFT.
    Section 4.3: post-annealing gives better results (+0.2 avg score).
    """

    # -------------------------------------------------------------------------
    # Hardware (Appendix B)
    # -------------------------------------------------------------------------
    num_gpus: int = 32
    """Number of GPUs: 4 H100 nodes × 8 GPUs (Appendix B)."""

    training_hours: float = 33.0
    """Approximate training time in hours (Appendix B)."""

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    output_dir: str = "outputs/sft"
    """Directory for saving SFT checkpoints."""

    wandb_project: str = "olmoe"
    """Weights & Biases project name."""

    run_name: str = "olmoe-1b-7b-sft"
    """Experiment run name."""

    save_every_steps: int = 500
    """Save checkpoints every N steps."""

    eval_every_steps: int = 500
    """Run evaluation every N steps."""

    log_every_steps: int = 1
    """Log metrics every N steps."""

    grad_clip: float = 1.0
    """Global gradient norm clipping threshold."""

    def __post_init__(self) -> None:
        """Validate SFT configuration."""
        if self.use_lb_loss:
            raise ValueError(
                "use_lb_loss must be False for SFT. "
                "Paper Section 4.3 shows load balancing loss hurts SFT performance."
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SFTConfig":
        """Construct SFTConfig from a dictionary.

        Args:
            d: Dictionary of configuration values. May be nested with 'sft:' key.

        Returns:
            SFTConfig instance.
        """
        if "sft" in d and isinstance(d["sft"], dict):
            d = d["sft"]

        known_fields = {f.name: f for f in fields(cls)}
        kwargs: Dict[str, Any] = {}

        for key, value in d.items():
            if key in known_fields:
                field_type = known_fields[key].type
                kwargs[key] = _cast_value(value, field_type)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class DPOConfig:
    """Direct Preference Optimization (DPO) configuration for OLMoE-1B-7B.

    All defaults match the DPO configuration from Appendix B and Section 4.3.
    """

    # -------------------------------------------------------------------------
    # Optimizer (Appendix B)
    # -------------------------------------------------------------------------
    optimizer: str = "adamw"
    """Optimizer type for DPO."""

    learning_rate: float = 5.0e-07
    """Constant learning rate for DPO (Appendix B)."""

    lr_schedule: str = "constant"
    """LR schedule: constant for DPO (Appendix B)."""

    adam_beta1: float = 0.9
    """AdamW beta1."""

    adam_beta2: float = 0.95
    """AdamW beta2."""

    adam_eps: float = 1.0e-08
    """AdamW epsilon."""

    weight_decay: float = 0.1
    """Weight decay applied to all parameters."""

    # -------------------------------------------------------------------------
    # Training Duration (Appendix B)
    # -------------------------------------------------------------------------
    num_epochs: int = 3
    """Number of DPO training epochs (Appendix B)."""

    # -------------------------------------------------------------------------
    # Batch Configuration (Appendix B)
    # -------------------------------------------------------------------------
    global_batch_size: int = 32
    """Global batch size: 4 nodes × 8 GPUs × per_device=1 (Appendix B)."""

    per_device_batch_size: int = 1
    """Per-device batch size (Appendix B)."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps (Appendix B)."""

    # -------------------------------------------------------------------------
    # DPO-Specific (Appendix B)
    # -------------------------------------------------------------------------
    dpo_beta: float = 0.1
    """DPO beta hyperparameter controlling deviation from reference (Appendix B)."""

    base_model: str = "sft_checkpoint"
    """Starting checkpoint for DPO: must be the SFT checkpoint (Section 4.3)."""

    # -------------------------------------------------------------------------
    # Mixed Precision (Appendix B)
    # -------------------------------------------------------------------------
    bf16: bool = True
    """Whether to use BF16 mixed precision."""

    # -------------------------------------------------------------------------
    # Auxiliary Losses (Section 4.3) — CRITICAL: both False for DPO
    # -------------------------------------------------------------------------
    use_lb_loss: bool = False
    """Whether to use load balancing loss during DPO.
    MUST be False: paper Section 4.3 shows this improves performance.
    """

    use_router_z_loss: bool = False
    """Whether to use router z-loss during DPO. Not used during adaptation."""

    # -------------------------------------------------------------------------
    # Hardware (Appendix B)
    # -------------------------------------------------------------------------
    num_gpus: int = 32
    """Number of GPUs: 4 H100 nodes × 8 GPUs (Appendix B)."""

    training_hours: float = 14.0
    """Approximate training time in hours (Appendix B)."""

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    output_dir: str = "outputs/dpo"
    """Directory for saving DPO checkpoints."""

    wandb_project: str = "olmoe"
    """Weights & Biases project name."""

    run_name: str = "olmoe-1b-7b-dpo"
    """Experiment run name."""

    save_every_steps: int = 100
    """Save checkpoints every N steps."""

    eval_every_steps: int = 100
    """Run evaluation every N steps."""

    log_every_steps: int = 1
    """Log metrics every N steps."""

    grad_clip: float = 1.0
    """Global gradient norm clipping threshold."""

    def __post_init__(self) -> None:
        """Validate DPO configuration."""
        if self.use_lb_loss:
            raise ValueError(
                "use_lb_loss must be False for DPO. "
                "Paper Section 4.3 shows load balancing loss hurts DPO performance."
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}."
            )
        if self.dpo_beta <= 0:
            raise ValueError(
                f"dpo_beta must be > 0, got {self.dpo_beta}."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DPOConfig":
        """Construct DPOConfig from a dictionary.

        Args:
            d: Dictionary of configuration values. May be nested with 'dpo:' key.

        Returns:
            DPOConfig instance.
        """
        if "dpo" in d and isinstance(d["dpo"], dict):
            d = d["dpo"]

        known_fields = {f.name: f for f in fields(cls)}
        kwargs: Dict[str, Any] = {}

        for key, value in d.items():
            if key in known_fields:
                field_type = known_fields[key].type
                kwargs[key] = _cast_value(value, field_type)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class KTOConfig:
    """KTO (Kahneman-Tversky Optimization) configuration for OLMoE-1B-7B.

    Alternative to DPO. All defaults from Appendix F and Table 14.
    """

    # -------------------------------------------------------------------------
    # Optimizer (Appendix F) — RMSProp, not Adam
    # -------------------------------------------------------------------------
    optimizer: str = "rmsprop"
    """Optimizer type for KTO. RMSProp instead of Adam (Appendix F, Table 14)."""

    learning_rate: float = 5.0e-07
    """Learning rate for KTO: same as DPO (Appendix B)."""

    lr_schedule: str = "constant"
    """LR schedule: constant."""

    # -------------------------------------------------------------------------
    # Training Duration (Appendix F, Table 14)
    # -------------------------------------------------------------------------
    num_steps: int = 5000
    """Number of KTO training steps: 1.3 epochs (Appendix F, Table 14)."""

    # -------------------------------------------------------------------------
    # Batch Configuration
    # -------------------------------------------------------------------------
    global_batch_size: int = 32
    """Global batch size."""

    per_device_batch_size: int = 1
    """Per-device batch size."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""

    # -------------------------------------------------------------------------
    # Mixed Precision
    # -------------------------------------------------------------------------
    bf16: bool = True
    """Whether to use BF16 mixed precision."""

    # -------------------------------------------------------------------------
    # Auxiliary Losses — both False for KTO
    # -------------------------------------------------------------------------
    use_lb_loss: bool = False
    """Whether to use load balancing loss during KTO."""

    use_router_z_loss: bool = False
    """Whether to use router z-loss during KTO."""

    # -------------------------------------------------------------------------
    # Hardware (Appendix B)
    # -------------------------------------------------------------------------
    num_gpus: int = 8
    """Number of GPUs: 8 H100 GPUs (Appendix B)."""

    training_hours: float = 30.0
    """Approximate training time in hours (Appendix B)."""

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    output_dir: str = "outputs/kto"
    """Directory for saving KTO checkpoints."""

    wandb_project: str = "olmoe"
    """Weights & Biases project name."""

    run_name: str = "olmoe-1b-7b-kto"
    """Experiment run name."""

    grad_clip: float = 1.0
    """Global gradient norm clipping threshold."""

    def __post_init__(self) -> None:
        """Validate KTO configuration."""
        if self.use_lb_loss:
            raise ValueError(
                "use_lb_loss must be False for KTO."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KTOConfig":
        """Construct KTOConfig from a dictionary.

        Args:
            d: Dictionary of configuration values. May be nested with 'kto:' key.

        Returns:
            KTOConfig instance.
        """
        if "kto" in d and isinstance(d["kto"], dict):
            d = d["kto"]

        known_fields = {f.name: f for f in fields(cls)}
        kwargs: Dict[str, Any] = {}

        for key, value in d.items():
            if key in known_fields:
                field_type = known_fields[key].type
                kwargs[key] = _cast_value(value, field_type)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class AblationConfig:
    """Configuration for ablation experiments (Section 4).

    Captures experiment-specific overrides on top of base OLMoEConfig
    and TrainingConfig.
    """

    experiment_name: str = "default_ablation"
    """Name of the ablation experiment (e.g., 'moe_vs