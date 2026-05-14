from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for the Gated Attention LLM models."""
    # Model architecture
    model_type: Literal["dense", "moe"] = "dense"
    n_layers: int = 28
    d_model: int = 2048
    n_query_heads: int = 32
    n_kv_heads: int = 4  # GQA
    d_head: int = 128
    d_ff: int = 5632  # FFN intermediate dimension
    vocab_size: int = 151936

    # MoE settings (for 15A2B model)
    n_experts: int = 128
    n_active_experts: int = 8
    expert_intermediate_dim: int = None  # will be computed as d_ff // n_experts_shared

    # Gating configuration for attention
    gating_position: Optional[Literal["G1", "G2", "G3", "G4", "G5"]] = None
    gating_granularity: Optional[Literal["elementwise", "headwise"]] = None
    gating_head_specific: bool = True
    gating_mode: Literal["multiplicative", "additive"] = "multiplicative"
    gating_activation: Literal["sigmoid", "silu", "identity", "ns_sigmoid"] = "sigmoid"

    # Stabilization
    use_sandwich_norm: bool = False
    use_qk_norm: bool = False

    # Position encoding
    rope_base: float = 10000.0
    max_seq_len: int = 4096

    # Normalization
    rms_norm_eps: float = 1e-6


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    max_lr: float = 4e-3
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    total_tokens: int = 3500000000000  # 3.5T
    batch_size: int = 1024  # global batch size in sequences
    seq_len: int = 4096
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    z_loss_coef: float = 0.001  # for MoE Z-loss
    load_balancing_loss_coef: float = 0.01

    # Data
    data_path: str = "data/tokens"

    # Logging
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000

    # Mixed precision
    dtype: str = "bfloat16"
    use_amp: bool = True


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment name
    exp_name: str = "gated_attention"

    def to_dict(self):
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "exp_name": self.exp_name,
        }


# Model size presets for paper configurations
def get_moe_15a2b_config(**kwargs) -> ModelConfig:
    """15B total, 2.54B activated MoE model."""
    return ModelConfig(
        model_type="moe",
        n_layers=48,
        d_model=4096,
        n_query_heads=32,
        n_kv_heads=4,
        d_head=128,
        d_ff=14336,
        n_experts=128,
        n_active_experts=8,
        **kwargs,
    )


def get_dense_1_7b_28l_config(**kwargs) -> ModelConfig:
    """1.7B dense model, 28 layers."""
    return ModelConfig(
        model_type="dense",
        n_layers=28,
        d_model=2048,
        n_query_heads=16,
        n_kv_heads=4,
        d_head=128,
        d_ff=5632,
        **kwargs,
    )


def get_dense_1_7b_48l_config(**kwargs) -> ModelConfig:
    """1.7B dense model, 48 layers."""
    return ModelConfig(
        model_type="dense",
        n_layers=48,
        d_model=1536,
        n_query_heads=24,
        n_kv_heads=8,
        d_head=64,
        d_ff=4096,
        **kwargs,
    )
