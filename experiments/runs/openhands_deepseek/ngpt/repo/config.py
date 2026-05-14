
"""Configuration for nGPT and baseline GPT experiments.

Hyperparameters from:
- Table 2: Model Parameters for 0.5B and 1B Models
- Table 3: Optimization Parameters
- Section 2.6: Summary of modifications
- Appendix A.6: Experimental Setup
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters matching Table 2."""
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 16
    d_mlp: int = 4096  # 4 * d_model
    vocab_size: int = 32000
    max_seq_len: int = 4096  # 4k context
    rope_base: int = 10000
    init_std: float = 0.02
    # nGPT-specific: std for normalized matrices
    init_std_norm: float = 0.0  # Set to 1/sqrt(d_model) in code since normalized afterwards

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def presets(cls) -> dict:
        """Return preset model configurations from Table 2."""
        return {
            "0.5B": cls(
                n_layers=24,
                d_model=1024,
                n_heads=16,
                d_mlp=4096,
            ),
            "1.0B": cls(
                n_layers=36,
                d_model=1280,
                n_heads=20,
                d_mlp=5120,
            ),
        }


@dataclass
class OptimConfig:
    """Optimization hyperparameters matching Table 3 and Section 2.6."""
    optimizer: Literal["adamw", "adam"] = "adamw"
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    lr_schedule: Literal["cosine"] = "cosine"
    initial_lr: float = 2.0e-3
    final_lr: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    global_batch_size: int = 512
    grad_clip: float = 1.0

    def to_gpt(self) -> "OptimConfig":
        """Convert to GPT baseline settings."""
        return OptimConfig(
            optimizer="adamw",
            weight_decay=0.1,
            warmup_steps=2000,
        )

    def to_ngpt(self) -> "OptimConfig":
        """Convert to nGPT settings (no weight decay, no warmup)."""
        return OptimConfig(
            optimizer="adam",
            weight_decay=0.0,
            warmup_steps=0,
        )


@dataclass
class nGPTConfig:
    """Configuration for the normalized Transformer specific parameters.

    See Section 2.6 for the summary of modifications.
    """
    # Eigen learning rates for attention and MLP blocks
    alpha_A_init: float = 0.05       # in order of 1/n_layers
    alpha_A_scale: float = 1.0       # 1/sqrt(d_model) at runtime
    alpha_M_init: float = 0.05
    alpha_M_scale: float = 1.0

    # QK scaling factors (equations 15, 16)
    s_qk_init: float = 1.0
    s_qk_scale: float = 1.0          # 1/sqrt(d_model) at runtime

    # MLP intermediate state scaling (equations 20, 21)
    s_u_init: float = 1.0
    s_u_scale: float = 1.0
    s_v_init: float = 1.0
    s_v_scale: float = 1.0

    # Logit scaling (equation 3)
    s_z_init: float = 1.0
    s_z_scale: float = 1.0           # 1/sqrt(d_model) at runtime

    # QK normalization enable
    qk_norm: bool = True

    # Use LERP (True) or SLERP (False) for hidden state update
    use_lerp: bool = True

    @classmethod
    def default(cls, d_model: int) -> "nGPTConfig":
        """Default initialization as described in Section 2.6."""
        return cls(
            alpha_A_init=0.05,
            alpha_A_scale=1.0 / (d_model ** 0.5),
            alpha_M_init=0.05,
            alpha_M_scale=1.0 / (d_model ** 0.5),
            s_qk_init=1.0,
            s_qk_scale=1.0 / (d_model ** 0.5),
            s_u_init=1.0,
            s_u_scale=1.0,
            s_v_init=1.0,
            s_v_scale=1.0,
            s_z_init=1.0,
            s_z_scale=1.0 / (d_model ** 0.5),
        )


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "openwebtext"
    seq_len: int = 4096
    tokenizer: str = "llama2"
    vocab_size: int = 32000


@dataclass
class TrainConfig:
    """Full training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    ngpt: nGPTConfig = field(default_factory=nGPTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    total_iters: int = 200000
    eval_interval: int = 1000
    log_interval: int = 100
    save_interval: int = 5000
    dtype: str = "bfloat16"
    grad_acc_steps: int = 1
    use_ngpt: bool = True

    def __post_init__(self):
        """Set nGPT config based on d_model after initialization."""
        if isinstance(self.ngpt, nGPTConfig):
            self.ngpt = nGPTConfig.default(self.model.d_model)
