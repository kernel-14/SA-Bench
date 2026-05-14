from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Architecture
    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 16
    vocab_size: int = 50304
    max_seq_len: int = 4096

    # MoE
    n_experts: int = 64
    n_experts_per_token: int = 8
    ffn_dim: int = 1024  # per-expert FFN hidden dim (SwiGLU intermediate)

    # Normalization
    norm_eps: float = 1e-5
    use_qk_norm: bool = True

    # RoPE
    rope_theta: float = 10000.0

    # Initialization
    init_std: float = 0.02
    init_trunc_factor: float = 3.0  # truncate at init_trunc_factor * init_std

    # Misc
    use_bias: bool = False
    tie_embeddings: bool = False

    # Loss weights
    load_balance_loss_weight: float = 0.01   # alpha
    router_z_loss_weight: float = 0.001      # beta


@dataclass
class TrainConfig:
    # Data
    data_path: str = "data/olmoe_mix"
    tokenizer_name: str = "allenai/gpt-neox-olmo-dolma-v1_5"
    num_workers: int = 4

    # Training duration
    total_tokens: int = 5_133_000_000_000   # 5.133T tokens
    annealing_tokens: int = 100_000_000_000  # 100B tokens for annealing

    # Batch
    seq_len: int = 4096
    batch_size_per_device: int = 1          # samples per device
    gradient_accumulation_steps: int = 1
    global_batch_size_tokens: int = 4_194_304  # ~4M tokens (~1024 samples * 4096)

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 4e-4
    min_lr: float = 4e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # LR schedule
    lr_schedule: str = "cosine"
    warmup_steps: int = 2500

    # Annealing
    annealing_schedule: str = "linear"
    annealing_min_lr: float = 0.0

    # Precision
    dtype: str = "bfloat16"
    grad_reduce_dtype: str = "float32"
    optimizer_state_dtype: str = "float32"

    # Checkpointing
    save_dir: str = "checkpoints"
    save_interval_steps: int = 5000
    resume_from: Optional[str] = None

    # Logging
    log_interval: int = 10
    eval_interval: int = 1000
    wandb_project: str = "olmoe"
    wandb_run_name: str = "olmoe-1b-7b"

    # Hardware
    seed: int = 42


@dataclass
class AdaptConfig:
    # SFT
    sft_data_path: str = "data/sft_mix"
    sft_epochs: int = 2
    sft_lr: float = 2e-5
    sft_batch_size: int = 128
    sft_max_seq_len: int = 4096
    sft_use_load_balance_loss: bool = False  # paper: not used during SFT

    # DPO
    dpo_data_path: str = "data/dpo_mix"
    dpo_epochs: int = 3
    dpo_lr: float = 5e-7
    dpo_batch_size: int = 32
    dpo_beta: float = 0.1
    dpo_use_load_balance_loss: bool = False

    # KTO
    kto_data_path: str = "data/kto_mix"
    kto_steps: int = 5000
    kto_lr: float = 5e-7
    kto_optimizer: str = "rmsprop"
    kto_use_load_balance_loss: bool = False

    # Common
    dtype: str = "bfloat16"
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    seed: int = 42
    save_dir: str = "checkpoints/adapted"

    # Checkpoint to start from (post-annealing)
    base_checkpoint: str = "checkpoints/final"


@dataclass
class OLMoEConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
