import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class ModelConfig:
    # Architecture
    attention_dim: int = 512
    mlp_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    num_routed_experts: int = 16
    num_shared_experts: int = 2
    top_k: int = 4
    patch_size: int = 8
    spatial_resolution: int = 128

    # Fourier layer
    fourier_modes: int = 16

    # MoE
    expert_kernel_size: int = 3

    # Derived
    total_params_m: float = 0.0
    activated_params_m: float = 0.0


@dataclass
class TrainingConfig:
    # Pre-training
    pretrain_epochs: int = 1000
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 20
    pretrain_warmup_epochs: int = 200
    num_timesteps_in: int = 10
    weight_decay: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.9

    # Fine-tuning
    finetune_epochs: int = 200
    finetune_lr: float = 1e-3
    finetune_warmup_epochs: int = 40

    # Downstream
    downstream_epochs: int = 500
    downstream_lr: float = 1e-3
    downstream_warmup_epochs: int = 100

    # Noise
    noise_eps: float = 1e-4

    # Load balance
    load_balance_weight: float = 0.1

    # Balanced sampling weight per dataset (all equal by default)
    dataset_weights: Dict[str, float] = field(default_factory=lambda: {
        "NS_1e-5": 1.0,
        "NS_1e-3": 1.0,
        "CNS_0.1_0.01": 1.0,
        "SWE": 1.0,
        "DR": 1.0,
        "CFDBench": 1.0,
    })

    # Storage
    save_dir: str = "./checkpoints"
    log_interval: int = 50


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "Tiny": ModelConfig(
        attention_dim=512,
        mlp_dim=512,
        num_layers=4,
        num_heads=4,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        total_params_m=30,
        activated_params_m=17,
    ),
    "Small": ModelConfig(
        attention_dim=1024,
        mlp_dim=1024,
        num_layers=6,
        num_heads=8,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        total_params_m=166,
        activated_params_m=90,
    ),
    "Medium": ModelConfig(
        attention_dim=1024,
        mlp_dim=2048,
        num_layers=8,
        num_heads=8,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        total_params_m=489,
        activated_params_m=288,
    ),
}

PRETRAIN_DATASETS = ["NS_1e-5", "NS_1e-3", "CNS_0.1_0.01", "SWE", "DR", "CFDBench"]

DATASET_SIZES: Dict[str, Tuple[int, int]] = {
    "NS_1e-5": (1000, 200),
    "NS_1e-3": (1000, 200),
    "CNS_0.1_0.01": (9000, 200),
    "SWE": (900, 60),
    "DR": (900, 60),
    "CFDBench": (9000, 1000),
}
