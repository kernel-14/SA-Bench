from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    attention_dim: int = 512
    mlp_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    num_routed_experts: int = 16
    num_shared_experts: int = 2
    top_k: int = 4
    patch_size: int = 8
    in_channels: int = 4       # max channels across datasets (padded)
    out_channels: int = 4
    spatial_size: int = 128    # H = W = 128 after preprocessing
    num_timesteps: int = 10    # T input frames
    load_balance_weight: float = 0.1
    dropout: float = 0.0


@dataclass
class TrainConfig:
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.9

    # Schedule
    num_epochs: int = 1000
    warmup_epochs: int = 200

    # Batch
    batch_size: int = 20       # total across all GPUs
    num_workers: int = 4

    # Noise injection (pre-training only)
    noise_scale: float = 0.01  # epsilon for noise std = epsilon * ||u^{<t}||

    # Data
    spatial_size: int = 128
    patch_size: int = 8
    num_timesteps: int = 10

    # Checkpointing
    save_every: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Multi-GPU
    num_gpus: int = 8
    seed: int = 42


@dataclass
class FinetuneConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.9
    num_epochs: int = 200
    warmup_epochs: int = 40
    batch_size: int = 20
    num_workers: int = 4
    freeze_router: bool = True   # freeze router-gating network during fine-tuning
    checkpoint_dir: str = "checkpoints_finetune"


@dataclass
class DownstreamConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.9
    num_epochs: int = 500
    warmup_epochs: int = 100
    batch_size: int = 20
    num_workers: int = 4
    freeze_router: bool = True
    checkpoint_dir: str = "checkpoints_downstream"


@dataclass
class DatasetConfig:
    name: str = ""
    path: str = ""
    train_size: int = 1000
    test_size: int = 200
    num_channels: int = 1
    spatial_size: int = 128
    num_timesteps: int = 20    # total frames available
    weight: float = 1.0        # importance weight for balanced sampling


# Pre-defined dataset configurations matching the paper
DATASET_CONFIGS = {
    "fno_ns_1e5": DatasetConfig(
        name="fno_ns_1e5",
        train_size=1000,
        test_size=200,
        num_channels=1,
        spatial_size=64,
        num_timesteps=20,
        weight=1.0,
    ),
    "fno_ns_1e3": DatasetConfig(
        name="fno_ns_1e3",
        train_size=1000,
        test_size=200,
        num_channels=1,
        spatial_size=64,
        num_timesteps=20,
        weight=1.0,
    ),
    "pdebench_cns": DatasetConfig(
        name="pdebench_cns",
        train_size=9000,
        test_size=200,
        num_channels=4,   # rho, u_x, u_y, p
        spatial_size=128,
        num_timesteps=21,
        weight=1.0,
    ),
    "pdebench_swe": DatasetConfig(
        name="pdebench_swe",
        train_size=900,
        test_size=60,
        num_channels=1,   # h
        spatial_size=128,
        num_timesteps=101,
        weight=1.0,
    ),
    "pdebench_dr": DatasetConfig(
        name="pdebench_dr",
        train_size=900,
        test_size=60,
        num_channels=2,   # u, v
        spatial_size=128,
        num_timesteps=101,
        weight=1.0,
    ),
    "cfdbench": DatasetConfig(
        name="cfdbench",
        train_size=9000,
        test_size=1000,
        num_channels=3,   # u_x, u_y, p
        spatial_size=128,
        num_timesteps=20,
        weight=1.0,
    ),
}

# Downstream task dataset configs
DOWNSTREAM_CONFIGS = {
    "fno_ns_1e4": DatasetConfig(
        name="fno_ns_1e4",
        train_size=2000,
        test_size=200,
        num_channels=1,
        spatial_size=64,
        num_timesteps=20,
        weight=1.0,
    ),
    "pdebench_cns_1_001": DatasetConfig(
        name="pdebench_cns_1_001",
        train_size=2000,
        test_size=200,
        num_channels=4,
        spatial_size=128,
        num_timesteps=21,
        weight=1.0,
    ),
    "pdearena": DatasetConfig(
        name="pdearena",
        train_size=2000,
        test_size=200,
        num_channels=3,
        spatial_size=64,
        num_timesteps=56,
        weight=1.0,
    ),
}


def get_model_config(size: str) -> ModelConfig:
    """Return model config for a given size (tiny, small, medium)."""
    configs = {
        "tiny": ModelConfig(
            attention_dim=512,
            mlp_dim=512,
            num_layers=4,
            num_heads=4,
            num_routed_experts=16,
            num_shared_experts=2,
            top_k=4,
            patch_size=8,
        ),
        "small": ModelConfig(
            attention_dim=1024,
            mlp_dim=1024,
            num_layers=6,
            num_heads=8,
            num_routed_experts=16,
            num_shared_experts=2,
            top_k=4,
            patch_size=8,
        ),
        "medium": ModelConfig(
            attention_dim=1024,
            mlp_dim=2048,
            num_layers=8,
            num_heads=8,
            num_routed_experts=16,
            num_shared_experts=2,
            top_k=4,
            patch_size=8,
        ),
    }
    size = size.lower()
    if size not in configs:
        raise ValueError(f"Unknown model size '{size}'. Choose from: {list(configs.keys())}")
    return configs[size]
