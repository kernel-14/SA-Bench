"""Configuration and hyperparameters for the generative PDE foundation model."""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class P2VAEConfig:
    """Pretrained Physics Variational Autoencoder configuration."""
    in_channels: int = 3
    latent_channels: int = 16
    spatial_size: int = 128
    latent_spatial_size: int = 16
    base_dim: int = 64  # 64 for 16M, 128 for 87M
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.0
    kl_weight: float = 1e-3
    z_channels: int = 16


@dataclass
class FMTConfig:
    """Flow Marching Transformer configuration."""
    latent_channels: int = 16
    latent_spatial_size: int = 16
    embed_dim: int = 512  # 256 (S), 512 (B), 768 (L)
    num_heads: int = 8
    head_dim: int = 64
    depth: int = 12
    mlp_ratio: float = 4.0
    max_seq_len: int = 4
    num_frames: int = 4
    pyramid_ratios: Tuple[int, ...] = (8, 4, 2, 1)
    rnn_hidden_dim: int = 512  # same as embed_dim


@dataclass
class FlowMarchingConfig:
    """Flow marching algorithm configuration."""
    num_sampling_steps: int = 100
    dt: float = 0.01
    eps: float = 1e-3
    num_frames: int = 4
    k_pred: float = 1.0  # deterministic prediction
    k_gen: float = 0.0   # fully stochastic generation


@dataclass
class TrainingConfig:
    """Training configuration."""
    # P2VAE training
    p2vae_batch_size: int = 256
    p2vae_lr: float = 1e-4
    p2vae_steps: int = 100_000
    p2vae_warmup_steps: int = 10_000
    p2vae_beta1: float = 0.9
    p2vae_beta2: float = 0.995
    p2vae_weight_decay: float = 1e-4
    p2vae_kl_weight: float = 1e-3

    # FMT training
    fmt_batch_size: int = 256
    fmt_lr: float = 1e-4
    fmt_steps: int = 100_000
    fmt_warmup_steps: int = 10_000
    fmt_beta1: float = 0.9
    fmt_beta2: float = 0.95
    fmt_weight_decay: float = 0.01

    # General
    grad_clip: float = 1.0
    fp16: bool = True
    num_workers: int = 4

    # Fine-tuning
    finetune_steps: int = 5_000
    finetune_lr: float = 1e-5
    finetune_kl_weight: float = 1e-3
    lambda_vae: float = 1.0


@dataclass
class DataConfig:
    """Dataset configuration."""
    spatial_size: int = 128
    in_channels: int = 3
    trajectory_length: int = 5  # 4 transitions need 5 frames (x0..x4)
    precision: str = "float16"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    datasets: List[str] = field(default_factory=lambda: [
        "FNO-v5", "FNO-v4", "FNO-v3",
        "PA-NS", "PA-NSC", "PA-SWE",
        "PB-CNSL", "PB-CNSH", "PB-SWE",
        "W-AM", "W-GS", "W-SWE", "W-RB", "W-SF", "W-TR", "W-VE",
    ])

    data_root: str = "./data"


@dataclass
class Config:
    """Master configuration."""
    p2vae: P2VAEConfig = field(default_factory=P2VAEConfig)
    fmt: FMTConfig = field(default_factory=FMTConfig)
    flow_marching: FlowMarchingConfig = field(default_factory=FlowMarchingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    use_amp: bool = True
    device: str = "cuda"
    seed: int = 42


def get_p2vae_config(size: str) -> P2VAEConfig:
    """Get P2VAE config by model size."""
    if size == "16M":
        return P2VAEConfig(base_dim=64)
    elif size == "87M":
        return P2VAEConfig(base_dim=128)
    else:
        raise ValueError(f"Unknown P2VAE size: {size}")


def get_fmt_config(size: str) -> FMTConfig:
    """Get FMT config by model size."""
    if size == "S":
        return FMTConfig(embed_dim=256, num_heads=4, depth=8,
                         rnn_hidden_dim=256)
    elif size == "B":
        return FMTConfig(embed_dim=512, num_heads=8, depth=12,
                         rnn_hidden_dim=512)
    elif size == "L":
        return FMTConfig(embed_dim=768, num_heads=12, depth=16,
                         rnn_hidden_dim=768)
    else:
        raise ValueError(f"Unknown FMT size: {size}")
