from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# P2VAE configurations
# ---------------------------------------------------------------------------

@dataclass
class P2VAEConfig:
    # Architecture
    in_channels: int = 3           # input physical field channels
    out_channels: int = 3          # reconstructed field channels
    latent_channels: int = 16      # latent space channels (c16p16)
    spatial_in: int = 128          # input spatial resolution
    spatial_latent: int = 16       # latent spatial resolution (8× compression)
    base_dim: int = 64             # base channel dimension (64 for 16M, 128 for 87M)
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)  # channel multipliers per stage
    num_res_blocks: int = 2        # ResBlocks per stage
    attn_resolutions: Tuple[int, ...] = (16,)  # spatial resolutions with attention
    dropout: float = 0.0
    z_channels: int = 16           # latent channels (same as latent_channels)
    double_z: bool = True          # encoder outputs mean + logvar

    # Training
    beta_kl: float = 1e-3          # KL divergence weight
    lr: float = 1e-4               # base learning rate (for batch_size=256)
    beta1: float = 0.9
    beta2: float = 0.995
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1       # fraction of steps for linear warmup
    max_steps: int = 100_000
    batch_size: int = 256
    grad_clip: float = 1.0


@dataclass
class P2VAEConfig16M(P2VAEConfig):
    base_dim: int = 64


@dataclass
class P2VAEConfig87M(P2VAEConfig):
    base_dim: int = 128


# ---------------------------------------------------------------------------
# FMT configurations
# ---------------------------------------------------------------------------

@dataclass
class FMTConfig:
    # Latent dimensions (from P2VAE)
    latent_channels: int = 16      # channels in latent space
    latent_spatial: int = 16       # spatial size of latent (16×16)

    # Temporal pyramid levels: (frame_idx → downsample_factor)
    # Frame 0: Down×8 → 2×2=4 tokens
    # Frame 1: Down×4 → 4×4=16 tokens
    # Frame 2: Down×2 → 8×8=64 tokens
    # Frame 3: full  → 16×16=256 tokens
    pyramid_factors: Tuple[int, ...] = (8, 4, 2, 1)
    n_frames: int = 4              # number of consecutive frames

    # Transformer
    embed_dim: int = 512           # embedding dimension
    depth: int = 12                # number of SiT blocks
    num_heads: int = 8             # attention heads (head_dim = embed_dim / num_heads = 64)
    head_dim: int = 64             # fixed head dimension
    mlp_ratio: float = 4.0        # SwiGLU hidden dim ratio (2/3 * mlp_ratio for SwiGLU)
    dropout: float = 0.0

    # Diffusion forcing GRU
    gru_dim: int = 512             # GRU hidden dim = embed_dim
    n_cross_attn_heads: int = 8    # heads for cross-attention state compression

    # Timestep embedding
    time_embed_dim: int = 256      # sinusoidal time embedding dimension

    # Training
    lr: float = 1e-4               # base learning rate (for batch_size=256)
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    max_steps: int = 100_000
    batch_size: int = 256
    grad_clip: float = 1.0

    # Inference
    n_euler_steps: int = 100       # Euler ODE discretization steps
    dt: float = 0.01               # Euler step size


@dataclass
class FMTConfigSmall(FMTConfig):
    embed_dim: int = 256
    depth: int = 12
    num_heads: int = 4             # 256 / 64 = 4 heads
    gru_dim: int = 256
    n_cross_attn_heads: int = 4
    time_embed_dim: int = 128


@dataclass
class FMTConfigBase(FMTConfig):
    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8             # 512 / 64 = 8 heads
    gru_dim: int = 512
    n_cross_attn_heads: int = 8
    time_embed_dim: int = 256


@dataclass
class FMTConfigLarge(FMTConfig):
    embed_dim: int = 768
    depth: int = 24
    num_heads: int = 12            # 768 / 64 = 12 heads
    gru_dim: int = 768
    n_cross_attn_heads: int = 12
    time_embed_dim: int = 384


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    root_dir: str = "/data/pde"
    spatial_size: int = 128
    n_channels: int = 3
    traj_len: int = 4              # consecutive frames per sample
    dtype: str = "float16"

    # Dataset names and trajectory counts (from paper Appendix A.2)
    datasets: List[str] = field(default_factory=lambda: [
        "fno_v5", "fno_v4", "fno_v3",
        "pa_ns", "pa_nsc", "pa_swe",
        "pb_cns_low", "pb_cns_high", "pb_swe",
        "w_am", "w_gs", "w_swe",
        "w_rb", "w_sf", "w_tr", "w_ve",
    ])

    # Approximate trajectory counts per dataset (from paper)
    traj_counts: dict = field(default_factory=lambda: {
        "fno_v5": 15_400,
        "fno_v4": 368_000,
        "fno_v3": 184_000,
        "pa_ns": 48_000,
        "pa_nsc": 120_000,
        "pa_swe": 470_000,
        "pb_cns_low": 299_000,   # half of 598k (low/high split)
        "pb_cns_high": 299_000,
        "pb_swe": 77_600,
        "w_am": 13_400,
        "w_gs": 92_200,
        "w_swe": 96_400,
        "w_rb": 266_600,
        "w_sf": 175_600,
        "w_tr": 7_000,
        "w_ve": 5_300,
    })

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 8
    pin_memory: bool = True


# ---------------------------------------------------------------------------
# Finetuning configuration (Kolmogorov turbulence)
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    # Dataset
    data_path: str = "/data/kolmogorov"
    n_train_trajs: int = 200
    n_test_trajs: int = 500

    # Training
    max_steps: int = 5_000
    batch_size: int = 32
    lr: float = 1e-4
    lambda_vae: float = 1.0        # weight for VAE loss in joint finetuning
    grad_clip: float = 1.0

    # Stop-gradient on VAE during CFM loss (REPA-E style)
    stop_grad_vae: bool = True


# ---------------------------------------------------------------------------
# Global training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Distributed training
    n_gpus: int = 4                # 4 H-100 GPUs
    mixed_precision: bool = True   # float16 training

    # Logging
    log_every: int = 100
    save_every: int = 5_000
    eval_every: int = 5_000
    wandb_project: str = "fmt-pde"
    checkpoint_dir: str = "checkpoints"

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_p2vae_config(model_size: str) -> P2VAEConfig:
    configs = {
        "16M": P2VAEConfig16M(),
        "87M": P2VAEConfig87M(),
    }
    if model_size not in configs:
        raise ValueError(f"Unknown P2VAE size: {model_size}. Choose from {list(configs)}")
    return configs[model_size]


def get_fmt_config(model_size: str) -> FMTConfig:
    configs = {
        "S": FMTConfigSmall(),
        "B": FMTConfigBase(),
        "L": FMTConfigLarge(),
    }
    if model_size not in configs:
        raise ValueError(f"Unknown FMT size: {model_size}. Choose from {list(configs)}")
    cfg = configs[model_size]
    # Ensure GRU dim matches embed dim
    cfg.gru_dim = cfg.embed_dim
    return cfg
